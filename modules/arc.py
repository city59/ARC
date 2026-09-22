"""ARC: query-personalized encoders for two-stage relation bottlenecks.

Only ordinary PyTorch operations are used; torch_scatter / PyG are not needed.
Each computation graph deduplicates (query user, graph node) at every depth.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from modules.ib_layers import GaussianMessageLayer, RelationBottleneck


@dataclass
class Block:
    targets: np.ndarray
    sources: np.ndarray
    self_index: np.ndarray
    target_index: np.ndarray
    source_index: np.ndarray
    relations: np.ndarray


class NeighborSampler:
    """Full CSR propagation by default; optional sampled diagnostic adjacency.

    ``fanout=0`` uses every directed edge at every propagation layer, including
    in the KL sum. Positive fanout opts into a sampled-neighborhood approximation
    and normalizes by sampled degrees. No padded slot is treated as an edge.
    """
    def __init__(self, graph, fanout, seed):
        self.n_nodes = len(graph.offsets) - 1
        if fanout < 0:
            raise ValueError("fanout must be nonnegative (0 means the full graph)")
        self.full_graph = fanout == 0
        if self.full_graph:
            self.rows = np.repeat(np.arange(self.n_nodes), np.diff(graph.offsets))
            self.neighbors = graph.neighbors
            self.relations = graph.relations
        else:
            self.neighbors, self.relations, self.valid = graph.sample_table(
                fanout, np.random.default_rng(seed))

    def blocks(self, queries, nodes, depth):
        queries = np.asarray(queries, dtype=np.int64)
        nodes = np.asarray(nodes, dtype=np.int64)
        if queries.shape != nodes.shape or nodes.ndim != 1:
            raise ValueError("queries and nodes must be equal-length vectors")
        if depth < 1 or (queries < 0).any():
            raise ValueError("depth must be positive and query indices nonnegative")
        if len(nodes) == 0 or (nodes < 0).any() or (nodes >= self.n_nodes).any():
            raise ValueError("roots must contain valid graph node IDs")
        if self.full_graph:
            # All nodes are needed for the exact all-edge information cost,
            # even when the ranking batch only reads a small subset of roots.
            query_ids = np.unique(queries)
            keys = (query_ids[:, None] * self.n_nodes
                    + np.arange(self.n_nodes)[None, :]).reshape(-1)
            offsets = np.arange(len(query_ids))[:, None] * self.n_nodes
            rows = (offsets + self.rows[None, :]).reshape(-1)
            neighbors = (offsets + self.neighbors[None, :]).reshape(-1)
            indices = np.arange(len(keys))
            block = Block(keys, keys, indices, rows, neighbors,
                          np.tile(self.relations, len(query_ids)))
            inverse = np.searchsorted(keys, queries * self.n_nodes + nodes)
            return [block] * depth, keys, inverse
        keys, inverse = np.unique(queries * self.n_nodes + nodes,
                                  return_inverse=True)
        blocks = []
        for _ in range(depth):
            node_ids = keys % self.n_nodes
            rows, slots = np.nonzero(self.valid[node_ids])
            neighbor_keys = ((keys[rows] // self.n_nodes) * self.n_nodes
                             + self.neighbors[node_ids[rows], slots])
            sources = np.union1d(keys, neighbor_keys)
            blocks.append(Block(
                keys, sources, np.searchsorted(sources, keys), rows,
                np.searchsorted(sources, neighbor_keys),
                self.relations[node_ids[rows], slots]))
            keys = sources
        return blocks, keys, inverse


def as_index(value, device):
    return torch.as_tensor(value, dtype=torch.long, device=device)


class ContextEncoder(nn.Module):
    def __init__(self, n_nodes, dim, layers):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.e_node = nn.Embedding(n_nodes, dim)
        nn.init.xavier_uniform_(self.e_node.weight)
        self.layers = nn.ModuleList([
            GaussianMessageLayer(dim, personalized=False) for _ in range(layers)])

    def encode(self, nodes, sampler, noise=True):
        blocks, leaves, inverse = sampler.blocks(
            np.zeros(len(nodes), dtype=np.int64), nodes, len(self.layers))
        device = self.e_node.weight.device
        e_node = self.e_node(as_index(leaves % sampler.n_nodes, device))
        e_output = e_node
        rate = e_node.sum() * 0.0
        for layer, block in zip(self.layers, reversed(blocks)):
            self_index = as_index(block.self_index, device)
            e_node, layer_rate = layer(
                e_node[self_index],
                e_node[as_index(block.source_index, device)],
                as_index(block.target_index, device), noise=noise)
            e_output = e_output[self_index] + e_node
            rate = rate + layer_rate
        return e_output[as_index(inverse, device)], rate

    def pair_scores(self, users, positives, negatives, n_users, sampler):
        nodes = np.concatenate((users, n_users + positives, n_users + negatives))
        e_node, rate = self.encode(nodes, sampler)
        e_user, e_pos, e_neg = e_node.chunk(3)
        return (e_user * e_pos).sum(-1), (e_user * e_neg).sum(-1), rate

    @torch.no_grad()
    def cache_context(self, n_users, sampler, samples=2, batch_size=512):
        if samples < 1 or batch_size < 1:
            raise ValueError("samples and batch_size must be positive")
        self.eval()
        result = self.e_node.weight.new_zeros((n_users, self.e_node.embedding_dim))
        # Full propagation already computes every node. Do it once per noise
        # draw rather than repeating the complete graph for each cache chunk.
        if sampler.full_graph:
            batch_size = n_users
        for _ in range(samples):
            for start in range(0, n_users, batch_size):
                nodes = np.arange(start, min(start + batch_size, n_users))
                e_node, _ = self.encode(nodes, sampler, noise=True)
                result[start:start + len(nodes)] += e_node / samples
        return result.detach()


class ARC(nn.Module):
    model_name = "ARC"

    def __init__(self, n_users, n_items, n_entities, n_relations, rho,
                 context, dim=64, layers=2, n_codes=4, n_blocks=4,
                 keep_blocks=2, mask_temperature=1.0, code_temperature=1.0,
                 gumbel_temperature=1.0, relation_context='personalized',
                 mask_mode='learned', mask_seed=2024):
        super().__init__()
        if layers < 1 or tuple(context.shape) != (n_users, dim):
            raise ValueError("layers must be positive; context shape must be [n_users, dim]")
        self.n_users, self.n_items = n_users, n_items
        self.n_nodes = n_users + n_entities
        if relation_context not in ('personalized', 'global'):
            raise ValueError("relation_context must be personalized or global")
        self.relation_context = relation_context
        self.register_buffer("e_context", context.detach().clone())
        self.e_node = nn.Embedding(self.n_nodes, dim)
        nn.init.xavier_uniform_(self.e_node.weight)
        self.e_ui = nn.Parameter(torch.empty(dim))
        nn.init.xavier_uniform_(self.e_ui.unsqueeze(0))
        self.relation = RelationBottleneck(
            n_relations, dim, n_codes, n_blocks, keep_blocks, rho,
            mask_temperature=mask_temperature, code_temperature=code_temperature,
            gumbel_temperature=gumbel_temperature, mask_mode=mask_mode,
            mask_seed=mask_seed)
        self.layers = nn.ModuleList([
            GaussianMessageLayer(dim, personalized=True) for _ in range(layers)])

    def relation_contexts(self, query_users):
        """Global ablation changes only the input to the relation encoder."""
        if self.relation_context == 'global':
            return self.e_context.mean(0, keepdim=True).expand(len(query_users), -1)
        return self.e_context[as_index(query_users, self.e_context.device)]

    def encode(self, query_users, root_queries, root_nodes, sampler,
               sample=True, noise=True, relation_state=None, query_weights=None):
        if sampler.n_nodes != self.n_nodes:
            raise ValueError("Joint sampler node universe differs from model")
        if len(query_users) == 0 or (np.asarray(query_users) < 0).any() or (
                np.asarray(query_users) >= self.n_users).any():
            raise ValueError("Invalid conditioning user IDs")
        if (np.asarray(root_queries) >= len(query_users)).any():
            raise ValueError("Root query index is outside conditioning user batch")
        if len(np.unique(query_users)) != len(query_users):
            raise ValueError("query_users must be unique so relation samples are shared")
        device = self.e_node.weight.device
        e_context = self.relation_contexts(query_users)
        state = (self.relation(e_context, sample=sample,
                               user_ids=as_index(query_users, device))
                 if relation_state is None else relation_state)
        if query_weights is None:
            query_weights = e_context.new_full((len(query_users),), 1.0 / len(query_users))
        else:
            query_weights = torch.as_tensor(query_weights, dtype=e_context.dtype, device=device)
            if query_weights.shape != (len(query_users),) or (query_weights < 0).any():
                raise ValueError("query_weights must be nonnegative and match query_users")
            if not torch.isfinite(query_weights).all() or not torch.isclose(
                    query_weights.sum(), query_weights.new_tensor(1.0)):
                raise ValueError("query_weights must be finite and sum to one")
        blocks, leaves, inverse = sampler.blocks(
            root_queries, root_nodes, len(self.layers))
        e_node = self.e_node(as_index(leaves % self.n_nodes, device))
        e_output = e_node
        rate = e_node.sum() * 0.0
        for layer, block in zip(self.layers, reversed(blocks)):
            target_queries = as_index(block.targets // self.n_nodes, device)
            target_index = as_index(block.target_index, device)
            relations = as_index(block.relations, device)
            edge_queries = target_queries[target_index]
            # UI uses its own code. Original relation embeddings never bypass Z.
            e_relation = state["code"][edge_queries, relations.clamp_min(0)]
            e_relation = torch.where((relations >= 0).unsqueeze(-1),
                                     e_relation, self.e_ui.unsqueeze(0))
            self_index = as_index(block.self_index, device)
            e_node, layer_rate = layer(
                e_node[self_index],
                e_node[as_index(block.source_index, device)], target_index,
                e_relation=e_relation, noise=noise,
                edge_weights=query_weights[edge_queries])
            e_output = e_output[self_index] + e_node
            rate = rate + layer_rate
        return e_output[as_index(inverse, device)], rate, state

    def pair_scores(self, users, positives, negatives, sampler):
        query_users, query_index = np.unique(users, return_inverse=True)
        root_queries = np.tile(query_index, 3)
        root_nodes = np.concatenate((users, self.n_users + positives,
                                     self.n_users + negatives))
        e_node, rate, state = self.encode(
            query_users, root_queries, root_nodes, sampler,
            query_weights=np.bincount(query_index) / len(users))
        e_user, e_pos, e_neg = e_node.chunk(3)
        pos = (e_user * e_pos).sum(-1)
        neg = (e_user * e_neg).sum(-1)
        # Uniform user sampling is used by the trainer. Weight repeated users
        # by their multiplicities instead of silently averaging only uniques.
        relation_rate = state["rate_per_user"][as_index(query_index, e_node.device)].mean()
        return pos, neg, rate, relation_rate

    @torch.no_grad()
    def score_all_items(self, user, sampler, samples=2, stochastic=True):
        if samples < 1:
            raise ValueError("samples must be positive")
        nodes = np.concatenate(([user], self.n_users + np.arange(self.n_items)))
        queries = np.zeros(len(nodes), dtype=np.int64)
        scores = self.e_node.weight.new_zeros(self.n_items)
        for _ in range(samples):
            e_node, _, _ = self.encode(np.array([user]), queries, nodes, sampler,
                                       sample=stochastic, noise=stochastic)
            scores += (e_node[:1] * e_node[1:]).sum(-1) / samples
        return scores


def parameter_penalty(model):
    return sum(parameter.square().sum() for parameter in model.parameters()
               if parameter.requires_grad)
