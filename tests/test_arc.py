"""CPU integration checks for graph sampling and the two-stage IB pipeline.

Run from the project directory with:
    python -m unittest discover -s tests -p 'test_arc.py' -v
"""
import io
import inspect
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from modules.arc import (
    ContextEncoder,
    NeighborSampler,
    ARC,
)
from utils.ib_data import load_dataset
from main import evaluate


class FixedGraph:
    """Small hand-specified sampled adjacency, including truly empty rows."""

    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.offsets = np.zeros(len(adjacency) + 1, dtype=np.int64)

    def sample_table(self, fanout, rng):
        width = min(fanout, max((len(row) for row in self.adjacency), default=0))
        neighbors = np.zeros((len(self.adjacency), width), dtype=np.int64)
        relations = np.full_like(neighbors, -1)
        valid = np.zeros_like(neighbors, dtype=bool)
        for node, row in enumerate(self.adjacency):
            for slot, (neighbor, relation) in enumerate(row[:width]):
                neighbors[node, slot] = neighbor
                relations[node, slot] = relation
                valid[node, slot] = True
        return neighbors, relations, valid


class DiagnosticLayer(nn.Module):
    """An exact arithmetic oracle makes wrong graph-layer indexing visible."""

    def __init__(self, self_scale):
        super().__init__()
        self.self_scale = self_scale

    def forward(self, e_self, e_source, target_index, e_relation=None,
                e_context=None, noise=True):
        messages = e_source if e_relation is None else e_source * e_relation
        aggregate = torch.zeros_like(e_self)
        aggregate.index_add_(0, target_index, messages)
        return self.self_scale * e_self + aggregate, e_self.sum() * 0.0


def tiny_graphs():
    # Users 0,1; item/entity nodes 2,3,4; extra entity node 5.
    # Item 4 is isolated: padding must not introduce an edge to user 0.
    ui = FixedGraph([[(2, -1)], [(3, -1)], [(0, -1)], [(1, -1)], []])
    joint = FixedGraph([
        [(2, -1)], [(3, -1)], [(0, -1), (5, 0)],
        [(1, -1), (5, 1)], [], [(2, 0), (3, 1)],
    ])
    return ui, joint


def make_model(context=None, layers=2):
    if context is None:
        context = torch.randn(2, 8)
    return ARC(
        n_users=2, n_items=3, n_entities=4, n_relations=2,
        rho=np.array([0.6, 0.4]), context=context,
        dim=8, layers=layers, n_codes=2, n_blocks=4, keep_blocks=2,
    )


class GraphIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_two_hop_blocks_preserve_layer_order_duplicates_and_isolates(self):
        graph = FixedGraph([[(1, -1)], [(2, -1)], [(3, -1)], [], []])
        sampler = NeighborSampler(graph, fanout=2, seed=4)
        encoder = ContextEncoder(5, 1, 2)
        encoder.layers = nn.ModuleList([DiagnosticLayer(10), DiagnosticLayer(100)])
        with torch.no_grad():
            encoder.e_node.weight.copy_(torch.arange(1.0, 6.0).unsqueeze(1))
        actual, rate = encoder.encode(np.array([0, 4, 0, 1]), sampler, noise=False)
        # 100 * (10*x_v + x_neighbor) + (10*x_neighbor + x_2hop).
        expected = torch.tensor([[1223.0], [5000.0], [1223.0], [2334.0]])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(rate.item(), 0.0)

    def test_personalized_query_indices_relation_routing_and_isolated_nodes(self):
        _, graph = tiny_graphs()
        sampler = NeighborSampler(graph, fanout=3, seed=5)
        model = make_model(layers=1)
        model.layers = nn.ModuleList([DiagnosticLayer(10)])
        with torch.no_grad():
            model.e_node.weight.copy_(torch.arange(1.0, 7.0)[:, None].expand(-1, 8))
            model.e_ui.fill_(2.0)
        # State follows query_users order, which deliberately differs from IDs.
        state = {"code": torch.tensor([[[3.0] * 8, [4.0] * 8],
                                       [[5.0] * 8, [6.0] * 8]])}
        actual, _, _ = model.encode(
            np.array([1, 0]), np.array([0, 1, 0, 1]),
            np.array([2, 2, 4, 3]), sampler,
            sample=False, noise=False, relation_state=state,
        )
        # Item 2 receives the shared UI code from user 0 and relation 0 from 5.
        expected = torch.tensor([50.0, 62.0, 50.0, 80.0])[:, None].expand(-1, 8)
        torch.testing.assert_close(actual, expected)

    def test_gaussian_encoder_handles_fully_edgeless_batch(self):
        graph = FixedGraph([[], [], []])
        sampler = NeighborSampler(graph, fanout=2, seed=3)
        encoder = ContextEncoder(3, 8, 2)
        actual, rate = encoder.encode(np.array([0, 2]), sampler, noise=True)
        self.assertEqual(tuple(actual.shape), (2, 8))
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertEqual(rate.item(), 0.0)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(encoder.e_node.weight.grad).all().item())

    def test_two_stage_gradient_flow_and_frozen_context(self):
        ui, joint = tiny_graphs()
        ctx_sampler = NeighborSampler(ui, fanout=3, seed=7)
        sampler = NeighborSampler(joint, fanout=3, seed=8)
        users, positives, negatives = (np.array([0, 1, 0]),
                                       np.array([0, 1, 0]), np.array([1, 2, 2]))
        encoder = ContextEncoder(5, 8, 2)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
        before = encoder.e_node.weight.detach().clone()
        positive, negative, rate = encoder.pair_scores(
            users, positives, negatives, 2, ctx_sampler)
        loss = -F.logsigmoid(positive - negative).mean() + 0.01 * rate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, encoder.e_node.weight.detach()))
        context = encoder.cache_context(2, ctx_sampler, samples=2, batch_size=1)
        self.assertFalse(context.requires_grad)
        context_before = context.clone()
        encoder_before = {name: parameter.detach().clone()
                          for name, parameter in encoder.named_parameters()}
        model = make_model(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        positive, negative, rate, rel_rate = model.pair_scores(
            users, positives, negatives, sampler)
        self.assertEqual(tuple(positive.shape), (3,))
        self.assertGreaterEqual(rel_rate.item(), -1e-6)
        loss = -F.logsigmoid(positive - negative).mean() + 0.01 * rate + rel_rate
        self.assertTrue(torch.isfinite(loss).item())
        optimizer.zero_grad()
        loss.backward()
        self.assertIsNotNone(model.e_node.weight.grad)
        gradients = [parameter.grad for parameter in model.relation.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
        optimizer.step()
        torch.testing.assert_close(model.e_context, context_before, rtol=0, atol=0)
        self.assertNotIn("e_context", dict(model.named_parameters()))
        for name, parameter in encoder.named_parameters():
            torch.testing.assert_close(parameter, encoder_before[name], rtol=0, atol=0)

    def test_checkpoint_roundtrip_includes_context_and_relation_prior(self):
        _, graph = tiny_graphs()
        sampler = NeighborSampler(graph, fanout=3, seed=9)
        original = make_model()
        original.eval()
        expected = original.score_all_items(1, sampler, samples=1, stochastic=False)
        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        restored = make_model(torch.zeros(2, 8))
        options = ({"weights_only": True}
                   if "weights_only" in inspect.signature(torch.load).parameters else {})
        restored.load_state_dict(torch.load(buffer, map_location="cpu", **options))
        restored.eval()
        actual = restored.score_all_items(1, sampler, samples=1, stochastic=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(restored.e_context, original.e_context, rtol=0, atol=0)
        torch.testing.assert_close(restored.relation.rho, original.relation.rho, rtol=0, atol=0)


class DataIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        directory = self.root / "tiny"
        directory.mkdir()
        # Non-contiguous raw IDs, one zero label, a test-only positive, a
        # catalog-only item, and duplicate KG triples exercise real loading.
        np.save(directory / "train_data.npy", np.array([
            [10, 100, 1], [10, 200, 1], [10, 300, 1],
            [20, 200, 1], [20, 400, 0],
        ], dtype=np.int64))
        np.save(directory / "test_data.npy", np.array([
            [10, 400, 1], [20, 300, 1], [20, 100, 0],
        ], dtype=np.int64))
        np.save(directory / "kg_final.npy", np.array([
            [100, 7, 900], [200, 7, 900], [300, 11, 901], [100, 7, 900],
        ], dtype=np.int64))
        np.savetxt(directory / "item_index2entity_id.txt", np.array([
            [51, 100], [52, 200], [53, 300], [54, 400], [55, 500],
        ]), fmt="%d")
        # This unrelated source must never replace the provided split.
        np.save(directory / "ratings_final.npy", np.array([[999, 999, 1]]))

    def test_source_split_label_filter_holdout_and_catalog_alignment(self):
        data = load_dataset(self.root, "tiny", seed=17, val_ratio=0.34, inverse=True)
        self.assertEqual((data.n_users, data.n_items, data.n_entities), (2, 5, 7))
        self.assertEqual(data.n_relations, 4)
        np.testing.assert_array_equal(data.item_ids, [100, 200, 300, 400, 500])
        np.testing.assert_allclose(data.rho, [1/3, 1/6, 1/3, 1/6])
        self.assertEqual(len(data.train_pairs), 3)
        self.assertEqual(len(data.valid_pairs), 1)
        self.assertEqual(data.test_user_items, {0: {3}, 1: {2}})
        self.assertEqual(data.train_user_items[1], {1})
        self.assertEqual(data.ui_graph.n_edges, 2 * len(data.train_pairs))
        for user in range(data.n_users):
            start, end = data.ui_graph.offsets[user:user+2]
            graph_items = set((data.ui_graph.neighbors[start:end] - data.n_users).tolist())
            self.assertEqual(graph_items, data.train_user_items.get(user, set()))
            self.assertFalse(graph_items & data.valid_user_items.get(user, set()))
            self.assertFalse(graph_items & data.test_user_items.get(user, set()))
        repeated = load_dataset(self.root, "tiny", seed=17, val_ratio=0.34, inverse=True)
        self.assertEqual(data.fingerprint, repeated.fingerprint)
        np.testing.assert_array_equal(data.train_pairs, repeated.train_pairs)

    def test_negatives_belong_to_same_user_and_do_not_consult_heldout_labels(self):
        data = load_dataset(self.root, "tiny", seed=17, val_ratio=0.34, inverse=False)
        before = data.sample_triples(512, np.random.default_rng(13))
        for user, positive, negative in zip(*before):
            self.assertIn(int(positive), data.train_user_items[int(user)])
            self.assertNotIn(int(negative), data.train_user_items[int(user)])
            self.assertTrue(0 <= negative < data.n_items)
        # Altering held-out labels must not change training samples.
        data.valid_user_items = {0: set(range(data.n_items))}
        data.test_user_items = {1: set(range(data.n_items))}
        after = data.sample_triples(512, np.random.default_rng(13))
        for expected, actual in zip(before, after):
            np.testing.assert_array_equal(actual, expected)

    def test_real_csr_graph_runs_both_encoders_after_remapping(self):
        data = load_dataset(self.root, "tiny", seed=17, val_ratio=0.34, inverse=True)
        torch.manual_seed(11)
        ctx_sampler = NeighborSampler(data.ui_graph, fanout=2, seed=11)
        sampler = NeighborSampler(data.joint_graph, fanout=2, seed=11)
        encoder = ContextEncoder(data.n_users + data.n_items, 8, 2)
        context = encoder.cache_context(data.n_users, ctx_sampler, samples=1)
        model = ARC(
            data.n_users, data.n_items, data.n_entities, data.n_relations,
            data.rho, context, dim=8, layers=2, n_codes=2,
            n_blocks=4, keep_blocks=2,
        )
        triples = data.sample_triples(4, np.random.default_rng(18))
        values = model.pair_scores(*triples, sampler)
        self.assertTrue(all(torch.isfinite(value).all().item() for value in values))
        scores = model.score_all_items(0, sampler, samples=1, stochastic=False)
        self.assertEqual(tuple(scores.shape), (data.n_items,))
        self.assertTrue(torch.isfinite(scores).all().item())


class RankingOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_node = nn.Embedding(1, 1)
        self.calls = []

    def score_all_items(self, user, sampler, samples=2):
        self.calls.append((user, samples))
        # Consume RNG as the real stochastic encoder does.
        torch.rand(3)
        # The first two (seen) items deliberately dominate both held-out items.
        return torch.tensor([100.0, 90.0, 1.0, 1.0, 0.0])


class RankingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            n_items=5,
            train_user_items={0: {0}},
            valid_user_items={0: {1}},
            test_user_items={0: {2}},
        )

    def test_full_catalog_filters_training_and_validation_positives(self):
        model = RankingOracle()
        validation = evaluate(model, self.data, None, split="valid", ks=(1, 3), samples=2)
        test = evaluate(model, self.data, None, split="test", ks=(1, 3), samples=2)
        # Validation excludes item 0; test additionally excludes item 1.
        # Equal scores of items 2/3 must be broken deterministically by item ID.
        for metrics in (validation, test):
            self.assertEqual(metrics["recall@1"], 1.0)
            self.assertEqual(metrics["ndcg@1"], 1.0)
            self.assertEqual(metrics["precision@3"], 1.0 / 3.0)
            self.assertTrue(metrics["full_catalog"])
            self.assertFalse(metrics["user_subset"])
            self.assertEqual(metrics["users"], 1)
        self.assertEqual(model.calls, [(0, 2), (0, 2)])

    def test_evaluation_restores_training_rng_and_rejects_empty_split(self):
        model = RankingOracle()
        torch.manual_seed(731)
        before = torch.random.get_rng_state().clone()
        evaluate(model, self.data, None, split="test", ks=(1,), samples=1)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.data.valid_user_items = {}
        with self.assertRaisesRegex(ValueError, "No eligible users"):
            evaluate(model, self.data, None, split="valid", ks=(1,), samples=1)


if __name__ == "__main__":
    unittest.main()
