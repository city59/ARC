"""Information-limited graph messages and personalized discrete relation codes.

Hard masks and hard categorical codes are used in the forward computation.
Their straight-through gradients are biased surrogate gradients.  In particular,
neither a soft codebook average nor an unmasked relation embedding is returned as
the relation representation used for graph propagation.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GaussianMessageLayer(nn.Module):
    """A graph layer with an independently noised Gaussian channel per edge.

    ``target_index`` maps each incoming edge to a row of ``e_target``.  The
    source tensor already contains the source state for each edge; this permits
    sampled bipartite blocks without constructing a dense adjacency matrix.
    The returned rate is the sum of exact Gaussian KL costs over incoming
    edges, as in Eqs. (3) and (7). Optional ``edge_weights`` affect this cost
    only, allowing callers to form expectations over target-user graphs.

    ``noise=False`` replaces messages by their conditional means.  This is an
    explicit deterministic inference approximation, not Monte Carlo inference.
    ``noise_values`` can supply the standard-normal draws when ``noise=True``.
    """

    def __init__(self, dim: int, personalized: bool = False) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.personalized = personalized
        self.eps = 1e-8
        self.message_projection = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        e_target: Tensor,
        e_source: Tensor,
        target_index: Tensor,
        e_relation: Optional[Tensor] = None,
        e_context: Optional[Tensor] = None,
        noise: bool = True,
        noise_values: Optional[Tensor] = None,
        edge_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if e_target.ndim != 2 or e_target.shape[1] != self.dim:
            raise ValueError("e_target must have shape [nodes, dim]")
        if e_source.ndim != 2 or e_source.shape[1] != self.dim:
            raise ValueError("e_source must have shape [edges, dim]")
        if target_index.ndim != 1 or target_index.numel() != e_source.shape[0]:
            raise ValueError("target_index must have one index per edge")
        if target_index.dtype != torch.long:
            raise ValueError("target_index must have dtype torch.long")
        if e_relation is not None and e_relation.shape != e_source.shape:
            raise ValueError("e_relation must have shape [edges, dim]")
        # Kept for compatibility with older callers; context conditions the
        # relation encoder, not the message gate in Eq. (1).
        if e_context is not None and e_context.shape != e_target.shape:
            raise ValueError("e_context, when supplied, must have shape [nodes, dim]")
        if edge_weights is not None:
            if edge_weights.shape != (e_source.shape[0],):
                raise ValueError("edge_weights must have shape [edges]")
            if not torch.isfinite(edge_weights).all() or (edge_weights < 0).any():
                raise ValueError("edge_weights must be finite and nonnegative")
        if noise_values is not None:
            if not noise:
                raise ValueError("noise_values requires noise=True")
            if noise_values.shape != e_source.shape:
                raise ValueError("noise_values must have shape [edges, dim]")

        e_aggregate = e_target.new_zeros(e_target.shape)
        if e_source.shape[0] == 0:
            # Keep rate.backward() valid even for an entirely empty block.
            rate = sum(parameter.sum() * 0.0 for parameter in self.parameters())
            return e_target, rate

        e_sender = e_source
        e_receiver = e_target.index_select(0, target_index)
        if e_relation is not None:
            e_sender = e_sender * e_relation
            e_receiver = e_receiver * e_relation
        # Clip the learned signal to the radius-sqrt(d) ball without expanding
        # small signals. The gate sees the endpoints, not the projected signal.
        e_signal = self.message_projection(e_sender)
        scale = (math.sqrt(self.dim) / e_signal.norm(
            p=2, dim=-1, keepdim=True
        ).clamp_min(self.eps)).clamp_max(1.0)
        e_raw = e_signal * scale
        gate_input = torch.cat([e_sender, e_receiver], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input)).clamp(1e-6, 1.0 - 1e-6)

        e_message = alpha.sqrt() * e_raw
        if noise:
            e_noise = torch.randn_like(e_message) if noise_values is None else noise_values
            e_message = e_message + (1.0 - alpha).sqrt() * e_noise

        e_aggregate.index_add_(0, target_index, e_message)
        degree = e_target.new_zeros(e_target.shape[0])
        degree.index_add_(0, target_index, e_target.new_ones(target_index.shape[0]))
        e_aggregate = e_aggregate / degree.clamp_min(1.0).sqrt().unsqueeze(-1)
        # Eq. (2) keeps the unmodulated receiver as an identity residual.
        e_next = e_target + e_aggregate

        # KL(N(sqrt(alpha)*e_raw, (1-alpha)I) || N(0,I)).  Retain
        # the norm term: exact zero and near-zero signals need not have norm d.
        norm_squared = e_raw.square().sum(dim=-1, keepdim=True)
        edge_rate = 0.5 * (
            alpha * (norm_squared - self.dim)
            - self.dim * torch.log1p(-alpha)
        )
        edge_rate = edge_rate.squeeze(-1)
        if edge_weights is not None:
            edge_rate = edge_rate * edge_weights
        return e_next, edge_rate.sum()


class RelationBottleneck(nn.Module):
    """User-conditioned block masking and a discrete relation bottleneck.

    One call computes all relation distributions for each *unique* user in the
    supplied context batch.  Callers index and reuse these sampled codes across
    graph edges and candidates, rather than resampling for each occurrence.
    ``sample=False`` returns MAP codes, a deterministic inference approximation.
    Relation information uses the complete relation marginal and differentiates
    through that marginal; it is not KL to a uniform output-code prior.
    """

    def __init__(
        self,
        n_relations: int,
        dim: int,
        n_codes: int,
        n_blocks: int,
        keep_blocks: int,
        rho: Tensor,
        mask_temperature: float = 1.0,
        code_temperature: float = 1.0,
        gumbel_temperature: float = 1.0,
        mask_mode: str = "learned",
        mask_seed: int = 2024,
    ) -> None:
        super().__init__()
        if n_relations <= 0 or dim <= 0:
            raise ValueError("n_relations and dim must be positive")
        if n_codes < 1:
            raise ValueError("n_codes must be at least one")
        if n_blocks < 1 or dim % n_blocks:
            raise ValueError("n_blocks must be positive and divide dim")
        if not 0 < keep_blocks <= n_blocks:
            raise ValueError("keep_blocks must satisfy 0 < keep_blocks <= n_blocks")
        if mask_mode not in {"learned", "all", "random"}:
            raise ValueError("mask_mode must be learned, all, or random")
        if mask_mode != "all" and keep_blocks == n_blocks:
            raise ValueError("learned/random masks require keep_blocks < n_blocks")
        if min(mask_temperature, code_temperature, gumbel_temperature) <= 0:
            raise ValueError("all temperatures must be positive")
        rho_tensor = torch.as_tensor(rho, dtype=torch.float32).detach().clone()
        if rho_tensor.shape != (n_relations,):
            raise ValueError("rho must have one entry per relation")
        if not torch.isfinite(rho_tensor).all() or (rho_tensor < 0).any():
            raise ValueError("rho must be finite and nonnegative")
        if rho_tensor.sum() <= 0:
            raise ValueError("rho must have positive total mass")
        self.register_buffer("rho", rho_tensor / rho_tensor.sum())
        self.n_relations = n_relations
        self.dim = dim
        self.n_codes = n_codes
        self.n_blocks = n_blocks
        self.keep_blocks = keep_blocks
        self.block_dim = dim // n_blocks
        self.mask_temperature = mask_temperature
        self.code_temperature = code_temperature
        self.gumbel_temperature = gumbel_temperature
        self.mask_mode = mask_mode
        self.mask_seed = int(mask_seed)

        self.relation_embedding = nn.Embedding(n_relations, dim)
        self.feature_projection = nn.Linear(dim, dim, bias=False)
        # Each output block implements one W_{c,j} from the method.
        self.context_projection = nn.Linear(dim, dim, bias=False)
        self.encoder_projection = nn.Linear(dim, dim, bias=False)
        self.codebook = nn.Embedding(n_codes, dim)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _random_mask(self, user_ids: Tensor, reference: Tensor) -> Tensor:
        """Fixed Rand-ablation masks keyed by actual user and relation IDs.

        Use private CPU generators so masks do not depend on batch ordering,
        global RNG state, graph sampling, or whether inference uses a GPU.
        """
        masks = []
        for user_id in user_ids.detach().cpu().tolist():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                (self.mask_seed + 1_000_003 * int(user_id)) % (2**63 - 1)
            )
            priorities = torch.rand(
                self.n_relations, self.n_blocks, generator=generator
            )
            selected = priorities.topk(self.keep_blocks, dim=-1).indices
            masks.append(torch.zeros_like(priorities).scatter_(-1, selected, 1.0))
        if not masks:
            return reference.new_empty((0, self.n_relations, self.n_blocks))
        return torch.stack(masks).to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        context: Tensor,
        sample: bool = True,
        user_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if context.ndim != 2 or context.shape[1] != self.dim:
            raise ValueError("context must have shape [users, dim]")
        users = context.shape[0]
        if user_ids is not None:
            if user_ids.shape != (users,) or user_ids.dtype != torch.long:
                raise ValueError("user_ids must be a torch.long vector of length users")
            if (user_ids < 0).any():
                raise ValueError("user_ids must be nonnegative")
        if self.mask_mode == "random" and user_ids is None:
            raise ValueError("random masking requires actual user_ids")
        e_features = self.feature_projection(self.relation_embedding.weight).reshape(
            self.n_relations, self.n_blocks, self.block_dim
        )
        e_context = self.context_projection(context).reshape(
            users, self.n_blocks, self.block_dim
        )
        scores = torch.einsum("ujd,rjd->urj", e_context, e_features)
        if self.mask_mode == "all":
            mask = torch.ones_like(scores)
        elif self.mask_mode == "random":
            mask = self._random_mask(user_ids, scores)
        else:
            selected = scores.topk(self.keep_blocks, dim=-1).indices
            mask_hard = torch.zeros_like(scores).scatter_(-1, selected, 1.0)
            mask_soft = torch.sigmoid(scores / self.mask_temperature)
            mask = mask_soft + (mask_hard - mask_soft).detach()
        e_masked = (mask.unsqueeze(-1) * e_features.unsqueeze(0)).reshape(
            users, self.n_relations, self.dim
        )
        e_encoded = self.encoder_projection(e_masked)

        # Squared distance via inner products avoids a [U,R,B,d] tensor.
        e_codes = self.codebook.weight
        distances = (
            e_encoded.square().sum(dim=-1, keepdim=True)
            + e_codes.square().sum(dim=-1).view(1, 1, -1)
            - 2.0 * torch.matmul(e_encoded, e_codes.t())
        )
        log_q = F.log_softmax(-distances / self.code_temperature, dim=-1)
        q = log_q.exp()
        if sample:
            assignments = F.gumbel_softmax(
                log_q, tau=self.gumbel_temperature, hard=True, dim=-1
            )
        else:
            assignments = F.one_hot(log_q.argmax(dim=-1), self.n_codes).to(q.dtype)
        code = torch.matmul(assignments, e_codes)

        if users == 0:
            rate = sum(parameter.sum() * 0.0 for parameter in self.parameters())
            rate_per_user = context.new_empty(0)
        elif self.n_codes == 1:
            # One symbol has zero information capacity, including in floating
            # point arithmetic, and still permits a differentiable zero loss.
            rate_per_user = q.sum(dim=(-1, -2)) * 0.0
            rate = rate_per_user.mean()
        else:
            # Log-space marginal also handles zero-frequency relation types.
            # No detach: the exact derivative includes the learned marginal.
            log_q_bar = torch.logsumexp(
                log_q + self.rho.log().view(1, -1, 1), dim=1, keepdim=True
            )
            relation_kl = (q * (log_q - log_q_bar)).sum(dim=-1)
            rate_per_user = (relation_kl * self.rho.view(1, -1)).sum(dim=-1)
            rate = rate_per_user.mean()
        return {
            "code": code,
            "q": q,
            "rate": rate,
            "rate_per_user": rate_per_user,
            "mask": mask,
            "assignments": assignments,
        }
