"""Focused mathematical and autograd checks; run with unittest discovery."""

import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.ib_layers import GaussianMessageLayer, RelationBottleneck


class GaussianMessageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_rate_matches_distribution_kl(self):
        layer = GaussianMessageLayer(4).double()
        e_target = torch.randn(3, 4, dtype=torch.float64)
        e_source = torch.randn(5, 4, dtype=torch.float64)
        e_source[0].zero_()  # The exact formula must also cover zero signal.
        target_index = torch.tensor([0, 1, 2, 0, 1])
        _, rate = layer(e_target, e_source, target_index, noise=False)
        e_signal = layer.message_projection(e_source)
        e_raw = math.sqrt(4) * e_signal / e_signal.norm(
            dim=-1, keepdim=True
        ).clamp_min(layer.eps)
        alpha = layer.gate(torch.cat([e_target[target_index], e_source], -1))
        alpha = alpha.sigmoid().clamp(1e-6, 1 - 1e-6)
        posterior = Independent(Normal(alpha.sqrt() * e_raw, (1 - alpha).sqrt()), 1)
        prior = Independent(Normal(torch.zeros_like(e_raw), torch.ones_like(e_raw)), 1)
        expected = kl_divergence(posterior, prior).mean()
        torch.testing.assert_close(rate, expected, rtol=1e-10, atol=1e-10)

    def test_supplied_noise_and_sqrt_degree_aggregation(self):
        layer = GaussianMessageLayer(2)
        with torch.no_grad():
            layer.message_projection.weight.copy_(torch.eye(2))
            layer.self_projection.weight.zero_()
            layer.aggregate_projection.weight.copy_(torch.eye(2))
            for parameter in layer.gate.parameters():
                parameter.zero_()
        e_target = torch.zeros(2, 2)
        e_source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        e_noise = torch.tensor([[0.2, -0.4], [0.6, 0.8]])
        e_next, rate = layer(
            e_target, e_source, torch.tensor([0, 0]), noise_values=e_noise
        )
        expected_aggregate = (e_source + e_noise / math.sqrt(2)).sum(0) / math.sqrt(2)
        torch.testing.assert_close(e_next[0], torch.nn.functional.elu(expected_aggregate))
        torch.testing.assert_close(e_next[1], torch.zeros(2))
        torch.testing.assert_close(rate, torch.tensor(math.log(2)))

    def test_empty_edges_are_finite_and_rate_backward_works(self):
        layer = GaussianMessageLayer(4, personalized=True)
        e_target = torch.randn(3, 4)
        e_next, rate = layer(
            e_target,
            torch.empty(0, 4),
            torch.empty(0, dtype=torch.long),
            e_context=torch.randn(3, 4),
        )
        self.assertTrue(torch.isfinite(e_next).all())
        self.assertEqual(rate.item(), 0)
        rate.backward()
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_personalized_relation_messages_backpropagate(self):
        layer = GaussianMessageLayer(4, personalized=True)
        e_target = torch.randn(3, 4, requires_grad=True)
        e_source = torch.randn(5, 4, requires_grad=True)
        e_relation = torch.randn(5, 4, requires_grad=True)
        e_context = torch.randn(3, 4, requires_grad=True)
        e_next, rate = layer(
            e_target, e_source, torch.tensor([0, 1, 2, 0, 1]),
            e_relation=e_relation, e_context=e_context,
        )
        (e_next.square().mean() + rate).backward()
        for tensor in (e_target, e_source, e_relation, e_context):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)


class RelationBottleneckTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    @staticmethod
    def make_layer(**kwargs):
        arguments = dict(
            n_relations=5, dim=12, n_codes=3, n_blocks=4, keep_blocks=2,
            rho=torch.tensor([0.1, 0.2, 0.3, 0.0, 0.4]),
        )
        arguments.update(kwargs)
        return RelationBottleneck(**arguments)

    def test_forward_masks_and_codes_are_hard(self):
        layer = self.make_layer()
        for sample in (True, False):
            result = layer(torch.randn(2, 12), sample=sample)
            self.assertEqual(result["code"].shape, (2, 5, 12))
            self.assertEqual(result["q"].shape, (2, 5, 3))
            self.assertTrue(((result["mask"] == 0) | (result["mask"] == 1)).all())
            torch.testing.assert_close(result["mask"].sum(-1), torch.full((2, 5), 2.0))
            self.assertTrue(((result["assignments"] == 0) | (result["assignments"] == 1)).all())
            torch.testing.assert_close(result["assignments"].sum(-1), torch.ones(2, 5))
            indices = result["assignments"].argmax(-1)
            torch.testing.assert_close(result["code"], layer.codebook(indices))

    def test_exact_information_matches_marginal_and_is_bounded(self):
        layer = self.make_layer()
        result = layer(torch.randn(4, 12))
        q = result["q"]
        q_bar = (q * layer.rho.view(1, -1, 1)).sum(1, keepdim=True)
        expected = (
            layer.rho.view(1, -1) * (q * (q.log() - q_bar.log())).sum(-1)
        ).sum(-1).mean()
        torch.testing.assert_close(result["rate"], expected, atol=1e-7, rtol=1e-5)
        self.assertGreaterEqual(result["rate"].item(), -1e-6)
        self.assertLessEqual(result["rate"].item(), math.log(layer.n_codes) + 1e-6)

    def test_independent_relations_have_zero_rate_even_for_nonuniform_codes(self):
        layer = self.make_layer()
        with torch.no_grad():
            layer.feature_projection.weight.zero_()
            layer.codebook.weight[0].zero_()
            layer.codebook.weight[1:].fill_(1.0)
        result = layer(torch.randn(2, 12))
        self.assertAlmostEqual(result["rate"].item(), 0.0, places=6)
        self.assertGreater(result["q"][0, 0, 0].item(), 0.99)
        # KL(q || uniform) would be large here, so it cannot replace this MI.
        uniform_kl = (result["q"] * (result["q"].log() + math.log(3))).sum(-1).mean()
        self.assertGreater(uniform_kl.item(), 1.0)

    def test_straight_through_gradients_reach_mask_context_and_codebook(self):
        layer = self.make_layer()
        context = torch.randn(3, 12, requires_grad=True)
        result = layer(context)
        (result["code"].square().mean() + result["rate"]).backward()
        for tensor in (
            context, layer.context_projection.weight,
            layer.feature_projection.weight, layer.relation_embedding.weight,
            layer.encoder_projection.weight, layer.codebook.weight,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_empty_users_and_invalid_configuration(self):
        layer = self.make_layer()
        result = layer(torch.empty(0, 12))
        self.assertEqual(result["code"].shape, (0, 5, 12))
        self.assertEqual(result["rate"].item(), 0.0)
        for invalid in (
            {"n_blocks": 5}, {"keep_blocks": 4}, {"keep_blocks": 0},
            {"n_codes": 1}, {"rho": torch.zeros(5)}, {"mask_temperature": 0},
        ):
            with self.assertRaises(ValueError):
                self.make_layer(**invalid)


if __name__ == "__main__":
    unittest.main()
