import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from net.ecosmh_promptir import (
    GMMStatisticalMemory,
    EcoDPLPromptIR,
    PromptFuser,
    SFTAdapter,
    extract_degradation_features,
)
from null_space import (
    apply_null_space_projection,
    compute_null_space_projectors,
    projector_from_covariance,
    snapshot_projected_weights,
)
from utils.derain_release import tiled_forward


class EcoSMHInvariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_high_frequency_proxy_is_nonnegative_and_handles_odd_shapes(self):
        features = extract_degradation_features(torch.randn(2, 3, 9, 11))
        self.assertEqual(tuple(features.shape), (2, 3, 5, 6))
        self.assertTrue(torch.all(features >= 0))

    def test_sft_adapter_is_identity_at_initialization(self):
        adapter = SFTAdapter(in_channels=3, prompt_channels=4)
        image = torch.randn(2, 3, 8, 8)
        prompt = torch.randn(2, 4, 8, 8)
        self.assertTrue(torch.equal(adapter(image, prompt), image))

    def test_sft_modulation_is_bounded(self):
        limit = 0.25
        adapter = SFTAdapter(
            in_channels=1, prompt_channels=1, modulation_limit=limit
        )
        with torch.no_grad():
            adapter.conv_beta[0].weight.fill_(1.0)
            adapter.conv_beta[2].weight.fill_(1e6)
        output = adapter(torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
        self.assertLessEqual(float(output.abs().max()), limit)

    def test_gmm_likelihood_is_finite_normalized_and_separates_tasks(self):
        memory = GMMStatisticalMemory(
            feature_dim=2,
            max_tasks=2,
            num_components=2,
            covariance_shrinkage=0.2,
        )
        task_zero = torch.tensor(
            [[1.0, -0.05], [1.0, 0.00], [1.0, 0.05], [0.9, 0.02]]
        )
        task_one = torch.tensor(
            [[-0.05, 1.0], [0.00, 1.0], [0.05, 1.0], [0.02, 0.9]]
        )
        memory.update_statistics(0, task_zero)
        memory.update_statistics(1, task_one)
        probabilities = memory.compute_task_scores(
            torch.tensor([[1.0, 0.01], [0.01, 1.0]])
        )
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1), torch.ones(2), atol=1e-6
            )
        )
        self.assertEqual(probabilities.argmax(dim=1).tolist(), [0, 1])

    def test_private_prompt_blocks_are_disjoint_and_frozen_slices_do_not_drift(self):
        fuser = PromptFuser(
            num_prompts=6,
            query_dim=3,
            value_shape=(2, 1, 1),
            max_tasks=3,
        )
        fuser.set_active_range(0, 2)
        fuser.update_task_mask(0)
        fuser.set_active_range(2, 4)
        fuser.backup_protected_prompts()
        before = fuser.values.detach().clone()

        optimizer = torch.optim.AdamW(fuser.parameters(), lr=0.1, weight_decay=0.1)
        fused, _ = fuser(torch.randn(4, 3))
        fused.square().mean().backward()
        fuser.zero_protected_grads()
        optimizer.step()
        fuser.restore_protected_prompts()

        self.assertTrue(torch.equal(fuser.values[:2], before[:2]))
        self.assertFalse(torch.equal(fuser.values[2:4], before[2:4]))
        self.assertTrue(torch.equal(fuser.values[4:], before[4:]))

        fuser.update_task_mask(1)
        self.assertEqual(
            int((fuser.task_prompt_mask[0] * fuser.task_prompt_mask[1]).sum()), 0
        )
        routed_mask = fuser.compute_soft_mask(torch.tensor([[0.0, 1.0]]))
        self.assertTrue(torch.equal(routed_mask[0, 2:4], torch.ones(2)))
        self.assertTrue(torch.equal(routed_mask[0, :2], torch.zeros(2)))

    def test_full_rank_hard_projector_is_zero_not_identity(self):
        projector, stats = projector_from_covariance(
            torch.eye(3), count=1, threshold=0.0, strength=1.0
        )
        self.assertEqual(stats["nullity"], 0)
        self.assertTrue(torch.equal(projector, torch.zeros_like(projector)))

    def test_projected_candidate_update_preserves_sampled_adapter_outputs(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = SFTAdapter(2, 2)

        model = TinyModel()
        with torch.no_grad():
            model.adapter.conv_gamma[2].weight.normal_()
            model.adapter.conv_beta[2].weight.normal_()

        prompt = torch.zeros(8, 2, 3, 3)
        prompt[:, 0] = torch.linspace(-1.0, 1.0, 8).view(-1, 1, 1)
        image = torch.randn(8, 2, 3, 3)
        model.adapter.set_nsp_collection(True)
        baseline = model.adapter(image, prompt).detach()
        model.adapter.set_nsp_collection(False)

        projectors = compute_null_space_projectors(
            model, threshold=1e-6, strength=1.0
        )
        old_weights = snapshot_projected_weights(model, projectors)
        with torch.no_grad():
            for parameter in model.adapter.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.2)
        apply_null_space_projection(model, projectors, old_weights)
        protected_output = model.adapter(image, prompt)
        self.assertTrue(
            torch.allclose(protected_output, baseline, atol=2e-5, rtol=2e-5)
        )

    def test_tiled_forward_reuses_one_routing_decision(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.received = []

            def forward(self, image, routing_probs=None):
                self.received.append(routing_probs)
                return image

        model = Recorder()
        routing_probs = torch.tensor([[0.25, 0.75]])
        image = torch.randn(1, 3, 12, 12)
        output = tiled_forward(
            model,
            image,
            tile_size=8,
            overlap=2,
            multiple=4,
            forward_kwargs={"routing_probs": routing_probs},
        )
        self.assertTrue(torch.allclose(output, image))
        self.assertGreater(len(model.received), 1)
        self.assertTrue(all(item is routing_probs for item in model.received))

    def test_small_end_to_end_forward_graph(self):
        class DummyVGG(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Conv2d(3, 256, 1))

        with patch("torchvision.models.vgg16", return_value=DummyVGG()):
            model = EcoDPLPromptIR(
                dim=8,
                num_blocks=[1, 1, 1, 1],
                num_refinement_blocks=1,
                heads=[1, 2, 4, 8],
                num_prompts=4,
                max_tasks=2,
            )
        model.set_active_prompt_range(0, 2)
        model.train()
        image = torch.randn(1, 3, 32, 32)
        output, auxiliary = model(image, return_aux=True)
        self.assertEqual(tuple(output.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIn("image_distance", auxiliary)
        model.update_task_statistics(0, [(image, image)], torch.device("cpu"))
        self.assertEqual(int(model.stat_memory.task_count), 1)
        self.assertEqual(int(model.image_fuser.task_prompt_mask[0].sum()), 2)
        self.assertGreater(int(model.image_prompt_adapter.nsp_input_count), 0)


if __name__ == "__main__":
    unittest.main()
