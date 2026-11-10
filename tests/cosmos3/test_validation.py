"""CPU checks for evaluation coverage, deterministic noise and training isolation.

The tiny test model implements the native CP-window contract; these tests do not
load Cosmos weights or claim to replace a full-model FSDP/CUDA smoke run.
"""
from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training.cosmos3.validation import evaluate_loss, isolated_evaluation, select_validation_rows, summarize_records, validation_protocol


def sample_rows():
    return [dict(domain=domain, target=f'{domain}/{i}.mp4', split='test', value=i + 1)
            for domain, count in [('droid', 3), ('robocasa', 5)] for i in range(count)]


class Mesh:
    def __init__(self, group, size):
        self.group, self.n = group, size

    def size(self):
        return self.n

    def get_group(self):
        return self.group


class TinyCPModel:
    def __init__(self, group=None, cp=1, items=3):
        self.items = items
        self.parallel_dims = (SimpleNamespace(cp_enabled=True, cp_mesh=Mesh(group, cp),
            cp_rank=dist.get_rank(group)) if cp > 1 else None)
        self._cp_window_slot = 0
        self._cp_local_training_payload = None
        self.net = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.5))
        with torch.no_grad():
            self.net[0].weight.fill_(.3)
            self.net[0].bias.fill_(.1)
        self.net.accum_image_sample_counter = 7
        self.net.accum_video_sample_counter = torch.tensor(11)
        self.net.train()
        self.net[1].eval()  # The original mixed module modes must be restored exactly.
        for param in self.net.parameters():
            param.grad = torch.ones_like(param)

    def _get_train_noise_level_vision(self, iteration=None):
        # Simulate upstream's optional rank/iteration generator. Evaluation must
        # pass None so noise remains identical when the CP owner rank changes.
        if iteration is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            return torch.rand((1, 1), generator=torch.Generator().manual_seed(100 + rank + iteration))
        return torch.rand((1, 1))

    def _add_noise_to_input(self, iteration=None):
        if iteration is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            return torch.randn((1, 1), generator=torch.Generator().manual_seed(200 + rank + iteration))
        return torch.randn((1, 1))

    def training_step(self, batch, iteration):
        assert not torch.is_grad_enabled() and not self.net.training
        cp = self.parallel_dims.cp_mesh.size() if self.parallel_dims else 1
        slot = self._cp_window_slot
        if slot == 0:
            assert batch
            self._cp_local_training_payload = dict(batch)
            # Emulate VAE preprocessing consuming a variable amount of RNG.
            torch.rand(batch['value'] + 2)
            np.random.rand()
            random.random()
        else:
            assert batch == {}
        payload = [self._cp_local_training_payload]
        if cp > 1:
            group = self.parallel_dims.cp_mesh.get_group()
            dist.broadcast_object_list(payload, src=dist.get_global_rank(group, slot), group=group)
        self._cp_window_slot = (slot + 1) % cp
        if self._cp_window_slot == 0:
            self._cp_local_training_payload = None
        self.net.accum_video_sample_counter += 1
        sigma = self._get_train_noise_level_vision(iteration=iteration)
        noise = self._add_noise_to_input(iteration=iteration)
        x0 = torch.tensor([[float(payload[0]['value'])]])
        xt = (1 - sigma) * x0 + sigma * noise
        mse = ((self.net(xt) - (noise - x0)) ** 2).mean()
        per_item = torch.stack([mse * 0] * (self.items - 1) + [mse])
        return dict(flow_matching_loss_vision_per_instance=per_item,
            condition_mask_vision=[torch.ones(1, 1, 1) for _ in range(self.items - 1)] + [torch.zeros(1, 1, 1)],
            sigma=sigma.expand(self.items, 1)), per_item.mean() * (10 * self.items / 3)


def snapshot(model):
    return dict(torch_rng=torch.get_rng_state().clone(), py_rng=random.getstate(),
        np_rng=np.random.get_state(), modes=[m.training for m in model.net.modules()],
        params=[p.detach().clone() for p in model.net.parameters()],
        grads=[p.grad.clone() for p in model.net.parameters()])


def check_unchanged(model, before):
    assert torch.equal(torch.get_rng_state(), before['torch_rng'])
    assert random.getstate() == before['py_rng']
    after_np = np.random.get_state()
    assert after_np[0] == before['np_rng'][0] and np.array_equal(after_np[1], before['np_rng'][1])
    assert after_np[2:] == before['np_rng'][2:]
    assert [m.training for m in model.net.modules()] == before['modes']
    assert model._cp_window_slot == 0 and model._cp_local_training_payload is None
    assert model.net.accum_image_sample_counter == 7
    assert model.net.accum_video_sample_counter.item() == 11
    assert '_get_train_noise_level_vision' not in model.__dict__
    assert '_add_noise_to_input' not in model.__dict__
    for param, value, grad in zip(model.net.parameters(), before['params'], before['grads']):
        assert torch.equal(param, value) and torch.equal(param.grad, grad)


def distributed_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        my_group = None
        # Noncontiguous CP groups exercise owner-global-rank mapping explicitly.
        for ranks in ([0, 2], [1, 3]):
            group = dist.new_group(ranks=ranks, backend='gloo')
            if rank in ranks:
                my_group = group
        for cp in (1, 2):
            model = TinyCPModel(my_group, cp)
            before = snapshot(model)
            summary, records = evaluate_loss(model, sample_rows(), dict, seed=1729, draws=2, device='cpu')
            check_unchanged(model, before)
            if rank == 0:
                Path(output, f'cp{cp}.json').write_text(json.dumps(dict(summary=summary, records=records)))
    finally:
        dist.destroy_process_group()


class ValidationTests(unittest.TestCase):
    def test_depth_evaluation_keeps_target_only_and_preserves_rgb_loss_scale(self):
        rows = [dict(r, scene_depth='scene.mp4', eef_depth='eef.mp4') for r in sample_rows()]
        before = snapshot(model := TinyCPModel(items=5))
        summary, records = evaluate_loss(model, rows, dict, seed=1729, draws=2, device='cpu')
        check_unchanged(model, before)
        rgb_summary, rgb_records = evaluate_loss(TinyCPModel(), sample_rows(), dict, seed=1729, draws=2, device='cpu')
        self.assertEqual(summary['evaluations'], rgb_summary['evaluations'])
        for depth, rgb in zip(records, rgb_records):
            self.assertEqual(depth['target_mse'], rgb['target_mse'])
            self.assertAlmostEqual(depth['loss'], rgb['loss'], places=4)
        with self.assertRaisesRegex(RuntimeError, 'condition streams'):
            evaluate_loss(TinyCPModel(), rows, dict, seed=1729, draws=1, device='cpu')

    def test_subset_is_stable_and_does_not_consume_training_rng(self):
        before = random.getstate()
        selected = select_validation_rows(sample_rows(), 2, 1729)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(selected, select_validation_rows(list(reversed(sample_rows())), 2, 1729))
        self.assertEqual(len(selected), 4)
        self.assertEqual(select_validation_rows(sample_rows(), 0, 1729), [])
        self.assertEqual(len(select_validation_rows(sample_rows(), 16, 1729)), 8)
        kwargs = dict(seed=1729, draws=2, per_domain=2, test_sha256='test')
        self.assertEqual(validation_protocol(selected, **kwargs), validation_protocol(selected, **kwargs))
        self.assertNotEqual(validation_protocol(selected, **kwargs)['id'],
                            validation_protocol(selected, **dict(kwargs, draws=3))['id'])

    def test_rejects_training_rows_and_duplicate_targets(self):
        rows = sample_rows()
        with self.assertRaises(ValueError):
            select_validation_rows([dict(r, split='train') for r in rows], 2, 1729)
        with self.assertRaises(ValueError):
            select_validation_rows(rows + [rows[0]], 2, 1729)

    def test_repeated_loss_and_training_state_are_reproducible(self):
        model = TinyCPModel()
        before = snapshot(model)
        summary, records = evaluate_loss(model, sample_rows(), dict, seed=1729, draws=2, device='cpu')
        check_unchanged(model, before)
        # Starting from unrelated training RNG must not change validation draws.
        torch.rand(27)
        np.random.rand(11)
        random.random()
        before = snapshot(model)
        _, repeated = evaluate_loss(model, sample_rows(), dict, seed=1729, draws=2, device='cpu')
        check_unchanged(model, before)
        self.assertEqual(records, repeated)
        self.assertEqual(summary['evaluations'], 16)
        self.assertAlmostEqual(summary['val_loss'],
            (summary['domains']['droid']['loss'] + summary['domains']['robocasa']['loss']) / 2)
        for record in records:
            self.assertAlmostEqual(record['loss'], record['target_mse'] * 10 / 3, places=4)
        self.assertTrue(any(records[i]['sigma'] != records[i + 1]['sigma'] for i in range(0, len(records), 2)))

    def test_restores_state_on_failure_and_rejects_mid_window(self):
        model = TinyCPModel()
        before = snapshot(model)
        calls = 0

        def broken_inputs(row):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('decode failure')
            return dict(row)

        with self.assertRaisesRegex(RuntimeError, 'decode failure'):
            evaluate_loss(model, sample_rows(), broken_inputs, seed=1729, draws=2, device='cpu')
        check_unchanged(model, before)
        model._cp_window_slot = 1
        with self.assertRaisesRegex(RuntimeError, 'boundary'):
            with isolated_evaluation(model, 'cpu'):
                self.fail('Mid-window evaluation must not run')
        self.assertEqual(model._cp_window_slot, 1)

    def test_duplicate_missing_and_nonfinite_measurements_are_rejected(self):
        _, records = evaluate_loss(TinyCPModel(), sample_rows(), dict, seed=1729, draws=2, device='cpu')
        for bad in (records[:-1], records + [records[0]],
                    [dict(records[0], loss=float('nan'))] + records[1:]):
            with self.assertRaises(RuntimeError):
                summarize_records(bad, sample_rows(), 2)

    def test_four_rank_cp1_cp2_match_serial_with_padding(self):
        reference, reference_records = evaluate_loss(TinyCPModel(), sample_rows(), dict,
            seed=1729, draws=2, device='cpu')
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_worker, args=(f'file://{folder}/rendezvous', folder), nprocs=4, join=True)
            for cp in (1, 2):
                result = json.loads(Path(folder, f'cp{cp}.json').read_text())
                self.assertEqual(result['records'], reference_records)
                self.assertEqual(result['summary']['val_loss'], reference['val_loss'])
                self.assertEqual(result['summary']['evaluations'], 16)
                self.assertEqual(result['summary']['world_size'], 4)
                self.assertEqual(result['summary']['cp_size'], cp)


if __name__ == '__main__':
    unittest.main()
