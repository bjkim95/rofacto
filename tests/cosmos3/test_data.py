"""Check distributed sample coverage/resume independent of expensive GPU training."""
import unittest
from training.cosmos3.data import WindowSampler, lr_factor


class SamplingTests(unittest.TestCase):
    def test_coverage_and_restart_across_uneven_epoch(self):
        size, world = 13, 4
        samplers = [WindowSampler(size, 42, world, rank) for rank in range(world)]
        consumed = [samplers[rank].index(window) for window in range(10) for rank in range(world)]
        for start in (0, 13, 26):
            self.assertEqual(set(consumed[start:start + size]), set(range(size)))
        resumed = [WindowSampler(size, 42, world, rank) for rank in range(world)]
        tail = [resumed[rank].index(window) for window in range(5, 10) for rank in range(world)]
        self.assertEqual(tail, consumed[20:])

    def test_cp_window_optimizer_boundary(self):
        # Four microsteps, CP=2, four ranks: eight distinct source clips/update.
        samplers = [WindowSampler(101, 42, 4, rank) for rank in range(4)]
        rows = [samplers[rank].index(micro // 2) for micro in (0, 2) for rank in range(4)]
        self.assertEqual(len(set(rows)), 8)
        self.assertEqual(WindowSampler(101, 42, 4, 0).index(2), samplers[0].index(2))

    def test_eight_gpu_preserves_effective_batch_and_source_order(self):
        four = [WindowSampler(101, 42, 4, r) for r in range(4)]
        eight = [WindowSampler(101, 42, 8, r) for r in range(8)]
        # Four ranks x 2 windows = eight ranks x 1 window, same eight examples.
        old = [four[r].index(w) for w in range(2) for r in range(4)]
        new = [eight[r].index(0) for r in range(8)]
        self.assertEqual(old, new)
        self.assertEqual(len(set(new)), 8)

    def test_lr_horizon_independent_of_pilot_stop(self):
        self.assertEqual(lr_factor(0, 100, 15000), .01)
        self.assertEqual(lr_factor(99, 100, 15000), 1.)
        self.assertGreater(lr_factor(999, 100, 15000), .98)
        self.assertEqual(lr_factor(14999, 100, 15000), 0.)


if __name__ == '__main__':
    unittest.main()
