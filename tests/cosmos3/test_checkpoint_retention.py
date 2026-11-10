"""CPU-only regression checks for destructive checkpoint retention decisions."""
from pathlib import Path
import tempfile
import unittest
from training.cosmos3.checkpoint import checkpoints_to_prune


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, step, complete=True):
        folder = self.root / f'checkpoint_step{step:06d}'
        folder.mkdir()
        if complete:
            (folder/'COMPLETE').write_text('all ranks saved\n')
        return folder

    def test_full_run_keeps_milestones_and_two_recent_states(self):
        steps = list(range(100, 15001, 100)) + [299]
        for step in steps:
            self.checkpoint(step)
        remove = checkpoints_to_prune(self.root, keep_steps=[200, 299, 500])
        retained = set(steps) - {int(p.name.removeprefix('checkpoint_step')) for p in remove}
        self.assertEqual(retained, set(range(1000, 15001, 1000)) | {200, 299, 500, 14900})

    def test_incomplete_new_save_cannot_displace_resume_states(self):
        a, b = self.checkpoint(600), self.checkpoint(700)
        self.checkpoint(800, complete=False)
        self.assertEqual(checkpoints_to_prune(self.root), [])
        self.assertTrue(a.is_dir() and b.is_dir())

    def test_unknown_names_and_symlinks_are_never_pruned(self):
        old = self.checkpoint(100)
        self.checkpoint(200)
        self.checkpoint(300)
        odd = self.root/'checkpoint_step_old'
        odd.mkdir(); (odd/'COMPLETE').touch()
        (self.root/'checkpoint_step000050').symlink_to(old, target_is_directory=True)
        self.assertEqual(checkpoints_to_prune(self.root, keep_every=0), [old])

    def test_reject_invalid_policy(self):
        for kwargs in ({'keep_recent':0}, {'keep_every':-1}, {'keep_steps':[0]}):
            with self.assertRaises(ValueError):
                checkpoints_to_prune(self.root, **kwargs)


if __name__ == '__main__':
    unittest.main()
