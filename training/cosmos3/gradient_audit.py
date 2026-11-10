"""Real-model independent gradients versus accumulated gradients, before clipping."""
import json
from pathlib import Path


def audit(model, params, optimizer, make_input, output, seed=172904):
    import torch
    import torch.distributed as dist
    from .checkpoint import local
    import random
    import numpy as np

    assert model._cp_size == 1 if hasattr(model, '_cp_size') else True
    saved = (torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate(), np.random.get_state())
    rank = dist.get_rank()

    def vector():
        pieces = []
        for value in params.values():
            assert value.grad is not None
            flat = local(value.grad).detach().flatten()
            stride = max(1, (flat.numel() + 127) // 128)
            pieces.append(flat[::stride].float().cpu().clone())
        return torch.cat(pieces)

    def backward(i):
        torch.manual_seed(seed + 100 * rank + i)
        random.seed(seed + 100 * rank + i)
        np.random.seed(seed + 100 * rank + i)
        auxiliary, loss = model.training_step(make_input(i), iteration=i)
        assert torch.isfinite(loss).all()
        (loss / 4).backward()
        value = loss.detach().float().item()
        del auxiliary, loss
        return value

    model.net.train()
    try:
        independent, losses = [], []
        for i in range(4):
            optimizer.zero_grad(set_to_none=True)
            losses.append(backward(i))
            independent.append(vector())
        expected = torch.stack(independent).sum(0)
        optimizer.zero_grad(set_to_none=True)
        accumulated_losses = [backward(i) for i in range(4)]
        actual = vector()
        denom = expected.norm().clamp_min(1e-20)
        relative_error = float((actual - expected).norm() / denom)
        last_only_error = float((independent[-1] - expected).norm() / denom)
        assert max(abs(a-b) for a, b in zip(losses, accumulated_losses)) < 1e-5
        record = dict(status='PASS' if relative_error < .025 and last_only_error > .1 else 'FAIL',
                      backend='torch_fsdp', microbatches=4, rank=rank,
                      relative_error=relative_error, last_only_relative_error=last_only_error,
                      expected_norm=float(denom), independent_losses=losses,
                      accumulated_losses=accumulated_losses, sampled_coordinates=actual.numel())
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(output, f'gradient_audit_rank{rank}.json').write_text(json.dumps(record, indent=2)+'\n')
        all_pass = torch.tensor(int(record['status'] == 'PASS'), device='cuda')
        dist.all_reduce(all_pass, op=dist.ReduceOp.MIN)
        assert all_pass.item(), 'Independent versus accumulated real-model gradients disagree'
        if rank == 0:
            print(json.dumps(dict(event='real_gradient_accumulation_audit_pass', **record)), flush=True)
        return record
    finally:
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(saved[0]); torch.cuda.set_rng_state(saved[1])
        random.setstate(saved[2]); np.random.set_state(saved[3])
