"""Stable clip ownership and conservative choices for a fixed global batch."""


def positions(window, world, rank, microbatch):
    assert 0 <= rank < world and microbatch > 0 and window >= 0
    return [(window * world + rank) * microbatch + i for i in range(microbatch)]


def select_case(results, global_batch=32):
    passing = [r for r in results if r.get('status') == 'PASS'
               and r['effective_batch'] == global_batch
               and r.get('gradient_audit_passed', False)
               and r['peak_allocated_gib'] < .95 * r['device_capacity_gib']
               and r['peak_reserved_gib'] < .985 * r['device_capacity_gib']]
    if not passing:
        raise RuntimeError('No batch configuration passed correctness, capacity and timing checks')
    fastest = min(r['mean_seconds'] for r in passing)
    competitive = [r for r in passing if r['mean_seconds'] <= fastest * 1.05]
    # Prefer no accumulation when it costs at most 5% throughput and fits safely.
    return min(competitive, key=lambda r: (r['accum'], r['mean_seconds']))
