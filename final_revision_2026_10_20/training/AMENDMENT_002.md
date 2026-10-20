# Sequential loading with bounded thread prefetch

Execution-only extension to Amendment 001. The original worker-memory failure
was avoided by sequential loading, but this exposes CPU preprocessing latency.
`run_prefetched.py` uses one thread to prefetch at most one additional batch.
It avoids spawned worker interpreters, preserves the same sampler and transform,
and bounds additional input storage to one batch. CPU tensor transforms use one
intra-op thread, matching the default used inside the former loader workers.
GPU training precision, batches, model states and all scientific rules remain
unchanged. Model initialization is still checked against the original run hash.

The adapter is tested against an actual persistent two-worker loader for three
epochs. On resumption, any partial epoch is replayed from the preceding committed
model, optimizer, scheduler and RNG checkpoint. Each run records the last
committed epoch before applying this adapter. Earlier execution records remain.
No new calibration or test output is available when this repair is specified.
