# Execution repair: memory-safe data loading

Recorded after permutation seed 17 completed epoch 1, before any new calibration
or test inference. The first epoch and its restart checkpoint are retained.
The second epoch failed in a spawned data-loader worker because Windows could
not allocate a 602,112-byte CPU tensor. A concurrent independent CPU test also
reported insufficient memory. This is an execution repair, not an outcome-based
change to the scientific conditions.

`run_memory_safe.py` executes the original frozen training module with an adapter
that uses zero spawned workers. It explicitly reconstructs the uninterrupted
original two-worker persistent-loader sample order: the first epoch consumes
the original loader base-seed draw before randperm; later epochs do not. The
original stateless image flips and all model/optimizer RNG states are retained.
The sequential loader uses its own generator so it cannot consume model RNG.
A synthetic actual two-worker comparison verifies every sample over three epochs.

All conditions use this same adapter. No images, targets, training seeds, batch
size, optimizer, learning-rate schedule, epoch budget, stopping criteria or
analysis endpoints change. Resuming the committed first epoch avoids dropping
or repeating optimization steps. Data-loader worker count is an execution
parameter, not an experimental treatment. The original training source and
freeze are unmodified; the adapter and proof are separately hashed.
