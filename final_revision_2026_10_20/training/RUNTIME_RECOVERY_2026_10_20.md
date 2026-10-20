# Runtime restart during the declared follow-up

During epoch 12 of eight-output Uniform, seed 89, the CUDA process exited with
an illegal-memory-access error reported while computing selection predictions.
The asynchronous CUDA error message does not identify its underlying cause.
No calibration or test inference had begun. The ten previously completed runs
and committed epochs 1--11 of this run remained intact.

The retained `restart.pt` was loaded in a fresh CPU process and verified to
contain committed epoch 11, populated optimizer state, scheduler epoch 11,
and CPU/CUDA random-generator states. Execution resumed on 20 October 2026
through the unchanged, frozen `run_prefetched.py --all` adapter. Completed runs
are skipped. The interrupted epoch is replayed from the preceding committed
checkpoint; its partial computation is not a retained experimental state.

No target, seed, sample order, batch size, precision, optimizer, schedule,
checkpoint rule, analysis rule or planned condition was changed. This is an
execution restart under the existing Amendment 002 recovery rule, not a new
scientific condition. The occurrence and recovery do not establish bitwise
reproducibility on other hardware or explain the CUDA failure.
