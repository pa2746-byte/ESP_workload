# Five-application expansion status

The requested scope is original application flow for explicit small inputs,
including the host-controlled CUDA launches and copies. Isolated replacement
kernel harnesses do not satisfy this scope. Host-only numerical preparation may
use documented, source-checked contracts; this is not CPU/GPU numerical execution.

Implemented and independently checked:

1. [Needleman–Wunsch](nw/README.md): original `main` and both wavefront loops.
2. [LU decomposition](lud/README.md): original `main` and the complete tiled
   factorization launch sequence for the `-s` input path.

The other three additions are **not implemented or validated**. Candidates
inspected in the same upstream revision are:

3. Backpropagation: retain the full driver and the CPU work between the two
   GPU phases. Needs host record/pointer handling and explicit accounting for
   the intermediate download and later uploads.
4. SRAD: retain all iteration, reduction, and update stages. The v1 code needs
   uploaded index-array handling and scalar downloads. The inspected v2 code
   contains unconditional out-of-range boundary loads before replacement;
   do not silently suppress these to obtain a graph.
5. CFD: retain initialization, step-factor, flux, and time-update phases.
   Needs vector/record device operations and data-dependent neighbor addressing.

These are engineering candidates, not a claim that arbitrary inputs can be
statically resolved. Unknown runtime decisions must remain explicit or fail
analysis. Each addition needs original-source provenance, an explicit scenario,
independent footprint/dependency tests, artifacts, and documented limits before
it is counted as supported. GPU timing and simulator runs remain separate work.
