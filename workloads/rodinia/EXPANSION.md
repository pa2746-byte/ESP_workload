# Five-application expansion status

All five additions have original application control flow for explicit small inputs, independent footprint/dependency tests, JSON artifacts, and rendered graphs:

1. [Needleman–Wunsch](nw/README.md)
2. [LU decomposition](lud/README.md)
3. [Backpropagation](backprop/README.md)
4. [SRAD v1](srad/README.md)
5. [CFD](cfd/README.md)

These are not replacement kernel harnesses. Source-checked host contracts abstract numerical preparation and CPU processing; they do not execute or validate that arithmetic. CFD has a documented iteration-macro guard and a synthetic mesh. Support is bounded to resolvable configured inputs, not arbitrary runtime datasets. Unknown decisions affecting addresses or launches must fail analysis.

Next: capture matched GPU runs for these scenarios, compare copy volumes and launch sequences, and validate simulator interpretation. GPU timing, numerical correctness, shared-memory behavior and simulator performance are not established by these source graphs.
