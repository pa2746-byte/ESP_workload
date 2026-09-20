This part of the project aims to automatically generate data-dependency graphs from program workloads using profiling and program-analysis techniques.

The immediate focus is on the graph-generation stage. 
Given a workload, the goal is to identify important computation and data-movement events, determine the dependencies between them, 
and export those relationships as a directed graph that can later be consumed by a hardware-performance simulator.

The project starts with small CUDA workloads whose dependency structure is easy to verify manually, including:

Vector addition — a simple input-to-compute-to-output flow
Two-stage pipeline — a linear producer-consumer dependency
Fork-join workload — independent branches that later synchronize
Reduction — hierarchical many-to-one dependencies

These workloads provide controlled test cases for evaluating different graph-generation approaches, 
such as static code analysis, dynamic instrumentation, and profiling tools.

The longer-term goal is to scale the same methodology to more realistic workloads such as matrix multiplication, 
attention, and transformer models, while preserving enough information about dependencies and data movement to support later hardware-independent performance analysis.
