# DWT2D: source-only logical transfer graph

The original complete driver evaluates the pinned 64x64 RGB input with `-d 64x64 -c 3 -l 1 -5 input.rgb`. The graph has 22 nodes and 23 edges. Four source-derived launches execute: conversion at grid 16/block 256, followed by three wavelet launches at grid 1x8/block 64. These counts are evaluation results, not constraints imported from GPU events.

Seven int planes occupy 16,384 bytes each. The RGB upload occupies 12,288 bytes. Initialization, component writes, three copies to a reused backup and wavelet reads/writes are represented. The captured build leaves `OUTPUT` undefined, so there are no result downloads. Logical byte footprints are not measured physical DRAM traffic.

`main_simulator.json` contains simulator nodes/edges. `main_metadata.json` contains buffer roles, byte ranges, source hashes, source-derived launch geometry and modeling assumptions. `source_scenario.json` identifies the capture used to recover inputs and records the two reviewed filename-allocation corrections that postdate it. No GPU metrics are attached and no corrected-source GPU rerun or numerical validation is claimed.

Reproduction and remaining analyzer limits are documented in `workloads/rodinia/REMAINING_EIGHT.md`. `tests/test_dwt2d_graph_contract.py` checks an independently specified source schedule.
