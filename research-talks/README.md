# Research Talks – Distributed Deep Learning

A series of paper-review talks on how to train large models efficiently across many GPUs. Each deck surveys recent systems papers and compares their approaches.

| Date | Talk | Papers covered |
|------|------|----------------|
| 2025-10-23 | [Evidence of mutual effects of data and model parallelism](2025-10-23-hybrid-data-and-model-parallelism.pptx) | Limits of pure DP/MP; hybrid schemes: DAPPLE, FlexFlow, Alpa, AutoDDL, Malleus, Uni-MoE, recommender-system training at scale |
| 2025-11-01 | [AutoDDL and automatic parallelization search](2025-11-01-autoddl-and-parallelization-search.pptx) | AutoDDL (IEEE TPDS 2024) and OneFlow's SBP (split / broadcast / partial-sum) abstraction with coordinate-descent search; ZeroPP, Optimus (2D tensor parallelism), FASOP |
| 2025-11-07 | [AutoDDL, Aceso and Helix](2025-11-07-autoddl-aceso-helix.pptx) | AutoDDL's α–β communication cost model; Aceso (EuroSys 2024) iterative bottleneck alleviation; Helix (ASPLOS 2025) LLM serving on heterogeneous GPUs via max-flow |
| 2026-05-20 | [NEST](2026-05-20-nest-device-placement.pptx) | NEST (MLSys 2026): network- and memory-aware device placement on hierarchical data-centre topologies, with a critique of Aceso and TopoOpt |
| 2026-06-03 | [Prefetching](2026-06-03-prefetching.pptx) | Parameter prefetching in fully sharded training (ZeRO-3 / FSDP, DeepCompile), input-pipeline data stalls, NVIDIA DALI and PyTorch `DataLoader` prefetching |
