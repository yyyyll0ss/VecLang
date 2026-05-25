# VecLang: Vector Map as Language

<p align="center">
  <b>A Unified Vision-Language Framework for Remote Sensing Vector Mapping</b>
</p>

<p align="center">
  <a href="#news">News</a> |
  <a href="#overview">Overview</a> |
  <a href="#results">Results</a> |
  <a href="#release-plan">Release Plan</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="95%">
</p>

## Overview

**VecLang** formulates remote sensing vector mapping as **structured language generation**. Instead of using category-specific polygon or graph representations, VecLang represents heterogeneous map entities with a shared **Structured Vector Language (SVL)**, where geometry, semantics, and topology are organized in a unified GeoJSON-like language space.

VecLang supports both closed-structure objects, such as **buildings** and **water bodies**, and network-like objects, such as **roads**, enabling unified modeling of diverse vector mapping tasks.

## Highlights

- **Vector Map as Language:** We introduce a language-based paradigm for remote sensing vector mapping, converting map elements into structured vector-language sequences.
- **Unified Representation:** SVL represents geometry, semantics, and topology in a shared format for both polygonal and network-like map entities.
- **Progressive Vectorization:** The Progressive Vectorization Framework localizes vectorization units before structured generation, improving stability and efficiency.
- **Hierarchical Optimization:** Hierarchical Vector Language Optimization improves syntactic validity, content consistency, and execution fidelity.
- **Broad Evaluation:** VecLang is evaluated on single-class, multiclass, cross-dataset, open-vocabulary, and large-scale vector mapping settings.

## News

- **2026-05:** Initial project page and repository skeleton released.
- **Coming soon:** Paper, benchmark data, pretrained checkpoints, and training/evaluation code will be released progressively.

## Method

<p align="center">
  <img src="assets/method_overview.png" width="95%">
</p>

VecLang contains three main components:

1. **Structured Vector Language (SVL):** A GeoJSON-like language for representing vector maps with explicit geometry, semantics, and topology.
2. **Progressive Vectorization Framework (PVF):** A localization-before-generation framework that first detects vectorization units and then generates structured vector outputs.
3. **Hierarchical Vector Language Optimization (HVLO):** A reward-based optimization strategy that improves syntactic validity, content consistency, and execution fidelity.

## Results

### Single-Class Vector Mapping

VecLang performs strong single-class vector mapping across different geometric forms, including building footprints, road networks, and water bodies.

<p align="center">
  <img src="assets/results/single_class_results.png" width="95%">
</p>

<p align="center">
  <b>Single-class vector mapping results.</b> VecLang generates regular building outlines, compact waterbody polygons, and continuous road structures within a unified language-based framework.
</p>

### Multi-Class Vector Mapping

VecLang jointly predicts buildings, roads, and water bodies in the same scene without using category-specific output heads or separate vector representations.

<p align="center">
  <img src="assets/results/multi_class_results.png" width="95%">
</p>

<p align="center">
  <b>Multi-class vector mapping results.</b> VecLang preserves both instance-level object boundaries and network connectivity under a shared structured representation.
</p>

### Large-Scale Vector Mapping

VecLang can be applied to large remote sensing scenes by progressively generating localized vectorization units and merging them into complete vector maps.

<p align="center">
  <img src="assets/results/large_scale_results.png" width="95%">
</p>

<p align="center">
  <b>Large-scale vector mapping results.</b> VecLang produces executable vector maps over large-area remote sensing imagery while maintaining compact and structured outputs.
</p>

## Release Plan

We are preparing a clean and reproducible release. The paper, data, checkpoints, and scripts will be released progressively.

| Component | Description | Status | Planned Release |
| --- | --- | --- | --- |
| Paper | Full paper and technical details | Coming soon | To be released |
| Code | Inference, training, and evaluation scripts | In progress | To be released |
| VecMap-Bench | Benchmark data for unified vector mapping | In progress | To be released |
| SVL Annotations | Structured Vector Language annotations | In progress | To be released |
| Model Weights | Pretrained and fine-tuned VecLang checkpoints | In progress | To be released |
| Visualization Examples | Qualitative results and demo samples | Partially available | Updating |
| Documentation | Usage instructions and data format description | In progress | Updating |

## Todo

- [ ] Release the paper.
- [ ] Release the full VecLang codebase.
- [ ] Release VecMap-Bench and SVL annotations.
- [ ] Release pretrained and fine-tuned model weights.
- [ ] Provide inference scripts for single-image and large-scale vector mapping.
- [ ] Provide training and post-training scripts.
- [ ] Provide evaluation scripts for polygonal objects and road networks.
- [ ] Add detailed examples for custom data conversion to SVL.

## Repository Structure

```text
VecLang/
├── assets/                     # Figures and visualization results
│   ├── teaser.png
│   ├── method_overview.png
│   └── results/
│       ├── single_class_results.png
│       ├── multi_class_results.png
│       └── large_scale_results.png
├── configs/                    # Configuration files
├── data/                       # Data preparation scripts
├── scripts/                    # Training, inference, and evaluation scripts
├── veclang/                    # Core implementation
├── README.md
└── LICENSE
