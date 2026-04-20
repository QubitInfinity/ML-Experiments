# FWHT Sieving for Transformer Weights

An exploratory post-training sparsification experiment that applies thresholding in the Fast Walsh–Hadamard Transform (FWHT) domain to linear layers of a pretrained transformer.

## Summary

This repository tests whether a simple **Hadamard-domain sieving** step can preserve language-model quality better than a straightforward unstructured magnitude-pruning baseline under similar sparsity settings.

The current implementation was tested on **SmolLM2-135M** and evaluated with **WikiText-2 perplexity**.

This is a **single-run, post-training experiment** intended primarily as a learning and research exercise, not a production compression pipeline.

## What the code does

For selected linear layers in a pretrained transformer:

1. Pad the input dimension to the next power of 2 where needed.
2. Apply a row-wise **FWHT** to the weight matrix.
3. Keep only the largest coefficients by absolute value according to a configurable **keep ratio**.
4. Evaluate the modified model without recovery fine-tuning.

Two execution variants are included:

### 1. Domain-kept sieving
A custom `SievedLinear` layer stores the sieved Hadamard-domain coefficients and applies FWHT to the input activations at inference time.

### 2. Spatial reconstruction
The sieved Hadamard coefficients are transformed back to the original weight domain, yielding an approximate dense weight matrix.

In the current formulation, these two variants are expected to produce the same outputs up to numerical effects; they mainly differ in representation and execution path.

## Baselines included

The project compares FWHT sieving against:

- **Unstructured magnitude pruning**
- **Simple simulated low-precision quantization**

## Current results

Model: **SmolLM2-135M**  
Evaluation: **WikiText-2**  
Metric: **Perplexity (lower is better)**

| Method                             | Perplexity |
|------------------------------------|-----------:|
| Baseline (FP32)                    | 20.59      |
| FWHT Domain Sieving                | 24.36      |
| FWHT Spatial Reconstruction        | 24.36      |
| Magnitude Pruning                  | 26.25      |
| Baseline Simulated 16-bit          | 21.28      |
| FWHT Spatial + Simulated 16-bit    | 25.54      |
| Magnitude Pruning + Simulated 16-bit | 26.92    |

### Interpretation

In this run, FWHT-based sieving degraded perplexity less than the magnitude-pruning baseline under the chosen settings.

That said, these numbers should be treated as **exploratory**, because they come from:

- one model
- one dataset
- one set of sparsity settings
- one-shot post-training modification
- no recovery fine-tuning
- no repeated trials or variance analysis

## Important implementation notes

- The FWHT is applied **row-wise** across the input dimension of each weight matrix.
- Layers such as token embeddings and the output head are intended to be excluded from sieving/pruning.
- Different **keep ratios** are used for attention and MLP layers:
  - attention layers: `0.70`
  - MLP layers: `0.85`

These are **keep ratios**, not prune ratios.

## Quantization note

The quantization baseline in this repository is a **simple simulated per-tensor uniform quantizer**.

It is **not**:
- true FP16 inference
- BF16 inference
- a hardware-aware quantization pipeline
- an optimized deployment format

It is included only as a lightweight comparison point.

## Scope and limitations

This repository is intentionally simple. It does **not** currently provide:

- recovery fine-tuning after sieving
- structured sparsity
- runtime benchmarking
- memory-footprint benchmarking
- kernel-level acceleration
- multi-model evaluation
- statistical analysis across seeds
- deployment-oriented quantization

Accordingly, the project should be read as a **proof-of-concept experiment**, not as evidence of a production-ready compression method.

## Repository features

- FWHT implementation based on the butterfly algorithm
- Automatic padding to the next power of 2
- Separate keep ratios for attention and MLP layers
- Custom `SievedLinear` layer for transformed-domain execution
- Spatial reconstruction path for approximate dense weights
- Baseline comparison against magnitude pruning
- Simple simulated quantization baseline
- CPU-friendly execution setup

## Suggested next steps

Natural extensions for this project would be:

- recovery fine-tuning after sieving
- per-layer or sensitivity-based keep-ratio selection
- evaluation on additional models and datasets
- runtime and memory benchmarking
- comparison against better quantization baselines
- blockwise or structured Hadamard-domain sparsification

## Motivation

This project was created as a hands-on exercise to build practical familiarity with transformer internals, compression ideas, and transform-domain thinking.

It also reflects an interest in applying concepts from signal processing and electrical engineering to modern machine-learning systems.

## Author

**Russel Maytham**

**Date:** 30 March 2026

## License

MIT License
