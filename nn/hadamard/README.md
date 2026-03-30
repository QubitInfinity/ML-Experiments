
# FWHT Sieving for Transformer Weights

A simple post-training experiment that applies Hadamard-domain sieving (thresholding in the FWHT domain) to linear layers of small language models.

## Overview

This repository contains code to apply **Hadamard-domain sieving** (thresholding in the FWHT domain) to the weights of a pretrained transformer. The goal is to investigate whether spectral decomposition via FWHT can identify structurally important patterns in weight matrices better than standard magnitude-based pruning.

Two variants are implemented:
- **Domain-kept sieving**: Uses a custom `SievedLinear` layer that operates directly in the transformed domain.
- **Spatial reconstruction**: Applies inverse FWHT to reconstruct approximate dense weights.

The project compares these approaches against:
- Unstructured magnitude pruning (baseline)
- Simple simulated low-precision quantization

Tested on **SmolLM2-135M** with WikiText-2 perplexity evaluation.

## Results (SmolLM2-135M)

=======================================================  
FINAL PERPLEXITY BENCHMARK                           
=======================================================  
1. Baseline (FP32)                    : 20.59
2. FWHT Domain Sieving                : 24.36
3. FWHT Spatial Reconstruction        : 24.36
4. Magnitude Pruning                  : 26.25
5. Baseline Q16-bit                   : 21.28
6. FWHT Spatial + Q16-bit             : 25.54
7. Magnitude Pruning + Q16-bit        : 26.92
====================================================== 

*Note: Results are from one-shot post-training pruning without any recovery fine-tuning.*

## Features

- Efficient in-place FWHT implementation (butterfly algorithm)
- Automatic padding to next power of 2
- Differentiated pruning ratios for attention and MLP layers
- Custom `SievedLinear` layer for operating in Hadamard domain
- Support for both domain-kept and reconstructed variants
- Simple simulated quantization for comparison

## License

MIT License

---

**Author:** Russel Maytham 
**Date:** 30 March 2026

This project was created primarily as a learning exercise to upskill in machine learning and AI coding, while experimenting with concepts from my electrical engineering background (particularly signal processing and transforms).

