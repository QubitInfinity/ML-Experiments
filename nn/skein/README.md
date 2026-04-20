# Skein-CNN: Constrained Mirror-Filter Convolutions

A small PyTorch experiment on MNIST using a custom convolutional block inspired by the **form** of the Conway skein relation.

## Summary

This repository explores a simple idea:

- apply a learned convolution
- apply the same convolution with the kernel spatially flipped
- subtract a smoothed projection of the input
- learn a scalar `z` controlling the strength of that smoothing term

The result is a **constrained convolutional network**, not a standard CNN and not an RNN.

This project is best understood as an exploratory architecture experiment rather than a mathematically rigorous implementation of knot-theoretic invariants.

## What the model actually is

The network consists of:

- two custom convolutional layers
- max-pooling after each layer
- one fully connected output layer

The custom layer computes three branches:

- **L+**: a standard convolution with learned weights
- **L-**: a convolution using the same kernel flipped across height and width
- **L0**: a smoothed version of the input produced by average pooling followed by a 1x1 projection

The layer output is:

```text
LeakyReLU(L+) - LeakyReLU(L-) - z * LeakyReLU(L0)
