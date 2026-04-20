# Skein-CNN: Constrained Mirror-Filter Convolutions

A small PyTorch experiment on MNIST using a custom convolutional block inspired by the form of the Conway skein relation.

## Summary

This repo tests a simple constrained convolution design:

- `L+`: a learned convolution
- `L-`: the same kernel flipped spatially
- `L0`: a smoothed projection of the input
- `z`: a learned scalar controlling the smoothing penalty

The layer output is:

~~~text
LeakyReLU(L+) - LeakyReLU(L-) - z * LeakyReLU(L0)
~~~

This is a **custom CNN block**, not an RNN and not a rigorous implementation of knot-theoretic invariants.

## Architecture

The model is:

1. `AuthenticSkeinConv(1, 16)`
2. `MaxPool2d(2)`
3. `AuthenticSkeinConv(16, 32)`
4. `MaxPool2d(2)`
5. `Linear(32 * 7 * 7, 10)`

## Dataset and training

- Dataset: **MNIST**
- Framework: **PyTorch**
- Optimizer: **Adam**
- Loss: **CrossEntropyLoss**
- Epochs: **5**
- Batch size: **128**

## Result

In the current run, the model reached **97.69% test accuracy after 5 epochs**.

That suggests the constrained layer can learn useful image features despite the mirrored-filter and smoothing constraints.

## What this is

This project is best described as:

> a constrained CNN using a learned filter, its spatially flipped counterpart, and a learned smoothing penalty

## Limitations

This is an exploratory experiment, not a strong research claim. It does not include:

- a matched baseline run in the repo
- ablation studies
- multiple seeds
- harder datasets
- evidence that the skein framing adds more than a structural bias

## License

MIT License

## Author

**Russel Maytham**
