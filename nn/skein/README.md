
# Skein-RNN: Topological Gating

A highly experimental neural network architecture that replaces standard LSTM/GRU sigmoid gates with a recursive knot-theory invariant (the Conway Skein Relation).

This repo contains the Python-based PyTorch training environment.

## The Math: Conway Skein Identity as a Recurrent Gate
Standard Recurrent Neural Networks (RNNs) use "forget gates" to process sequences. This experiment asks: *What if we treat a data sequence as a physical braid being tied in real-time?*

Instead of a standard activation function, the network's hidden state is updated by resolving the topological interference of the data using the Conway Skein equation:
`L+ - L- = z * L0`

Where the current state and the incoming data row are projected into three matrices:
* **L+ (Over-crossing):** The positive interference.
* **L- (Under-crossing):** The negative interference.
* **L0 (Smoothed):** The baseline/unknotted state.
* **z:** A learned parameter balancing the structural tension.

By calculating the difference at every step, the network "unties" the sequence to find its fundamental invariant.
