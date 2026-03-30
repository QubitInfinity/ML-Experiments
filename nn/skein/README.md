
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

## The Output:
Training 'Authentic' Skein-CNN on cpu...
Epoch 1 | Accuracy: 96.87% | Z-val: 0.887
Epoch 2 | Accuracy: 96.91% | Z-val: 0.831
Epoch 3 | Accuracy: 96.99% | Z-val: 0.808
Epoch 4 | Accuracy: 97.26% | Z-val: 0.785
Epoch 5 | Accuracy: 97.69% | Z-val: 0.758

Success. The model is now using constrained weights to simulate topological crossings.

==================================================
--- Authentic Skein-CNN Inference Test ---
==================================================
Final Learned 'Z' Smoothing Weight (Layer 1): 0.7581
Final Learned 'Z' Smoothing Weight (Layer 2): 1.8039

Sample   | Target   | Predicted  | Result
---------------------------------------------
1        | 7        | 7          | ✅ Match
2        | 2        | 2          | ✅ Match
3        | 1        | 1          | ✅ Match
4        | 0        | 0          | ✅ Match
5        | 4        | 4          | ✅ Match
6        | 1        | 1          | ✅ Match
7        | 4        | 4          | ✅ Match
8        | 9        | 9          | ✅ Match
9        | 5        | 5          | ✅ Match
10       | 9        | 9          | ✅ Match
11       | 0        | 0          | ✅ Match
12       | 6        | 6          | ✅ Match
13       | 9        | 9          | ✅ Match
14       | 0        | 0          | ✅ Match
15       | 1        | 1          | ✅ Match
---------------------------------------------
Inference Accuracy on this subset: 15/15 (100.0%)
==================================================

Process finished with exit code 0
