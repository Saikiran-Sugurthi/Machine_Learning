# CNN & Deep Learning — My ML Notes
> Covers everything from neurons → layers → training → CNN architecture → feature vectors

---

## 1. What is Machine Learning?

- **Traditional programming:** You write the rules → computer follows them
- **Machine Learning:** You give examples + labels → computer figures out the rules itself
- **Deep Learning** is a type of ML that uses **Neural Networks** — powerful for images, audio, language

---

## 2. The Neuron — Building Block

A single neuron does this:

```
output = activation( w₁×x₁ + w₂×x₂ + ... + wₙ×xₙ + bias )
```

| Part | What it is |
|------|-----------|
| x₁, x₂… | Inputs (data values) |
| w₁, w₂… | Weights — how important each input is (LEARNED) |
| bias | A starting offset (LEARNED) |
| activation | A function that adds non-linearity (ReLU, Sigmoid…) |

**Key:** Weights and bias are what the network **learns** during training.

---

## 3. Why Activation Functions?

Without activation: stacking layers = still one linear equation (useless)

```
W₃(W₂(W₁x)) = Wx  ← just one linear layer no matter how deep!
```

Activation functions add **non-linearity** so the network can learn complex patterns.

| Activation | Formula | Use where |
|-----------|---------|-----------|
| ReLU | max(0, x) | Hidden layers — default choice |
| Sigmoid | 1/(1+e⁻ˣ) | Binary classification output |
| Softmax | eˣⁱ/Σeˣʲ | Multi-class output (probabilities sum to 1) |
| Linear (none) | x | Regression output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs |

**Vanishing gradient:** Sigmoid/Tanh squash gradients → early layers stop learning. ReLU fixes this.

**Dying ReLU:** Neurons always get negative input → output always 0 → fixed by Leaky ReLU.

---

## 4. Layers in a Neural Network

### Input Layer
- NOT a real neuron — just holds raw data values
- No weights, no activation, no computation
- One node per input value (e.g. 784 nodes for 28×28 image)

### Hidden Layers
- Where actual computation happens
- Each neuron: receives all outputs from previous layer → weighted sum → activation → output
- Earlier layers detect simple features, deeper layers detect complex features
- This is why it's called **Deep** Learning

### Output Layer
- Formats the final answer for your specific task
- Number of neurons = number of classes (or 1 for regression)
- Uses a different activation than hidden layers

> **Note:** When people say "2-layer network" they mean 1 hidden + 1 output. Input layer is NOT counted.

---

## 5. Why Multiple Hidden Layers?

One layer *can* learn anything but needs impossibly many neurons.
Multiple layers build understanding **step by step**:

```
Layer 1 → edges & lines
Layer 2 → shapes & curves  (built from Layer 1's edges)
Layer 3 → parts like eyes, nose  (built from Layer 2's shapes)
Layer 4 → full object like "face"  (built from Layer 3's parts)
```

Each layer only needs to learn a small step beyond the previous. Efficient, reusable.

---

## 6. How Training Works

### Full Training Loop:

**Step 0** — You set hyperparameters. Weights initialised to random small numbers.

**Step 1 — Forward Pass**
Data flows left → right through every layer. Each layer computes its output and passes it to the next. Final layer gives a prediction.

**Step 2 — Compute Loss**
Compare prediction to the **real label from the dataset**.
Loss = how wrong the prediction was. High loss = very wrong.

**Step 3 — Backpropagation**
Error signal travels right → left through ALL layers simultaneously.
Calculus (chain rule) figures out which weights caused the error.

**Step 4 — Gradient Descent**
Nudge all weights slightly in the direction that reduces loss.

**Step 5 — Repeat**
Do steps 1–4 for every batch, for many epochs. Weights stop being random and become meaningful detectors.

---

## 7. Parameters vs Hyperparameters ⚠️ Common Interview Question

| | Parameters | Hyperparameters |
|--|-----------|----------------|
| **What** | Weights, biases, kernel values | Learning rate, batch size, #layers, filter size |
| **Who sets it** | Learned automatically by backprop | You set before training |
| **Changes during training?** | Yes — every step | No — fixed throughout |
| **Examples** | 9 values in a 3×3 filter | filter size = 3×3 (that's your choice) |

---

## 8. Why FNN Fails for Images

A 1920×1080 RGB image = **6,220,800 input values** (pixels × 3 colour channels).

An FNN would need 6.2M input neurons, each connected to every hidden neuron → billions of weights → computationally impossible.

Also FNN has no concept of spatial structure — a pixel's neighbours matter, but FNN treats all inputs independently.

**CNN solves this with:**
1. Small filters (3×3 = only 9 weights)
2. Parameter sharing (same 9 weights reused across entire image)

---

## 9. CNN — Convolutional Neural Network

### Full Architecture Flow:
```
Image → [Conv + ReLU] → [Pooling] → [Conv + ReLU] → [Pooling] → Flatten → FC Layer → Output
```

---

### 9a. Convolutional Layer

A small filter/kernel (e.g. 3×3) slides across the image.
At each position: **element-wise multiply + sum = one output value**

Example with 3×3 filter on 5×5 image:
```
At position (0,0):
(img values × filter values) summed = one number in output
```

**Output size formula:**
```
output = (n - f + 1) × (n - f + 1)
5×5 image, 3×3 filter → (5-3+1) = 3 → output is 3×3
```

**Stride** = how many pixels the filter moves each step
- Stride 1 → output size = (n-f+1)
- Larger stride → smaller output, faster

**Key properties:**
- Kernels are **learnable parameters** (w, b) — learned via backprop, not hand-designed
- **Parameter sharing** — same 9 weights reused across entire image → massive efficiency
- Each filter learns to detect ONE type of feature (edge, curve, texture...)

---

### 9b. Multiple Filters

A CNN layer uses MANY filters simultaneously (32, 64, 128...):
- Filter 1 → detects vertical edges → produces one feature map
- Filter 2 → detects horizontal edges → produces another feature map
- Filter 3 → detects curves → another feature map
- … and so on

All filters are the SAME filters for ALL images — no "cat filters" or "dog filters".
The same "edge detector" fires on a cat's ear AND a dog's ear AND a building's wall.

**Feature map** = the output grid from one filter applied to one image = shows WHERE that pattern was detected.

---

### 9c. Pooling Layer

Reduces the size of feature maps (downsampling) while keeping the important information.

**Max Pooling** (most common): take the maximum value in each 2×2 region.

Why it helps:
- Reduces computation
- Makes detection position-independent (cat's ear slightly shifted = still detected)
- Reduces overfitting

---

### 9d. Flatten

After Conv + Pooling, output is a 3D block (height × width × num_filters).
FC layer needs a 1D list.
**Flatten** just unrolls that 3D block into one long vector. No computation — just reshaping.

---

### 9e. FC Layer (Fully Connected / ANN part)

Takes the flattened feature vector → does regular FNN classification.

This is where class decisions are made:
- Learned: "HIGH edge score + HIGH pointy-ear score + LOW snout score → cat"
- Learned: "MED edge + LOW pointy-ear + HIGH snout → dog"

---

## 10. Feature Vectors — The Core Insight

After flatten, each image becomes a **list of numbers = feature vector (embedding)**.

This is the image's mathematical fingerprint:

```
Cat 1  →  [0.91, 0.87, 0.95, 0.08]
Cat 2  →  [0.88, 0.83, 0.91, 0.05]  ← similar to Cat 1!
Dog    →  [0.71, 0.55, 0.10, 0.92]  ← very different
```

Same class → similar vectors → FC layer maps to same output.

**This concept is everywhere in modern AI:**
- Face recognition (same person = similar vectors)
- Spotify recommendations (similar songs = similar vectors)
- LLMs like ChatGPT (similar meaning = similar vectors)
- Google image search

---

## 11. Neural Network Architectures — Quick Reference

| Architecture | Best for | Key idea |
|---|---|---|
| **FNN / MLP** | Tabular data, structured numbers | Fully connected layers |
| **CNN** | Images, video | Filters slide across image, parameter sharing |
| **RNN / LSTM** | Sequences (old approach) | Hidden state memory passed forward step by step |
| **Transformer** | Language, modern everything | Self-attention — every token looks at every other simultaneously |
| **GAN** | Generating new content | Generator vs Discriminator compete |

---

## 12. Key Terms Cheat Sheet

| Term | Meaning |
|------|---------|
| Epoch | One full pass through entire training dataset |
| Batch size | Number of examples processed before one weight update |
| Learning rate | How big each gradient descent step is |
| Loss function | Measures how wrong predictions are (Cross-Entropy for classification) |
| Optimizer | Algorithm that updates weights (Adam is most popular) |
| Overfitting | Memorises training data, fails on new data |
| Underfitting | Model too simple to learn the pattern |
| Kernel / Filter | Small grid of weights that slides across image |
| Feature map | Output of one filter applied to one image |
| Stride | How many pixels filter moves each step |
| Flatten | Reshape 3D block into 1D vector |
| Feature vector | The list of numbers after flatten — image's fingerprint |
| Embedding | Same as feature vector — numerical representation of data |
| Inference | Using a trained model on new data (deployment) |
| Parameter sharing | Same filter weights reused across the entire image |

---

## 13. Interview Quick Answers

**Q: Why activation functions?**
Without them, stacking layers = one linear equation. Activation adds non-linearity so the network can learn complex patterns.

**Q: Parameters vs hyperparameters?**
Parameters (weights, biases, kernel values) are learned automatically by backprop. Hyperparameters (learning rate, filter size, #layers) are set by the engineer before training.

**Q: Why CNN over FNN for images?**
FNN needs one neuron per pixel → millions of weights. CNN uses small filters (9 weights) reused across entire image via parameter sharing → efficient and spatially aware.

**Q: Do filters change per class (cat filter vs dog filter)?**
No. One shared set of filters for all images. Filters detect low-level features (edges, textures). The FC layer learns which combination of filter outputs maps to which class.

**Q: What is a feature vector?**
The flattened output after all conv+pool layers — a list of numbers that is the image's numerical fingerprint. Similar images produce similar feature vectors.

**Q: What is backpropagation?**
After computing loss, the error signal travels backward through ALL layers simultaneously using chain rule (calculus), calculating each weight's contribution to the error so gradient descent can update them.

---

*Notes compiled from hands-on learning — CNN Architecture, Apna College*
