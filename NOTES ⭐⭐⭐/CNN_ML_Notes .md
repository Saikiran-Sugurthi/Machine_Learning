# 🧠 CNN & Deep Learning — ML Notes

> From neurons → layers → training → CNN architecture → feature vectors  
> Everything I need to remember. Built through hands-on learning.

---

## Table of Contents
1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [The Neuron — Building Block](#2-the-neuron--building-block)
3. [Why Activation Functions?](#3-why-activation-functions)
4. [Layers in a Neural Network](#4-layers-in-a-neural-network)
5. [Why Multiple Hidden Layers?](#5-why-multiple-hidden-layers)
6. [How Training Works](#6-how-training-works)
7. [Parameters vs Hyperparameters](#7-parameters-vs-hyperparameters-)
8. [Why FNN Fails for Images](#8-why-fnn-fails-for-images)
9. [CNN — Full Architecture](#9-cnn--full-architecture)
10. [Important vs Unwanted Pixels](#10-important-vs-unwanted-pixels)
11. [Feature Vectors — The Core Insight](#11-feature-vectors--the-core-insight)
12. [All Architectures — Quick Reference](#12-all-architectures--quick-reference)
13. [Key Terms Cheat Sheet](#13-key-terms-cheat-sheet)
14. [Interview Quick Answers](#14-interview-quick-answers)

---

## 1. What is Machine Learning?

| | Traditional Programming | Machine Learning |
|--|--|--|
| **Approach** | You write the rules → computer follows | You give labelled examples → model learns rules |
| **Example** | `IF temp > 38°C → fever` | Show 10,000 patient records + labels → model figures it out |
| **Limitation** | Can't handle complex patterns | Needs lots of data |

**Deep Learning** is a subset of ML using **Neural Networks** — powerful for images, audio, language.

```
AI  ⊃  Machine Learning  ⊃  Deep Learning (Neural Networks)
```

---

## 2. The Neuron — Building Block

```
output = activation( w₁×x₁ + w₂×x₂ + ... + wₙ×xₙ + bias )
```

<img src="https://raw.githubusercontent.com/neuron-diagram/placeholder/main/neuron.svg" alt="neuron diagram" />

<!-- GitHub renders SVG files. Inline SVG below works on GitHub too: -->

<svg viewBox="0 0 600 150" xmlns="http://www.w3.org/2000/svg" width="100%">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#7c83f5" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <circle cx="70" cy="44" r="20" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="70" y="49" text-anchor="middle" font-size="13" fill="#2e7d32" font-family="monospace" font-weight="bold">x₁</text>
  <circle cx="70" cy="96" r="20" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="70" y="101" text-anchor="middle" font-size="13" fill="#2e7d32" font-family="monospace" font-weight="bold">x₂</text>
  <circle cx="70" cy="140" r="16" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="70" y="145" text-anchor="middle" font-size="12" fill="#2e7d32" font-family="monospace">x₃</text>
  <line x1="90" y1="44" x2="252" y2="82" stroke="#388e3c" stroke-width="2" marker-end="url(#ar)"/>
  <line x1="90" y1="96" x2="252" y2="90" stroke="#388e3c" stroke-width="3" marker-end="url(#ar)"/>
  <line x1="86" y1="136" x2="252" y2="100" stroke="#388e3c" stroke-width="1" marker-end="url(#ar)"/>
  <text x="158" y="58" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif">w₁=0.3</text>
  <text x="165" y="90" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif">w₂=0.8 (most important)</text>
  <text x="158" y="128" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif">w₃=0.1</text>
  <circle cx="295" cy="90" r="36" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="295" y="84" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif" font-weight="bold">Σ + bias</text>
  <text x="295" y="100" text-anchor="middle" font-size="10" fill="#6a1b9a" font-family="sans-serif">→ activate</text>
  <line x1="331" y1="90" x2="395" y2="90" stroke="#7c83f5" stroke-width="2" marker-end="url(#ar)"/>
  <circle cx="422" cy="90" r="24" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="422" y="86" text-anchor="middle" font-size="11" fill="#1565c0" font-family="sans-serif" font-weight="bold">out</text>
  <text x="422" y="100" text-anchor="middle" font-size="11" fill="#1565c0" font-family="monospace">0.72</text>
  <text x="70" y="148" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">Inputs</text>
  <text x="158" y="148" text-anchor="middle" font-size="9" fill="#f57c00" font-family="sans-serif">Weights (LEARNED)</text>
  <text x="295" y="140" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">Neuron</text>
  <text x="422" y="128" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">Output</text>
</svg>

| Part | What it is | Learned? |
|------|-----------|----------|
| x₁, x₂… | Inputs — raw data values | ❌ No — you provide them |
| w₁, w₂… | Weights — importance of each input | ✅ Yes — by backprop |
| bias | Starting offset so neuron isn't stuck at 0 | ✅ Yes — by backprop |
| activation | Adds non-linearity | ❌ No — you choose which one |

---

## 3. Why Activation Functions?

> **Without activation:** No matter how many layers you stack, it collapses to one linear equation.

```
W₃(W₂(W₁x)) = Wx   ← same as ONE layer. Useless for complex data.
```

<svg viewBox="0 0 640 115" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="640" height="115" fill="#fafafa" rx="10"/>
  <text x="155" y="18" text-anchor="middle" font-size="11" fill="#c62828" font-family="sans-serif" font-weight="bold">❌ Without activation — collapses</text>
  <rect x="20" y="28" width="62" height="28" rx="5" fill="#fff" stroke="#ccc" stroke-width="1"/>
  <text x="51" y="47" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">Layer 1</text>
  <rect x="118" y="28" width="62" height="28" rx="5" fill="#fff" stroke="#ccc" stroke-width="1"/>
  <text x="149" y="47" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">Layer 2</text>
  <rect x="216" y="28" width="62" height="28" rx="5" fill="#fff" stroke="#ccc" stroke-width="1"/>
  <text x="247" y="47" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">Layer 3</text>
  <line x1="82" y1="42" x2="118" y2="42" stroke="#bbb" stroke-width="1.5"/>
  <line x1="180" y1="42" x2="216" y2="42" stroke="#bbb" stroke-width="1.5"/>
  <text x="155" y="82" text-anchor="middle" font-size="11" fill="#c62828" font-family="sans-serif">= just Wx (one line, no matter how deep)</text>
  <text x="155" y="100" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">Can only draw straight decision boundaries</text>
  <text x="487" y="18" text-anchor="middle" font-size="11" fill="#2e7d32" font-family="sans-serif" font-weight="bold">✅ With ReLU — truly deep</text>
  <rect x="352" y="28" width="62" height="28" rx="5" fill="#fff" stroke="#ccc" stroke-width="1"/>
  <text x="383" y="47" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">Layer 1</text>
  <rect x="450" y="20" width="36" height="44" rx="6" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="468" y="46" text-anchor="middle" font-size="15" fill="#2e7d32" font-family="sans-serif" font-weight="bold">σ</text>
  <rect x="522" y="28" width="62" height="28" rx="5" fill="#fff" stroke="#ccc" stroke-width="1"/>
  <text x="553" y="47" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">Layer 2</text>
  <line x1="414" y1="42" x2="450" y2="42" stroke="#2e7d32" stroke-width="1.5"/>
  <line x1="486" y1="42" x2="522" y2="42" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="487" y="82" text-anchor="middle" font-size="11" fill="#2e7d32" font-family="sans-serif">Non-linear! Learns complex patterns</text>
  <text x="487" y="100" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">Can separate any decision boundary</text>
</svg>

| Activation | Formula | Use where | Watch out |
|-----------|---------|-----------|-----------|
| **ReLU** | max(0, x) | Hidden layers — default | Dying ReLU |
| Leaky ReLU | max(0.01x, x) | Fix for dying ReLU | — |
| Sigmoid | 1/(1+e⁻ˣ) | Binary classification output | Vanishing gradient |
| Softmax | eˣⁱ/Σeˣʲ | Multi-class output | Output layer only |
| Linear | x | Regression output | — |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs | Vanishing gradient |
| GELU | smooth ReLU | Transformers / LLMs | — |

> ⚠️ **Vanishing gradient:** Sigmoid/Tanh squash gradients to <0.25. Multiply across 50 layers → gradient → 0 → early layers stop learning.  
> ⚠️ **Dying ReLU:** Neuron always gets negative input → always outputs 0 → permanently dead. Fix: Leaky ReLU.

---

## 4. Layers in a Neural Network

<svg viewBox="0 0 680 195" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="680" height="195" fill="#fafafa" rx="10"/>
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#7c83f5" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="78" y="18" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Input layer</text>
  <circle cx="78" cy="55" r="22" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="78" y="59" text-anchor="middle" font-size="12" fill="#00695c" font-family="monospace" font-weight="bold">x₁</text>
  <circle cx="78" cy="112" r="22" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="78" y="116" text-anchor="middle" font-size="12" fill="#00695c" font-family="monospace" font-weight="bold">x₂</text>
  <circle cx="78" cy="165" r="22" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="78" y="169" text-anchor="middle" font-size="12" fill="#00695c" font-family="monospace" font-weight="bold">x₃</text>
  <text x="78" y="194" text-anchor="middle" font-size="9" fill="#c62828" font-family="sans-serif">no computation</text>
  <text x="310" y="18" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Hidden layer (real neurons)</text>
  <circle cx="310" cy="45" r="24" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="310" y="40" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="sans-serif">Σ+ReLU</text>
  <text x="310" y="54" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="monospace">→0.62</text>
  <circle cx="310" cy="110" r="24" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="310" y="105" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="sans-serif">Σ+ReLU</text>
  <text x="310" y="119" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="monospace">→0.14</text>
  <circle cx="310" cy="170" r="24" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="310" y="165" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="sans-serif">Σ+ReLU</text>
  <text x="310" y="179" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="monospace">→0.87</text>
  <text x="310" y="194" text-anchor="middle" font-size="9" fill="#2e7d32" font-family="sans-serif">weights × inputs + bias → activate</text>
  <line x1="100" y1="55"  x2="286" y2="47"  stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="55"  x2="286" y2="112" stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="55"  x2="286" y2="170" stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="112" x2="286" y2="47"  stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="112" x2="286" y2="112" stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="112" x2="286" y2="170" stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="165" x2="286" y2="47"  stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="165" x2="286" y2="112" stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <line x1="100" y1="165" x2="286" y2="170" stroke="#00695c" stroke-width="0.5" opacity="0.6"/>
  <text x="540" y="18" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Output layer</text>
  <circle cx="540" cy="82" r="26" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="540" y="78" text-anchor="middle" font-size="11" fill="#c62828" font-family="sans-serif" font-weight="bold">cat</text>
  <text x="540" y="92" text-anchor="middle" font-size="11" fill="#c62828" font-family="monospace">92%</text>
  <circle cx="540" cy="148" r="26" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="540" y="144" text-anchor="middle" font-size="11" fill="#c62828" font-family="sans-serif" font-weight="bold">dog</text>
  <text x="540" y="158" text-anchor="middle" font-size="11" fill="#c62828" font-family="monospace">8%</text>
  <line x1="334" y1="45"  x2="514" y2="80"  stroke="#6a1b9a" stroke-width="0.5" opacity="0.6"/>
  <line x1="334" y1="45"  x2="514" y2="148" stroke="#6a1b9a" stroke-width="0.5" opacity="0.6"/>
  <line x1="334" y1="110" x2="514" y2="80"  stroke="#6a1b9a" stroke-width="0.5" opacity="0.6"/>
  <line x1="334" y1="110" x2="514" y2="148" stroke="#6a1b9a" stroke-width="0.5" opacity="0.6"/>
  <line x1="334" y1="170" x2="514" y2="80"  stroke="#6a1b9a" stroke-width="0.5" opacity="0.6"/>
  <line x1="334" y1="170" x2="514" y2="148" stroke="#6a1b9a" stroke-width="0.5" opacity="0.6"/>
  <text x="540" y="194" text-anchor="middle" font-size="9" fill="#f57c00" font-family="sans-serif">Sigmoid/Softmax → probabilities</text>
</svg>

> 📝 **Counting layers:** Input layer is NOT counted. A "2-layer network" = 1 hidden + 1 output.

---

## 5. Why Multiple Hidden Layers?

<svg viewBox="0 0 680 150" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="680" height="150" fill="#fafafa" rx="10"/>
  <defs>
    <marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="10" y="50" width="100" height="55" rx="8" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="60" y="74" text-anchor="middle" font-size="11" fill="#00695c" font-family="sans-serif" font-weight="bold">Raw pixels</text>
  <text x="60" y="92" text-anchor="middle" font-size="10" fill="#555" font-family="monospace">0.2, 0.8…</text>
  <line x1="110" y1="78" x2="140" y2="78" stroke="#555" stroke-width="1.5" marker-end="url(#ar3)"/>
  <rect x="144" y="36" width="108" height="84" rx="8" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="198" y="60" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif" font-weight="bold">Layer 1</text>
  <text x="198" y="78" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif">edges &amp;</text>
  <text x="198" y="94" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif">lines</text>
  <text x="198" y="132" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">simple features</text>
  <line x1="252" y1="78" x2="282" y2="78" stroke="#555" stroke-width="1.5" marker-end="url(#ar3)"/>
  <rect x="286" y="36" width="108" height="84" rx="8" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="340" y="60" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif" font-weight="bold">Layer 2</text>
  <text x="340" y="78" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif">shapes &amp;</text>
  <text x="340" y="94" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif">curves</text>
  <text x="340" y="132" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">medium features</text>
  <line x1="394" y1="78" x2="424" y2="78" stroke="#555" stroke-width="1.5" marker-end="url(#ar3)"/>
  <rect x="428" y="36" width="108" height="84" rx="8" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="482" y="60" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif" font-weight="bold">Layer 3</text>
  <text x="482" y="78" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif">eyes, nose</text>
  <text x="482" y="94" text-anchor="middle" font-size="11" fill="#6a1b9a" font-family="sans-serif">ears…</text>
  <text x="482" y="132" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">complex features</text>
  <line x1="536" y1="78" x2="566" y2="78" stroke="#555" stroke-width="1.5" marker-end="url(#ar3)"/>
  <rect x="570" y="54" width="96" height="48" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="618" y="76" text-anchor="middle" font-size="12" fill="#2e7d32" font-family="sans-serif" font-weight="bold">FACE</text>
  <text x="618" y="92" text-anchor="middle" font-size="11" fill="#2e7d32" font-family="monospace">97%</text>
</svg>

> 💡 Each layer only needs to learn a **small step** beyond the previous. No starting from scratch. This is why it's called **Deep** Learning.

---

## 6. How Training Works

<svg viewBox="0 0 680 95" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="680" height="95" fill="#fafafa" rx="10"/>
  <defs>
    <marker id="ar4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="10" y="28" width="96" height="40" rx="8" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="58" y="46" text-anchor="middle" font-size="10" fill="#00695c" font-family="sans-serif" font-weight="bold">Labelled</text>
  <text x="58" y="60" text-anchor="middle" font-size="10" fill="#00695c" font-family="sans-serif">Dataset</text>
  <line x1="106" y1="48" x2="128" y2="48" stroke="#555" stroke-width="1.5" marker-end="url(#ar4)"/>
  <rect x="132" y="28" width="96" height="40" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="180" y="46" text-anchor="middle" font-size="10" fill="#1565c0" font-family="sans-serif" font-weight="bold">Forward</text>
  <text x="180" y="60" text-anchor="middle" font-size="10" fill="#1565c0" font-family="sans-serif">Pass →</text>
  <line x1="228" y1="48" x2="250" y2="48" stroke="#555" stroke-width="1.5" marker-end="url(#ar4)"/>
  <rect x="254" y="28" width="96" height="40" rx="8" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="302" y="46" text-anchor="middle" font-size="10" fill="#c62828" font-family="sans-serif" font-weight="bold">Compute</text>
  <text x="302" y="60" text-anchor="middle" font-size="10" fill="#c62828" font-family="sans-serif">Loss</text>
  <line x1="350" y1="48" x2="372" y2="48" stroke="#555" stroke-width="1.5" marker-end="url(#ar4)"/>
  <rect x="376" y="28" width="96" height="40" rx="8" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="424" y="46" text-anchor="middle" font-size="10" fill="#6a1b9a" font-family="sans-serif" font-weight="bold">Backprop</text>
  <text x="424" y="60" text-anchor="middle" font-size="10" fill="#6a1b9a" font-family="sans-serif">← blame</text>
  <line x1="472" y1="48" x2="494" y2="48" stroke="#555" stroke-width="1.5" marker-end="url(#ar4)"/>
  <rect x="498" y="28" width="96" height="40" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="546" y="46" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif" font-weight="bold">Update</text>
  <text x="546" y="60" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif">Weights</text>
  <path d="M594 48 Q650 48 650 82 Q650 90 302 90 Q10 90 10 82 Q10 70 10 68" fill="none" stroke="#aaa" stroke-width="1.2" stroke-dasharray="5 4" marker-end="url(#ar4)"/>
  <text x="620" y="88" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">repeat</text>
</svg>

| Step | What happens |
|------|-------------|
| **0 — Init** | You set hyperparameters. Weights set to random small numbers. |
| **1 — Forward pass** | Data flows left→right through all layers. Get a prediction. |
| **2 — Compute loss** | Compare prediction to **real label from dataset**. Measure how wrong. |
| **3 — Backprop** | Error travels right←left through ALL layers via chain rule (calculus). |
| **4 — Update weights** | Gradient descent nudges all weights to reduce loss. |
| **5 — Repeat** | Thousands of times until weights become meaningful detectors. |

> 📌 **Dataset is used every single step.** Model always compares against real labels — never learns from its own outputs alone.

---

## 7. Parameters vs Hyperparameters ⚠️

> **Most common interview question. Do NOT mix these up.**

| | Parameters | Hyperparameters |
|--|-----------|----------------|
| **What** | Weights, biases, kernel values | Learning rate, batch size, #layers, filter size |
| **Who sets it** | Learned automatically by backprop | **You** set before training |
| **Changes during training?** | ✅ Yes — every step | ❌ No — fixed throughout |
| **Example** | The 9 values inside a 3×3 filter | The decision to use a 3×3 filter at all |

---

## 8. Why FNN Fails for Images

<svg viewBox="0 0 640 100" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="640" height="100" fill="#fafafa" rx="10"/>
  <rect x="10" y="18" width="110" height="64" rx="8" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="65" y="44" text-anchor="middle" font-size="11" fill="#00695c" font-family="sans-serif" font-weight="bold">1920×1080</text>
  <text x="65" y="60" text-anchor="middle" font-size="11" fill="#00695c" font-family="sans-serif">RGB image</text>
  <text x="65" y="74" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">×3 channels</text>
  <text x="155" y="54" text-anchor="middle" font-size="20" fill="#aaa" font-family="sans-serif">→</text>
  <rect x="182" y="8" width="152" height="84" rx="8" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="258" y="32" text-anchor="middle" font-size="11" fill="#c62828" font-family="sans-serif" font-weight="bold">FNN input layer</text>
  <text x="258" y="52" text-anchor="middle" font-size="16" fill="#c62828" font-family="monospace" font-weight="bold">6,220,800</text>
  <text x="258" y="70" text-anchor="middle" font-size="10" fill="#c62828" font-family="sans-serif">neurons needed!</text>
  <text x="258" y="84" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">each connected to every hidden neuron = billions of weights</text>
  <text x="370" y="54" text-anchor="middle" font-size="20" fill="#aaa" font-family="sans-serif">→</text>
  <rect x="390" y="18" width="240" height="64" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="510" y="40" text-anchor="middle" font-size="11" fill="#2e7d32" font-family="sans-serif" font-weight="bold">✅ CNN solution</text>
  <text x="510" y="58" text-anchor="middle" font-size="11" fill="#2e7d32" font-family="sans-serif">3×3 filter = only 9 weights</text>
  <text x="510" y="74" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif">same 9 reused across entire image (parameter sharing)</text>
</svg>

---

## 9. CNN — Full Architecture

```
Image → [Conv+ReLU] → [Pooling] → [Conv+ReLU] → [Pooling] → Flatten → FC Layer → Output
          ↑                           ↑
     eyes,nose,ears              head, body
     (simple features)         (complex features)
```

### 9a. Convolution — how it works

<svg viewBox="0 0 580 190" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="580" height="190" fill="#fafafa" rx="10"/>
  <text x="85" y="16" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Image (5×5)</text>
  <rect x="10" y="22" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="26" y="42" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="42" y="22" width="32" height="32" rx="3" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="58" y="42" text-anchor="middle" font-size="12" fill="#1565c0" font-family="monospace" font-weight="bold">0</text>
  <rect x="74" y="22" width="32" height="32" rx="3" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="90" y="42" text-anchor="middle" font-size="12" fill="#1565c0" font-family="monospace" font-weight="bold">1</text>
  <rect x="106" y="22" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="122" y="42" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <rect x="138" y="22" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="154" y="42" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="10" y="54" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="26" y="74" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="42" y="54" width="32" height="32" rx="3" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="58" y="74" text-anchor="middle" font-size="12" fill="#1565c0" font-family="monospace" font-weight="bold">0</text>
  <rect x="74" y="54" width="32" height="32" rx="3" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="90" y="74" text-anchor="middle" font-size="12" fill="#1565c0" font-family="monospace" font-weight="bold">1</text>
  <rect x="106" y="54" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="122" y="74" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <rect x="138" y="54" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="154" y="74" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="10" y="86" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="26" y="106" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="42" y="86" width="32" height="32" rx="3" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="58" y="106" text-anchor="middle" font-size="12" fill="#1565c0" font-family="monospace" font-weight="bold">0</text>
  <rect x="74" y="86" width="32" height="32" rx="3" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="90" y="106" text-anchor="middle" font-size="12" fill="#1565c0" font-family="monospace" font-weight="bold">1</text>
  <rect x="106" y="86" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="122" y="106" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <rect x="138" y="86" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="154" y="106" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="10" y="118" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="26" y="138" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <rect x="42" y="118" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="58" y="138" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="74" y="118" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="90" y="138" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <rect x="106" y="118" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="122" y="138" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="138" y="118" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="154" y="138" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <rect x="10" y="150" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="26" y="170" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="42" y="150" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="58" y="170" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="74" y="150" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="90" y="170" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <rect x="106" y="150" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="122" y="170" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">0</text>
  <rect x="138" y="150" width="32" height="32" rx="3" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="154" y="170" text-anchor="middle" font-size="12" fill="#aaa" font-family="monospace">1</text>
  <text x="85" y="188" text-anchor="middle" font-size="9" fill="#1565c0" font-family="sans-serif">blue = current filter position</text>
  <text x="275" y="16" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Filter 3×3 (edge detector)</text>
  <rect x="236" y="22" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="252" y="42" text-anchor="middle" font-size="13" fill="#f57c00" font-family="monospace" font-weight="bold">1</text>
  <rect x="268" y="22" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="284" y="42" text-anchor="middle" font-size="13" fill="#f57c00" font-family="monospace" font-weight="bold">0</text>
  <rect x="300" y="22" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="316" y="42" text-anchor="middle" font-size="12" fill="#f57c00" font-family="monospace" font-weight="bold">-1</text>
  <rect x="236" y="54" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="252" y="74" text-anchor="middle" font-size="13" fill="#f57c00" font-family="monospace" font-weight="bold">1</text>
  <rect x="268" y="54" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="284" y="74" text-anchor="middle" font-size="13" fill="#f57c00" font-family="monospace" font-weight="bold">0</text>
  <rect x="300" y="54" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="316" y="74" text-anchor="middle" font-size="12" fill="#f57c00" font-family="monospace" font-weight="bold">-1</text>
  <rect x="236" y="86" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="252" y="106" text-anchor="middle" font-size="13" fill="#f57c00" font-family="monospace" font-weight="bold">1</text>
  <rect x="268" y="86" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="284" y="106" text-anchor="middle" font-size="13" fill="#f57c00" font-family="monospace" font-weight="bold">0</text>
  <rect x="300" y="86" width="32" height="32" rx="3" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="316" y="106" text-anchor="middle" font-size="12" fill="#f57c00" font-family="monospace" font-weight="bold">-1</text>
  <text x="275" y="138" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">9 weights total</text>
  <text x="275" y="154" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif">learned via backprop</text>
  <text x="380" y="70" text-anchor="middle" font-size="22" fill="#aaa" font-family="sans-serif">→</text>
  <text x="490" y="16" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Output (3×3)</text>
  <rect x="448" y="22" width="40" height="40" rx="4" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="468" y="46" text-anchor="middle" font-size="14" fill="#2e7d32" font-family="monospace" font-weight="bold">-3</text>
  <rect x="490" y="22" width="40" height="40" rx="4" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="510" y="46" text-anchor="middle" font-size="14" fill="#2e7d32" font-family="monospace" font-weight="bold">-3</text>
  <rect x="532" y="22" width="40" height="40" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="552" y="46" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">…</text>
  <rect x="448" y="64" width="40" height="40" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="468" y="88" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">…</text>
  <rect x="490" y="64" width="40" height="40" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="510" y="88" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">…</text>
  <rect x="532" y="64" width="40" height="40" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="552" y="88" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">…</text>
  <text x="490" y="125" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif">feature map — WHERE edge found</text>
</svg>

```
Output size = (n - f + 1) × (n - f + 1)
5×5 image, 3×3 filter → (5-3+1) = 3 → output is 3×3

Calculation at position (0,0):
(0×1)+(0×0)+(1×-1)+(0×1)+(0×0)+(1×-1)+(0×1)+(0×0)+(1×-1) = -3
```

### 9b. Max Pooling

<svg viewBox="0 0 440 115" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="440" height="115" fill="#fafafa" rx="10"/>
  <text x="88" y="16" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Feature map (4×4)</text>
  <rect x="10" y="22" width="38" height="38" rx="4" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="29" y="45" text-anchor="middle" font-size="14" fill="#2e7d32" font-family="monospace" font-weight="bold">9</text>
  <rect x="48" y="22" width="38" height="38" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="67" y="45" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">2</text>
  <rect x="86" y="22" width="38" height="38" rx="4" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="105" y="45" text-anchor="middle" font-size="14" fill="#2e7d32" font-family="monospace" font-weight="bold">8</text>
  <rect x="124" y="22" width="38" height="38" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="143" y="45" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">3</text>
  <rect x="10" y="60" width="38" height="38" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="29" y="83" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">4</text>
  <rect x="48" y="60" width="38" height="38" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="67" y="83" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">1</text>
  <rect x="86" y="60" width="38" height="38" rx="4" fill="#f5f5f5" stroke="#ccc" stroke-width="0.5"/>
  <text x="105" y="83" text-anchor="middle" font-size="14" fill="#aaa" font-family="monospace">5</text>
  <rect x="124" y="60" width="38" height="38" rx="4" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="143" y="83" text-anchor="middle" font-size="14" fill="#2e7d32" font-family="monospace" font-weight="bold">7</text>
  <text x="88" y="112" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">bold = max in each 2×2 block</text>
  <text x="185" y="55" text-anchor="middle" font-size="22" fill="#aaa" font-family="sans-serif">→</text>
  <text x="300" y="16" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">After pooling (2×2)</text>
  <rect x="222" y="28" width="58" height="58" rx="6" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="251" y="62" text-anchor="middle" font-size="20" fill="#2e7d32" font-family="monospace" font-weight="bold">9</text>
  <rect x="286" y="28" width="58" height="58" rx="6" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="315" y="62" text-anchor="middle" font-size="20" fill="#2e7d32" font-family="monospace" font-weight="bold">8</text>
  <text x="390" y="48" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif">Half the</text>
  <text x="390" y="62" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif">size!</text>
  <text x="390" y="78" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">keeps strongest</text>
  <text x="390" y="90" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">signal only</text>
</svg>

---

## 10. Important vs Unwanted Pixels

> **Key insight:** In any image, most pixels are background — sky, walls, floors, empty surroundings — that carry zero useful information about the object. Convolution is specifically designed to suppress these and amplify only the important pixels.

<svg viewBox="0 0 680 195" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="680" height="195" fill="#fafafa" rx="10"/>
  <defs>
    <marker id="ar7" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="76" y="14" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Raw face photo</text>
  <rect x="10" y="22" width="130" height="130" rx="8" fill="#f5f5f5" stroke="#ccc" stroke-width="1"/>
  <rect x="14" y="26" width="122" height="40" rx="4" fill="#e3f2fd" opacity="0.6"/>
  <text x="75" y="48" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">background / sky</text>
  <text x="75" y="62" text-anchor="middle" font-size="9" fill="#aaa" font-family="sans-serif">(irrelevant pixels)</text>
  <rect x="30" y="74" width="90" height="68" rx="4" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="75" y="96" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif" font-weight="bold">face pixels</text>
  <text x="75" y="111" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif">eyes, nose, mouth</text>
  <text x="75" y="126" text-anchor="middle" font-size="9" fill="#f57c00" font-family="sans-serif">edges, textures</text>
  <text x="75" y="140" text-anchor="middle" font-size="9" fill="#f57c00" font-family="sans-serif">HIGH information!</text>
  <text x="75" y="164" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">all pixels: 6.2M values</text>
  <text x="75" y="178" text-anchor="middle" font-size="9" fill="#c62828" font-family="sans-serif">most = useless background</text>
  <line x1="140" y1="87" x2="188" y2="87" stroke="#555" stroke-width="1.5" marker-end="url(#ar7)"/>
  <text x="165" y="80" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="sans-serif">Conv</text>
  <text x="165" y="100" text-anchor="middle" font-size="9" fill="#6a1b9a" font-family="sans-serif">filters</text>
  <text x="258" y="14" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">After convolution</text>
  <rect x="193" y="22" width="130" height="130" rx="8" fill="#f5f5f5" stroke="#2e7d32" stroke-width="1.5"/>
  <rect x="197" y="26" width="122" height="40" rx="4" fill="#f5f5f5" opacity="0.3"/>
  <text x="258" y="48" text-anchor="middle" font-size="10" fill="#bbb" font-family="monospace">≈ 0</text>
  <text x="258" y="62" text-anchor="middle" font-size="9" fill="#bbb" font-family="sans-serif">(background suppressed)</text>
  <rect x="210" y="74" width="100" height="68" rx="4" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="260" y="96" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif" font-weight="bold">HIGH activations</text>
  <text x="260" y="111" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif">where edges and</text>
  <text x="260" y="126" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif">features detected</text>
  <text x="258" y="164" text-anchor="middle" font-size="9" fill="#2e7d32" font-family="sans-serif">only meaningful signal</text>
  <text x="258" y="178" text-anchor="middle" font-size="9" fill="#2e7d32" font-family="sans-serif">background gone!</text>
  <line x1="323" y1="87" x2="371" y2="87" stroke="#555" stroke-width="1.5" marker-end="url(#ar7)"/>
  <text x="348" y="80" text-anchor="middle" font-size="9" fill="#f57c00" font-family="sans-serif">Pooling</text>
  <text x="348" y="100" text-anchor="middle" font-size="9" fill="#f57c00" font-family="sans-serif">shrinks</text>
  <text x="426" y="14" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">After pooling</text>
  <rect x="376" y="36" width="100" height="100" rx="8" fill="#f5f5f5" stroke="#6a1b9a" stroke-width="1.5"/>
  <rect x="390" y="50" width="72" height="72" rx="4" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1"/>
  <text x="426" y="84" text-anchor="middle" font-size="10" fill="#6a1b9a" font-family="sans-serif" font-weight="bold">compact</text>
  <text x="426" y="100" text-anchor="middle" font-size="10" fill="#6a1b9a" font-family="sans-serif">feature</text>
  <text x="426" y="116" text-anchor="middle" font-size="10" fill="#6a1b9a" font-family="sans-serif">map</text>
  <text x="426" y="152" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">small but info-rich</text>
  <line x1="476" y1="87" x2="510" y2="87" stroke="#555" stroke-width="1.5" marker-end="url(#ar7)"/>
  <text x="494" y="80" text-anchor="middle" font-size="9" fill="#1565c0" font-family="sans-serif">flatten</text>
  <rect x="514" y="46" width="30" height="90" rx="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="529" y="70" text-anchor="middle" font-size="9" fill="#1565c0" font-family="monospace">0.9</text>
  <text x="529" y="84" text-anchor="middle" font-size="9" fill="#1565c0" font-family="monospace">0.0</text>
  <text x="529" y="98" text-anchor="middle" font-size="9" fill="#1565c0" font-family="monospace">0.8</text>
  <text x="529" y="112" text-anchor="middle" font-size="9" fill="#1565c0" font-family="monospace">0.0</text>
  <text x="529" y="126" text-anchor="middle" font-size="9" fill="#1565c0" font-family="monospace">0.7</text>
  <text x="529" y="152" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">feature</text>
  <text x="529" y="164" text-anchor="middle" font-size="9" fill="#888" font-family="sans-serif">vector</text>
  <line x1="544" y1="87" x2="576" y2="87" stroke="#555" stroke-width="1.5" marker-end="url(#ar7)"/>
  <rect x="580" y="62" width="90" height="50" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="625" y="84" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif" font-weight="bold">face: 96%</text>
  <text x="625" y="100" text-anchor="middle" font-size="9" fill="#2e7d32" font-family="sans-serif">cat: 3%</text>
</svg>

**Why background pixels produce ≈ 0 activation:**
- Uniform regions (plain sky, flat wall) have **no variation** → edge/curve filter finds nothing → output ≈ 0
- Facial edges (eye outline, lip boundary) have **sharp colour changes** → filter activates strongly → HIGH output
- Texture (skin, fur) → specific filters fire → captured clearly

> ✅ **This is also why CNNs are position-independent.** An FNN memorises "koala's ear is in pixel 342,198." CNN learns "there's a round-ear shape" and finds it wherever it appears — top-left or bottom-right.

---

## 11. Feature Vectors — The Core Insight

> After flatten, each image becomes a **list of numbers = its mathematical fingerprint.** Same class → similar fingerprint → FC layer maps to same output.

<svg viewBox="0 0 660 175" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="660" height="175" fill="#fafafa" rx="10"/>
  <text x="330" y="14" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Same filters → different images → different number sequences → ANN identifies the pattern</text>
  <rect x="10" y="28" width="86" height="36" rx="6" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="53" y="50" text-anchor="middle" font-size="12" fill="#00695c" font-family="sans-serif" font-weight="bold">Cat 1 🐱</text>
  <text x="230" y="52" text-anchor="middle" font-size="12" fill="#333" font-family="monospace">[<tspan fill="#2e7d32" font-weight="bold">0.91</tspan>, <tspan fill="#2e7d32" font-weight="bold">0.87</tspan>, <tspan fill="#2e7d32" font-weight="bold">0.95</tspan>, <tspan fill="#c62828">0.08</tspan>]</text>
  <text x="490" y="52" text-anchor="middle" font-size="12" fill="#2e7d32" font-family="sans-serif" font-weight="bold">→ CAT ✓</text>
  <rect x="10" y="78" width="86" height="36" rx="6" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="53" y="100" text-anchor="middle" font-size="12" fill="#00695c" font-family="sans-serif" font-weight="bold">Cat 2 🐱</text>
  <text x="230" y="102" text-anchor="middle" font-size="12" fill="#333" font-family="monospace">[<tspan fill="#2e7d32" font-weight="bold">0.88</tspan>, <tspan fill="#2e7d32" font-weight="bold">0.83</tspan>, <tspan fill="#2e7d32" font-weight="bold">0.91</tspan>, <tspan fill="#c62828">0.05</tspan>]</text>
  <text x="490" y="102" text-anchor="middle" font-size="12" fill="#2e7d32" font-family="sans-serif" font-weight="bold">→ CAT ✓</text>
  <rect x="560" y="42" width="80" height="68" rx="8" fill="none" stroke="#2e7d32" stroke-width="1.5" stroke-dasharray="5 3"/>
  <text x="600" y="80" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif">similar!</text>
  <rect x="10" y="128" width="86" height="36" rx="6" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="53" y="150" text-anchor="middle" font-size="12" fill="#c62828" font-family="sans-serif" font-weight="bold">Dog 🐶</text>
  <text x="230" y="152" text-anchor="middle" font-size="12" fill="#333" font-family="monospace">[<tspan fill="#f57c00">0.71</tspan>, <tspan fill="#f57c00">0.55</tspan>, <tspan fill="#c62828">0.10</tspan>, <tspan fill="#2e7d32" font-weight="bold">0.92</tspan>]</text>
  <text x="490" y="152" text-anchor="middle" font-size="12" fill="#c62828" font-family="sans-serif" font-weight="bold">→ DOG ✓</text>
  <text x="330" y="172" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">FC learned: high pointy-ear + low snout = cat. High snout + low pointy-ear = dog.</text>
  <text x="106" y="52" text-anchor="middle" font-size="11" fill="#aaa" font-family="sans-serif">→</text>
  <text x="106" y="102" text-anchor="middle" font-size="11" fill="#aaa" font-family="sans-serif">→</text>
  <text x="106" y="152" text-anchor="middle" font-size="11" fill="#aaa" font-family="sans-serif">→</text>
</svg>

> 🌍 **This concept is everywhere in modern AI:**
> - Face recognition → same person = similar vectors
> - Spotify → similar songs = similar vectors  
> - Google image search → similar images = similar vectors
> - ChatGPT / Claude → similar meaning = similar vectors

---

## 12. All Architectures — Quick Reference

<svg viewBox="0 0 680 118" xmlns="http://www.w3.org/2000/svg" width="100%">
  <rect x="0" y="0" width="680" height="118" fill="#fafafa" rx="10"/>
  <defs>
    <marker id="ar8" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#aaa" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="270" y="8" width="140" height="34" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="340" y="29" text-anchor="middle" font-size="13" fill="#1565c0" font-family="sans-serif" font-weight="bold">ANN (umbrella)</text>
  <line x1="290" y1="42" x2="110" y2="72" stroke="#aaa" stroke-width="1" marker-end="url(#ar8)"/>
  <line x1="310" y1="42" x2="228" y2="72" stroke="#aaa" stroke-width="1" marker-end="url(#ar8)"/>
  <line x1="340" y1="42" x2="340" y2="72" stroke="#aaa" stroke-width="1" marker-end="url(#ar8)"/>
  <line x1="370" y1="42" x2="452" y2="72" stroke="#aaa" stroke-width="1" marker-end="url(#ar8)"/>
  <line x1="390" y1="42" x2="578" y2="72" stroke="#aaa" stroke-width="1" marker-end="url(#ar8)"/>
  <rect x="22" y="74" width="174" height="36" rx="6" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="109" y="89" text-anchor="middle" font-size="10" fill="#2e7d32" font-family="sans-serif" font-weight="bold">FNN / MLP</text>
  <text x="109" y="103" text-anchor="middle" font-size="9" fill="#555" font-family="sans-serif">tabular / structured data</text>
  <rect x="200" y="74" width="100" height="36" rx="6" fill="#e0f2f1" stroke="#00695c" stroke-width="1.5"/>
  <text x="250" y="89" text-anchor="middle" font-size="10" fill="#00695c" font-family="sans-serif" font-weight="bold">CNN</text>
  <text x="250" y="103" text-anchor="middle" font-size="9" fill="#555" font-family="sans-serif">images / video</text>
  <rect x="306" y="74" width="100" height="36" rx="6" fill="#fff8e1" stroke="#f57c00" stroke-width="1.5"/>
  <text x="356" y="89" text-anchor="middle" font-size="10" fill="#f57c00" font-family="sans-serif" font-weight="bold">RNN/LSTM</text>
  <text x="356" y="103" text-anchor="middle" font-size="9" fill="#555" font-family="sans-serif">sequences (older)</text>
  <rect x="412" y="74" width="112" height="36" rx="6" fill="#ede7f6" stroke="#6a1b9a" stroke-width="1.5"/>
  <text x="468" y="89" text-anchor="middle" font-size="10" fill="#6a1b9a" font-family="sans-serif" font-weight="bold">Transformer</text>
  <text x="468" y="103" text-anchor="middle" font-size="9" fill="#555" font-family="sans-serif">language / LLMs</text>
  <rect x="530" y="74" width="140" height="36" rx="6" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="600" y="89" text-anchor="middle" font-size="10" fill="#c62828" font-family="sans-serif" font-weight="bold">GAN</text>
  <text x="600" y="103" text-anchor="middle" font-size="9" fill="#555" font-family="sans-serif">generating content</text>
</svg>

| Architecture | Best for | Key idea | Famous examples |
|---|---|---|---|
| **FNN / MLP** | Tabular / structured data | Fully connected layers | Basic classifiers |
| **CNN** | Images, video | Filters slide + parameter sharing | ResNet, VGG, YOLO |
| **RNN / LSTM** | Sequences (older) | Memory state looped forward | Old translation models |
| **Transformer** | Language, modern AI | Self-attention — all tokens see all others | GPT, BERT, Claude |
| **GAN** | Generating new content | Generator vs Discriminator compete | DALL-E early, deepfakes |

---

## 13. Key Terms Cheat Sheet

| Term | Meaning |
|------|---------|
| **Epoch** | One full pass through entire training dataset |
| **Batch size** | Number of examples before one weight update (e.g. 32, 64) |
| **Learning rate** | How big each gradient step is. Too high = overshoot. Too low = slow. |
| **Loss function** | Measures how wrong predictions are. Cross-Entropy for classification, MSE for regression. |
| **Optimizer** | Algorithm updating weights. Adam is most popular. |
| **Overfitting** | Memorises training data, fails on new data |
| **Underfitting** | Model too simple to learn the pattern |
| **Kernel / Filter** | Small grid of learnable weights that slides across image |
| **Feature map** | Output of one filter — shows WHERE that pattern was detected |
| **Stride** | How many pixels filter moves each step |
| **Parameter sharing** | Same filter weights reused across entire image — key CNN efficiency |
| **Flatten** | Reshape 3D block → 1D vector. No computation, just reshaping. |
| **Feature vector / Embedding** | List of numbers after flatten — image's mathematical fingerprint |
| **Inference** | Using a trained model on new data (deployment) |
| **Vanishing gradient** | Gradients → 0 in deep networks. Sigmoid/Tanh suffer from this. |
| **Dying ReLU** | Neurons permanently output 0. Fix: Leaky ReLU. |

---

## 14. Interview Quick Answers

**Q: Why activation functions?**  
Without them, stacking layers collapses to one linear equation. Activation adds non-linearity so the network learns complex patterns.

---

**Q: Parameters vs hyperparameters?**  
Parameters (weights, biases, kernel values) are learned automatically by backprop. Hyperparameters (learning rate, filter size, #layers) are set by the engineer before training and stay fixed.

---

**Q: Why CNN over FNN for images?**  
FNN needs 6.2M neurons for a 1080p image — billions of weights. CNN uses 3×3 filters (just 9 weights) reused across the entire image (parameter sharing). Also CNN understands spatial structure; FNN treats all pixels independently.

---

**Q: Do CNN filters change per class — cat filters vs dog filters?**  
No. One shared set of filters for all images, always. Filters detect low-level features (edges, textures). The FC layer learns which combination = which class.

---

**Q: What is a feature vector?**  
The list of numbers after flattening — the image's mathematical fingerprint. Similar images produce similar vectors. This concept powers face recognition, Spotify, Google image search, and LLMs.

---

**Q: What is backpropagation?**  
After computing loss, the error travels backward through ALL layers simultaneously via chain rule. Each weight is told how much it contributed to the error. Gradient descent then updates all weights.

---

**Q: Why do images have unwanted pixels and how does CNN handle them?**  
Most pixels in an image are background (sky, walls, floors) with no useful info. Convolution filters produce near-zero activations on flat uniform regions and HIGH activations where sharp edges/textures exist. By the time data reaches the FC layer, background noise is suppressed and only meaningful signal remains.

---

**Q: What does pooling do?**  
Max pooling takes the maximum value in each 2×2 region. Halves the size, keeps the strongest signal, makes the network position-independent, reduces overfitting.

---

*Notes compiled through hands-on learning · CNN Architecture · Apna College*  
*Add to this as you learn RNN, Transformers, and GANs next! 🚀*
