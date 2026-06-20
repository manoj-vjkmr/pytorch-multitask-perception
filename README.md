# Multi-Task Visual Perception Pipeline in PyTorch

A cohesive, multi-stage computer vision system built from scratch in PyTorch without using other builtin functions to simultaneously perform **image classification, single-object localization, and semantic segmentation** from a single shared backbone. 

This architecture leverages a custom-built VGG11 encoder paired with a symmetric U-Net style expansive decoder, optimized to handle the multi-task learning dynamics of the Oxford-IIIT Pet Dataset.

---
### Key Technical Implementations

* **Custom Low-Level Primitives:** To maximise the mathematical learning/understanding, core framework utilities were built manually rather than using native PyTorch abstractions:
  * **Inverted Dropout Layer:** Built a custom `nn.Module` managing statistical binary masks and deterministic evaluation scaling when `self.training = False`.
  * **Vectorized IoU Loss:** Developed a numerically stable Intersection over Union (IoU) loss function for bounding box coordinate regression.
* **Symmetric Expansive Path:** Implemented a full U-Net decoder utilizing Transposed Convolutions for learnable upsampling, complete with high-resolution skip-connection feature fusion along the channel dimension.
* **Unified Multi-Task Forward Pass:** Engineered a `forward(self, x)` flow branching into three specialized task heads, allowing simultaneous gradient generation while monitoring task interference.

---

## Experiment Insights

### 1. Transfer Learning Strategies (Encoder Adaptation)
Evaluated three distinct training regimes when adapting the pre-trained VGG11 backbone for semantic segmentation:
* **Strict Feature Extractor:** Frozen backbone; only trained the decoder. Fast epochs but bounded performance.
* **Partial Fine-Tuning:** Unlocked later convolutional blocks to refine high-level domain-specific shapes (snouts, ears) while preserving lower-level edge extractors.
* **Full Fine-Tuning:** End-to-end optimization. Yielded the highest Dice Scores, though vulnerable to initial gradient volatility.

### 2. Metric Evaluation: Dice vs. Pixel Accuracy
Empirical validation revealed that Pixel Accuracy remained artificially high ($>85\%$) in early training epochs while the Dice Coefficient lagged. This mathematically demonstrated the impact of class imbalance in dense trimaps (where background pixels dominate the loss space), validating why the **Dice Similarity Coefficient** is a superior metric for tracking genuine boundary segmentation.

### 3. Feature Map Interpretability
Extracted and visualized hidden layer representations. The network demonstrates a clear hierarchical transition: early layers capture localized, high-frequency edges and textures, while deeper layers compress spatial dimensions to extract abstract semantic features.

---

WandB report link:
https://wandb.ai/ns26z119-indian-insti/da6401-assignment-2/reports/NS26Z119-Assignment-2-Report--VmlldzoxNjUwMTU1MA?accessToken=0eilp7pybbi6x1qqms8ooa5r7x8cvoa3qynkmuz9tks82ogk2ova7hdiwuwcadf1

Github repo: 
https://github.com/manoj-vjkmr/da6401_assignment2
