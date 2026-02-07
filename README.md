# Off-Road Semantic Segmentation

This project implements a **semantic segmentation pipeline** designed for **unstructured environments** using a **DINOv2 foundation model** backbone. It is ideal for off-road robotics, autonomous vehicles, or any application where terrain understanding is critical.

---

## Features

- **DINOv2 Integration**  
  Uses self-supervised ViT features as a **frozen backbone** for robust feature extraction.

- **Efficient Training**  
  Implements a **feature caching system** that pre-calculates backbone tokens, significantly reducing GPU compute during head training.

- **Lightweight Head**  
  A custom convolutional **SegHead** maps high-dimensional tokens to class labels.

---

## Technology Stack

- Core Framework: PyTorch  
- Backbone: dinov2_vits14 for self-supervised feature extraction  
- Head Architecture: Custom convolutional **SegHead** with bilinear interpolation for high-resolution mask generation  
- Optimization: Feature caching to accelerate training by pre-computing transformer tokens  

---

## Performance Visualization

- IoU Curves: Track validation progress over epochs  
- Per-Class Bar Charts: Identify which terrains the model masters best  
- Qualitative Overlays: Side-by-side comparisons of Input, Ground Truth, and Model Predictions  

---

## Requirements

- Framework: PyTorch  
- Hardware: CUDA-enabled GPU (recommended) or CPU  
- Environment: Jupyter Notebook or any Python 3.12+ IDE
