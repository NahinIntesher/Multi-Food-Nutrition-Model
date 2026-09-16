# Multi-Food Recognition Using YOLO + ViT-LSTM Hybrid Deep Learning Pipeline

A hybrid deep learning framework for real-world multi-food recognition using **YOLO-based instance segmentation** and **ViT-LSTM based food classification refinement**.

The proposed pipeline uses **YOLO26n-Seg** as the primary food detection and segmentation model to localize multiple food items from a single image. To improve classification performance on uncertain detections, a previously trained **Vision Transformer (ViT)-LSTM classifier** is integrated as a fallback refinement module.

## Key Features

**Multi-food detection and segmentation**

  * Detects multiple food regions from complex meal images
  * Generates bounding boxes and segmentation masks

**Adaptive YOLO + ViT-LSTM Fusion**

  * High-confidence YOLO predictions are directly accepted
  * Low-confidence detections are forwarded to ViT-LSTM for improved classification

 **Fine-grained food classification**

  * Uses transformer-based visual representation learning
  * Improves recognition of visually similar food categories

**Comprehensive Evaluation**

  * Precision
  * Recall
  * F1 Score
  * mAP@50
  * mAP@50-95

## System Pipeline

```
Input Food Image
        |
        ↓
YOLO26n-Seg
(Food Detection + Instance Segmentation)
        |
        ↓
Confidence-Based Decision
        |
 ┌───────────────┐
 │               │
High Confidence  Low Confidence
 │               │
 ↓               ↓
YOLO Output   ViT-LSTM Refinement
 │               │
 └───────┬───────┘
         ↓
 Final Food Prediction
```

## Dataset

The model is trained and evaluated on a merged multi-food dataset containing diverse food images with multiple food instances per image.

Evaluation is performed on a test set of **2786 images**.
