"""
Run this script once to train and save the Brain Tumor CNN model.

Usage:
    python train_cnn.py

Before running:
1. Download Brain Tumor MRI Dataset from Kaggle:
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Extract to data/brain_tumor/ folder
   Structure should be:
   data/brain_tumor/
       glioma/
       meningioma/
       notumor/
       pituitary/
"""

from utils.cnn_model import train_cnn

DATA_DIR = "data/brain_tumor/Training"

print("=" * 50)
print("Brain Tumor MRI Classification")
print("Model: ResNet50 Transfer Learning")
print("Classes: Glioma, Meningioma, No Tumor, Pituitary")
print("=" * 50)
print("Starting training...")

model, history = train_cnn(DATA_DIR)

val_acc = max(history.history["val_accuracy"])
print("=" * 50)
print(f"Training complete!")
print(f"Best validation accuracy: {val_acc * 100:.2f}%")
print(f"Model saved to: models/brain_tumor_cnn.h5")
print("=" * 50)
