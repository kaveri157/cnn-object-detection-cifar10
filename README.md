# CNN-Based Object Detection using CIFAR-10

A Convolutional Neural Network (CNN) built with TensorFlow and Keras that classifies images from the CIFAR-10 dataset into 10 object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

Mini-Project for the subject Artificial Neural Network, AI & Data Science Engineering, Savitribai Phule Pune University (2025-26).

## Model Architecture

Input (32x32x3) → Conv2D (32 filters, 3x3, ReLU) → MaxPooling (2x2) → Conv2D (64 filters, 3x3, ReLU) → MaxPooling (2x2) → Flatten → Dense (64, ReLU) → Output Dense (10, Softmax)

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## How to Run

```bash
pip install -r requirements.txt
python cnn_object_detection.py
```

The CIFAR-10 dataset downloads automatically on the first run, so you need an internet connection.

## What the Script Does

1. Loads and normalizes the CIFAR-10 dataset
2. Builds and trains the CNN for 5 epochs
3. Evaluates accuracy on the test set
4. Plots training vs validation accuracy
5. Predicts the class of sample test images (airplane and cat)

## Results

The model reaches roughly 68-70% test accuracy after 5 epochs.

## Future Improvements

- Add more convolution layers, dropout and data augmentation
- Train for more epochs
- Use real object detection models such as YOLO or Faster R-CNN

## Author

Kaveri Dhatrak
