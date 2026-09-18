# Handwriting Digit Recognition (CNN)

Recognize handwritten digits (0â€“9) with a **convolutional neural network**
trained on the MNIST dataset. This repository contains **two ways to use it**:

- a **CLI script** that trains the model and shows a sample prediction, and
- a **Tkinter GUI** where you can draw digits by hand and classify them.

![Python](https://img.shields.io/badge/Python-3.x-3776AB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)
![CNN](https://img.shields.io/badge/CNN-Keras-FF6F00)
![GUI](https://img.shields.io/badge/GUI-Tkinter-blue)

## Features

### CLI (train + evaluate)

- Loads and preprocesses the MNIST dataset
- Trains a CNN and reports **test accuracy**
- Plots a random test image with its predicted label

### GUI (draw and predict)

- Draw a digit on a grid canvas
- One click to **Predict** with confidence percentage
- **All-class probability bars** showing the model's reasoning
- **Retrain** options, **test-set grid**, and **confusion matrix** views

## Model architecture

```
Input (28Ã—28Ã—1)
  â†’ Conv2D 32 + ReLU â†’ MaxPool 2Ã—2
  â†’ Conv2D 64 + ReLU â†’ MaxPool 2Ã—2
  â†’ Flatten â†’ Dense 128 + ReLU â†’ Dense 10 + Softmax
```

## Getting Started

### Requirements

Install the dependencies:

```bash
pip install tensorflow matplotlib numpy
```

### Run the CLI version

Trains the model (1 epoch) and displays a sample prediction:

```bash
python app_CLI_based.py
```

### Run the GUI version

```bash
python Handwriting_Digit_Recognition/GUI_hand_writing_digit_recognition_app.py
```

draw a digit, then press **Predict**.

> Want live prediction while drawing, pen sizes, an eraser, and a dark theme?
> Check out the [Enhanced Handwriting Digit Recognition](https://github.com/Plabon-Basak/enhanced_handwriting_digit_recognition_app) app.

## Project structure

```
handwriting_digit_recognition_using_CNN_Model/
â”œâ”€â”€ app_CLI_based.py                          # CLI training/evaluation script
â””â”€â”€ Handwriting_Digit_Recognition/
    â”œâ”€â”€ GUI_hand_writing_digit_recognition_app.py  # Tkinter GUI
    â””â”€â”€ digit_model.keras                     # Trained model
```

## License

This project is open-source and available under the [MIT License](LICENSE).