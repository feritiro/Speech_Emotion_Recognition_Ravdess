# Speech Emotion Recognition with RAVDESS (Happy, Angry, Neutral)

This project implements a deep learning model to classify speech emotions using the [RAVDESS dataset](https://zenodo.org/record/1188976). The model is trained to recognize three emotional states: **Happy**, **Angry**, and **Neutral**, achieving high performance on the test set.

## Objective

To build a robust emotion classification system capable of identifying **Happy**, **Angry**, and **Neutral** emotions from raw speech using only audio features.

---

## Results

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 92.79%    |
| F1 Score     | 92.61%    |
| Loss         | 0.2249    |
| Test Accuracy| 93.61%    |

---

## Dataset

- **Name:** RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song
- **Source:** [Zenodo - RAVDESS](https://zenodo.org/record/1188976)
- **Selected Emotions:** Happy, Angry, Neutral
- **Preprocessing:**
  - Downsampling to 16 kHz
  - Extracting MFCC features
  - Normalization and padding

---

## Model Overview

- **Type:** Convolutional Neural Network (CNN)
- **Input:** MFCCs extracted from audio clips
- **Output:** One-hot classification into 3 emotion classes
- **Libraries Used:** 
  - `TensorFlow`, `Keras` for model development
  - `Librosa` for audio feature extraction
  - `NumPy`, `Scikit-learn`, `Matplotlib` for preprocessing and evaluation

---

