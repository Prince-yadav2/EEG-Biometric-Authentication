# EEG-Based Multi-User Biometric Authentication using Deep Learning

## Overview

This project presents a **Multi-User EEG Biometric Authentication System** that utilizes **Digital Signal Processing (DSP)** and **Deep Learning (CNN)** for secure user authentication. The system processes Electroencephalography (EEG) signals, extracts discriminative time-frequency features using the **Continuous Wavelet Transform (CWT)**, and classifies registered users using a **Convolutional Neural Network (CNN)**. It also supports **unknown user rejection** through a majority-voting authentication mechanism.

---

## Features

- Multi-user EEG biometric authentication
- EEG preprocessing using DSP techniques
- Bandpass filtering (8–30 Hz Butterworth IIR Filter)
- Continuous Wavelet Transform (CWT)
- Scalogram-based feature extraction
- Patch-based image generation (avoids information loss)
- Multi-channel CNN classification
- Unknown user rejection
- GUI-based authentication system
- Performance evaluation using standard metrics

---

## Project Workflow

```
EEG Signal
     │
     ▼
Epoching
     │
     ▼
Normalization
     │
     ▼
Bandpass Filter (8–30 Hz)
     │
     ▼
Wavelet Transform (CWT)
     │
     ▼
Scalogram Generation
     │
     ▼
Patch Extraction
     │
     ▼
Multi-Channel Image Stacking
     │
     ▼
CNN Classification
     │
     ▼
Majority Voting
     │
     ▼
Authentication
(Authorized / Unauthorized)
```

---

## Digital Signal Processing Pipeline

### Preprocessing

- EEG Epoching
- Z-score Normalization
- Butterworth Bandpass Filtering (8–30 Hz)

### Feature Extraction

- Fast Fourier Transform (FFT)
- Spectrogram (STFT)
- Continuous Wavelet Transform (CWT)
- Scalogram Generation

After comparative analysis, **Wavelet Transform with Scalogram representation** was selected because it preserves both time and frequency information effectively.

---

## Deep Learning Architecture

The proposed CNN consists of:

- Input Layer
- Convolution Layer (3×3)
- ReLU Activation
- Max Pooling Layer
- Convolution Layer
- Max Pooling Layer
- Flatten Layer
- Fully Connected Layer
- Dropout Layer
- Softmax Output Layer

---

## Authentication Strategy

The trained CNN predicts the class of each extracted EEG patch.

The final authentication decision is obtained using **majority voting**:

- If the majority of patches belong to a registered user:
  - **Authorized**
- Otherwise:
  - **Unauthorized**

---

## Dataset

**Dataset:** EEG Motor Movement/Imagery Dataset

Dataset Format:

- EDF (European Data Format)

Dataset Information:

- 64 EEG Channels
- Sampling Frequency: 160 Hz
- Multiple Subjects
- Multiple Recording Sessions

> **Note:** The dataset is not included in this repository. Please download it separately and organize it in the following structure.

```
data/
│
├── S001/
│   ├── *.edf
│
├── S002/
│   ├── *.edf
│
├── S003/
│   ├── *.edf
│
└── ...
```

---

## Repository Structure

```
EEG-Biometric-Authentication/
│
├── data/
├── test_data/
│
├── GUI_Multi_User_Authentication.py
├── Multi_User_Biometric_Authentication.ipynb
├── DSP_filter_and_feature_extraction_technique.ipynb
├── Single_User_BA.py
├── Single_User_Biometric_Authentication.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Prince-yadav2/EEG-Biometric-Authentication.git
```

Go to the project directory:

```bash
cd EEG-Biometric-Authentication
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the GUI:

```bash
python GUI_Multi_User_Authentication.py
```

The GUI allows you to:

- Select an EEG (.edf) file
- Authenticate the user
- Display the authentication result

---

## Performance Metrics

The proposed system is evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- AUC-ROC
- Matthews Correlation Coefficient (MCC)
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Confusion Matrix
- Error Rate
- Latency

> **Note:** Replace the values below with your experimental results.

| Metric | Value |
|---------|-------|
| Accuracy | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |
| Specificity | XX.XX% |
| F1 Score | XX.XX% |
| MCC | XX.XX |
| FAR | XX.XX |
| FRR | XX.XX |
| Latency | XX ms |

---

## Results

### Filter Comparison

_Insert your filter comparison figure here._

---

### Feature Extraction

_Insert FFT, Spectrogram, Wavelet, and Scalogram figures here._

---

### User-Level Confusion Matrix

_Insert your confusion matrix here._

---

### GUI

_Insert screenshots of the GUI here._

---

## Future Work

- Increase the number of registered users
- Improve unknown-user rejection
- Optimize CNN architecture
- Real-time EEG acquisition
- Hardware implementation using embedded systems
- Deploy the authentication system as a standalone desktop application

---

## Author

**Prince Yadav**

Department of Electrical Engineering

---

## Acknowledgements

- PhysioNet EEG Motor Movement/Imagery Dataset
- MNE-Python
- TensorFlow / Keras
- Scikit-learn
- PyWavelets

---

## License

This project is intended for educational and research purposes.
