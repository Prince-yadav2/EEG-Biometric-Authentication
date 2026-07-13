import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

import os
import random
from collections import Counter

import mne
import numpy as np
import pywt
import tensorflow as tf

from scipy.signal import butter, filtfilt
from mne import make_fixed_length_epochs

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

channels = [0, 1, 2]
patch_size = 32

train_data_path = "data"

def bandpass_filter(signal, fs):
    low = 8 / (fs / 2)
    high = 30 / (fs / 2)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def generate_scalogram(signal, fs):
    scales = np.arange(1, 64)
    coeffs, _ = pywt.cwt(signal, scales, 'morl', 1/fs)
    scalogram = np.abs(coeffs)

    scalogram = (scalogram - np.min(scalogram)) / (np.max(scalogram) - np.min(scalogram) + 1e-8)

    return scalogram

def create_patches(scalogram, patch_size=32):
    patches = []
    h, w = scalogram.shape

    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = scalogram[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches

def process_file(file_path, label):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    fs = int(raw.info['sfreq'])

    epochs = make_fixed_length_epochs(raw, duration=2.0, overlap=1.0)
    epochs_data = epochs.get_data()

    X = []
    y = []

    for epoch in epochs_data:

        epoch = (epoch - np.mean(epoch, axis=1, keepdims=True)) / (np.std(epoch, axis=1, keepdims=True) + 1e-8)

        # Filter all channels
        filtered = [bandpass_filter(ch, fs) for ch in epoch]
        filtered = np.array(filtered)

        # Generate scalograms per channel
        scalograms = [generate_scalogram(filtered[ch], fs) for ch in channels]

        # Create patches per channel
        channel_patches = [create_patches(s) for s in scalograms]

        # Combine patches across channels
        for i in range(len(channel_patches[0])):
            patch_stack = np.stack([
                channel_patches[0][i],
                channel_patches[1][i],
                channel_patches[2][i]
            ], axis=-1)

            X.append(patch_stack)
            y.append(label)

    return X, y

X = []
y = []

user_folders = sorted(os.listdir(train_data_path))

for label, user in enumerate(user_folders):
    user_path = os.path.join(train_data_path, user)

    for file in os.listdir(user_path):
        if file.endswith(".edf"):
            file_path = os.path.join(user_path, file)
            Xi, yi = process_file(file_path, label)

            X.extend(Xi)
            y.extend(yi)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_classes = len(np.unique(y))

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(patch_size, patch_size, 3)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16
)

label_map = {i: user_folders[i] for i in range(len(user_folders))}

def authenticate_user(file_path):

    raw = mne.io.read_raw_edf(file_path, preload=True)
    fs = int(raw.info['sfreq'])

    epochs = make_fixed_length_epochs(raw, duration=2.0, overlap=1.0)
    epochs_data = epochs.get_data()

    all_patches = []

    for epoch in epochs_data:

        # Normalize
        epoch = (epoch - np.mean(epoch, axis=1, keepdims=True)) / (np.std(epoch, axis=1, keepdims=True) + 1e-8)

        # Filter
        filtered = [bandpass_filter(ch, fs) for ch in epoch]
        filtered = np.array(filtered)

        # Scalograms
        scalograms = [generate_scalogram(filtered[ch], fs) for ch in channels]

        # Patches
        channel_patches = [create_patches(s) for s in scalograms]

        for i in range(len(channel_patches[0])):

            patch_stack = np.stack([
                channel_patches[0][i],
                channel_patches[1][i],
                channel_patches[2][i]
            ], axis=-1)

            all_patches.append(patch_stack)

    # Convert once
    all_patches = np.array(all_patches)

    # SINGLE prediction call
    predictions = model.predict(
        all_patches,
        verbose=0
    )

    predicted_classes = np.argmax(predictions, axis=1)

    # Majority voting
    counter = Counter(predicted_classes)

    final_class = counter.most_common(1)[0][0]

    votes = counter.most_common(1)[0][1]

    vote_ratio = votes / len(predicted_classes)

    if vote_ratio < 0.75:

        return "Unauthorized", None

    else:

        return "Authorized", label_map[final_class]
    
root = tk.Tk()
root.title("EEG Biometric Authentication")
root.geometry("700x500")
root.configure(bg="#EAF2F8")

heading = tk.Label(
    root,
    text="EEG Multi-User Authentication System",
    font=("Arial", 22, "bold"),
    bg="#EAF2F8",
    fg="#154360"
)

heading.pack(pady=20)

result_label = tk.Label(
    root,
    text="Select an EDF file for authentication",
    font=("Arial", 18),
    bg="#EAF2F8",
    fg="black"
)

result_label.pack(pady=40)

def run_authentication(file_path):

    result_label.config(
        text="Processing...",
        fg="blue"
    )

    root.update_idletasks()

    status, user = authenticate_user(file_path)

    if status == "Unauthorized":

        result_label.config(
            text="Unauthorized",
            fg="red"
        )

    else:

        result_label.config(
            text=f"{user} - Authorized",
            fg="green"
        )

def select_file():

    file_path = filedialog.askopenfilename(
        filetypes=[("EDF Files", "*.edf")]
    )

    if file_path:

        threading.Thread(
            target=run_authentication,
            args=(file_path,),
            daemon=True
        ).start()

button = tk.Button(
    root,
    text="Select EEG File",
    command=select_file,
    font=("Arial", 16, "bold"),
    bg="#1F618D",
    fg="white",
    padx=20,
    pady=10
)

button.pack(pady=20)

exit_button = tk.Button(
    root,
    text="Exit",
    command=root.destroy,
    font=("Arial", 14),
    bg="#922B21",
    fg="white",
    padx=15,
    pady=5
)

exit_button.pack(pady=20)

root.mainloop()