import zipfile
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

"""
# Path to the zip file containing the dataset
zip_file_path = "/content/archive.zip"
 
# Destination directory to extract the contents
extracted_dir = "/content/barbie_vs_puppy"
 
# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)


# Set the path to dataset folder
data_dir = "/content/barbie_vs_puppy/barbie_vs_puppy"
"""
data_dir = "/Users/mac/Desktop/audio_barbie_puppy_classification/barbie_vs_puppy"
# Load and preprocess audio data using spectrograms
labels = os.listdir(data_dir)
audio_data = []
target_labels = []
print(labels)

for label in labels:
    label_dir = os.path.join(data_dir, label)
    for audio_file in os.listdir(label_dir):
        audio_path = os.path.join(label_dir, audio_file)
        y, sr = librosa.load(audio_path, duration=3)  # Load audio and limit to 3 seconds
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        # Transpose the spectrogram to have the shape (timesteps, n_mels)
        spectrogram = spectrogram.T
        audio_data.append(spectrogram)
        target_labels.append(label)


# Encode target labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(target_labels)
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_data, encoded_labels, test_size=0.2, random_state=42)
 
# Ensure all spectrograms have the same shape
max_length = max([spec.shape[0] for spec in audio_data])
X_train = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_train]
X_test = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_test]
 
# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)


# Count the number of samples in each class
class_counts = [len(os.listdir(os.path.join(data_dir, label))) for label in labels]
# Define colors for each class
class_colors = ['blue', 'green']
# Create a bar chart to visualize class distribution
plt.figure(figsize=(5, 3))
plt.bar(labels, class_counts, color=class_colors)
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")
plt.show()

# Define a function to plot spectrograms for a class
def plot_spectrograms(label, num_samples=3):
    label_dir = os.path.join(data_dir, label)
    plt.figure(figsize=(7, 4))
    plt.suptitle(f"Spectrogram Comparison for Class: {label}")
 
    for i, audio_file in enumerate(os.listdir(label_dir)[:num_samples]):
        audio_path = os.path.join(label_dir, audio_file)
        y, sr = librosa.load(audio_path, duration=3)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
 
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.title(f"Spectrogram {i + 1}")
        plt.imshow(spectrogram, cmap="viridis")
        plt.colorbar(format="%+2.0f dB")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
 
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.title(f"Audio Waveform {i + 1}")
        plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
 
 
# Visualize spectrograms and audio waveforms for &quot;barbie&quot; class
# adjust num_samples parameter to see desired number of  visualization of samples
plot_spectrograms("barbie", num_samples=1)
print('\n')
# Visualize spectrograms and audio waveforms for &quot;puppy&quot; class
plot_spectrograms("puppy", num_samples=1)


# Convert the data to a flat 2D shape
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
 
# Create Gradient Boosting classifier
model = GradientBoostingClassifier(random_state=42)
# Train the model
model.fit(X_train_flat, y_train)
 
# Make predictions
y_pred = model.predict(X_test_flat)
 
# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
 
print("Accuracy: {:.4f}".format(accuracy))
print("F1 score: {:.4f}".format(f1))