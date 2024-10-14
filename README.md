# Music Genre Classification Using Deep Learning
![image](https://github.com/user-attachments/assets/257aeb64-6a2f-458a-96cf-4f6d04ffcc4f)
![image](https://github.com/user-attachments/assets/e1a9625a-7cdd-471c-902b-3dc49cd263ce)

## Overview

In the age of digital music, classifying audio tracks into genres presents a significant challenge due to the sheer volume of available content. The ability to accurately categorize music not only enhances user experiences through personalized recommendations but also aids in organizing vast libraries of audio files. This project aims to build an automated music genre classification system utilizing **deep learning techniques**, specifically employing **Convolutional Neural Networks (CNNs)** and **Mel-Spectrograms**.

## Problem Statement

Manual classification of music genres is labor-intensive and often subjective, relying heavily on metadata that may not accurately reflect the audio content. Traditional approaches often struggle to capture the intricate features of music, leading to inconsistent classification results. This project addresses these challenges by employing a data-driven approach that analyzes the audio signal directly, resulting in a robust and reliable genre classification system.

## Methodology

### 1. Feature Extraction: Mel-Spectrograms

The first step in our approach involves transforming raw audio data into a visual representation that captures essential features of sound. We generate **Mel-Spectrograms**, which are time-frequency representations of audio that highlight the energy distribution across various frequency bands, aligned with human auditory perception (Mel scale). This method effectively emphasizes relevant features that distinguish different genres.

To create the Mel-Spectrograms, we utilize the **Librosa** library, which allows for efficient audio processing. The Mel-Spectrograms serve as image-like data inputs for our CNN, preserving the time and frequency relationships crucial for genre differentiation.

### 2. Chunking Audio Data

To enhance model performance, we split each audio track into **15 segments**, each lasting 4 seconds with 2 seconds overlap with previous one. This chunking technique allows the CNN to learn from localized features within the audio, capturing transient changes and patterns that might indicate genre-specific characteristics. Each chunk is converted into its corresponding Mel-Spectrogram, resulting in a robust dataset for training.

### 3. Convolutional Neural Network (CNN)

The heart of our classification system lies in a deep **Convolutional Neural Network (CNN)**, which excels at processing image data. The architecture consists of several layers:

- **Convolutional Layers**: Extract spatial features from the Mel-Spectrograms, allowing the model to recognize patterns such as rhythms and harmonic structures.
- **Max-Pooling Layers**: Reduce the dimensionality of the feature maps, focusing on the most salient features while maintaining essential information.
- **Fully Connected Layers**: Interpret the learned representations and map them to the 10 target genres, outputting a probability distribution for each class.

### 4. Model Training

The model is trained on the **GTZAN dataset**, which includes 1,000 audio tracks spanning 10 genres. Each track is 30 seconds long, providing a diverse representation of musical styles. The training process involves optimizing the CNN's parameters using the **Adam optimizer** and minimizing the **categorical cross-entropy loss** function, suitable for multi-class classification tasks.

### 5. Predictions and Evaluation

After training, the model is evaluated on a separate test set to assess its performance. Predictions are made for each audio chunk, and the final genre classification for a track is derived by averaging the predictions of its individual chunks.

## Results

The CNN model demonstrated a commendable accuracy of **90%** on the test set, showcasing its ability to learn complex audio features effectively. By utilizing both **Mel-Spectrograms**  the model can capture the intricate characteristics of music genres, leading to improved classification performance.

## Installation and Setup

### Prerequisites

To successfully run this project, ensure you have the following prerequisites installed:

- **Python 3.x**: Make sure you have Python version 3.x installed. You can download it from the official [Python website](https://www.python.org/downloads/).
- **Required Libraries**: This project relies on several libraries. You can install them using the `requirements.txt` file included in the repository, which contains the following:
  - TensorFlow
  - Librosa
  - Numpy
  - Matplotlib
  - Scikit-learn

### Steps to Run the Project

1. **Clone the Repository**:  
   Begin by cloning the project repository to your local machine. Open your terminal and execute the following commands:
   ```bash
   git clone https://github.com/Muhammadfarooq297/Music_Genre_Classification
   cd music-genre-classifier

## Conclusion

In this project, we successfully developed a Music Genre Classification system using Convolutional Neural Networks (CNNs) and Mel-Spectrograms as input features. By leveraging the GTZAN dataset, we trained a robust model capable of accurately identifying various music genres. The modular structure of the code allows for easy enhancements and experimentation, inviting further exploration in the field of audio classification. This project not only demonstrates the power of deep learning in analyzing audio data but also serves as a foundation for more advanced applications in music analysis and recommendation systems. We encourage you to experiment with different architectures, hyperparameters, and datasets to further improve classification performance and explore the fascinating world of audio processing.

