# Sign Language Interpreter Using Deep Learning

### A CNN-Based Hand Gesture Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" />
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Keras-CNN-red?logo=keras" />
  <img src="https://img.shields.io/badge/Status-Prototype-success" />
</p>

## Overview

The **Sign Language Interpreter** is a machine learning project that uses computer vision and deep learning to recognize predefined hand gestures through a webcam.

The system captures hand images, processes them using OpenCV, and passes the extracted hand representation to a trained Convolutional Neural Network (CNN). The model predicts the corresponding gesture class and displays the result.

The project demonstrates how supervised machine learning and deep learning can be applied to image classification and real-time gesture recognition.

## Key Features

- Real-time webcam-based hand gesture recognition.
- Image preprocessing and hand segmentation using OpenCV.
- CNN-based image classification using TensorFlow and Keras.
- Supervised learning with labeled gesture images.
- Automatic feature extraction through convolutional layers.
- Softmax-based multi-class prediction.
- SQLite database for gesture labels.
- Model training, validation, and testing workflow.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam capture and image processing |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and training |
| NumPy | Numerical operations and image arrays |
| Scikit-learn | Dataset splitting and preprocessing |
| SQLite | Gesture label storage |

## System Architecture

The project consists of two major workflows: model training and real-time inference.

### 1. Model Training Pipeline

```text
Gesture Image Collection
          |
          v
     Image Dataset
          |
          v
   Image Preprocessing
          |
          v
 Dataset Labeling & Splitting
          |
          v
    CNN Architecture
          |
          v
 Forward Propagation
          |
          v
   Loss Calculation
          |
          v
   Backpropagation
          |
          v
   Adam Optimization
          |
          v
   Trained CNN Model
```

### 2. Real-Time Recognition Pipeline

```text
      Webcam Input
           |
           v
     Frame Capture
           |
           v
    Hand Region Selection
           |
           v
   HSV Conversion and
 Histogram-Based Segmentation
           |
           v
      Binary Mask
           |
           v
   Image Resizing (50x50)
           |
           v
     Normalization
           |
           v
      Trained CNN
           |
           v
   Softmax Classification
           |
           v
   Predicted Gesture Label
```

## Machine Learning Implementation

### Dataset Collection

The initial training experiment used a dataset of 750 hand gesture images distributed across five classes.

| Gesture | Images |
|---|---:|
| Namaste | 150 |
| This is | 150 |
| Hard | 150 |
| You | 150 |
| OK | 150 |
| **Total** | **750** |

Each image was associated with a class label, allowing the CNN to learn the relationship between hand representations and their corresponding gesture categories.

*Note: These figures describe the initial five-class training experiment. Update this section if you publish a newly collected dataset with different classes.*

### Image Preprocessing

OpenCV is used to prepare images before they are passed to the CNN.

The preprocessing stages include:

1. Capturing frames from the webcam.
2. Selecting the Region of Interest (ROI) containing the hand.
3. Converting the selected region into HSV color space.
4. Applying histogram back-projection for hand segmentation.
5. Generating a binary mask.
6. Applying filtering and morphological operations to reduce noise.
7. Resizing the processed image to 50 × 50 pixels.
8. Normalizing pixel values from 0–255 to 0–1.

The final input is a 50 × 50 grayscale image with one channel.

### CNN Model Architecture

The project uses a Convolutional Neural Network to automatically learn spatial features from hand gesture images.

| Layer | Configuration |
|---|---|
| Input | 50 × 50 × 1 |
| Conv2D | 16 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 32 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | 1,024 features |
| Dense | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output | 5 neurons, Softmax |

**Total trainable parameters: 155,141**

### Model Training Configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Model | Convolutional Neural Network |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Batch Size | 32 |
| Maximum Epochs | 30 |
| Early Stopping | Enabled |
| Dropout Rate | 0.30 |
| Input Dimensions | 50 × 50 × 1 |

### How the Model Learns

The CNN follows a supervised learning approach.

1. A labeled gesture image is passed through the network.
2. Convolutional layers extract visual features.
3. The dense layer uses the extracted features for classification.
4. Softmax generates probabilities for the output classes.
5. Sparse categorical cross-entropy calculates the prediction error.
6. Backpropagation calculates gradients.
7. Adam updates the model weights to minimize the loss.

This process is repeated over multiple batches and epochs until the model learns useful patterns from the training dataset.

## Training Results

The initial five-class training experiment achieved the following results:

| Metric | Result |
|---|---:|
| Training Dataset | 750 images |
| Best Validation Accuracy | 100% |
| Test Accuracy | **98.23%** |
| Total Model Parameters | 155,141 |
| Training Epochs Completed | 10 |
| Saved Model | `cnn_model_keras2.h5` |

The model reached 100% validation accuracy by epoch 4 and achieved 98.23% accuracy on the held-out test dataset.

These results represent the initial experimental dataset and should not be interpreted as guaranteed accuracy under all real-world webcam conditions.

## Project Structure

```text
sign-language-interpreter-cnn/
│
├── Code/
│   ├── gestures/
│   ├── create_gestures.py
│   ├── load_images.py
│   ├── cnn_model_train.py
│   ├── recognize_gestures.py
│   ├── webcam_test.py
│   ├── gesture_db.db
│   ├── hist
│   └── cnn_model_keras2.h5
│
├── README.md
├── requirements.txt
└── .gitignore
```

The exact files included may vary depending on the published version of the project. The trained model, histogram file, and database are required by the corresponding inference implementation.

## Installation and Setup

### Prerequisites

- Python 3.10
- Working webcam
- Git
- Windows, Linux, or macOS

### 1. Clone the repository

```bash
git clone https://github.com/mahrooscyber/sign-language-interpreter-cnn.git
cd sign-language-interpreter-cnn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the webcam test

```bash
python Code/webcam_test.py
```

### 5. Collect gesture images

```bash
cd Code
python create_gestures.py
```

Follow the prompts to register gesture labels and collect training images.

### 6. Prepare the dataset

```bash
python load_images.py
```

### 7. Train the CNN

```bash
python cnn_model_train.py
```

The trained model is saved as:

```text
cnn_model_keras2.h5
```

### 8. Run real-time recognition

```bash
python recognize_gestures.py
```

Press `Q` to close the webcam window.

*Note: The recognition step requires the compatible trained model, histogram file, and gesture database. If these files are not included in the repository, generate them using the collection and training workflow.*

## Limitations

- The current prototype recognizes a limited set of predefined static gestures.
- Recognition performance depends on lighting conditions and hand positioning.
- Histogram-based segmentation can be affected by background colors and environmental changes.
- The initial dataset contains a relatively small number of images.
- The system does not currently interpret complete sign language sentences or dynamic gesture sequences.

## Future Enhancements

- Expand the gesture vocabulary.
- Collect larger and more diverse training datasets.
- Introduce data augmentation to improve model robustness.
- Explore MediaPipe hand landmark detection.
- Implement LSTM or Transformer-based architectures for dynamic gesture recognition.
- Add text-to-speech functionality.
- Optimize the model for mobile and edge-device deployment.

## Learning Outcomes

Through this project, we explored:

- Supervised Machine Learning.
- Deep Learning and CNN architectures.
- Image preprocessing and segmentation.
- Feature extraction and classification.
- Forward propagation and backpropagation.
- Loss functions and optimization algorithms.
- Model evaluation and generalization.
- Real-time computer vision applications.

## Contributors

**Developed as a Machine Learning academic project.**

Salman NS 
Hashim

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

<p align="center">
  <b>Built with Python, OpenCV and TensorFlow.</b>
  <br>
  Exploring the application of Deep Learning in gesture recognition.
</p>
