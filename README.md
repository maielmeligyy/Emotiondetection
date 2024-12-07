# **Emotion Detection Using FER2013 Dataset**

## **Overview**
This project explores facial emotion recognition using the **FER2013 dataset**, implementing and comparing the performance of two machine learning models:
1. **Support Vector Machine (SVM)**  
2. **Convolutional Neural Network (CNN)**  

The project demonstrates the effectiveness of CNNs in learning spatial hierarchies for image-based tasks and highlights the challenges of using SVMs for emotion recognition.

---

## **Project Structure**
```
.
├── project.ipynb        # Jupyter Notebook containing the full implementation
├── README.md            # Project README file
├── requirements.txt     # List of required Python packages
├── fer2013/             # Folder for the FER2013 dataset
│   ├── train/           # Training images
│   ├── test/            # Test images

```

---

## **Dataset**
The **FER2013 dataset** contains 48x48 grayscale facial images labeled with seven emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise  

The dataset is available on Kaggle: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013).

---

## **Key Features**
- **Preprocessing**: Images are resized, normalized, and augmented for CNN training. Labels are one-hot encoded for compatibility.
- **Models Implemented**:
  - **SVM**: Evaluated using flattened image arrays.
  - **CNN**: Designed with convolutional, pooling, dropout, and dense layers.
- **Performance Evaluation**:
  - Confusion matrices, training/validation curves, and feature map visualizations.
  - Metrics: Accuracy, precision, recall, F1-score.

---

## **Setup and Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion-detection-fer2013.git
   cd emotion-detection-fer2013
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the FER2013 dataset and place it in the `dataset/` directory.

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook project.ipynb
   ```

---

## **Results**
- **CNN Accuracy**: 51.03%  
- **SVM Accuracy**: 15.62%  

| Metric      | SVM     | CNN     |
|-------------|---------|---------|
| Accuracy    | 15.62%  | 51.03%  |
| Precision   | 30.00%  | 65.00%  |
| Recall      | 15.62%  | 51.03%  |
| F1-Score    | 18.00%  | 56.00%  |

**Visualizations**:
- Feature maps from CNN layers show hierarchical learning of edges, shapes, and facial features.
- Confusion matrices highlight the CNN's strengths in recognizing "Happy" and "Neutral" emotions.

---

## **Usage**
1. **Train Models**: Run the notebook to preprocess data, train the SVM and CNN models, and generate results.
2. **Visualize Results**: View training curves, feature maps, and confusion matrices.
3. **Modify Hyperparameters**: Experiment with CNN architecture and SVM kernels.

---

## **Future Enhancements**
- Implement transfer learning with pre-trained models (e.g., ResNet, MobileNet).
- Address class imbalance in the dataset.
- Explore real-time emotion detection applications.

---

## **References**
- Kaggle. "FER2013 Dataset." [Link](https://www.kaggle.com/datasets/msambare/fer2013)
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media, 2019.
- TensorFlow Documentation. [Link](https://www.tensorflow.org)

---

## **License**
This project has no license.

---
