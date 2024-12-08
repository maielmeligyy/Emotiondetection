# README for Emotion Recognition Model Evaluation

## **Project Overview**
This project is part of the **KH6006CEM - Machine Learning and Related Applications** module. It focuses on evaluating three machine learning models for emotion recognition: **Support Vector Machine (SVM)**, **Random Forest (RF)**, and **Convolutional Neural Network (CNN)**. The models are trained and evaluated on the FER2013 dataset for facial emotion recognition.

## **Dataset**
The FER2013 dataset contains grayscale images of faces, categorized into multiple emotion classes:
- **Training Directory**: `/fer2013/train`
- **Testing Directory**: `/fer2013/test`

Each image is resized to `48x48` and normalized to scale pixel values between 0 and 1.

## **Key Features**
- **Models Evaluated**:
  1. **SVM**: Fine-tuned using GridSearchCV for hyperparameter optimization.
  2. **Random Forest**: Fine-tuned using GridSearchCV with a range of tree-based parameters.
  3. **CNN**: A custom deep learning architecture using Keras.

- **Metrics**:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **ROC-AUC**

- **Visualizations**:
  - Confusion Matrices for SVM, RF, and CNN.
  - Feature Importance for Random Forest.
  - CNN Feature Maps.
  - Bar charts comparing models' metrics.

## **Directory Structure**
- **Dataset**: FER2013, with `train` and `test` subdirectories.
- **Scripts**: Python scripts for training, evaluating, and visualizing models.
- **Output**:
  - Saved models (`emotion_recognition_model.h5`)
  - Training accuracy plots
  - Model comparison results in text files

## **Setup Instructions**
1. **Install Dependencies**:
   - Python 3.8+
   - Required libraries:
     ```bash
     pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
     ```

2. **Data Preprocessing**:
   - Images are rescaled and resized to `48x48`.
   - For SVM and RF, images are flattened into vectors.

3. **Training Models**:
   - Run the script to train and fine-tune all models.

4. **Evaluation**:
   - Metrics are printed to the console.
   - Results are saved to text files.

5. **Visualization**:
   - Confusion matrices and feature maps are plotted.
   - Bar charts for metric comparison.

## **Model Architectures**
### **1. Support Vector Machine (SVM)**
- Kernel: Linear and RBF
- Hyperparameters:
  - `C` (Regularization): [0.1, 1, 10]
- Outputs probabilities using `probability=True`.

### **2. Random Forest**
- Parameters:
  - Number of Trees: [50, 100, 200]
  - Maximum Depth: [None, 10, 20]
- Fine-tuned using GridSearchCV.

### **3. Convolutional Neural Network (CNN)**
- Layers:
  - **Conv2D**: Two layers with 32 and 64 filters, respectively.
  - **Pooling**: MaxPooling layers after each Conv2D layer.
  - **Dropout**: 25% and 50% dropout rates.
  - **Dense**: Fully connected layer with 128 neurons.
  - **Output Layer**: Softmax activation.
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical Crossentropy
- Epochs: 10

## **Evaluation Results**
### Metrics:
| Model          | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------|----------|-----------|--------|----------|---------|
| SVM            | 0.2188   | 0.0479    | 0.2188 | 0.0785   | 0.4722  |
| Random Forest  | 0.2812   | 0.1727    | 0.2812 | 0.2128   | 0.5771  |
| CNN            | 0.1344   | 0.1489    | 0.1344 | 0.0748   | 0.4939  |


### Visualizations:
- **Confusion Matrices**: Show model performance for each class.
- **Feature Importance**: Visualized for Random Forest.
- **Bar Charts**: Compare metrics across models.

## **Usage**
1. **Run Training and Evaluation**:
   ```bash
   python train_and_evaluate.py
   ```
2. **View Results**:
   - Metrics: `model_results.txt`
   - Visualization: Plots saved as `.png` files.

## **Contact**
For any questions or feedback, please contact:
- **Mai Elmeligy** (ID: 202101086)
- **Course**: KH6006CEM - Machine Learning and Related Applications
