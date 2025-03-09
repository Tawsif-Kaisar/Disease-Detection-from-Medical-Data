# **Parkinson's Disease Detection using Machine Learning**

## **Overview**
This project develops a **machine learning model** to detect **Parkinson’s disease** using **biomedical voice features**. The dataset consists of 22 voice measurements from patients and healthy individuals. After preprocessing (feature selection, scaling, and **PCA**), models like **SVM, Random Forest, and Neural Networks** were trained, achieving **95.2% accuracy with SVM**. The best model was deployed for real-time predictions, highlighting AI’s role in **early Parkinson’s detection**.

## **Dataset**
- **Source:** Parkinson’s Disease dataset
- **Features:** 22 voice-related biomedical attributes
- **Target Variable:** `status` (0 = Healthy, 1 = Parkinson’s Disease)

## **Technologies Used**
- Python, Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn (SVM, Random Forest, Logistic Regression, PCA)
- Neural Networks (MLPClassifier)
- SMOTE (for data balancing)
- Joblib (for model saving & deployment)

## **Project Workflow**
1. **Data Preprocessing:** Remove irrelevant features, handle missing values, scale data, and apply **PCA**.
2. **Model Training:** Train **SVM, Random Forest, Logistic Regression, and Neural Networks**, using **GridSearchCV** for optimization.
3. **Performance Evaluation:** Evaluate models using **accuracy, precision, recall, F1-score, and confusion matrix**.
4. **Model Deployment:** Save the best model and test it on **new patient data**.

## **Results**
| Model                  | Accuracy  |
|------------------------|-----------|
| Random Forest         | 93.5%     |
| Support Vector Machine (SVM) | 95.2%     |
| Logistic Regression   | 90.1%     |
| Neural Network (MLP)  | 94.3%     |

**SVM was the best-performing model, achieving 95.2% accuracy.**

## **How to Use**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Parkinsons-Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python disease_detection_from_medical_data.py
   ```
4. Test with a new patient sample:
   ```python
   loaded_model = joblib.load("best_svm.pkl")
   prediction = loaded_model.predict(new_patient_data)
   print("Predicted:", "Parkinson's" if prediction[0] == 1 else "Healthy")
   ```

## **Future Improvements**
- Expand the dataset with additional biomarkers
- Experiment with deep learning for improved accuracy
- Deploy a real-time detection system using a web app
