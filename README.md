# Cardiovascular Disease Prediction Project

## Problem Statement
The objective of this project is to predict the likelihood of cardiovascular disease (CVD) in patients based on clinical and demographic factors. Cardiovascular diseases are a leading cause of mortality worldwide, and early detection is critical for timely intervention and preventive healthcare.

### Challenges
- CVD prediction is complex due to the interplay of medical, genetic, and lifestyle parameters.
- Imbalanced data, outliers, and overlapping feature distributions pose additional challenges.

## Dataset
**Source**: Kaggle - Cardiovascular Disease Dataset  
**Description**:  
- **Features**:  
  - Objective: Age, Height, Weight, Gender  
  - Examination: Blood Pressure, Cholesterol, Glucose  
  - Subjective: Smoking, Alcohol intake, Physical activity  
- **Target Variable**: `Cardio` (0 = No Disease, 1 = Disease)  
- **Size**: 70,000 samples, 12 features  

The entire dataset is used after careful preprocessing to retain maximum information.

## Proposed Solution
This project is **predictive**, leveraging historical medical and lifestyle data to classify individuals as at-risk or not at-risk of CVD.

### Machine Learning Techniques
The following ML models are implemented for predictive analysis:
1. **Logistic Regression**: Baseline model for binary classification.
2. **Decision Tree**: Captures non-linear relationships and interprets feature importance.
3. **K-Nearest Neighbor (KNN)**: Non-parametric method for proximity-based prediction.
4. **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.

### Preprocessing Strategy
1. Handle missing values and remove duplicates.
2. Convert age into years for better interpretability.
3. Handle outliers using IQR and percentile methods for features like height, weight, BMI, and blood pressure.
4. Standardize features such as blood pressure and BMI.

### Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**  
5-fold cross-validation is performed to ensure robust evaluation.

## Relevance
- Accurate prediction of cardiovascular diseases can significantly reduce medical emergencies.
- Enhances preventive healthcare measures, lowering healthcare costs and improving patient outcomes.

## Libraries Used
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

## Project Workflow
1. **Data Preprocessing**: Clean, transform, and standardize the data.
2. **Model Implementation**: Develop and evaluate Logistic Regression, Decision Tree, KNN, and SVM models.
3. **Feature Analysis**: Identify key predictors contributing to cardiovascular risk.
4. **Model Comparison**: Compare the performance of each model using standardized metrics.
5. **Visualization and Documentation**: Generate insightful visualizations and comprehensive reports.

## References
- [Cardiovascular Disease Dataset on Kaggle](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)
- [Research on Decision Trees and SVM for CVD Prediction](https://www.sciencedirect.com/science/article/pii/S2772963X24004113)
- [KNN for Cardiovascular Risk Prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9206502)
- [Comparative Analysis of KNN and Decision Trees](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9855428)
- [20 Models for CVD Prediction](https://www.kaggle.com/code/vbmokin/20-models-for-cdv-prediction)

## How to Run
1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the Jupyter Notebook or Python script to preprocess the data, train models, and evaluate results.
