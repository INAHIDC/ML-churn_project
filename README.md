# Telco CX Churn Prediction Project

Customer Churn Prediction project, Shows whether a customer will churn (stop doing business with the company) based on their demographic, account, and service usage information.

![churn_distribution png](https://github.com/user-attachments/assets/6d774eb1-ee21-44c1-a88d-59e74b015cd5)




---

## Content

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset)
3. [Project Structure](#file-tree)
4. [Installation](#file-tree)
5. [Data Analysis (EDA)](#Analysis)
6. [Preprocessing](#data-preprocessing)
7. [Feature ](#feature-engineering)
8. [Modeling](#modeling)
9. [Evaluation](#evaluation)
10. [Conclusion](#conclusion)
11. [Acknowledgements](#source)

---

Customer churn is a issue in the telecommunications industry. Understanding why customers leave and predicting churn can help companies come up with new strategies to retain customers. This builds a machine learning model to predict customer churn based on various data provided in the csv file.

---

## Dataset

The dataset contains information about a fictional tele communication company's customers (CX). It includes customer demographic info, account details, and the services they have subscribed to.

- **Features Include**:
  - **CustomerID**
  - **Demographic**: Gender, Senior Citizen status, Partner, and Dependents.
  - **Account Information**: Tenure, Contract type, Payment method, Monthly charges, and Total charges.
  - **Services**: Phone service, Multiple lines, Internet service, Online security, Online backup, Device protection, Tech support, Streaming TV, and Streaming movies.
  
- **Target Variable**:
  - **Churn**: If the customer churned (Yes) or not (No).

---

## file tree

```
churn_prediction_project/
├── data/
│   └── telco_churn.csv
├── images/
│   ├── churn_distribution.png
│   ├── numerical_distributions.png
│   └── correlation_matrix.png
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md
```

- **data/**: dataset
- **images/** 
- **src/**: modules
- **main.py**: main logic
- **requirements.txt**: packages needed to run this
- **README.md**

---

## TO START

1. **clone the repo**

   ```bash
   git clone https://github.com/INAHIDC/ML-churn_project.git
   cd churn_prediction_project
   ```

2. **Dependencies** ;skip if you have it already

   i didnt use a virtual env for this but its recommended!

   ```bash
   pip install -r requirements.txt
   ```

3. **run it**

   ```bash
   python main.py
   ```

---

## Analysis

Understanding the data is important before going into modeling. 

### 1. Churn Distribution

We start by examining the distribution of the target variable.

![Churn Distribution] ![churn_distribution png](https://github.com/user-attachments/assets/62854ba5-f933-4fd0-9fa9-7a3723d8e30c)


- **Observation**: The dataset is imbalanced, with a higher number of customers who did not churn.

### 2. Numerical Features Distribution

We plotted histograms for numerical features like `tenure`, `MonthlyCharges`, and `TotalCharges`.

![Numerical Distributions] ![numerical_distributions png](https://github.com/user-attachments/assets/28894f66-9967-4858-9497-21e8b643ed0e)


- **Observation**: 
  - **Tenure**: lots of customers are either new or have been with the company for a long time.
  - **MonthlyCharges**: a wide range of monthly charges.
  - **TotalCharges**: Similar to tenure, reflects the cumulative amount charged.

### 3. Correlation Matrix

analyzed correlations between numerical variables.

![Correlation Matrix]![correlation_matrix png](https://github.com/user-attachments/assets/2346a9f0-5261-42f5-994d-1791a9ac54d4)


- **Observation**:  correlation between `TotalCharges` and `tenure`, which makes sense as longer tenure typically results in higher total charges.

---

## Data Preprocessing

preprocessing include handling missing values, encoding categorical variables

### 1. Handling Missing Values

- Replaced empty strings with `NaN`.
- Dropped rows with missing values.

### 2. Encoding Categorical Variables

- **Label**: binary categorical variables.
- **One Hot**:  categorical variables with more than two categories.

### 3. Feature Scaling

- **standardization**: added to numerical features to normalize the data.

---

## Feature Engineering

- **TotalServices**: Summed up the number of services a customer has subscribed to.

---

## Modeling

two models to predict customer churn.

### 1. Logistic Regression

simple model to establish a baseline.

- **Pros**: Easy to understand.
- **Cons**: does not show complex relationships

### 2. Random Forest Classifier

An ensemble model to improve performance.

- **Pros**:  nonlinear relationships.
- **Cons**: not understable/ or intuitive 

---

## Evaluation

evaluated the models using classification reports, confusion matrices, and ROC curves.

### 1. Logistic Regression Results

- **Accuracy**: Moderate.
- **Precision and Recall**: the model's performance on predicting churned CX.
- **Confusion Matrix and ROC Curve**: shows true positives, false positives etc.

### 2. Random Forest

- **Accuracy**: Improved compared to Logistic Regression.
- helps find which features contribute most to predictions.
- **Confusion Matrix and ROC Curve**: Showed better performance in distinguishing churned customers.

---

## Conclusion

- **What do i think?**
  - Forest model outperformed Logistic Regression in predicting customer churn.
  - Features like contract type, tenure, and monthly charges are important predictors.

- **Implications**:
  - **Cx Retention**: Focus on customers with month to month contracts and high monthly charges.
  - **Service Improvement**: improve services that are linked to higher churn rates.

---

## source

- **Dataset Source**: [IBM Cognos Analytics Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).

