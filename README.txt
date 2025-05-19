# Customer Churn Prediction with Data Visualization

This project aims to analyze telecom customer data to *predict customer churn* using machine learning and visualize key insights using *Power BI*.

## Overview

- *Goal*: Identify customers who are likely to stop using the service (churn) and understand the factors that influence churn.
- *Technologies Used*:
  - Python (Pandas, Seaborn, Scikit-learn, SMOTE, Matplotlib)
  - Power BI (for dashboard visualizations)
  - Random Forest Classifier
  - Git & GitHub for version control

## Dataset

- *Source*: Kaggle - Telco Customer Churn dataset
- *Columns*: Demographic info, service details, and churn status.
- *Target Variable*: Churn (Yes = 1, No = 0)

## Python Workflow

1. *Data Cleaning*
   - Converted TotalCharges to numeric
   - Handled missing values
   - Dropped irrelevant columns
   - Encoded categorical data

2. *Data Processing*
   - Feature scaling using StandardScaler
   - Balanced data using SMOTE

3. *Model Building*
   - Used RandomForestClassifier to train and test churn prediction model
   - Achieved *84% accuracy*
   - Exported the trained model to churn_prediction_model.pkl
   - Saved final dataset with predictions to Processed_Customer_Churn.csv

## Power BI Dashboard

- Created interactive charts to visualize:
  - Churn distribution
  - Churn by contract type
  - Monthly charges vs churn
- Added custom colors, titles, and layout for clarity.

## How to Use

1. Clone the repo or download the files.
2. Open the Python script (churn_prediction_code.py) to understand the model workflow.
3. Open Processed_Customer_Churn.csv in Power BI to explore the dashboard or use your own data.
4. Predict churn by feeding new data into the trained model (churn_prediction_model.pkl).

## Author

- Mohamed Faheem
- BCA Final Year Student | Aspiring Data Analyst
- GitHub: https://github.com/Faheem23663

## License

This project is for academic purposes only.