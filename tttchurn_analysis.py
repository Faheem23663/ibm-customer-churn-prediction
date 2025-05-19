import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert 'TotalCharges' column to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop irrelevant column
df.drop(columns=["customerID"], inplace=True, errors="ignore")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Encode categorical variables dynamically
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove("Churn")  # Exclude target variable

encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Define features and target variable
X = df.drop(columns=["Churn"])  # Features
y = df["Churn"]  # Target variable

# Normalize numerical columns
scaler = StandardScaler()
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {accuracy_rf:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:\n", cm_rf)

# Feature Importance Analysis
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Feature Importances")
plt.show()

# Add Predictions to Original DataFrame
df["Predicted_Churn"] = rf_model.predict(X)

# Save the final dataset with predictions
df.to_csv("Processed_Customer_Churn.csv", index=False)
print("✅ Processed dataset saved as 'Processed_Customer_Churn.csv'")

# Save the model for future use
joblib.dump(rf_model, "churn_prediction_model.pkl")
print("✅ Model saved as 'churn_prediction_model.pkl'")

# Visualize Churn Distribution
sns.countplot(x="Churn", hue="Churn", data=df, palette="coolwarm", legend=False)
plt.title("Customer Churn Distribution")
plt.show()
