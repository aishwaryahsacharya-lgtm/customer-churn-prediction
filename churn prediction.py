import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# LOAD DATASET
df = pd.read_csv("C:/Users/hsgee/Downloads/Telco-Customer-Churn.csv")
print("Dataset shape:", df.shape)
print(df.head())

# DATA CLEANING
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# DROP MISSING VALUES
df.dropna(inplace=True)

# DROP CUSTOMER ID
df.drop("customerID", axis=1, inplace=True)

# ENCODING
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# FEATURES & TARGET
x = df.drop("Churn", axis=1)
y = df["Churn"]

# TRAIN-TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# MODEL TRAINING
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# EVALUATION
y_pred = model.predict(x_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:", classification_report(y_test, y_pred))

# PREDICTION FUNCTION
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data], columns=x.columns)
    prediction = model.predict(input_df)

    return "Churn" if prediction[0] == 1 else "No churn"

# PREDICTION
sample = x.iloc[0].to_dict()
print("\nSample Prediction:", predict_churn(sample))

        
