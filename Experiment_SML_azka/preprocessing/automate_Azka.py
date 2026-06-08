
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop_duplicates()

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df["Dependents"] = df["Dependents"].astype(int)

    label_cols = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Loan_Status"
    ]

    encoder = LabelEncoder()

    for col in label_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    scale_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term"
    ]

    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    df_preprocessed = X.copy()
    df_preprocessed["Loan_Status"] = y

    df_preprocessed.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess("train_u6lujuX_CVtuZ9i.csv", "dataset_preprocessing.csv")
