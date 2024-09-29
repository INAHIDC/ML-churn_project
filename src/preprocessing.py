import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler

def handle_missing_values(df):
    df.replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def encode_categorical_variables(df):
    le = LabelEncoder()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in categorical_features:
        categorical_features.remove('customerID')
    for col in categorical_features:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df

def scale_features(df, numerical_features):
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df