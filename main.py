
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loading import load_data
from src.eda import plot_churn_distribution, plot_numerical_distributions, plot_correlation_matrix
from src.preprocessing import handle_missing_values, encode_categorical_variables, scale_features
from src.feature_engineering import create_total_services_feature
from src.modeling import build_logistic_regression_model, build_random_forest_model
from src.evaluation import evaluate_classification_model, plot_roc_curve

def main():
    
    df = load_data('data/telco_churn.csv')
    plot_churn_distribution(df)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    plot_numerical_distributions(df, numerical_features)
    plot_correlation_matrix(df)

    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Churn' in numerical_features:
        numerical_features.remove('Churn')
    df = scale_features(df, numerical_features)

    df = create_total_services_feature(df)

 
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #build
    logreg_model = build_logistic_regression_model(X_train, y_train)
    rf_model = build_random_forest_model(X_train, y_train)

    #evaluation
    evaluate_classification_model(logreg_model, X_test, y_test, 'Logistic Regression')
    plot_roc_curve(logreg_model, X_test, y_test, 'Logistic Regression')

    evaluate_classification_model(rf_model, X_test, y_test, 'Random Forest')
    plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')

if __name__ == '__main__':
    main()
