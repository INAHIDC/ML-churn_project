from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_logistic_regression_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def build_random_forest_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
