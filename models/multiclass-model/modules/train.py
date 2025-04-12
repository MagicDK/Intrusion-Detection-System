import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def train_model_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model, "Random Forest"

def train_model_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
    n_estimators=100,  
    max_depth=10,  
    learning_rate=0.1,
    objective="multi:softmax", 
    eval_metric="mlogloss",
    random_state=42
    )
    model.fit(X_train, y_train)
    return model, "Extreme Gradient Boosting"