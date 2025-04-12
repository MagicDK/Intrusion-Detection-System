import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def train_model_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model, "Random Forest"

def train_model_xgb(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        n_estimators=100,  
        learning_rate=0.1,  
        max_depth=6,
        gamma=0.2,
        subsample=0.9,  
        colsample_bytree=0.7,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  
        early_stopping_rounds=15  
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, "Extreme Gradient Boosting"