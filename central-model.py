import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    return train_df, test_df

# Preprocess dataset
def preprocess_data(df, label_encoder=None):
    column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
                    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
                    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
                    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                    "dst_host_srv_rerror_rate", "label", "difficulty_level"]
    df.columns = column_names
    
    # Drop unnecessary columns
    df = df.drop(columns=["difficulty_level"], errors="ignore")
    
    # Convert to binary classification
    df["label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
    
    # Encode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Encode labels (binary classification: 0 = normal, 1 = attack)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])
    else:
        df["label"] = label_encoder.transform(df["label"])
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=["label"]))
    y = df["label"].values
    
    return X, y, scaler, label_encoders, label_encoder

# Train model
def train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        n_estimators=500,  
        learning_rate=0.1,  
        max_depth=6,
        gamma=0.2,
        subsample=0.9,  
        colsample_bytree=0.7,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  
        early_stopping_rounds=15  
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Main execution
train_data, test_data = load_data("KDDTrain+.txt", "KDDTest+.txt")
X_train, y_train, scaler, label_encoders, label_encoder = preprocess_data(train_data)
X_test, y_test, _, _, _ = preprocess_data(test_data, label_encoder)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train & Evaluate
model = train_model(X_train, y_train, X_val, y_val)
evaluate_model(model, X_test, y_test)
