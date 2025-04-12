import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, accuracy_score, classification_report, confusion_matrix

# Load dataset
def load_data(train_path, test_path):
    # Correct column names based on NSL-KDD dataset format
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
    ]
    
    train_data = pd.read_csv("KDDTrain+.txt", names=columns)
    test_data = pd.read_csv("KDDTest+.txt", names=columns)
    
    # Drop the "difficulty_level" column (not needed)
    train_data = train_data.drop(columns=["difficulty_level"])
    test_data = test_data.drop(columns=["difficulty_level"])

    return train_data, test_data

# Preprocess dataset
def preprocess_data(df, label_encoder=None):
    # Drop unused columns
    df = df.drop(columns=["num_outbound_cmds"], errors='ignore')

    # Identify categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]

    # Apply Label Encoding for categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 🚀 Fix: Handle unseen labels in test data
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])  # Train encoding
    else:
        df["label"] = df["label"].apply(lambda x: x if x in label_encoder.classes_ else "unknown")  
        label_encoder.classes_ = np.append(label_encoder.classes_, "unknown")  # Add "unknown" class
        df["label"] = label_encoder.transform(df["label"])  

    # Convert all remaining columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    # Fill remaining NaN values with column mean
    df.fillna(df.mean(), inplace=True)

    # Remove infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=["label"]))
    y = df["label"].values

    return X, y, scaler, label_encoders, label_encoder

# Train model
def train_model_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

# Train model
def train_model_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
    n_estimators=100,  # More boosting rounds for better accuracy
    max_depth=10,  # Keep depth moderate to prevent overfitting
    learning_rate=0.1,  # Controls step size during training
    scale_pos_weight=1,  # Can adjust for imbalanced classes (try different values)
    objective="multi:softmax",  # Multiclass classification
    eval_metric="mlogloss",
    random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    #print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(" Accuracy:", accuracy_score(y_test, y_pred))  
    print(" Precision (macro):", precision_score(y_test, y_pred, average="macro"))  
    print(" Recall (macro):", recall_score(y_test, y_pred, average="macro"))  
    #print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Main execution
train_data, test_data = load_data("KDDTrain+.txt", "KDDTest+.txt")
X_train, y_train, scaler, label_encoders, label_encoder = preprocess_data(train_data)
X_test, y_test, _, _, _ = preprocess_data(test_data, label_encoder)

# Train-validation split
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train & Evaluate
model = train_model_xgb(X_train, y_train)
evaluate_model(model, X_test, y_test)
