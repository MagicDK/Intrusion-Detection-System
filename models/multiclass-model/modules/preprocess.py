import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

    # Drop unused columns
    df = df.drop(columns=["difficulty_level"], errors='ignore')

    # Identify categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]

    # Apply Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Handle unseen labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])  # Train encoding
    else:
        df["label"] = df["label"].apply(lambda x: x if x in label_encoder.classes_ else "unknown")  
        label_encoder.classes_ = np.append(label_encoder.classes_, "unknown")  # Add "unknown" class
        df["label"] = label_encoder.transform(df["label"])  

    # Convert remaining columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns that are NaN
    df = df.dropna(axis=1, how="all")

    # Fill remaining NaN values with column mean
    df.fillna(df.mean(), inplace=True)

    # Remove infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Normalize numerical features
    scaler = StandardScaler()
    x = scaler.fit_transform(df.drop(columns=["label"]))
    y = df["label"].values

    return x, y, scaler, label_encoders, label_encoder