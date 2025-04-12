import pandas as pd
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
    
    # Encode labels ( binary classification: 0 = normal, 1 = anomaly (attack) )
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])
    else:
        df["label"] = label_encoder.transform(df["label"])
    
    # Convert columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=["label"]))
    y = df["label"].values
    
    return X, y, scaler, label_encoders, label_encoder