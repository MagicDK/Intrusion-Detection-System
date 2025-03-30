import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    train_arff = arff.loadarff(train_path)
    test_arff = arff.loadarff(test_path)
    train_df = pd.DataFrame(train_arff[0])
    test_df = pd.DataFrame(test_arff[0])
    train_df.head()
    test_df.head()
    return train_df, test_df

def preprocess_data(df, label_encoder=None, scaler=None, fit=True, feature_columns=None):
    df = df.copy()
    
    # 1. Preserve the 'class' column separately
    y_values = df['class'].str.decode('utf-8') if df['class'].dtype == object else df['class']
    df = df.drop(columns=['class'])  # Remove early to avoid interference
    
    # 2. Process binary features
    binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype(int)
    
    # 3. One-hot encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # 4. Align features if this is test data
    if not fit:
        missing_cols = set(feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[feature_columns]
    else:
        feature_columns = df.columns
    
    # 5. Scale numerical features
    numerical_cols = [col for col in df.columns if col not in binary_cols + categorical_cols]
    if fit:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # 6. Encode labels
    if fit:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_values)
    else:
        y = label_encoder.transform(y_values)
    
    return df.values, y, scaler, label_encoder, feature_columns

# Federated Data Preparation
def create_federated_data(X, y, num_clients=5):
    # Shuffle data to ensure IID distribution across clients
    indices = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[indices], y[indices]
    
    clients = []
    data_per_client = len(X) // num_clients
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client if i != num_clients - 1 else len(X)
        client_X = X_shuffled[start:end]
        client_y = y_shuffled[start:end]
        
        # Convert to TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
        dataset = dataset.batch(32)  # Remove .repeat() to avoid infinite loops
        clients.append(dataset)
    return clients

# Create TFF model
def create_keras_model(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def model_fn():
    keras_model = create_keras_model(input_dim=X_train.shape[1])
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Load data
train_df, test_df = load_data("KDDTrain+.arff", "KDDTest+.arff")

# Preprocess training data (fit=True)
X_train, y_train, scaler, label_encoder, feature_columns = preprocess_data(train_df, fit=True)

# Preprocess test data (fit=False, use saved scaler/label_encoder/feature_columns)
X_test, y_test, _, _, _ = preprocess_data(
    test_df,
    label_encoder=label_encoder,
    scaler=scaler,
    fit=False,
    feature_columns=feature_columns
)

# After preprocessing, check shapes
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)

# Create federated data (IID)
federated_train_data = create_federated_data(X_train, y_train, num_clients=5)

# Initialize trainer
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(0.02),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(1.0)
)

state = trainer.initialize()
for round_num in range(10):  # Simulating 10 communication rounds
    state, metrics = trainer.next(state, federated_train_data)
    print(f'Round {round_num+1}, Metrics: {metrics}')

# 1. Extract weights from the trained TFF state
# For TFF 0.87.0, the weights are stored as a list of lists
tff_weights = state.global_model_weights

# Evaluating the trained model
print("Training Class Balance:", np.unique(y_train, return_counts=True))
print("Test Class Balance:", np.unique(y_test, return_counts=True))

# 2. Create and compile your Keras model
central_model = create_keras_model(X_train.shape[1])
central_model.compile(loss='binary_crossentropy', 
                     metrics=['accuracy', 
                             tf.keras.metrics.Precision(name='precision'),
                             tf.keras.metrics.Recall(name='recall')])

# 3. Convert and assign weights (special handling for TFF 0.87.0 structure)
if isinstance(tff_weights[0], list):
    # Handle nested list structure in TFF 0.87+
    flat_weights = [item for sublist in tff_weights for item in sublist]
else:
    flat_weights = tff_weights

# Assign weights to Keras model
central_model.set_weights([w.numpy() if hasattr(w, 'numpy') else w for w in flat_weights])

# 4. Run evaluation
print("\n=== Test Set Evaluation ===")
results = central_model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}")  # Crucial for anomaly detection
print(f"Recall: {results[3]:.4f}")     # Attack detection rate