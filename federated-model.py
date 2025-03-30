import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
# Assumes the dataset is in the same format as used in the centralized model

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

# Load and preprocess data
train_df, test_df = load_data("KDDTrain+.arff", "KDDTest+.arff")
#print("Raw training columns:", train_df.columns.tolist())
#print("Raw test columns:", test_df.columns.tolist())

#print("Actual columns:", [f"'{col}'" for col in train_df.columns])
#train_df = train_df.rename(columns={'class ': 'class'})

#print("\n=== DATA VALIDATION ===")
#print("Train columns:", train_df.columns.tolist())
#print("Test columns:", test_df.columns.tolist())
#print("'class' in train?", 'class' in train_df.columns)
#print("'class' in test?", 'class' in test_df.columns)
#print("Sample class values:", train_df['class'].head(3))

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

# Test 1 round
#state = trainer.initialize()
#result = trainer.next(state, federated_train_data)
#print("Round 1 metrics:", result.metrics)  # Check for loss/accuracy

# Evaluating the trained model
#central_model = create_keras_model()
#central_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#central_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
#print("Evaluation on test data:", central_model.evaluate(X_test, y_test))