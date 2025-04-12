import pandas as pd
from sklearn.model_selection import train_test_split
from modules import preprocess
from modules import train
from modules import evaluate

def main():
    # Load data
    train_data = pd.read_csv("data/KDDTrain+.txt", header=None)
    test_data = pd.read_csv("data/KDDTest+.txt", header=None)

    # Main execution
    X_train, y_train, scaler, label_encoders, label_encoder = preprocess.preprocess_data(train_data)
    X_test, y_test, _, _, _ = preprocess.preprocess_data(test_data, label_encoder)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train & Evaluate
    rf_model, classifier1 = train.train_model_rf(X_train, y_train)
    xgb_model, classifier2 = train.train_model_xgb(X_train, y_train, X_val, y_val)
    evaluate.evaluate_model(rf_model, X_test, y_test, classifier1)
    evaluate.evaluate_model(xgb_model, X_test, y_test, classifier2)

if __name__ == '__main__':
    main()