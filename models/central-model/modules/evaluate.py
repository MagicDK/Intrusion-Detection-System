from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

def evaluate_model(model, X_test, y_test, classifier):
    y_pred = model.predict(X_test)
    print("\n--------------------------------------------------")
    print("\n{class_type} Classifier:".format(class_type=classifier))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n--------------------------------------------------\n")