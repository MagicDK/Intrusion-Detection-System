from sklearn.metrics import recall_score, precision_score, accuracy_score
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def evaluate_model(model, X_test, y_test, classifier):
    y_pred = model.predict(X_test)
    print("\n--------------------------------------------------")
    print("\n{class_type} Classifier:".format(class_type=classifier))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="macro"))
    print("Recall:", recall_score(y_test, y_pred, average="macro"))
    print("\n--------------------------------------------------\n")