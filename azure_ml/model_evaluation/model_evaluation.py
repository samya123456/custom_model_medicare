import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd


def evaluate_model(input_model, input_data, output_evaluation):
    # Load model
    model = joblib.load(input_model)

    # Load preprocessed data
    data = pd.read_pickle(input_data)
    X_test, y_test = data['X_test'], data['y_test']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Save evaluation metrics
    evaluation = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    with open(f"{output_evaluation}/metrics.txt", 'w') as f:
        for key, value in evaluation.items():
            f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', type=str, required=True)
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_evaluation', type=str, required=True)
    args = parser.parse_args()

    evaluate_model(args.input_model, args.input_data, args.output_evaluation)
