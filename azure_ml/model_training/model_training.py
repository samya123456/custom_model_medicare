import argparse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd


def train_model(input_data, output_model):
    # Load preprocessed data
    data = pd.read_pickle(input_data)
    X_train, y_train = data['X_train'], data['y_train']

    # Initialize and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, output_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_model', type=str, required=True)
    args = parser.parse_args()

    train_model(args.input_data, args.output_model)
