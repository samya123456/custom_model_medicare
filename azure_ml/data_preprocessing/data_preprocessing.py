import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def preprocess_data(input_data, output_data):
    # Load the data
    df = pd.read_csv(input_data)

    # Process data here as per your preprocessing logic
    # Example: Drop certain columns and impute missing values
    X = df.drop('V28HCCCODED', axis=1)
    y = df['V28HCCCODED']

    # Preprocessing steps
    numerical_cols = [
        col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ])

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Save preprocessed data
    preprocessed_data = {'X_train': X_train,
                         'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    pd.to_pickle(preprocessed_data, output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_data', type=str, required=True)
    args = parser.parse_args()

    preprocess_data(args.input_data, args.output_data)
