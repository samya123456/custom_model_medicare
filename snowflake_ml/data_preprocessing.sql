CREATE OR REPLACE FUNCTION data_preprocessing(input_data string)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('pandas','scikit-learn','snowflake-snowpark-python')
HANDLER = 'preprocess_data'
AS
$$
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from snowflake.snowpark import Session
from snowflake.snowpark.files import SnowflakeFile
from sklearn.linear_model import LogisticRegression
import json
def preprocess_data(input_data):
    model_file = "@MEDICARE.PUBLIC.DATA/dataset1.csv"
    with SnowflakeFile.open(model_file, 'rb', require_scoped_url = False) as f:
        df = pd.read_csv(f)
        print('*********** ', df.head(10))
    X = df.drop('V28HCCCODED', axis=1)
    y = df['V28HCCCODED']
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply transformations
    X_transformed = preprocessor.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(
         X_transformed, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    model_path = '/tmp/logistic_regression_model.pkl'
    joblib.dump(model, model_path)
    return model_path

  
$$;