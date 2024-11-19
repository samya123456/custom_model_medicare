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

import json
def preprocess_data(input_data):
    model_file = "@MEDICARE.PUBLIC.DATA/dataset1.csv"
    with SnowflakeFile.open(model_file, 'rb', require_scoped_url = False) as f:
        df = pd.read_csv(f)
        print('*********** ', df.head(10))
    

  
$$;