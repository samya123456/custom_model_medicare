import gzip
import joblib
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from snowflake_conn_util import SnowflakeUtil  # Import the SnowflakeUtil class


class ModelHandler:
    """
    A class to handle the execution of Snowflake notebooks, retrieving models, 
    extracting compressed files, and uploading them to Azure Blob Storage.
    """

    @staticmethod
    def execute_notebook(notebook_name: str):
        """
        Executes a Snowflake notebook.

        Args:
            notebook_name (str): Fully qualified name of the Snowflake notebook to execute.
        """
        conn = SnowflakeUtil.get_connection()

        try:
            query = f'EXECUTE NOTEBOOK {notebook_name}()'
            result = SnowflakeUtil.call_function(query)
            print(f"Notebook executed successfully: {result}")
        except Exception as e:
            print(f"Error while executing notebook: {e}")
        finally:
            conn.close()

    @staticmethod
    def get_model(stage_path: str, local_path: str):
        """
        Downloads a model file from a Snowflake stage to a local directory.

        Args:
            stage_path (str): Path to the file in the Snowflake stage (e.g., @STAGE/PATH/FILE).
            local_path (str): Local directory where the file should be downloaded.
        """
        conn = SnowflakeUtil.get_connection()

        try:
            query = f'GET {stage_path} file://{local_path}'
            result = SnowflakeUtil.call_function(query)
            print(f"Model downloaded successfully: {result}")
        except Exception as e:
            print(f"Error while downloading model: {e}")
        finally:
            conn.close()

    @staticmethod
    def extract_and_upload_model(compressed_model_path: str, container_name: str, blob_name: str, account_url: str, credential: str) -> str:
        """
        Extracts a .pkl.gz file, decompresses it, and uploads the decompressed model to Azure Blob Storage.

        Args:
            compressed_model_path (str): Local path to the compressed .pkl.gz file.
            container_name (str): Name of the Azure Blob Storage container.
            blob_name (str): Name of the blob (file) to upload to the container.
            account_url (str): Azure Blob Storage account URL.
            credential (str): Azure Blob Storage access credential.

        Returns:
            str: A message indicating success or failure.
        """
        try:
            # Validate compressed file exists
            if not os.path.exists(compressed_model_path):
                raise FileNotFoundError(
                    f"Compressed model file not found: {compressed_model_path}")

            # Decompress the .pkl.gz file
            decompressed_model_path = compressed_model_path.replace(
                '.gz', '')  # e.g., 'model.pkl.gz' -> 'model.pkl'
            with gzip.open(compressed_model_path, 'rb') as compressed_file:
                with open(decompressed_model_path, 'wb') as decompressed_file:
                    decompressed_file.write(compressed_file.read())
            print(f"Decompressed model saved to {decompressed_model_path}")

            # Upload the decompressed model to Azure Blob Storage
            blob_service_client = BlobServiceClient(
                account_url=account_url, credential=credential)
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name)

            with open(decompressed_model_path, "rb") as model_file:
                blob_client.upload_blob(model_file, overwrite=True)
            print(
                f"Model uploaded successfully to Azure Blob Storage: {blob_name}")

            return f"Decompressed and uploaded model successfully: {blob_name}"

        except Exception as e:
            return f"Error occurred during extraction and upload: {e}"


# Parameters
NOTEBOOK_NAME = "MEDICARE.PUBLIC.MY_NOTEBOOK"
STAGE_PATH = "@MEDICARE.PUBLIC.DATA/logistic_regression_model.pkl"
LOCAL_PATH = "./snowflake_ml/model/"
COMPRESSED_MODEL_PATH = "./snowflake_ml/model/logistic_regression_model.pkl.gz"
CONTAINER_NAME = "models"
BLOB_NAME = "logistic_regression_model.pkl"
ACCOUNT_URL = "https://medicarestorageaccml.blob.core.windows.net"
CREDENTIAL = "DwMWqEcsrCeVBZefG0aN1nDwwgjOrZLYoOaKtpMHLkRPE8l30n6QF54Ypd1JArtHO+h20mcNL92q+AStl+QJ2w=="

# Execution Workflow
if __name__ == "__main__":
    # Execute Snowflake notebook
    ModelHandler.execute_notebook(NOTEBOOK_NAME)

    # Retrieve model from Snowflake stage
    ModelHandler.get_model(STAGE_PATH, LOCAL_PATH)

    # Extract and upload model to Azure Blob Storage
    result = ModelHandler.extract_and_upload_model(
        COMPRESSED_MODEL_PATH, CONTAINER_NAME, BLOB_NAME, ACCOUNT_URL, CREDENTIAL
    )
    print(result)
