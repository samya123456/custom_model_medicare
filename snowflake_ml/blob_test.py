from azure.storage.blob import BlobServiceClient
container_name = "models"
blob_name = "data_preprocessing.sql"
model_path = './snowflake_ml/data_preprocessing.sql'
blob_service_client = BlobServiceClient(
    account_url="https://medicarestorageaccml.blob.core.windows.net",
    credential="DwMWqEcsrCeVBZefG0aN1nDwwgjOrZLYoOaKtpMHLkRPE8l30n6QF54Ypd1JArtHO+h20mcNL92q+AStl+QJ2w=="
)
blob_client = blob_service_client.get_blob_client(
    container=container_name, blob=blob_name)
with open(model_path, "rb") as model_file:
    blob_client.upload_blob(model_file, overwrite=True)

    # Return Azure Blob Storage path for reference
