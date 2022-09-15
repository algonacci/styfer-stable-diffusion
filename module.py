from dotenv import dotenv_values
from google.oauth2 import service_account
from google.cloud import storage
import time

config = dotenv_values(".env")

CLOUD_STORAGE_BUCKET = config["CLOUD_STORAGE_BUCKET"]
credentials = service_account.Credentials.from_service_account_file(
    "styfer.json")

timestr = time.strftime("%Y%m%d-%H%M%S")


def upload_result_image(filename):
    saved_image_path = filename
    gcs = storage.Client(credentials=credentials)
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(saved_image_path)
    blob.upload_from_filename(saved_image_path)
    blob.make_public()
    return blob.public_url
