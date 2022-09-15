from dotenv import dotenv_values
from google.oauth2 import service_account
from google.cloud import storage
import matplotlib.pyplot as plt
import time

config = dotenv_values(".env")

CLOUD_STORAGE_BUCKET = config["CLOUD_STORAGE_BUCKET"]
credentials = service_account.Credentials.from_service_account_file(
    "styfer.json")

timestr = time.strftime("%Y%m%d-%H%M%S")
