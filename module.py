from dotenv import dotenv_values
from google.oauth2 import service_account
from google.cloud import storage
import time
from torch import autocast
from diffusers import StableDiffusionPipeline

config = dotenv_values(".env")

CLOUD_STORAGE_BUCKET = config["CLOUD_STORAGE_BUCKET"]
credentials = service_account.Credentials.from_service_account_file(
    "styfer.json")

time_stamp = time.strftime("%Y%m%d-%H%M%S")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
).to("cuda")


def run_inference(prompt):
    with autocast("cuda"):
        image = pipe(prompt).images[0]
    image_path = "static/" + time_stamp+"_"+prompt.replace(" ", "_")+".png"
    image.save(image_path)
    return image_path


def upload_result_image(filename):
    saved_image_path = filename
    gcs = storage.Client(credentials=credentials)
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(saved_image_path)
    blob.upload_from_filename(saved_image_path)
    blob.make_public()
    return blob.public_url
