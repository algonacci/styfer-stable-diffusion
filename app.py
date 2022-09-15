from flask import Flask, request, send_file
import io
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


app = Flask(__name__)
assert torch.cuda.is_available()

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
).to("cuda")


def run_inference(prompt):
    with autocast("cuda"):
        image = pipe(prompt)["sample"][0]
    img_data = io.BytesIO()
    image.save(img_data, "PNG")
    img_data.seek(0)
    return img_data


@app.route("/")
def index():
    return {
        "status_code": 200,
        "message": "Success!"
    }


@app.route('/prompt')
def prompt():
    if "prompt" not in request.args:
        return "Please specify a prompt parameter", 400
    prompt = request.args["prompt"]
    img_data = run_inference(prompt)
    return send_file(img_data, mimetype='image/png')


@app.errorhandler(400)
def bad_request(error):
    return {
        "status_code": 400,
        "error": error,
        "message": "Client side error!"
    }, 400


@app.errorhandler(404)
def not_found(error):
    return {
        "status_code": 404,
        "message": "URL not found"
    }, 404


@app.errorhandler(405)
def method_not_allowed(error):
    return {
        "status_code": 405,
        "message": "Request method not allowed!"
    }, 405


@app.errorhandler(500)
def internal_server_error(error):
    return {
        "status_code": 500,
        "message": "Server error"
    }, 500


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
