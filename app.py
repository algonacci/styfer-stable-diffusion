import os
from flask import Flask, request, send_file, jsonify
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
        image = pipe(prompt).images[0]
    image.save("test.png")
    return image


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


@app.route("/generate", methods=["POST"])
def generate():
    input_request = request.get_json()
    prompt = input_request["prompt"]
    if request.method == "POST":
        if prompt == "":
            json = {
                "data": [],
                "message": "Prompt cannot be empty",
                "status_code": 400
            }
            return jsonify(json)
        else:
            result_image = run_inference(prompt=prompt)
            json = {
                "message": "Success generated image based on the prompt",
                "status_code": 200
            }
    else:
        json = {
            "data": "",
            "message": "Method not allowed",
            "status_code": 405
        }
        return jsonify(json)


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
