import os
from flask import Flask, request, jsonify
import torch
import time
import module as md


app = Flask(__name__)
assert torch.cuda.is_available()

time_stamp = time.strftime("%Y%m%d-%H%M%S")


@app.route("/")
def index():
    return {
        "status_code": 200,
        "message": "Success!"
    }


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
            result_image = md.run_inference(prompt=prompt)
            processed = md.upload_result_image(filename=result_image)
            json = {
                "message": "Success generated image based on the prompt",
                "status_code": 200,
                "image_url": processed
            }
            return jsonify(json)
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
