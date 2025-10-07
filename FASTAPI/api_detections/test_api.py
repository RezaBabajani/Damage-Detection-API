import json
import requests
import argparse

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--token", type=str, default=None)
parser.add_argument("--url", type=str, default='http://127.0.0.0')
parser.add_argument("--port", type=str, default='8080')
parser.add_argument("--image", type=str, default=None)

args = parser.parse_args()

if args.token is None:
    raise ValueError('Token is required')

url = f"{args.url}:{args.port}/plate_detection"

if args.image is None:
    raise ValueError('Image path is required')

with open(args.image, "r") as file:
    base64_img = file.read()

frame = {
    "frame": base64_img,
    "token": args.token
}

response = requests.post(url, json=frame)
# Print the response
print(response.json())

