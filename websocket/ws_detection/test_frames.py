import json
from dotenv import load_dotenv
import os
import jwt
import uuid
import time
import websockets
import asyncio
import argparse

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--token", type=str, default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3Mjk2OTAxNTIuNjU3ODQ4NCwiU0VDUkVUX0tFWSI6Ijg3ZDdhODlhZTVhOGIzYzg5MzdlNmJiNzRhNDc3MjIwNjUwMTIxZWJlMjI1YmYzYjU5NjMzMGVjNjdkMGI4ZDgiLCJ1dWlkIjoiMzgyMmEzZWEtYTUwYS00YTQ1LWI1NmUtOGNiZDI1YmU0NzNhIiwiZW1haWwiOiJyLmJhYmFqYW5pQG9taWEuZnIifQ.hXPd4l7J8jo7koruoERPWjHTCeJG4VjPA5wZbxssigA")
parser.add_argument("--url", type=str, default='ws://127.0.0.1')
parser.add_argument("--port", type=str, default='7000')
parser.add_argument("--image", type=str, default='image.txt')

args = parser.parse_args()

if args.token is None:
    raise ValueError('Token is required')

user_id = str(uuid.uuid4())
ws_uri = f"{args.url}:{args.port}/websocket?token={args.token}&id={user_id}"

if args.image is None:
    raise ValueError('Image path is required')

with open(args.image, "r") as file:
    base64_img = file.read()

load_dotenv()

SECRET_KEY = str(os.getenv('JWT_SECRET_CODE'))
# Expire the token for one month
expire_time = time.time() + 60 * 60 * 24 * 30


async def send_frame():
    with open("image.txt", "r") as file:
        base64_img = file.read()
    async with websockets.connect(ws_uri) as websocket:
        frame = {
            "frame": base64_img,
            "confidence": 0.25,
            "id": user_id,
        }
        await websocket.send(json.dumps(frame))
        response = await websocket.recv()
        print(response)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(send_frame())
