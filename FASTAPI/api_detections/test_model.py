import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Add parent directory to Python path
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from dotenv import load_dotenv
import jwt
from database_management.config import connect
from nn_models.plate_detector_yolonas import TextDetectorYoloNAS
from base64_properties import readb64, bas64_encode
from aiocache import Cache
from aiocache.backends.redis import RedisCache
from datetime import datetime, timedelta
import time
import redis
from prometheus_redis_client import REGISTRY

load_dotenv()
# REGISTRY.set_redis(redis.from_url("redis://localhost:6379"))
# redis_conn = redis.from_url("redis://localhost:6379")

# plate detection models
# car_plate_detector = TextDetector()
car_plate_detector = TextDetectorYoloNAS()
conn = connect()


def plate_detection(frame, token):
    start_time = time.time()

    try:
        jwt.decode(token, str(os.getenv('SECRET_KEY')), algorithms=['HS256'])
        frame = str(frame)
        input_frame = readb64(frame)
        if not car_plate_detector.gpu_busy:
            output_frame, plate_photo, text = car_plate_detector.apply_model(input_frame)
            output_frame_encoded = bas64_encode(output_frame)
            time_difference = (time.time() - start_time) * 1000
            print(f"Time taken: {time_difference} ms")

        else:
            return print('GPU is busy')
    except Exception as e:
        print('Token is not valid')


if __name__ == "__main__":
    # read image.txt file
    with open("image.txt", "r") as file:
        base64_img = file.read()
    token = "39135b18-9f54-4284-870c-5c560573e050"
    plate_detection(base64_img, token)
