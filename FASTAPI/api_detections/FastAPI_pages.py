import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Add parent directory to Python path
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from dotenv import load_dotenv
import jwt
# from database_management.config import connect
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# from nn_models.plate_detector import TextDetector
from nn_models.damage_detector import YoloDetector
from base64_properties import readb64, bas64_encode
import time
from aiocache import Cache
from aiocache.backends.redis import RedisCache
from bayesian_tracker.bayesian_filter import BayesianFilter
from datetime import datetime, timedelta

app = FastAPI()

load_dotenv()
cache = Cache(RedisCache, endpoint="redis", port=6379, namespace="main")

# plate detection models
# car_plate_detector = TextDetector()
car_damage_detector = YoloDetector()
bayesian_filter = BayesianFilter()
# conn = connect()

damage_color = {"Scratch": "#f05d02",
                "Dent": "#005FBD",
                "Missing part": "#A0A411",
                "Deformed": "#88896B",
                "Backlight": "#8B0534",
                "Frontlight": "#59444C",
                "Crack": "#590676",
                "Misplaced": "#515231",
                "Varnish": "#D16901",
                "Rust": "#50755C",
                "Glass": "#000000",
                }

class_names = ['Scratch', 'Backlight', 'Deformed', 'Frontlight', 'Crack', 'Misplaced', 'Dent', 'Varnish', 'Rust',
               'Missing part', 'Glass']


class Meta:
    def __init__(self):
        pass

    async def get_state_bayesianfilter(self, uuid):
        return await cache.get(uuid)

    async def save_state_bayesianfilter(self, uuid, state):
        # Set the bayesian filter into the cache and keep it for 2 minutes and if it is not used, delete it
        await cache.set(uuid, state, ttl=120)  # await cache.set(uuid, pickle.dumps([0,1,2,3]), ttl=120)


meta = Meta()


@app.api_route("/damages", methods=["GET", "POST"])
async def damage_detection(request: Request):
    damage_data = await request.json()
    frame = damage_data["frame"]
    #  Confidence used by the model
    model_confidence = damage_data["confidence"]
    # Confidence applied for extracting boxes in Bayesian filtering
    box_confidence = 0
    uuid = damage_data["id"]

    try:
        token = damage_data["token"]
        jwt.decode(token, str(os.getenv('SECRET_KEY')), algorithms=['HS256'])
    except Exception as e:
        return JSONResponse({'error': 'Token is not valid'})

    # Try to fetch the frame number, and refuse to process a new frame if a more recent has already been treated.
    # This is done in order to optimize executio, but above all the main purpose is to use the correct sequence for the bayesian tracker
    frame_number = damage_data.get("frame_number", 0)

    last_frame_number = await cache.get(f"{uuid}_frame_number")
    if last_frame_number is None:
        last_frame_number = 0
    if frame_number < last_frame_number:
        msg = f"A more recent frame has already been processed for {uuid} ({last_frame_number} > {frame_number})"
        print(msg)
        return JSONResponse({
            "msg": msg,
            'processed': False
        })

    input_frame = readb64(frame)

    human_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
    print(f"Frame received from {uuid} at {human_datetime}")

    # Apply raw model
    results = car_damage_detector.apply_model(input_frame, model_confidence)
    # Convert to dict
    # results_as_dict = [{
    #     "x": int(res[0]),
    #     "y": int(res[1]),
    #     "width": int(res[2] - res[0]),
    #     "height": int(res[3] - res[1]),
    #     "confidence": float(res[4]),
    #     "labels": class_names[int(res[5])]
    # } for res in results]
    results_as_dict = [{
        "x": res[0][0],
        "y": res[0][1],
        "width": res[0][2],
        "height": res[0][3],
        "confidence": res[1],
        "labels": res[2]} for res in results]

    if damage_data['tracking_method'] == 'Bayesian':

        # Load cached masks for Bayesian filtering, if they exist
        state = await meta.get_state_bayesianfilter(uuid)
        if state is None:
            bayesian_filter.reset_state()
            print("Reset filter")
        else:
            print("Load filter cache")
            bayesian_filter.load_state(state)

        # Apply threshold defined by the request
        bayesian_filter.threshold = 0
        bayesian_filter.box_threshold = box_confidence

        # Apply filtering
        bayesian_filter.update_frame(input_frame)
        bayesian_filter.update_predictions(results_as_dict)
        info_damage = bayesian_filter.get_boxes()

        # Store masks in cache
        new_state = bayesian_filter.export_state()
        await cache.set(uuid, new_state, ttl=120)

    # Some formatting
    for i in range(len(info_damage)):
        info_damage[i]["color"] = damage_color[info_damage[i]['labels']]
        info_damage[i]["confidence"] = float(info_damage[i]["confidence"])

    print(info_damage)
    await cache.set("{uuid}_frame_number", frame_number, ttl=120)
    return JSONResponse({
        "defauts": info_damage,
        'processed': True
    })
