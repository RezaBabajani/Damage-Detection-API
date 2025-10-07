import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import tornado.httpclient
import socket
import asyncio
import threading
import ssl  # Import the ssl module
import json
import requests
from datetime import datetime
import jwt
import time
from dotenv import dotenv_values
from uuid import uuid4
import os
import shutil
import redis
# from prometheus_redis_client import REGISTRY

# REGISTRY.set_redis(redis.from_url("redis://redis:6379"))

import os
# from prometheus_redis_client import Counter, CommonGauge

# REQUEST_OPENED_COUNTER = Counter("n_websocket_opened_session", "Number of websocket sessions opened",
#                                  labelnames=["source"])
# REQUEST_CLOSED_COUNTER = Counter("n_websocket_closed_session", "Number of websocket sessions closed",
#                                  labelnames=["source"])
# FRAMES_RECEIVED_COUNTER = Counter("n_websocket_frames_received", "Number of websocket frames received",
#                                   labelnames=["source"])
# FRAMES_SENT_COUNTER = Counter("n_websocket_frames_sent", "Number of websocket frames sent", labelnames=["source"])
# FRAMES_PER_SECOND_GAUGE = CommonGauge("websocket_frames_per_second", "Number of websocket frames per second",
#                                       labelnames=["uuid", "type"])
# LATENCY_GAUGE = CommonGauge("websocket_latency", "Latency of websocket frames", labelnames=["uuid", "type"])

# Initialize Redis connection
# redis_conn = redis.from_url("redis://redis:6379")
# if redis_conn.get("opened_wss_counter") is None:
#     redis_conn.set("opened_wss_counter", 0)

# Load environment variables
config = dotenv_values(".env")

# Parse the port number from the passed arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7000)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

# Create an SSL context
# ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
# ssl_ctx.load_cert_chain(
#     os.path.join(config["CERT_PATH"], 'fullchain.pem'),
#     os.path.join(config["CERT_PATH"], 'privkey.pem')
# )
# Load from .env
JWT_SECRET_CODE = config["JWT_SECRET_CODE"]
JWT_AUTH = json.loads(config["JWT_AUTH"])

test_endpoint = '/openapi.json'
inference_endpoint = '/damages'

# List of fastapi servers
fastapi_servers = json.loads(config["FASTAPI_SERVERS"])


class SingleItemQueue:
    def __init__(self):
        self.item = None
        self.lock = threading.Lock()
        self.opened = False

    def put(self, item):
        """Add or overwrite an item in the queue, using a lock for thread safety."""
        with self.lock:
            # Check if there is already a more recent frame in the queue
            if self.item is not None and item["frame_number"] < self.item["frame_number"]:
                return
            self.item = item

    def get(self):
        """Get the item without removing it, using a lock for thread safety."""
        with self.lock:
            return self.item


class WSHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        print("Init websocket")
        super().__init__(*args, **kwargs)
        self.detection_results = {}
        self.start_time = {'real': {}, 'double': {}}
        self.source = "Unspecified"
        self.opened = False
        self.uuid = None
        self.last_frame_number = 0
        self.user_info = None

        # Create a queue of processed
        self.processed_frame = SingleItemQueue()
        self.connection_closing = False
        self.connection_closing_lock = threading.Lock()

        self.http_client = tornado.httpclient.AsyncHTTPClient()
        self.url_inference = None

        self.opened_wss_counter = 0

    def safe_close(self):
        """Close the connection safely, using a lock for thread safety."""
        with self.connection_closing_lock:
            if not self.connection_closing:
                if self.opened:
                    self.start_time['real'].pop(self.uuid)
                    self.start_time['double'].pop(self.uuid)
                    # REQUEST_CLOSED_COUNTER.labels(source=self.source).inc()
                    # redis_conn.set("print_status", "--------- Safe close function called ---------")
                    # self.opened_wss_counter = int(redis_conn.get("opened_wss_counter"))
                    self.opened_wss_counter -= 1
                    self.opened_wss_counter = max(0, self.opened_wss_counter)
                    # redis_conn.set("opened_wss_counter", self.opened_wss_counter)

                self.connection_closing = True
                self.close()

    @tornado.gen.coroutine
    def open(self):
        token = self.get_argument("token", default=None)
        self.source = self.get_argument("source", default='Unspecified')

        uuid = self.get_argument("id", default=None)
        self.uuid = uuid

        self.start_time['real'][uuid] = [time.perf_counter()]
        self.start_time['double'][uuid] = [time.perf_counter()]

        msg = "New connection established"
        #  If JWT authentication is enabled, check if the token is valid
        if JWT_AUTH:
            if token is None:
                print("Connection failed due to missing UUID or token")
                yield self.write_message({
                    'type': 'connection_failed',
                    'message': 'Connection failed due to missing UUID or token'
                })
                self.safe_close()
                return  # Important to stop further execution after closing the connection

            # Check if the token is valid
            try:
                self.user_info = jwt.decode(token, JWT_SECRET_CODE, algorithms=['HS256'])
            except Exception as e:
                print(f"Error in decoding token: {e}")
                yield self.write_message({
                    'type': 'connection_failed',
                    'message': 'Connection failed due to invalid token'
                })

                self.safe_close()
                return  # Again, stop further execution
        else:  # If JWT authentication is disabled

            msg = "New connection established with no authentication"

        # Choose a server to use for inference
        any_server = yield self.select_server()
        if not any_server:
            msg = "Connection failed due to no server available for inference"
            print(msg)
            yield self.write_message({
                'type': 'connection_failed',
                'message': msg
            })
            self.safe_close()
            return

        print(msg)
        #     HERE I WANT TO INCREASE THE COUNTER OF NUMBER OF WS OPENED
        try:
            yield self.write_message({
                'type': 'connection',
                'message': msg
            })
            self.opened = True
            # REQUEST_OPENED_COUNTER.labels(source=self.source).inc()
            # redis_conn.set("print_status", "--------- Open function called ---------")
            # self.opened_wss_counter = int(redis_conn.get("opened_wss_counter"))
            self.opened_wss_counter += 1
            # redis_conn.set("opened_wss_counter", self.opened_wss_counter)

        except Exception as e:
            print(f"Error in writing message: {e}")
            self.safe_close()
            return

    @tornado.gen.coroutine
    def select_server(self):
        """Select the server to use for inference"""
        # Send a request to each server to check if it is available
        available_servers = []
        for server in fastapi_servers:
            print(f"Checking server {server['name']}")
            # response = requests.get(server['url'] + test_endpoint, timeout=server['wait_time'])
            # if response.status_code == 200:
            # Use tornado httpclient instead of requests
            request = tornado.httpclient.HTTPRequest(server['url'] + test_endpoint, method="GET",
                                                     request_timeout=server['wait_time'])
            try:
                response = yield self.http_client.fetch(request)
                print(f"Server {server['name']} is available")
                available_servers.append(server)
            except Exception as e:
                print(f"Server {server['name']} is not available")
        # Select a server randomly based on the priority
        if len(available_servers) == 0:
            return False

        # Make a random weighted choice
        weights = [server['priority'] for server in available_servers]
        import random
        selected_server = random.choices(available_servers, weights=weights)[0]
        print(f"Selected server {selected_server['name']} with priority {selected_server['priority']}")
        self.url_inference = selected_server['url'] + inference_endpoint
        return True

    @tornado.gen.coroutine
    def async_inference_httpclient(self, frame_data):
        human_datetime = datetime.now().strftime("%H:%M:%S.%f %Y-%m-%d")
        print(f"{human_datetime} - Sending frame {frame_data['frame_number']} of {frame_data['id']} to API")

        try:
            print(f"calculating the fps for uuid {self.start_time['real']}")
            fps = len(self.start_time['real'][frame_data['id']])
            if fps <= 10:
                print("Start sending the frame to the API")
                request = tornado.httpclient.HTTPRequest(
                    self.url_inference,
                    method="POST",
                    body=json.dumps(frame_data),
                    headers={'Content-Type': 'application/json'},
                    request_timeout=3
                )

                response = yield self.http_client.fetch(request)
                print(f"Response received from the API")
                detection = json.loads(response.body)
                processed = {
                    "results": detection,
                    "frame_number": frame_data["frame_number"],
                    "uuid": frame_data["id"]
                }
                self.save_damage_data_in_database(frame_data['frame'], detection)
                self.processed_frame.put(processed)
                print(f"{human_datetime} - Received frame {frame_data['frame_number']} of {frame_data['id']} from API")
            else:
                print(f"FPS: {fps}")
                print("FPS is too high, skipping frame")
        # catch the error if the connection is closed
        except tornado.httpclient.HTTPClientError as e:
            if "closed" in str(e):
                # If the connection is closed, reinitialize the client
                print(f"{human_datetime} - Connection restarting for uuid {self.uuid}")
                self.http_client = tornado.httpclient.AsyncHTTPClient()
                # Retry the request after reinitializing the client
                yield self.async_inference_httpclient(frame_data)
            else:
                print(
                    f"{human_datetime} - Error in sending/receiving HTTP request for frame {frame_data['frame_number']} of {frame_data['id']} : {e}")
        except Exception as e:
            print(
                f"{human_datetime} - Error in sending/receiving HTTP request for frame {frame_data['frame_number']} of {frame_data['id']} : {e}")

    def blocking_inference(self, frame_data):
        """The non-async version of the inference function, using requests"""

        human_datetime = datetime.now().strftime("%H:%M:%S.%f %Y-%m-%d")
        print(f"{human_datetime} - Sending frame {frame_data['frame_number']} of {frame_data['id']} to API")

        try:
            response = requests.post(self.url_inference, json=frame_data)
            detection = response.json()
            processed = {}
            processed["results"] = detection
            processed["frame_number"] = frame_data["frame_number"]
            processed["uuid"] = frame_data["id"]
            self.processed_frame.put(processed)
            print(f"{human_datetime} - Received frame {frame_data['frame_number']} of {frame_data['id']} from API")
        except Exception as e:
            print(f"Error in sending/receiving HTTP request: {e}")

    def save_damage_data_in_database(self, input_frame, outputs):
        try:
            cursor = self.database.cursor()
            print(self.user_info)
            # insert a new user into the users table
            cursor.execute("INSERT INTO models_api.saved_damage "
                           "(x1, width, y1, height, type, confidence, user_email, added_time, photo) "
                           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                           (outputs['x1'], outputs['width'], outputs['y1'], outputs['height'], outputs['labels'],
                            outputs['confidence'], self.user_info['email'], datetime.now(), input_frame))
            self.database.commit()
            cursor.close()
        except Exception as e:
            print(f"Error in saving licence plate in database: {e}")

    @tornado.gen.coroutine
    def on_message(self, message):
        # Check if the connection is closing
        if self.connection_closing:
            return
        data = json.loads(message)
        frame = data["frame"]
        # check if confidence is in the data
        if 'confidence' in data:
            confidence = data['confidence']
        else:
            confidence = 0.25
        if 'frame_number' in data:
            frame_number = data["frame_number"]
        else:
            frame_number = 0

        if 'id' in data:
            uuid = data["id"]
        else:
            uuid = str(uuid4())
        frame_data = {
            "frame": frame,
            "confidence": confidence,
            "id": uuid,
            "frame_number": frame_number,
            "tracking_method": "Bayesian"
        }

        #  Make a nice informative print
        human_datetime = datetime.now().strftime("%H:%M:%S.%f %Y-%m-%d")
        print(f"{human_datetime} - Received frame {frame_number} from {uuid}")
        # FRAMES_RECEIVED_COUNTER.labels(source=self.source).inc()
        # # Blocking version with requests
        # thread = threading.Thread(target=self.blocking_inference, args=(frame_data,))
        # thread.start()

        # # Non-blocking version with httpclient
        # Schedule the task to run in the background
        tornado.ioloop.IOLoop.current().spawn_callback(self.async_inference_httpclient, frame_data)

        # Get the most recent result of the Queue
        result = self.processed_frame.get()
        if result is None:
            # If there is no result, wait for the inference to finish
            print(f"Waiting for inference to finish for uuid {uuid}")
            while result is None:
                result = self.processed_frame.get()
                yield asyncio.sleep(0.1)
        print(f"{human_datetime} - Returning frame {result['frame_number']} from {result['uuid']} to {uuid}: {result}")
        try:
            if result['frame_number'] != self.last_frame_number:
                self.last_frame_number = result['frame_number']
                self.start_time['real'][uuid].append(time.perf_counter())
                if self.start_time['real'][uuid][-1] - self.start_time['real'][uuid][0] >= 1:
                    fps = len(self.start_time['real'][uuid])
                    latency = 1000 / fps
                    # FRAMES_PER_SECOND_GAUGE.labels(uuid=uuid, type='effective').set(fps)
                    # LATENCY_GAUGE.labels(uuid=uuid, type='effective').set(latency)
                    self.start_time['real'][uuid] = [time.perf_counter()]
            self.start_time['double'][uuid].append(time.perf_counter())
            if self.start_time['double'][uuid][-1] - self.start_time['double'][uuid][0] >= 1:
                fps = len(self.start_time['double'][uuid])
                latency = 1000 / fps
                # FRAMES_PER_SECOND_GAUGE.labels(uuid=uuid, type='double').set(fps)
                # LATENCY_GAUGE.labels(uuid=uuid, type='double').set(latency)
                self.start_time['double'][uuid] = [time.perf_counter()]
            # FRAMES_SENT_COUNTER.labels(source=self.source).inc()
            yield self.write_message({
                "type": "frame",
                "message": "Frame received",
                "results": result["results"],
                'processed': True,
                "frame_number": result["frame_number"],
            })
        except Exception as e:
            print(f"Error in writing message: {e}")
            self.safe_close()
            return

    # def on_close(self):
    #     global opened_wss_counter
    #     # I don't know if this is necessary, as it is already done in on_connection_close
    #     self.connection_closing = True
    #     self.http_client.close()
    #     human_datetime = datetime.now().strftime("%H:%M:%S.%f %Y-%m-%d")
    #     print(f"{human_datetime} - Connection closed (on_close)")
    #     if self.opened:
    #         self.start_time['real'].pop(self.uuid)
    #         self.start_time['double'].pop(self.uuid)
    #         REQUEST_CLOSED_COUNTER.labels(source=self.source).inc()
    #         redis_conn.set("print_status", "--------- On close function called ---------")
    #         opened_wss_counter -= 1
    #         opened_wss_counter = max(0, opened_wss_counter)
    #         redis_conn.set("opened_wss_counter", opened_wss_counter)
    #     self.close()

    def check_origin(self, origin):
        return True

    # Deal with abruptly closed connections
    def on_connection_close(self):
        # print(REGISTRY.output())
        self.connection_closing = True
        self.http_client.close()
        human_datetime = datetime.now().strftime("%H:%M:%S.%f %Y-%m-%d")
        print(f"{human_datetime} - Connection closed (on_connection_close) for uuid {self.uuid}")
        if self.opened:
            self.start_time['real'].pop(self.uuid)
            self.start_time['double'].pop(self.uuid)
            # REQUEST_CLOSED_COUNTER.labels(source=self.source).inc()
            # redis_conn.set("print_status", "--------- On connection close function called ---------")
            # self.opened_wss_counter = int(redis_conn.get("opened_wss_counter"))
            self.opened_wss_counter -= 1
            self.opened_wss_counter = max(0, self.opened_wss_counter)
            # redis_conn.set("opened_wss_counter", self.opened_wss_counter)
        self.close()
        # Decrease the number of ws opened


# Publish metrics = REGISTRY.output() on path /metrics
#  using tornado


class MetricsHandler(tornado.web.RequestHandler):

    def get(self):
        print("Metrics requested")
        # self.write(REGISTRY.output())


def start_metrics_server():
    metrics_app = tornado.web.Application([(r'/metrics', MetricsHandler)])
    metrics_app.listen(8000)
    print("Metrics server running on port 8000")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    application = tornado.web.Application([
        (r'/websocket', WSHandler),
        (r'/metrics', MetricsHandler),
    ])

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.bind(args.port)
    http_server.start(args.workers)  # Forks multiple sub-processes
    myIP = socket.gethostbyname(socket.gethostname())
    print(f'*** Websocket Server Started at %s:{args.port} workers={args.workers}***' % myIP)
    tornado.ioloop.IOLoop.instance().start()