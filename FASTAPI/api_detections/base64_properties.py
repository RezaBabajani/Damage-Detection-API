import io
from PIL import Image
import base64
import cv2
import numpy as np


def readb64(base64_string):
    """ bas64 decoded: convert sended frame from HTMl (type string + base64) to numpy array """

    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]
    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def bas64_encode(img):
    imgencode = cv2.imencode('.jpeg', img)[1]
    imgencode = imgencode.tobytes()
    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData
    return stringData