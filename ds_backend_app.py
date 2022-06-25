import requests
from flask import Flask, request, Response
import numpy as np
import pandas as pd
from flask_cors import CORS
from requests import Request, Session
import os

import json

from ds_pipeline import process_video

app = Flask(__name__)
cors = CORS(app)



@app.route(('/analyze'))
def file_content():
    filename = request.args.get('input_video_path', type=str)
    return Response(process_video(filename))


@app.route(('/kill_flask'))
def kill_flask():
    raise ValueError('Server was killed')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
