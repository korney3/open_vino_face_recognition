import requests
from flask import Flask, request, Response, jsonify
import numpy as np
import pandas as pd
from flask_cors import CORS
from requests import Request, Session
import os

import json

from ds_pipeline import process_video

app = Flask(__name__)
cors = CORS(app)


@app.route('/class-analyze', methods=['POST'])
def file_content():
    root_dir = "/home/alex/ezee-emotions-analyzer-backend/"
    data = request.json

    filename = data['filename']#request.args.get('filename', type=str)
    absolute_filename = os.path.join(root_dir, filename)
    results = process_video(absolute_filename)

    return results


@app.route(('/kill_flask'))
def kill_flask():
    raise ValueError('Server was killed')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
