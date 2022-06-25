import pytest
from flask import Flask
from ds_backend_app import app

# @pytest.fixture()
# def app():
#     app = Flask(__name__)
#     app.config.update({
#         "TESTING": True,
#     })
#
#     # other setup can go here
#
#     yield app
#
#     # clean up / reset resources here


# @pytest.fixture()
# def client(app):
#     return app.test_client()
#
#
# @pytest.fixture()
# def runner(app):
#     return app.test_cli_runner()


def test_request_example():
    response = app.test_client().get("/analyze", query_string={"input_video_path": "./demo_video/class_fussy.mp4"})
    print(response)
    assert b"<h2>Hello, World!</h2>" in response.data

