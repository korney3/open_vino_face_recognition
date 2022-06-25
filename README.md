# Install
```
conda create --name open_vino_face_recognition python=3.9
conda activate open_vino_face_recognition
pip install openvino
git clone https://github.com/openvinotoolkit/open_model_zoo.git
cd open_model_zoo
git submodule update --init --recursive
cd ..
pip install openvino-dev
cd open_model_zoo/demos/face_recognition_demo/python/
omz_downloader --list models.lst
cd ../../../..
pip install open_model_zoo/demos/common/python
pip install tensorflow
pip install flask
pip install requests
pip install pytest
pip install Flask-Cors
```