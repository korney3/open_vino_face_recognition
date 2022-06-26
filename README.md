# Data Science pipeline

## Install
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
pip install timm
```

## Description

### Запуск

В репозитории находится код для распознавания лиц на видео, идентификации и классификации эмоций.

Для запуска обработки видео вызовите функцию из `ds_pipeline.py`

```
process_video(input_video_path)
```

### Описание моделей

### Детекция лиц

1. [Face recognition](https://docs.openvino.ai/2021.3/omz_models_model_facenet_20180408_102900.html) - MobileNetV2 SSD, pretrained
2. [Face identification](https://docs.openvino.ai/2019_R1/_face_reidentification_retail_0095_description_face_reidentification_retail_0095.html) - MobileNetV2, pretrained

### Распознавание эмоций
1. 3xVGG blocks. Trained on FER2013, Fine-tune on custom teenage dataset
2. ConvNext. Pretrained on ImageNet1k, Fine-tune FER2013