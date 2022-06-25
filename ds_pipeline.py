import json
import logging as log
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import requests

from src.face_recognition.frame_processor import FrameProcessor
from src.utils import center_crop, draw_detections, save_detections_to_json

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

common_module_path = str(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'open_model_zoo/demos/common/python'))
sys.path.append(common_module_path)

from open_model_zoo.demos.face_recognition_demo.python.face_identifier import FaceIdentifier

import monitors
from helpers import resolution
from images_capture import open_images_capture

from openvino.model_zoo.model_api.models import OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'MYRIAD', 'HETERO', 'HDDL']


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', required=False,
                         help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    general.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    general.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    general.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    general.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    general.add_argument('--no_show', default=True, action='store_true',
                         help="Optional. Don't show output.")
    general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
                         help='Optional. Crop the input stream to this resolution.')
    general.add_argument('--match_algo', default='HUNGARIAN', choices=('HUNGARIAN', 'MIN_DIST'),
                         help='Optional. Algorithm for face matching. Default: HUNGARIAN.')
    general.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', default='', help='Optional. Path to the face images directory.')
    gallery.add_argument('--run_detector', action='store_true',
                         help='Optional. Use Face Detection model to find faces '
                              'on the face images, otherwise use full images.')
    gallery.add_argument('--allow_grow', default=True, action='store_true',
                         help='Optional. Allow to grow faces gallery and to dump on disk. '
                              'Available only if --no_show option is off.')

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', type=Path, required=False,
                        help='Required. Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', type=Path, required=False,
                        help='Required. Path to an .xml file with Facial Landmarks Detection model.')
    models.add_argument('-m_reid', type=Path, required=False,
                        help='Required. Path to an .xml file with Face Reidentification model.')
    models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection model for '
                             'reshaping. Example: 500 700.')

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Detection model. '
                            'Default value is CPU.')
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Facial Landmarks Detection '
                            'model. Default value is CPU.')
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Reidentification '
                            'model. Default value is CPU.')
    infer.add_argument('-v', '--verbose', action='store_true',
                       help='Optional. Be more verbose.')
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help='Optional. Probability threshold for face detections.')
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help='Optional. Cosine distance threshold between two vectors '
                            'for face identification.')
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for bboxes passed to face recognition.')
    return parser


def process_video(input_video_filename: str):
    script_path = os.path.dirname(os.path.realpath(__file__))
    time_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    result_path = os.path.join(script_path, "logs", f"{time_str}_log")
    jsons_path = os.path.join(result_path, "frames_processed")
    os.makedirs(jsons_path, exist_ok=True)

    models_path = os.path.join(script_path, "open_model_zoo/demos/face_recognition_demo/python")

    face_detector_model_path = os.path.join(models_path,
                                            "./intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml")
    landmarks_model_path = os.path.join(models_path,
                                        "./intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml")
    reidentification_model_path = os.path.join(models_path,
                                               "./intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml")

    args = build_argparser().parse_args([])

    args.fg = os.path.join(result_path, "photos_db")
    os.makedirs(args.fg, exist_ok=True)
    args.input = input_video_filename
    args.m_fd = face_detector_model_path
    args.m_lm = landmarks_model_path
    args.m_reid = reidentification_model_path
    args.output = os.path.join(script_path, f"./demo_video/{time_str}.webm")
    cap = open_images_capture(input_video_filename, args.loop)
    fps = cap.fps()
    frame_processor = FrameProcessor(args)

    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None
    if args.crop_size[0] > 0 and args.crop_size[1] > 0:
        input_crop = np.array(args.crop_size)
    elif not (args.crop_size[0] == 0 and args.crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')
    video_writer = cv2.VideoWriter()

    while True:
        start_time = perf_counter()
        frame = cap.read()

        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break

        if frame_num % 24 == 0 and frame_num != 0:
            frame_num += 1
            continue

        if input_crop:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'vp80'),
                                                     cap.fps(), output_resolution):
                raise RuntimeError("Can't open video writer")

        detections = frame_processor.process(frame)
        results = save_detections_to_json(detections, frame_num, log_dir=jsons_path, db_path=args.fg)

        r = requests.post('http://127.0.0.1:3000/api/result-class', json=results)
        print(r)
        # yield results
        presenter.drawGraphs(frame)
        frame = draw_detections(frame, frame_processor, detections, output_transform,
                                unknown_id=FaceIdentifier.UNKNOWN_ID)
        metrics.update(start_time, frame)

        frame_num += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frame_num <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Face recognition demo', frame)
            key = cv2.waitKey(1)
            # Quit
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

    response = {
        'status': 'finished',
        'output_src': args.output
    }
    r = requests.post('http://127.0.0.1:3000/api/result-class', json=response)
    # yield json.dumps(response, ensure_ascii=False)
    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)
    return args.output


script_path = os.path.dirname(os.path.realpath(__file__))

# process_video("./demo_video/class_front_view.mp4")
