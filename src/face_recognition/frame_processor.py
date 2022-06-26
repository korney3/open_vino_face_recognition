import logging as log
import sys
from pathlib import Path

from openvino.runtime import Core, get_version

from src.emotions_classification.emotion_classifier import EmotionClassifier, EmotionClassifierConvNext

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))


from open_model_zoo.demos.face_recognition_demo.python.utils import crop

from open_model_zoo.demos.face_recognition_demo.python.landmarks_detector import LandmarksDetector
from open_model_zoo.demos.face_recognition_demo.python.face_detector import FaceDetector
from open_model_zoo.demos.face_recognition_demo.python.faces_database import FacesDatabase
from open_model_zoo.demos.face_recognition_demo.python.face_identifier import FaceIdentifier

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)



class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow #and not args.no_show
        self.current_id = 0

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)
        self.emotion_classifier = EmotionClassifierConvNext()#EmotionClassifier()

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))

        emotions = self.emotion_classifier.process(frame, rois)

        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        # landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois))#, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                # name = self.faces_database.ask_to_save(crop_image)
                name = f"Pupil_id_{self.current_id}"
                self.current_id+=1
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        return [rois, face_identities, emotions] #landmarks, face_identities]

