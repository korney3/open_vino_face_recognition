import json
import os
import time
from datetime import datetime

import cv2


def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2: (fh + crop_size[1]) // 2,
           (fw - crop_size[0]) // 2: (fw + crop_size[0]) // 2,
           :]


def draw_detections(frame, frame_processor, detections, output_transform, unknown_id):
    labels_translation = {'Злость': 'Angry', 'Отвращение':'Disgust', 'Страх':'Fear', 'Радость':'Happy', 'Нейтральное состояние':'Neutral',
                            'Грусть':'Sad', 'Удивление':'Surprise'}
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    rois, face_identities, emotions = detections
    if len(rois) != len(face_identities):
        return
    for i in range(len(rois)):
        roi = rois[i]
        identity = face_identities[i]
        emotion = emotions[i]
        main_emotion, main_emotion_score = get_main_emotion(emotion)

        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != unknown_id:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
        text += f" {labels_translation[main_emotion]}: {round(main_emotion_score, 3)}"
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        # for point in landmarks:
        #     x = xmin + output_transform.scale(roi.size[0] * point[0])
        #     y = ymin + output_transform.scale(roi.size[1] * point[1])
        #     cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame


def get_main_emotion(emotion):
    sorted_emotion = sorted(emotion.items(), key=lambda item: item[1])
    return sorted_emotion[-1]


def save_detections_to_json(detections, frame_num, log_dir="", fps=1, db_path=""):
    time_str = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')

    rois, face_identities, emotions = detections
    if len(rois) != len(face_identities):
        return
    results = []
    for i in range(len(rois)):
        result = {}
        roi = rois[i]
        face_identity = face_identities[i]
        emotion = emotions[i]

        result['time'] = frame_num / fps

        # result["image_id"] = int(roi.image_id)
        # result["label"] = int(roi.label)
        # result["confidence"] = float(roi.confidence)
        # result["position_x"] = float(roi.position[0])
        # result["position_y"] = float(roi.position[1])
        # result["size_width"] = float(roi.size[0])
        # result["size_height"] = float(roi.size[1])

        result["id"] = int(face_identity.id)
        result["image_src"] = os.path.join(db_path, f"pupil_id_{id}-0.jpg")
        # result["distance"] = float(face_identity.distance)
        # result["descriptor"] = face_identity.descriptor.tolist()

        for key in emotion.keys():
            result[key] = float(emotion[key])

        results.append(result)
    # with open(os.path.join(log_dir, f"res_{time_str}.json"), "w") as f:
    #     json.dump(results, f)
    response = {
        'status': 'processing',
        'data': results
    }
    return json.dumps(response, ensure_ascii=False)
