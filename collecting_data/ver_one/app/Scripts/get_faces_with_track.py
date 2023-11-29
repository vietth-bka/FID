import os
import cv2
import numpy as np
import insightface
from .sort import Sort

class MyDetection():
    def __init__(self, step=1):
        """
        :param step: 1 means detect on every frames, 2 means 
        """
        # model = insightface.model_zoo.get_model('retinaface_r50_v1')
        model = insightface.model_zoo.get_model('retinaface_mnet025_v2')
        model.prepare(ctx_id = 0, nms=0.4)
        self.model = model
        self.step = step

    def get_frames(self, cap, roi=None):
        """
        with each processed frame, yeild a dict of following keys
        - frame: a dict of keys: frame_count, data (the actual image)
        - faces: a list of dict of: trackid, data, bbox, landmark
        - roi: region of interest, x, y, w, h
        when not at step, yeild None
        :param cap: cv2.VideoCapture object
        """
        mot_tracker = Sort(max_age=3, min_hits=3)
        frame_count = 0 # postgres is 1-indexed
        while cap.isOpened():
            frame_count += 1
            if frame_count % (self.step) != 0:
                if False == cap.grab():
                    break
                yield None
                continue

            ret, frame = cap.read()
            if ret == False:
                break

            result = {}
            result['frame'] = {
                "frame_count": frame_count,
                "data": frame,
            }
            result['faces'] = []

            # if frame_count % (self.step) != 0:
            #     yield result
            #     continue
            if roi is not None:
                roi_frame = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            else:
                roi_frame = frame

            h, w, c = roi_frame.shape
            scale = 1280 / h # scale image to fullHD
            bboxes, landmarks = self.model.detect(roi_frame, threshold=0.8, scale=scale)
            trackers = mot_tracker.update(np.array(bboxes))

            confidents = [i[4] for i in bboxes]


            faces = []
            for tracker in trackers:
                bbox = tracker['box']
                landmark = landmarks[tracker['index']]
                face = insightface.utils.face_align.norm_crop(roi_frame, landmark)
                if roi is not None:
                    bbox[0] = bbox[0] + roi[0]
                    bbox[1] = bbox[1] + roi[1]
                    bbox[2] = bbox[2] + roi[0]
                    bbox[3] = bbox[3] + roi[1]
                    for i in range(5):
                        landmark[i][0] = landmark[i][0] + roi[0]
                        landmark[i][1] = landmark[i][1] + roi[1]
                trackId = tracker['trackId']
                confident = confidents[tracker['index']]
                faces.append({
                    "trackid": trackId, 
                    "data": face, 
                    "bbox": bbox, 
                    "landmark": landmark,
                    "confident": confident
                })
            result['faces'] = faces
            yield result
        return

    def show_result(self, result, roi=None):
        """show the result
        :param result is a dict of following keys
        - frame: a dict of keys: frame_count, data (the actual image)
        - faces: a list of dict of: trackid, data, bbox, landmark, confident
        - roi: region of interest
        """
        if result == None:
            return np.array([])
        frame = result['frame']['data'].copy()
        assert(frame.size != 0)
        if roi is not None:
            cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 0, 255), 2)
        for face in result['faces']:
            bbox_int = [int(b) for b in face['bbox']]
            cv2.rectangle(frame, tuple(bbox_int[0:2]), tuple(bbox_int[2:4]), (0, 255, 0), 2) 
            for i in range(5):
                cv2.circle(frame, tuple(face['landmark'][i].astype(np.int32)), 5, (0, 255, 0), -1)

            cv2.putText(frame, str(face['confident']), (bbox_int[0], bbox_int[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0))
        return cv2.resize(frame, (1280, 720))


    def get_single_face(self, frame):
        """input a single frame, output face image in the frame,
        only use when select Reference 
        :param frame: numpy frame, should include only one face. 
        If not, only the first face will be return
        :return a dict of keys (trackid, data, bbox, landmark, confident)
        return None if no face is detected
        """
        bboxes, landmarks = self.model.detect(frame, threshold=0.8, scale=1)
        if len(bboxes) == 0:
            return None
        face = insightface.utils.face_align.norm_crop(frame, landmarks[0])
        bbox = bboxes[0][:-1]
        confident = bboxes[0][-1]
        ret = {
            "data": face, 
            "bbox": bbox, 
            "landmark": landmarks[0],
            "confident": confident
        }
        return ret