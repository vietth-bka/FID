"""main app"""
import logging
import re
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from the_class import Laymau

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

source = 'rtsp://admin:123456a%40@172.21.111.111'
# source = 'rtsp://admin:123456a%40@172.21.104.112'

if(re.search("GStreamer:\s*NO", cv2.getBuildInformation())):
    logger.info("cv2 is built without gstreamer")
    cap = cv2.VideoCapture(source)
else:
    pipeline = "rtspsrc location={} latency=100 ! queue ! rtph265depay" \
            " ! h265parse ! avdec_h265 ! videoscale ! video/x-raw,width=1920,height=1080" \
            " ! videoconvert ! video/x-raw,format=BGR" \
            " ! appsink emit-signals=true sync=false async=false drop=true".format(source)

    logger.info("playing pipeline: %s", pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

def show_and_wait(a_frame: np.ndarray) -> bool:
    """just show frame, return True if wanted to quit"""
    # hd_frame = cv2.resize(a_frame, (1920, 1080))
    cv2.imshow("frame", a_frame)
    if 27 == cv2.waitKey(3): # esc
        sys.exit(0)

def select_roi(_cap: cv2.VideoCapture) -> np.ndarray: 
    """
    select interested region using the mouse

    Args:
        _cap (cv2.VideoCapture): the cap whose frames will be use to choose the roi

    Returns:
        np.ndarray: roi in (x, y, w, h) format
    """
    
    first_ret, first_frame = _cap.read()
    if first_frame is not None:
        first_frame = cv2.resize(first_frame, (1920, 1080))
    if not first_ret:
        logger.error("video has no more frames")
        # sys.exit(0)
        return None
        
    roi = cv2.selectROI("select region of interest", first_frame)    
    print(roi)
    cv2.destroyWindow('select region of interest')
    if roi is None:
        logger.error("no region of interest")
        sys.exit(0)
    return roi

file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "save_image"))
min_samples = 9
sampler = Laymau(file_name, min_samples)

while True:
    sampler.reset()
    cv2.destroyAllWindows()
    num_of_person = int(input("\nEnter number of sample person, or enter 0 to exit: "))
    if num_of_person <= 0:
        break    
    roi = select_roi(cap)
    if roi is None:
        continue
    done = False
    pbar = tqdm(total=min_samples * num_of_person, desc="collecting face")
    while cap.isOpened() and not done:        
        a_ret, a_frame = cap.read()
        a_frame = cv2.resize(a_frame, (1920, 1080))
        if not a_ret:
            logger.error("video has no more frames")
            break

        roi_frame = a_frame[
            int(roi[1]) : int(roi[1] + roi[3]),
            int(roi[0]) : int(roi[0] + roi[2]),
        ]
        done, num_accumulated, bboxes = sampler.process_one_frame(roi_frame, num_of_person)
        pbar.update(len(bboxes))
        # draw
        cv2.rectangle(a_frame, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 0, 255), 1,)
        for bbox in bboxes:
            bbox_int = [int(b) for b in bbox]
            bbox_int[0] += roi[0]
            bbox_int[1] += roi[1]
            bbox_int[2] += roi[0]
            bbox_int[3] += roi[1]
            cv2.rectangle(a_frame, tuple(bbox_int[0:2]), tuple(bbox_int[2:4]), (0, 255, 0), 1)

        show_and_wait(a_frame)
