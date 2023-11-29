"""
when a face is extracted, all the exceed task will be process here
why? cause this class will be another thread
"""

from threading import Thread
# import insightface
import numpy as np
from .blury import is_blur
from .get_emb.get_embedding import CustomEmbedding

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'WHENET_TRT'))
from headpose_trt import Headposetrt


class FaceHandlerThread(Thread):
    thread_count = -1
    def __init__(self, queue, queue_cluster, dbhandler, pbar_queue=None):
        """
        :param queue: queue to get object from to run
        each queue item is a dict with keys: (frameid, trackid, data, bbox, landmark, confident)
        :param queue_cluster: queue for clustering. This thread should put objects in queue
        :param dbhandler: DbHandler object
        :param pbar_queue: optional queue for progress check
        """
        super().__init__()
        self.queue = queue
        self.dbhandler = dbhandler
        self.queue_cluster = queue_cluster

        # em_model = insightface.model_zoo.get_model('arcface_r100_v1')
        # em_model.prepare(ctx_id=0)
        em_model = CustomEmbedding()
        self.em_model = em_model
        self.pbar_queue = pbar_queue

        self.headposer = Headposetrt(serialized_plan=os.path.join(os.path.dirname(__file__), 'WHENET_TRT/saved_model.plan'), 
                binding_names=os.path.join(os.path.dirname(__file__), 'WHENET_TRT/saved_model_bindings.txt'))
        FaceHandlerThread.thread_count = FaceHandlerThread.thread_count + 1
        self.thread_count = FaceHandlerThread.thread_count
    def run(self):
        """do all intensive tasks for each face, including:
        - extracting embedding
        - blury calculation
        - head pose calculation
        """
        while True:
            item = self.queue.get()
            # print('\nitem get at facehandler #{}'.format(self.thread_count), flush=True)
            # santinel check for return
            if item == 0:
                print('FaceHandlerThread quit')
                self.queue_cluster.put(0)
                self.queue.task_done()
                break
            face_image = item['data']

            # TODO: blury handling
            isBlur, blurry = is_blur(face_image)
            if isBlur:
                self.queue.task_done()
                continue

            # raw_emb = self.em_model.get_embedding(face_image)
            # norm = np.linalg.norm(raw_emb) 
            # emb = raw_emb / norm
            emb, norm = self.em_model.get_embedding(face_image)

            faceinfo = item
            faceinfo['embedding'] = emb
            faceinfo['embedding_norm'] = float(norm)

            yaw, pitch, roll = self.headposer.get_pose(face_image)
            faceinfo['pose_yaw'] = yaw
            faceinfo['pose_pitch'] = pitch
            faceinfo['pose_roll'] = roll
            # faceinfo = {
            #     'data': item['data'],
            #     'bbox': item['bbox'],
            #     'landmark': item['landmark'],
            #     'confident': item['confident'],
            #     'frameid': item['frameid'],
            #     'trackid': item['trackid'],
            #     'embedding': emb.tobytes(),
            #     'embedding_norm': float(norm)
            # }
            self.dbhandler.add_face_v2(faceinfo, save_remote=True)
            if self.pbar_queue is not None:
                self.pbar_queue.put(1)
            self.queue_cluster.put(faceinfo)
            self.queue.task_done()

    


