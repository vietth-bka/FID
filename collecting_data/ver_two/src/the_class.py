"""the laymau class"""
import os
import re
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from detection import FaceDetector
from facefeature.facefeature import  FaceFeature
from maskpose.maskpose import MaskPose
from alignment.face_align import norm_crop
from typing import Tuple
from shutil import rmtree
from typing import List
from pydantic import BaseModel

DETECTION_THRESHOLD = 0.95

class Metadata(BaseModel):
    yaw: float
    pitch: float
    roll: float

class Laymau:
    """the laymau class"""

    static_id = 0

    def __init__(self, save_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "save_image")), min_imgs: int = 0):
        """constructor

        ### Args:
            save_dir (str, optional): _description_. Defaults to os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "save_image")).
            min_imgs: minimum number of images of each staff
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for subfolder in os.listdir(save_dir):
            if re.search("^ID_\d+$", subfolder):
                rmtree(os.path.join(save_dir, subfolder))

        self.detector = FaceDetector()
        self.maskpose = MaskPose()
        self.featurer = FaceFeature()
        self.reset()
        Laymau.static_id = 0

        self.num_person = 0
        self.min_imgs = min_imgs

    def reset(self):
        self.list_embedding, self.list_image = [], []
        self.list_pose: List[Metadata] = []

    def process_one_frame(self, roi: np.ndarray, num_person: int = 1) -> Tuple:
        """
        process one frame
        ### Args:
            roi (np.ndarray): region of interest
            num_person (int, optional): number of person to be collected. Defaults to 1.

        ### Returns:
            Tuple: type of (done, num_accumulated, display_boxes) in which
                done: True if done
                num_accumulated: number of accumulated face
                display_boxes: list of bouding box to be displayed
        """

        self.num_person = num_person
        assert isinstance(num_person, int)
        num_person = num_person
        max_num_face = self.min_imgs * num_person
        #################
        ### detection ###
        #################
        done = False
        display_faces = []
        bboxes, landmarks = self.detector.detect(roi, threshold=DETECTION_THRESHOLD, scale=1)
        if len(bboxes) == 0:
            return done, len(self.list_image), display_faces

        #################
        ### alignment ###
        #################
        for (bbox, landmark) in zip(bboxes, landmarks):
            face = norm_crop(roi, landmark)
            ### remove masked face and large yaw pose
            masked, yaw, pitch, roll = self.maskpose.get_pose(face)
            if masked == 0 and abs(yaw) <= 35 and abs(pitch) <= 25 and not self.is_blur(face)[0]:
                display_faces.append(bbox)
                #################
                ### feature ###
                #################
                feature, norm = self.featurer.get(face)
                self.list_embedding.append(feature)
                self.list_image.append(face)
                self.list_pose.append(Metadata(
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll
                ))

        if len(self.list_image) >= max_num_face  and (len(self.list_image) - max_num_face)%20 == 0:
            # occasionally check clusters
            done, clustering = self.cluster()

        #################
        ### save images ###
        #################
        if done:
            self.save_imgs(clustering)

        return done, len(self.list_image), display_faces

    def cluster(self):
        """cluster"""
        clustering = DBSCAN(eps=0.25, min_samples=5, metric='cosine').fit(self.list_embedding)
        print('cluster done for image', len(self.list_embedding))
        groups_dict = {}
        for idx, c in enumerate(clustering.labels_):
            if c >= 0:
                if c in groups_dict:
                    groups_dict[c].append(idx)
                else:
                    groups_dict[c] = [idx]

        print('Check len clusters:', [len(groups_dict[c]) for c in groups_dict])
        if len(groups_dict) > 0:
            valid_grs = [k for k in groups_dict if self.check_one_cluster(groups_dict[k])]
            if len(valid_grs) == int(self.num_person):
                print('Valid groups:', valid_grs)
                return True, clustering
            elif len(valid_grs) > int(self.num_person):
                print('Too much valid groups:', valid_grs, '> num_of_person:', int(self.num_person), \
                        ', it should be manually merging/removing the groups!')
                return True, clustering
            else:
                pass        
        return False, None
        
    
    def save_imgs(self, clustering):
        assert clustering is not None, 'clustering is None !'
        for i in range(len(self.list_embedding)):
            print(i)
            if clustering.labels_[i] >= 0:
                id_path = 'ID_' + str(clustering.labels_[i] + Laymau.static_id)
            else:
                id_path = '.noise'
            id_path = os.path.join(self.save_dir, id_path)
            if not os.path.exists(id_path):
                os.makedirs(id_path)
            cv2.imwrite(os.path.join(id_path, str(i) + '.jpg'), self.list_image[i])
            with open(os.path.join(id_path, str(i) + '.txt'), 'w') as f:
                f.write(str(self.list_embedding[i]))
            with open(os.path.join(id_path, str(i) + '.json'), 'w') as f:
                f.write(self.list_pose[i].json())

    def check_one_cluster(self, ids):
        if len(ids) < self.min_imgs: return False
        valid_poses = [self.list_pose[i] for i in ids]
        front_views, left_views, right_views = 0, 0, 0
        for vp in valid_poses:
            if abs(vp.yaw) <= 15.:
                front_views += 1
            elif vp.yaw > 15. and vp.yaw <= 35.:
                left_views += 1
            elif vp.yaw < -15. and vp.yaw >= -35.:
                right_views += 1
        
        assert front_views + left_views + right_views == len(valid_poses), 'wrong sum !'
        print('Total images:', len(valid_poses), f'front_view: {front_views}/{self.min_imgs//3}', \
                                                f'left_view:, {left_views}/{self.min_imgs//3}', \
                                                f'right_view: {right_views}/{self.min_imgs//3}')
        if front_views >= self.min_imgs//3 and left_views >= self.min_imgs//3  and right_views >= self.min_imgs//3:
            return True
        else:
            return False

    def is_blur(sefl, img):
        """Decide if the input image is blur or not
        :param img: input image
        :type img: numpy.array
        :returns True if image is blur, return blury as well
        """        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplace = cv2.Laplacian(gray_img, cv2.CV_64F)
        blury = laplace.var()

        # the more blur, the smaller output
        if blury <= 75.:
            # This image is blur ==> need to remove!
            return True, blury
        else:
            # 'This image has good quality ==> retain!'
            return False, blury