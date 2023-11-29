from threading import Thread
from .customize_cluster import CustomizeCluster
import numpy as np

SENTINEL = 0

class CustomizeClusterThread(Thread):
    def __init__(self, queue, message_queue):
        """
        :param queue: data queue,
        :param message_queue: queue to put result message in
        """
        super().__init__()
        self.queue = queue
        self.cluster = CustomizeCluster()
        self.message_queue = message_queue

        self.update_size = 50
        self.update_pos = 0 # should be from 0 to self.update_size - 1
        self.addition_points = np.empty((self.update_size, 512)) # avoid using np.concatenate

        self.all_yaw = [] # store all gotten pose_yaw
        self.ref = [] # index of 3 reference points, have to set at the first update


    def run(self):
        while True:
            faceinfo = self.queue.get()
            # santinel check for return
            if faceinfo == SENTINEL:
                # print('CustomizeClusterThread quitting...')
                self.queue.task_done()
                # while not self.queue.empty:
                    # self.queue.get()
                    # self.queue.task_done()
                # print('CustomizeClusterThread quitting...done')
                break
            
            self.all_yaw.append(faceinfo['pose_yaw'])
            if len(self.ref) < 1 and faceinfo['isReference'] == True:
                self.ref.append(self.update_pos)
            
            # copy embedding to addition_points[update_pos]
            self.addition_points[self.update_pos] = faceinfo['embedding']
            self.update_pos += 1

            if self.update_pos >= self.update_size:
                # update
                self.cluster.update(self.addition_points, self.ref)
                # count result
                result = self.cluster.get_result()

                frontal_count = 0
                threefourth_count = 0
                sideview_count = 0
                other_count = 0
                for i in result:
                    yaw = self.all_yaw[i]
                    if abs(yaw) <= 20:
                        frontal_count += 1
                    elif abs(yaw) <= 40:
                        threefourth_count += 1
                    elif abs(yaw) < 50: 
                        sideview_count += 1
                    else:
                        other_count += 1
                # send result to main app
                self.message_queue.put({'frontal_count': frontal_count,
                                'threefourth_count': threefourth_count,
                                'sideview_count': sideview_count,
                                'other_count': other_count})
                # reset cache
                self.update_pos = 0

            self.queue.task_done()