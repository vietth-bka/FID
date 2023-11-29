import fire
import os
import cv2
import numpy as np
from tqdm import tqdm
from time import time, sleep
import sys
from Scripts.utils import getmd5
from Scripts.get_faces_with_track import MyDetection
from Scripts.dbhandler import DbHanler
from Scripts.storagehandler import StorageHandler
from Scripts.facehandler import FaceHandlerThread
from Scripts.customize_cluster import CustomizeCluster
from Scripts.customize_cluster_thread import CustomizeClusterThread
from Scripts.blury import is_blur
import queue, threading
from unidecode import unidecode
from tqdm import tqdm
import pandas as pd
script_dir = os.path.dirname(__file__)

class Collectioner():
    """
    Collection facial data
    """
    def __init__(self, local_storage="../db/images", step=1, num_facehandler=1):
        """
        :param step: 1 means detect on every frames, 3 means detect on 1 frame and then drop 2 next frame \n
        :param num_facehandler: number of thread doing embedding extractiong
        """
        self.local_storage = local_storage
        self.step = step
        self.num_facehandler = num_facehandler

    def __display_pbars(self):
        pbar = tqdm(desc='Processed faces', leave=True)
        while True:
            try:
                item = self.__pbarqueue.get(timeout=5.0)
            except queue.Empty:
                pbar.close()
                break
            pbar.update(item)
            self.__pbarqueue.task_done() 

    def __check_cluster_result(self):
        """query message from __cluster_messagequeue
        """
        while self.__isRunning:
            try:
                item = self.__cluster_messagequeue.get(timeout=1)
                print('\ncurrent status: #frontal: {} \t #threeforth: {} \t #sideview: {} \t #other: {}'.format(
                    item['frontal_count'], item['threefourth_count'], item['sideview_count'], item['other_count']
                ))
                if item['frontal_count'] >= 30 \
                    and item['threefourth_count'] >= 50 \
                    and item['sideview_count'] >= 2:
                    self.__isRunning = False
                    print('\nENOUGHHHHHHHHHHHHHHHHHHHHH!\n')
                self.__cluster_messagequeue.task_done()
            except queue.Empty:
                pass

    def __process_detected_result(self, result, videosrcid, save_frame=False, save_local=False, save_remote=False, pbar=None):
        """do following tasks
        - save entire frame to db
        - enqueue faces, which will be processed in another thread
        :param result \n
        :param videosrcid \n
        :param save_frame: save 4k frame or not, this saving process might take times \n
        :param save_local: save images to local filesystem \n
        :param save_remove: save images to minio S3 storage \n
        :param pbar: custom progressbar \n
        """
        if result == None:
            return
        frameinfo = {"src_id": videosrcid,
                    "frame_count": result['frame']['frame_count'],
                    "data": result['frame']['data']}

        if len(result['faces']) > 0:
            frameid = self.__dbhandler.add_frame_v2(frameinfo, save_remote=True)
            # put face to __facequeue
            for face in result['faces']:
                faceinfo = face
                face['frameid'] = frameid
                face['src_id'] = videosrcid
                self.__facequeue.put(faceinfo)
        else:
            self.__dbhandler.add_frame_v2(frameinfo, save_remote=True, commit=False)

    def __init_detector(self):
        self.__detector = MyDetection(step=self.step)
        self.local_storage = os.path.join(script_dir, self.local_storage)
        if not os.path.exists(self.local_storage):
            import errno
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.local_storage)
        self.__dbhandler = DbHanler(local_storage=self.local_storage)
        
    def __preprocess(self):
        """Just init:
        - db
        - queues
        - Face processing threads
        """
        
        self.__facequeue = queue.Queue()
        self.__pbarqueue = queue.Queue() # just for display progresses
        self.__cluster_facequeue = queue.Queue() # data to be clustering
        self.__cluster_messagequeue = queue.Queue() # send message when #face is enough
        self.facehandlers = []
        for i in range(self.num_facehandler):
            __facehandler = FaceHandlerThread(queue=self.__facequeue,
                queue_cluster=self.__cluster_facequeue,
                dbhandler=self.__dbhandler, 
                pbar_queue=self.__pbarqueue)
            __facehandler.start()
            self.facehandlers.append(__facehandler)
        self.__cluster_thread = CustomizeClusterThread(self.__cluster_facequeue, self.__cluster_messagequeue)
        self.__cluster_thread.start()
        self.t1 = threading.Thread(target=self.__display_pbars)
        self.t2 = threading.Thread(target=self.__check_cluster_result)
        self.__isRunning = True

    def __join(self):
        for i in range(self.num_facehandler):
            self.__facequeue.put(0) # sentinel for stop thread
        for __facehandler in self.facehandlers:
            __facehandler.join()
            assert not __facehandler.is_alive()
        self.__cluster_thread.join()
        assert not self.__cluster_thread.is_alive()
        print('cluster thread quitted')
        self.__facequeue.join() # block until all face are processed
        print('facequeue quitted')
        while not self.__cluster_facequeue.qsize() == 0:
            self.__cluster_facequeue.get()
            self.__cluster_facequeue.task_done()
            sleep(0.01)
        self.__cluster_facequeue.join()
        print('cluster face queue quitted')
        while not self.__pbarqueue.qsize() == 0:
            self.__pbarqueue.get()
            self.__pbarqueue.task_done()
        self.__pbarqueue.join()
        print('progressbar quit')
        try:
            self.t1.join()
            assert not self.t1.is_alive()
        except RuntimeError:
            pass
        print('display bar quit')
        try:
            self.t2.join()
            assert not self.t2.is_alive()
        except RuntimeError:
            pass
        print('result check thread quit')
        self.__cluster_messagequeue.join()
        print('cluster message queue quitted')
        # del self.__dbhandler
        # print('del __dbhandler')

    def __dbscan(self, sourceid):
        """
        Do DBSCAN
        """
        # TODO: get ref faces, and faces from sourceid
        faces = self.__dbhandler.get_all_face(videosrcid=sourceid)
        print('number of face belong to source {} is {}'.format(sourceid, len(faces)))
        ref_indexs = []
        embedding_arr = []
        for i in range(len(faces)):
            if faces[i].isReference == True:
                ref_indexs.append(i)
            em = np.frombuffer(faces[i].embedding)
            embedding_arr.append(em)
        print('ref indexs', ref_indexs)
        embedding_arr = np.array(embedding_arr)
        
        # TODO: do something with customize_cluster
        final_cluster = CustomizeCluster()
        final_cluster.update(embedding_arr, ref_indexs)
        toSave = final_cluster.get_result()
        print('toSave', len(toSave))

        # update
        for i in toSave:
            faces[i].isDBSCANgood = True

        self.__dbhandler.commit()

    def __count_result(self, sourceid):
        """do pose calculation on all face with isDBSCANgood=True
        :param sourceid: id of source 
        :returns False mean still not enough faces, and the count
        """
        faces = self.__dbhandler.get_all_face(videosrcid=sourceid, isDBSCANgood=True)
        frontal_count = 0
        threefourth_count = 0
        sideview_count = 0
        other_count = 0
        for face in faces:
            yaw = face.pose_yaw
            if abs(yaw) <= 20:
                frontal_count += 1
            elif abs(yaw) <= 40:
                threefourth_count += 1
            elif abs(yaw) < 50: 
                sideview_count += 1
            else:
                other_count += 1

        print('\n\nhead pose calculated: #frontal {}, #threefourth {}, #side view {}, #other {}'.format(\
            frontal_count, threefourth_count, sideview_count, other_count))
        return frontal_count >= 30 and threefourth_count >= 50 and sideview_count >= 2, frontal_count + threefourth_count + sideview_count

        # return frontal_count >= 70 and threefourth_count >= 70 and sideview_count >= 10, frontal_count + threefourth_count + sideview_count

    def select_ref(self, cap, wanted_person_id, videosrcid):
        """select reference faces for `wanted_person_id`
        :param cap: cv2.VideoCapture obejct
        :param wanted_person
        :param videosrcid:
        """
        try:
            isPlaying = True
            num_of_ref = 0
            while cap.isOpened():
                if isPlaying:
                    ret, frame = cap.read()
                if ret == False:
                    print("Video capture read error")
                    sys.exit(0)
                    break
                h, w, c = frame.shape
                if h > 1080:
                    display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                else:
                    display_frame = frame
                cv2.imshow("select ref", display_frame)
                key = cv2.waitKey(3)
                # if key == 27:
                    # break
                # if key == 32: # space
                if key == 112: # space is conflict with selectROI, use 'p' instead
                    isPlaying = not isPlaying
                if not isPlaying:
                    cv2.namedWindow("select region of interest", cv2.WINDOW_AUTOSIZE)
                    roi = cv2.selectROI("select region of interest", display_frame)
                    cv2.destroyWindow('select region of interest')
                    if h > 1080:
                        roi = tuple([2 * i for i in roi])
                    print(roi)
                    imCrop = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                    # detect ref image
                    face = self.__detector.get_single_face(imCrop)
                    if face == None :
                        print('\nselected roi has no image, do it again', flush=True)
                    else:
                        # check blurry
                        isBlur, _ = is_blur(face['data'])
                        if isBlur:
                            print('\nselected roi has no image, do it again', flush=True)
                        else:
                            cv2.imshow('ref' + str(num_of_ref), face['data'])
                            # save it to db
                            faceinfo = face
                            faceinfo['isReference'] = True
                            faceinfo['nameid'] = wanted_person_id
                            faceinfo['src_id'] = videosrcid
                            self.__facequeue.put(faceinfo)

                            num_of_ref = num_of_ref + 1
                            if num_of_ref > 0:
                                break
                    isPlaying = not isPlaying
            # block until all 3 ref faces is processed, and then commit
            self.__facequeue.join()
            self.__dbhandler.commit()
            self.__dbhandler.update_presigned_url()
            if not 1 == len(self.__dbhandler.get_ref_frame_of_source(videosrcid)):
                print("not enough reference frame")
                self.__join()
                cv2.destroyAllWindows()
                sys.exit(0)
            cv2.destroyWindow('select ref')
        except KeyboardInterrupt:
            print('Keyboard Interrupted')
            try:
                self.__join()
                cv2.destroyAllWindows()
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    

    def process(self, source, wanted_person, width=3840, height=2160, redo=False, show=True, 
        save_frame=False, save_local=False, save_remote=True):
        """
        Do face detection on `source`, and save data to db \n
        :param source: could be a file or a rtsp stream,   
        for example `rtsp://admin:123456a%40@10.61.166.189` \n
        :param width: width of the rtsp stream
        :param height: height of the rtsp stream
        :param wanted_person: short name of the person who we want to extract face from this source
        for example `datnt527` or 218777\n
        :param save_frame: save 4k frame or not, this saving process might take times \n
        :param save_local: save images to local filesystem \n
        :param save_remove: save images to minio S3 storage \n
        """
        print(source)
        try:
            self.__preprocess()
            source = str(source)
            if isinstance(source, int):
                srctype = "device"
            elif source.startswith("rtsp"):
                srctype = "rtsp"
                # srctype = "rtspngungoc"
            else:
                srctype = "file"

            if srctype == "file" and not redo:
                if self.__dbhandler.is_videosrc_exists(source):
                    print('video processed')
                    return 0

            if srctype == "file":
                md5 = getmd5(source)
            else:
                md5 = ''

            wanted_person_id = self.__dbhandler.get_staff_id(wanted_person)
           

            if srctype == "rtsp":
                cap_width = width
                cap_height = height
                pipeline = "rtspsrc location={} latency=100 ! queue ! rtph265depay \
                            ! h265parse ! nvh265dec ! videoconvert ! video/x-raw,width={},height={},format=BGR \
                            ! appsink emit-signals=true sync=false async=false drop=true".format(\
                                source, cap_width, cap_height)
                print('playing pipeline:', pipeline)
                try:
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                except cv2.error as err:
                    sys.exit(0)
                except Exception as e:
                    sys.exit(0)
            else:
                print('\n\npipeline doesnt work, use opencv instead with device', flush=True)
                cap = cv2.VideoCapture(source)
                cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not cap.isOpened():
                print('\n\ncannot open stream, abort now', flush=True)
                sys.exit(0)
            

            videosrc_info = {
                "hash_md5": md5,
                "uri": source, 
                "time_from": time(), "time_to": "", "location": "", 
                "width": cap_width, "height": cap_height,
                "wanted_person_id": wanted_person_id,
                "status": "qdt140621"}
                
            if srctype == "rtsp":
                videosrc_info['from'] = str(time())
            videosrcid = self.__dbhandler.add_videosrc_v2(videosrc_info)

            self.select_ref(cap, wanted_person_id, videosrcid)

            # select a ROI for laymau
            aRet, aFrame = cap.read()
            if aRet == False:
                self.__isRunning = False
                self.__join()
                self.__dbhandler.commit()
                self.__dbhandler.update_presigned_url()
                sys.exit(0)
            if cap_height > 1080:
                aFrame = cv2.resize(aFrame, None, fx=0.5, fy=0.5)
            roi = cv2.selectROI("select region of interest", aFrame)
            cv2.destroyWindow('select region of interest')
            if cap_height > 1080:
                roi = tuple([2 * i for i in roi])
            

            self.t1.start()
            self.t2.start()
            pbar = tqdm(desc='Number of saved Frame', leave=True)
            # try:
            for result in self.__detector.get_frames(cap, roi):
                try:
                    if self.__isRunning == False:
                        break

                    if show:
                        show_mat = self.__detector.show_result(result, roi)
                        if show_mat.size != 0:
                            cv2.imshow('result', show_mat)
                            if 27 == cv2.waitKey(1):
                                self.__isRunning = False
                                break

                    pbar.update(1)

                    # save result to db
                    self.__process_detected_result(result, videosrcid, 
                        save_frame=save_frame, save_local=save_local, save_remote=save_remote)

                except KeyboardInterrupt:
                    print("Bye")
            # except RuntimeError as e:
            #     print("EOF?")
            pbar.close()
                
            self.__join()
            self.__dbhandler.commit()
            self.__dbhandler.update_presigned_url()

            # dbscan
            self.__dbscan(videosrcid)

            # calculate pose
            result, goodcount = self.__count_result(videosrcid)
            print(str(wanted_person) + ' done' if result else str(wanted_person) + ': still not enough images')

            # TODO: update goodcount to wanted_person_id
            self.__dbhandler.update_staff(wanted_person_id, num_good_face=goodcount)

        except KeyboardInterrupt:
            print('Interrupted')
            try:
                return 0
                # sys.exit(0)
            except SystemExit:
                os._exit(0)
        
        print("Done for", wanted_person)
        cv2.destroyAllWindows()
        return 0

    def process_all(self, source):
        print(source)
        try:
            self.__init_detector()
            while (1):
                print("Nhap ma so nhan vien: ")
                wanted_person = input()
                if wanted_person == 'x':
                    sys.exit(0)
                self.process(source, wanted_person)
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)   

    def test(self, videosourceid):
        # return
        self.__dbhandler = DbHanler(local_storage=self.local_storage)
        self.__dbhandler.update_staff(232, num_good_face=30)


class Shower():
    """
    Show collected facial data
    """
    def __init__(self, local_storage="../db/images"):
        self.local_storage = local_storage
        local_storage = os.path.join(script_dir, local_storage)
        if not os.path.exists(local_storage):
            import errno
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), localdb)
        self.__dbhandler = DbHanler(local_storage=local_storage)

    def source(self, videosourceid=-1):
        """show the result in many way
        :param videosourceid:
        """
        if videosourceid == -1:
            querysourceid = self.__dbhandler.get_latest_video_src_id()
            print('latest source id', querysourceid)
        else:
            querysourceid = videosourceid
        faces = self.__dbhandler.get_all_face(videosrcid=querysourceid, isDBSCANgood=True, hasPose=True)
        for frame, face in self.__dbhandler.download_face_data(faces):
            cv2.imshow("face", frame)
            if 27 == cv2.waitKey():
                break

    def ref(self, videosourceid):
        print(self.__dbhandler.get_ref_frame_of_source(videosourceid))

    def nothing(self, videosourceid):
        def path_to_image_html(path):
            return '<img src="' + path + '" width="112">'
        import pandas as pd
        df = pd.read_sql_query('select * from "FACE" where "src_id" = {}; /* and "isDBSCANgood" is null */'.format(videosourceid), con=self.__dbhandler.get_engine())
        df = df.drop(['embedding', 'landmark', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_w', 'bbox_h'], axis=1)
        df.to_html('/tmp/webpage.html',escape=False, formatters=dict(file_remote=path_to_image_html))
        print('file://tmp/webpage.html')
        return

    def download_frame(self, staff_id):
        staff_id = str(staff_id)
        if len(staff_id) == 5:
            staff_id = '0' + staff_id
        staff = self.__dbhandler.get_staff_by_id(str(staff_id))
        if staff is None:
            return staff_id, "person is not in db"

        # save_folder = "/media/dat/503FB8550318C5B1/chieu160821"
        save_folder = "/media/v100/DATA2/vietth/chieu160821"
        outfolder = os.path.join(save_folder, str(staff_id) + "_" + unidecode(staff.fullname))
        # frame_folder = os.path.join(outfolder, 'frame')
        # face_folder = os.path.join(outfolder, 'face')

        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        # if not os.path.exists(frame_folder):
        #     os.makedirs(frame_folder)
        # if not os.path.exists(face_folder):
        #     os.makedirs(face_folder)

        frames = self.__dbhandler.get_face_by_staff_id(str(staff_id))
        if len(frames) != 0:
            # self.__dbhandler.download_face_images_v2(face_folder, frame_folder, frames)
            self.__dbhandler.download_face_images_v2(outfolder, "", frames)
        # for frame in frames:
        #     print(frame.FACE.bbox_x1)
        #     print(frame.FACE.bbox_y1)
        #     print(frame.FACE.bbox_x2)
        #     print(frame.FACE.bbox_y2)
        #     print(os.path.join(frame_folder, frame.FACE.file_name))
        return len(frames), str(staff_id) + "_" + unidecode(staff.fullname)
    def download_all_tang4(self):
        # csv = pd.read_excel("/home/dat/source/faceid/laymau/chieu210521.xlsx")
        ids = ["289249",
                "288938",
                "272733",
                "288939",
                "288939",
                "285624",
                "285519"]

        # for mnv in csv['staff_id']:
        for mnv in ids:
            number_of_imgs, name = self.download_frame(mnv)
            print(name, number_of_imgs)
    def get_staff_by_id(self, staff_id):
        staff = self.__dbhandler.get_staff_by_id(str(staff_id))
        print(staff.fullname, staff.staff_id)

    def download(self, videosourceid, outfolder):
        """download all faces of videosourceid to outfolder
        :param videosourceid: id of source
        :param outfolder: path to where to save
        """
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        ref_faces = self.__dbhandler.get_ref_frame_of_source(videosourceid)
        for ref_face in ref_faces:
            print(ref_face.file_name)
            em = np.frombuffer(ref_face.embedding)
            print(em)
            np.savetxt(ref_face.file_name[:-3] + 'txt', em)
        
        ref_filenames = [i.file_name for i in ref_faces]
        with open(os.path.join(outfolder, 'reference.txt'), 'w') as f:
            f.write('\n'.join(ref_filenames))
            f.write('\n')

        self.__dbhandler.download_face_images(outfolder, videosrcid=videosourceid)

    def config(self):
        from scripts.dbhandler import conf
        print(str(conf.POSTGRES_HOSTNAME))

if __name__ == "__main__":
    # fire.Fire(Collectioner)
    fire.Fire({
        'collect': Collectioner,
        'show': Shower,
    })
