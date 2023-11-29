from minio import Minio
from minio.error import S3Error
from io import BytesIO
import cv2
import numpy as np
import threading, queue
import sys
from datetime import timedelta
from .config import getConfig
SENTINEL = 0

class StorageHandler:
    def __init__(self, num_thread=2, debug=False):
        """helper class to store image to MinIO server
        :param num_thread: number of thread do all the encoding, and uploading
        :param debug: print debug message or not
        """
        conf = getConfig()
        client = Minio(
            conf.MINIO_HOSTNAME + ':' + str(conf.MINIO_PORT),
            access_key=conf.MINIO_ACCESS_KEY,
            secret_key=conf.MINIO_SECRET_KEY,
            secure=False)
        # Create a client with the MinIO server playground, its access key and secret key.
        # client = Minio(
        #     "192.168.1.51:9000",
        #     access_key="ucQbeazlRY2PdwUp",
        #     secret_key="xYdCTj1SJVhbnkmM",
        #     secure=False,
        # )
        # client = Minio(
            # "192.168.1.15:9001",
            # access_key="minio",
            # secret_key="minio123",
            # secure=False,
        # )
        # bucketname = "laymau"
        bucketname = conf.MINIO_BUCKETNAME
        # Make bucketname bucket if not exist.
        found = client.bucket_exists(bucketname)
        if not found:
            client.make_bucket(bucketname)
        else:
            print("Bucket `{}` already exists".format(bucketname))

        self.client = client
        self.bucketname = bucketname
        self.queue = queue.Queue()
        self.count = 1
        self.threads = []
        for i in range(num_thread):
            th = threading.Thread(target=self.__do_upload_in_background, args=(self.queue, ))
            self.threads.append(th)
            th.start()
        self.debug = debug

    def drawing_down(self):
        for i in range(len(self.threads)):
            self.queue.put(SENTINEL)
        for i in self.threads:
            i.join()
        while self.queue.qsize() != 0:
            self.queue.get()
            self.queue.task_done()

    def __del__(self):
        if self.debug:
            print('\n\nStorageHandler done, unsaved pic: {}\n\n'.format(self.queue.qsize()), flush=True)

    def upload_stream(self, object_name, numpy_frame):
        """
        Uploads data from a stream to an object in a bucket.
        :param object_name <str>: Object name to be store in the bucket.
        :param numpy_frame
        """
        # convert numpy_frame to BytesIO first
        ret, encoded_frame = cv2.imencode(".jpg", numpy_frame)
        assert(ret == True)
        frame_as_stream = BytesIO(encoded_frame)
        try:
            writeResult = self.client.put_object(self.bucketname, object_name, frame_as_stream, length=frame_as_stream.getbuffer().nbytes,
                    content_type="image/jpeg")
        except Exception as e:
            print(e, file=sys.stderr)
            raise e
        self.count = self.count + 1
        if self.debug:
            print("created {0} object; etag: {1}, version-id: {2}".format(
                writeResult.object_name, writeResult.etag, writeResult.version_id))

    def __do_upload_in_background(self, q):
        """run in background, continuously retrive item from queue `q`
        each item in q is a dict with keys: object_name, numpy_frame
        """
        while True:
            try:
                item = q.get()
                if item == SENTINEL:
                    if self.debug:
                        print('In the end, queue size {}, num saved {}'.format(self.queue.qsize(), self.count), flush=True)
                    q.task_done()
                    break
                object_name = item['object_name']
                numpy_frame = item['numpy_frame']
                self.upload_stream(object_name, numpy_frame)
                q.task_done()

                if self.debug:
                    print('queue size {}, saved {}'.format(self.queue.qsize(), self.count), flush=True)
            except Exception as e:
                print(e, flush=True)

    def upload_stream_async(self, object_name, numpy_frame):
        self.queue.put({'object_name': object_name,
            'numpy_frame': numpy_frame})

    def download_stream(self, object_name):
        """Download data from bucket
        :param object_name
        :return numpy frame
        """
        try:
            response = self.client.get_object(self.bucketname, object_name)
            frame = np.asarray(bytearray(response.read()), dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        except Exception as e:
            print(e, file=sys.stderr)
            raise e
        finally:
            response.close()
            response.release_conn()

        return frame

    def download_to_file(self, object_name, file_name):
        """Download data from bucket to file
        :param object_name
        :param file_name: name of file to be save
        """
        try:
            response = self.client.fget_object(self.bucketname, object_name, file_name)
        except Exception as e:
            print(e, file=sys.stderr)
            raise e

    def get_presigned_url(self, object_name):
        try:
            url = self.client.get_presigned_url("GET", self.bucketname, object_name)
        except Exception as e:
            url = ''
            print(e)
            raise e
        finally:
            return url