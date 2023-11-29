import os
import cv2
import numpy as np
import psycopg2
import sqlalchemy
from sqlalchemy import Column, VARCHAR, FLOAT, BIGINT, VARCHAR, INT, BOOLEAN 
from sqlalchemy import and_, or_, not_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4
from .storagehandler import StorageHandler
from .config import getConfig

conf = getConfig()
_sql_string = "postgresql://" + conf.POSTGRES_USER_NAME + ":" + conf.POSTGRES_USER_PASS \
            + '@' + conf.POSTGRES_HOSTNAME + ':' + str(conf.POSTGRES_PORT) \
            + '/' + conf.POSTGRES_DATABASE
# engine = create_engine("postgres://postgres:123456@localhost:5432/laymau")
engine = create_engine(_sql_string)
Base = declarative_base(engine)


class VIDEO_SRC(Base):
    __tablename__ = 'VIDEO_SRC'
    __table_args__ = {'autoload': True}

class FACE(Base):  
    __tablename__ = 'FACE'
    __table_args__ = {'autoload': True}

class FRAME(Base):
    __tablename__ = 'FRAME'
    __table_args__ = {'autoload': True}

class STAFF(Base):
    __tablename__ = 'STAFF'
    __table_args__ = {'autoload': True}

class DbHanler:
    def __init__(self, local_storage, debug=False):
        """ a helper class
        :param local_storage: path to top level local storage filesystem
        note that attribute session is public, so one could use it to do customized query
        """
        self.local_storage = local_storage
        
        conn = psycopg2.connect(
            host=conf.POSTGRES_HOSTNAME,
            port=conf.POSTGRES_PORT,
            database=conf.POSTGRES_DATABASE,
            user=conf.POSTGRES_USER_NAME,
            password=conf.POSTGRES_USER_PASS)
        # create a cursor
        cur = conn.cursor()
        
	    # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
    
        self.conn = conn
        self.debug = debug
        self.cur = cur

        # use sqlalchemy
        # db_string = "postgres://postgres:123456@localhost:5432/laymau"
        # engine = sqlalchemy.create_engine(db_string)
        # metadata = Base.metadata
        Session = sessionmaker(bind=engine)  
        session = Session()
        # base.metadata.create_all(db)

        # meta = MetaData()
        # meta.reflect(bind=engine)
        # tables = meta.tables

        self.session = session
        # self.tables = tables
        self.storagehandler = StorageHandler(num_thread=5)

    def __del__(self):
        self.storagehandler.drawing_down()

    def get_engine(self):
        return engine

    def commit(self):
        self.conn.commit()
        self.session.commit()

    def run_custom_query(self, query):
        """run whatever you want
        """
        self.cur.execute(query)
        # self.conn.commit()
        return cur.fetchall()

    def add_department(self, departments):
        sql_string = """INSERT INTO "DEPARTMENT" (name) VALUES (%s) RETURNING id;"""
        for department in departments:
            query = cur.mogrify(sql_string, (str(department),))
            if self.debug:
                print(query)  
            self.cur.execute(query)
            self.conn.commit()

    def add_staff(self, staffs):
        """add staffs into database
        staffs is list of (dict of (staff_id, fullname, department_id))
        """
        sql_string = """INSERT INTO "STAFF" (staff_id, fullname, department_id) values ('{}', '{}', {}) RETURNING *;"""
        for staff in staffs:
            query = sql_string.format(staff['staff_id'], staff['fullname'], staff['department_id'])
            if self.debug:
                print(query)  
            self.cur.execute(query)
            self.conn.commit()

    def add_camera(self, cameras):
        """add cameras into database
        cameras is list of (dict of (ip, location, floor, width, height, department_id))
        """
        sql_string = """INSERT INTO "CAMERA" (ip, location, floor, width, height, department_id) 
                        values ('{}', '{}', '{}'::floor_name, {}, {}, {}) RETURNING *;"""
        sql_string2 = """INSERT INTO "CAMERA" (ip, location, floor, width, height) 
                        values ('{}', '{}', '{}'::floor_name, {}, {}) RETURNING *;"""
        for cam in cameras:
            try:
                query = sql_string.format(cam['ip'], cam['location'], cam['floor'], cam['width'], cam['height'], cam['department_id'])
            except KeyError:
                query = sql_string2.format(cam['ip'], cam['location'], cam['floor'], cam['width'], cam['height'])
            if self.debug:
                print(query)  
            self.cur.execute(query)
            self.conn.commit()

    def add_videosrc(self, video):
        """add a single video source into database
        :param video is dict of (hash_md5, uri, time_from, time_to, location, width, height)
        :return id
        """
        sql_string = """INSERT INTO "VIDEO_SRC" (hash_md5, uri, time_from, time_to, location, width, height)
                        values ('{}', '{}', 'infinity'::timestamp, 'infinity'::timestamp, '{}', {}, {}) RETURNING id;"""
        query = sql_string.format(video['hash_md5'], video['uri'], video['location'], video['width'], video['height'])
        if self.debug:
            print(query)
        self.cur.execute(query)
        self.conn.commit()
        return self.cur.fetchone()[0]

    def add_videosrc_v2(self, videosrcinfo):
        """add a single video source into database using sqlalchemey
        :param videosrcinfo is dict of (hash_md5, uri, time_from, time_to, location, width, height, wanted_person_id)
        :return id
        """
        _video = VIDEO_SRC(hash_md5 = videosrcinfo['hash_md5'],
                           uri      = videosrcinfo['uri'],
                        #    time_from= video['time_from'],
                        #    time_to  = video['time_to'],
                           wanted_people=videosrcinfo['wanted_person_id'],
                           location = videosrcinfo['location'],
                           width    = videosrcinfo['width'],
                           height   = videosrcinfo['height'],
                           status   = videosrcinfo['status'],)
        self.session.add(_video)
        self.session.commit()
        self.session.refresh(_video)
        return _video.id

    def add_frame(self, frame):
        """add a single frame source into database
        :param frame is dict of (src_id, frame_count, file_local, file_remote)
        :retur id
        """
        sql_string = """INSERT INTO "FRAME" (src_id, frame_count, file_local, file_remote)
                        values ({}, {},  '{}', '{}') RETURNING id;"""
        query = sql_string.format(frame['src_id'], frame['frame_count'], frame['file_local'], frame['file_remote'])
        if self.debug:
            print(query)
        self.cur.execute(query)
        # self.conn.commit()
        return self.cur.fetchone()[0]

    def add_frame_v2(self, frameinfo, save_local=False, save_remote=False, commit=True):
        """add a single frame source into database using sqlalchemey
        :param frameinfo is dict of (src_id, frame_count, data)
        :param save_local: where save the frame to local filesystem or not
        :param save_remote: where save the frame to S3 server or not
        :commit set to False if only add frame without commit \n
        when set to False, the performance can increase but the return value if not correct anymore
        :return id
        """
        framename = str(uuid4()) + '.jpg'
        framefile = os.path.join(self.local_storage, framename)
        if save_local:
            cv2.imwrite(framefile, frameinfo['data'])
        if save_remote:
            self.storagehandler.upload_stream_async(framename, frameinfo['data'])

        _frame = FRAME(src_id       = frameinfo['src_id'], 
                        frame_count = frameinfo['frame_count'],
                        file_name   = framename,
                        file_local  = framefile,
                        file_remote = '')
        self.session.add(_frame)
        if commit:
            self.session.commit()
            self.session.refresh(_frame)
        return _frame.id

    def add_face(self, face):
        """add a single face source into database
        :param face is dict of (trackid, file_local, file_remote, detection_confident,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_w, bbox_h
                        landmark, frameid)
        :return nothing
        """
        sql_string = """INSERT INTO "FACE" (trackid, file_local, file_remote, 
            detection_confident, 
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_w, bbox_h, landmark, frameid)
            values ({}, '{}', '{}', {}, {}, {}, {}, {}, {}, {}, '{}', {});"""

        query = sql_string.format(face['trackid'], face['file_local'], face['file_remote'], 
            face['detection_confident'],
            face['bbox_x1'], face['bbox_y1'], face['bbox_x2'], face['bbox_y2'], face['bbox_w'], face['bbox_h'], 
            face['landmark'], face['frameid'])
        if self.debug:
            print(query)
        self.cur.execute(query)
        # self.conn.commit()

    def add_face_v2(self, faceinfo, save_local=False, save_remote=False):
        """add a single face source into database using sqlalchemey
        :param faceinfo is dict of (trackid, detection_confident,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_w, bbox_h
                        landmark, frameid, isReference)
        :return nothing
        """
        facename = str(uuid4()) + '.jpg'
        facefile = os.path.join(self.local_storage, facename)
        if save_local:
            cv2.imwrite(facefile, faceinfo['data'])
        if save_remote:
            self.storagehandler.upload_stream_async(facename, faceinfo['data'])

        _face = FACE(
                    file_name = facename,
                    file_local = facefile,
                    file_remote = '',
                    detection_confident = faceinfo['confident'],
                    bbox_x1 = faceinfo['bbox'][0], 
                    bbox_y1 = faceinfo['bbox'][1], 
                    bbox_x2 = faceinfo['bbox'][2], 
                    bbox_y2 = faceinfo['bbox'][3], 
                    bbox_w  = faceinfo['bbox'][2] - faceinfo['bbox'][0], 
                    bbox_h  = faceinfo['bbox'][3] - faceinfo['bbox'][1],
                    landmark = np.array2string(faceinfo['landmark']))
        if 'src_id' in faceinfo:
            _face.src_id = faceinfo['src_id']
        if 'trackid' in faceinfo:
            _face.trackid = faceinfo['trackid']
        if 'frameid' in faceinfo:
            _face.frameid = faceinfo['frameid']
        if 'isReference' in faceinfo:
            _face.isReference = faceinfo['isReference']
        if 'nameid' in faceinfo:
            _face.nameid = faceinfo['nameid']
        if 'embedding' in faceinfo:
            _face.embedding = faceinfo['embedding'].astype(np.float64).tobytes()
        if 'embedding_norm' in faceinfo:
            _face.embedding_norm = faceinfo['embedding_norm']
        if 'pose_yaw' in faceinfo:
            _face.pose_yaw = faceinfo['pose_yaw']
        if 'pose_pitch' in faceinfo:
            _face.pose_pitch = faceinfo['pose_pitch']
        if 'pose_roll' in faceinfo:
            _face.pose_roll = faceinfo['pose_roll']
        self.session.add(_face)

    def is_videosrc_exists(self, video):
        """ check if a video file is existed
        :param video: path to video file or rtsp stream uri
        """
        sql_string = """SELECT EXISTS(
	                    SELECT 1 FROM "VIDEO_SRC" WHERE hash_md5 = '{}');"""
        query = sql_string.format(video)
        self.cur.execute(query)
        return self.cur.fetchone()[0]

    def update_videosrc_done(self, video_id):
        """change status of a videosrc to done
        :param video_id: id of about to change videosrc
        :return true if success
        """
        sql_string = """UPDATE "VIDEO_SRC" SET status = 'done' WHERE id = {} RETURNING status;"""
        query = sql_string.format(video_id)
        if self.debug:
            print(query)
        self.cur.execute(query)
        self.conn.commit()
        return self.cur.fetchone()[0] == 'done'

    def update_videosrc_done_v2(self, video_id):
        """change status of a videosrc to done
        :param video_id: id of about to change videosrc
        :return true if success
        """
        _video = self.session.query(VIDEO_SRC).filter_by(id=video_id).first()
        _video.status = "done"
        self.session.commit()

    def get_frame_by_staff_id(self, staff_id):
        frames = self.session.query(FACE, VIDEO_SRC, STAFF, FRAME).filter(
            STAFF.staff_id == staff_id,
            STAFF.id == VIDEO_SRC.wanted_people,
            VIDEO_SRC.id == FACE.src_id,
            FRAME.id == FACE.frameid
        )
        return frames.all()

    def get_face_by_staff_id(self, staff_id):
        frames = self.session.query(FACE, VIDEO_SRC, STAFF).filter(
            STAFF.staff_id == staff_id,
            STAFF.id == VIDEO_SRC.wanted_people,
            VIDEO_SRC.id == FACE.src_id,
        )
        return frames.all()
    
    def get_staff_by_id(self, staff_id):
        staff = self.session.query(STAFF).filter(
            STAFF.staff_id == staff_id
        ).first()
        return staff
    def get_all_face(self, videosrcid=None, frameid=None, isDBSCANgood=None, hasPose=None):
        """get all face object
        :param videosrcid: id of videosrc to be query
        :param frameid: id of frame to be query, if `videosrcid` is specified, `frameid` will be ignore
        :param hasPose: if True, filter only face with calculated pose, if False, filter only face with no pose
        :returns list of face objects
        """
        if videosrcid is not None:
            faces = self.session.query(FACE).filter_by(
                src_id = videosrcid
            )
        elif frameid is not None:
            faces = self.session.query(FACE).filter_by(
                frameid = frameid
            )
        else:
            faces = self.session.query(FACE)

        if isDBSCANgood is not None:
            faces = faces.filter_by(isDBSCANgood=isDBSCANgood)

        if hasPose is not None:
            if hasPose == True:
                faces = faces.filter(
                    FACE.pose_yaw != None,
                    FACE.pose_pitch != None,
                    FACE.pose_roll != None)
            if hasPose == False:
                faces = faces.filter(
                    FACE.pose_yaw == None,
                    FACE.pose_pitch == None,
                    FACE.pose_roll == None)
        
        return faces.all()
    
    def download_img(self, img_path, save_path):
        data = self.storagehandler.download_to_file(img_path, save_path)
        yield data

    def download_face_data(self, faces):
        """download face data
        :param faces: list of FACE object
        :yields tuple of data, and curent face item
        """
        for face in faces:
            data = self.storagehandler.download_stream(face.file_name)
            yield data, face

    def download_face_images(self, folder, videosrcid=None, frameid=None):
        """get all face
        :param folder: where to save downloaded images
        :param videosrcid: id of videosrc to be query
        :param frameid: id of frame to be query, if `videosrcid` is specified, `frameid` will be ignore
        :returns generator of numpy frames
        """
        faces = self.get_all_face(videosrcid=videosrcid, frameid=frameid)
        for face in faces:
            self.storagehandler.download_to_file(face.file_name, os.path.join(folder, face.file_name))
    def download_face_images_v2(self, face_folder, frame_folder, faces):
        """get all face
        :param folder: where to save downloaded images
        :param videosrcid: id of videosrc to be query
        :param frameid: id of frame to be query, if `videosrcid` is specified, `frameid` will be ignore
        :returns generator of numpy frames
        """
        # faces = self.get_all_face(videosrcid=videosrcid, frameid=frameid)
        for face in faces:
            if face.FACE.isReference:
                face_name = "_ref_" + face.FACE.file_name
            else:
                face_name = face.FACE.file_name
            self.storagehandler.download_to_file(face.FACE.file_name, os.path.join(face_folder, face_name))
            # self.storagehandler.download_to_file(face.FRAME.file_name, os.path.join(frame_folder, face.FACE.file_name))
            with open(os.path.join(face_folder, face_name)[0:-3] + 'txt', 'w') as f:
                f.write(str(face.FACE.pose_yaw) + "\n")
                f.write(str(face.FACE.pose_pitch) + "\n")
                f.write(str(face.FACE.pose_roll) + "\n")

    def update_presigned_url(self, add_new_only=True):
        """update remote_file for all face and frame which haven't have presigned url
        :param add_new_only: True=only add new remote_file for face and frame which haven't have url yet;
        False=update all remote_file
        :returns None
        """
        faces = self.session.query(FACE)
        if add_new_only:
            faces = faces.filter_by(file_remote='')
        for face in faces.all():
            url = self.storagehandler.get_presigned_url(face.file_name)
            # print(face.file_name, url)
            face.file_remote = url
        frames = self.session.query(FRAME)
        if add_new_only:
            frames = frames.filter_by(file_remote='')
        for frame in frames.all():
            url = self.storagehandler.get_presigned_url(frame.file_name)
            # print(frame.file_name, url)
            frame.file_remote = url

        self.commit()

    def get_ref_frame_of_source(self, source_id):
        """get reference frames belonging to a source
        :param source_id
        :return array of faces
        """
        ref_faces = self.session.query(FACE).filter(
            and_(
                FACE.isReference == True,
                FACE.src_id == source_id,
            )
        ).all()

        return ref_faces

    def get_staff_id(self, name_or_staffid):
        """get id of person
        :param name_or_staffid: can be 21877 or datnt527
        :return id (id in the database, not 218777)
        """
        this_staff = self.session.query(STAFF).filter(
            or_(
               STAFF.staff_id.like(str(name_or_staffid)),
               STAFF.shortname.like(str(name_or_staffid))
            )
        ).all()

        assert len(this_staff) == 1, "person"
        id = this_staff[0].id
        return id

    def update_staff(self, id, staff_id=None, shortname=None, fullname=None, department_id=None, 
        num_good_face=None, face_poitraid_id=None, comment=None):
        _staff = self.session.query(STAFF).filter_by(id=id).first()
        if staff_id is not None:
            _staff.staff_id = staff_id
        if shortname is not None:
            _staff.shortname = shortname
        if fullname is not None:
            _staff.fullname = fullname
        if department_id is not None:
            _staff.department_id = department_id
        if num_good_face is not None:
            _staff.num_good_face = num_good_face
        if face_poitraid_id is not None:
            _staff.face_poitraid_id = face_poitraid_id
        if comment is not None:
            _staff.comment = comment
        self.session.commit()

    def get_latest_video_src_id(self):
        """get latest video id
        """
        return self.session.query(VIDEO_SRC).order_by(VIDEO_SRC.id.desc()).first().id

if __name__ == "__main__":
    dbhandler = DbHanler()
    faces = dbhandler.read_face_all()
    for face in faces:
        print(face.id)

    dbhandler.update_videosrc_done_v2(1)
