import unittest
import sys
import numpy as np
sys.path.append('../app/scripts')

unittest.TestLoader.sortTestMethodsUsing = None

from dbhandler import DbHanler, FACE

class TestDB(unittest.TestCase):
    def setUp(self):
        try:
            self.db = DbHanler('../db/images')
        except Exception:
            print(Exception)
            self.fail("DbHandler() raised Exception unexpectedly")

    def test_save_load_embedding(self):
        testarray =  np.array([1, 2, 3, 4], dtype=np.float64)
        _face = FACE(id=1, embedding=testarray.tobytes())
        self.db.session.add(_face)
        self.db.commit()
        query_face = self.db.session.query(FACE).filter_by(
                id = 1
            ).first()
        array = np.frombuffer(query_face.embedding)
        print(testarray)
        print(array)
        

if __name__ == "__main__":
    unittest.main()