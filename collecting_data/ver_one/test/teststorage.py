import unittest
import sys
import numpy as np
sys.path.append('../app/scripts')

unittest.TestLoader.sortTestMethodsUsing = None

from storagehandler import StorageHandler

class TestDB(unittest.TestCase):
    def setUp(self):
        try:
            self.storage = StorageHandler()
        except Exception as e:
            print(Exception)
            raise e
            self.fail("DbHandler() raised Exception unexpectedly")

    def test_uploadstream(self):
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        print(self.storage.uploadstream("test.jpg", img))

if __name__ == "__main__":
    unittest.main()