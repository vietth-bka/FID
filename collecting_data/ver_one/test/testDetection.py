import unittest
import sys
sys.path.append('../app/scripts')


from get_faces_with_track import MyDetection

class TestDB(unittest.TestCase):
    def setUp(self):
        try:
            myDetection = MyDetection()
        except Exception:
            print(Exception)
            self.fail("MyDetection() raised Exception unexpectedly")
        self.myDetection = myDetection

    def test_video(self):
        try:
            self.myDetection.get_frames('test.mp4')
        except Exception:
            print(Exception)
            self.fail("MyDetection() raised Exception unexpectedly")

if __name__ == "__main__":
    unittest.main()