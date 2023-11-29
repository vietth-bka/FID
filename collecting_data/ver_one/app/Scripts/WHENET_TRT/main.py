from headpose_trt import Headposetrt
import cv2
input_file_path = 'test.jpg'
img = cv2.imread(input_file_path)
headpost = Headposetrt()
print(headpost.get_pose(img))