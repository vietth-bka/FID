import cv2


cap = cv2.VideoCapture('rtsp://admin:123456a%40@172.21.111.104')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('end of stream')
        break

    resized = cv2.resize(frame, None, None, 0.4, 0.4)
    cv2.imshow('resized', resized)
    if 27 == cv2.waitKey(3):
        break