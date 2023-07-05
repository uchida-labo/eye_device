import cv2
from functions import image_processing, data_acquisition, judge_area, frame_diff

# Trimming area
xmin, xmax = 220, 420  #100 , 500
ymin, ymax = 180, 240  #100 , 300

# Setting of USB camera
cap = cv2.VideoCapture(0)  # 0+cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 15) # カメラFPS設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # カメラ画像の横幅設定  1280pxel
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # カメラ画像の縦幅設定  720pxel

fps, width, height = judge_area.video_parameter(cap)

avg = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray, bin, edges, contours = image_processing.img_process(frame, xmin, xmax, ymin, ymax)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    thresh = frame_diff.diff(gray, avg)

