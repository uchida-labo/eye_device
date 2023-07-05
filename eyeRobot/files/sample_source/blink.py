import cv2
from matplotlib import pyplot as plt
from function import filter
import time

# Font type setting
fontType = cv2.FONT_HERSHEY_COMPLEX

# Trimming　area
xmin, xmax = 240, 400  #100 , 500
ymin, ymax = 180, 240  #100 , 300

# Setting of USB camera
cap = cv2.VideoCapture(0)  # 0+cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 10) # カメラFPS設定
width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # カメラ画像の横幅設定  1280pxel
height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # カメラ画像の縦幅設定  720pxel

# List setting

while True :
    ret , frame = cap.read()
    gray, bin, edges, contours = filter(frame, xmin, xmax, ymin, ymax)
    # fgmask = fgbg.apply(bin)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    "最小外接円"
    # contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    # contours = list(filter(lambda x : cv2.contourArea(x) > 100, contours))
    for i, cnt in enumerate(contours):
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow('output', frame)
    # cv2.imshow('fgmask', fgmask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()