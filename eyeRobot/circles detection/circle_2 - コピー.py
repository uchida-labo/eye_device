import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import squeeze

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 70)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

while True:
    ret, frame = cap.read()              # フレームを取得
    frame = cv2.resize(frame, dsize=(640, 480))     #ハフ変換用にリサイズ
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  #グレースケール変換
    gray = cv2.GaussianBlur(gray, (33, 33), 1)      #平滑化
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, 80, 100)

    if circles is not None:
        for cx, cy, r in circles.squeeze(axis = 0).astype(int):
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 2)
    cv2.imshow('output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
