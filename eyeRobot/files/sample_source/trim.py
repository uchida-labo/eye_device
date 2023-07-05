import cv2
from cv2 import bitwise_not
import numpy as np
from matplotlib.pyplot import xlim
from IPython import display

# Pタイル法
def p_tile_threshold(img_gry, per):  #img_gry → 2値化対象のグレースケール画,per → 2値化対象が画像で占める割合,return → img_thr: 2値化した画像
    # ヒストグラム取得
    img_hist = cv2.calcHist([img_gry], [0], None, [256], [0, 256])

    # 2値化対象が画像で占める割合から画素数を計算
    all_pic = img_gry.shape[0] * img_gry.shape[1]
    pic_per = all_pic * per

    # Pタイル法による2値化のしきい値計算
    p_tile_thr = 0
    pic_sum = 0

    # 現在の輝度と輝度の合計(高い値順に足す)の計算
    for hist in img_hist:
        pic_sum += hist

        # 輝度の合計が定めた割合を超えた場合処理終了
        if pic_sum > pic_per:
            break

        p_tile_thr += 1

    # Pタイル法によって取得したしきい値で2値化処理
    ret, img_thr = cv2.threshold(img_gry, p_tile_thr, 200, cv2.THRESH_BINARY)
    return img_thr

# 描画設定
def draw(video, point, radius):
    pts = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(video, pts, r, (100, 255, 0), 2)
    cv2.circle(video, pts, 2, (0,0,255), 3)     #中心点を描画

# カメラ設定
capture = cv2.VideoCapture(2)   #USBカメラ接続時は → 0+cv2.CAP_DSHOW
width = int(capture.set(cv2.CAP_PROP_FRAME_WIDTH, 540))
height = int(capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360))
fps = capture.set(cv2.CAP_PROP_FPS, 15)

# 切り取り領域設定
xmin, xmax = 240, 320  #100 , 500
ymin, ymax = 180, 240  #100 , 300

# 背景差分設定
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))

while True :
    ret, frame = capture.read()
    if not ret :
        break

    gray = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    ptile = p_tile_threshold(gray, 0.3)
    # rev = bitwise_not(ptile)
    # rev = cv2.resize(rev, (700, 200))
    fgmask = fgbg.apply(ptile)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    contours = list(filter(lambda x : cv2.contourArea(x) > 100, contours))

    for i, cnt in enumerate(contours):
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
        center, radius = cv2.minEnclosingCircle(cnt)
        # if radius < 20:
        #     draw(frame, (center[0], center[1]), radius)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    

    cv2.imshow('output', frame)
    # cv2.imshow('gray', gray)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()
