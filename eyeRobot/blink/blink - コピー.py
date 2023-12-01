import cv2
from cv2 import bitwise_not
import numpy as np
from matplotlib import pyplot as plt

# Trimming　area
xmin, xmax = 240, 400  #100 , 500
ymin, ymax = 180, 240  #100 , 300

# Setting of USB camera
cap = cv2.VideoCapture(2)  # 0+cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 100) # カメラFPS設定
width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # カメラ画像の横幅設定  1280pxel
height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # カメラ画像の縦幅設定  720pxel

# Function of P-tile threshold 
def p_tile_threshold(img_gry, per):
    """
    Pタイル法による2値化処理
    :param img_gry: 2値化対象のグレースケール画像
    :param per: 2値化対象が画像で占める割合
    :return img_thr: 2値化した画像
    """

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

# Function of drawing circle
def draw(video, point, radius):
    pts = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(video, pts, r, (100, 255, 0), 2)
    cv2.circle(video, pts, 2, (0,0,255), 3)     #中心点を描画

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))

frame_past = None
frame_now = None

while True :
    ret , frame = cap.read()
    #GaussianFilter　
    filter = cv2.GaussianBlur(frame, (5, 5), 0)    
    #Grayscale変換
    gray = cv2.cvtColor(filter[ymin:ymax, xmin:xmax], cv2.COLOR_RGB2GRAY)
    ret2, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    fgmask = fgbg.apply(frame[ymin:ymax, xmin:xmax])
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    "最小外接円"
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    # contours = list(filter(lambda x : cv2.contourArea(x) > 100, contours))
    for i, cnt in enumerate(contours):
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow('output', frame)
    cv2.imshow('fgmask', fgmask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()