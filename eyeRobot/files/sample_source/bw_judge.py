import cv2
import numpy as np
from matplotlib import pyplot as plt
import serial
import matplotlib.pyplot as plt

# Trimming area
xmin, xmax = 220, 420  #100 , 500
ymin, ymax = 180, 240  #100 , 300

# Setting of USB camera
cap = cv2.VideoCapture(0)  # 0+cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 15) # カメラFPS設定
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

# Function of getting points
def getpoint(xcenter, ycenter):
    wm, hm = 540, 360
    x = int((xcenter * wm) / 1000 * width)
    y = int((ycenter * hm) / 1000 * height)

    return x, y

def judge(frame, x, y, width, height):
    imagesize = width * height
    trim_frame = cv2.cvtColor(frame[y:y+height, x:x+width], cv2.COLOR_RGB2GRAY)
    ret_a, trim = cv2.threshold(trim_frame, 65, 255, cv2.THRESH_BINARY)
    white = cv2.countNonZero(trim)
    black = imagesize - white
    white_ratio = (white/imagesize) * 100
    black_ratio = (black/imagesize) * 100

    return white_ratio, black_ratio

while True :
    #frame取得
    ret , frame = cap.read()

    #GaussianFilter　平滑化
    filter = cv2.GaussianBlur(frame, (5, 5), 1)    

    #Grayscale変換
    gray = cv2.cvtColor(filter[ymin:ymax, xmin:xmax], cv2.COLOR_RGB2GRAY)  #trimmnig
    # gray = cv2.cvtColor(filter, cv2.COLOR_RGB2GRAY)
    ret2, bin = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
    "最小外接円"    
    #Canny法　edge検出
    edges = cv2.Canny(gray, 100, 170)  #C1-205では220 , 330
    # edgeを膨張させる(Dilation)  morphology変換
    # edges = cv2.dilate(edges, kernel=np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    for i, cnt in enumerate(contours) :
        center, radius = cv2.minEnclosingCircle(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        white_ratio, black_ratio = judge(frame, x, y, w, h)
        if white_ratio < 30 and black_ratio > 70:
            if radius < 17 and radius > 10:
                cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 255, 0), 4)
                # draw(frame, (center[0], center[1]), radius)
                intarray = np.asarray(center, dtype = int)
                x, y = getpoint(intarray[0], intarray[1])
                print('( x , y ) = ', x , y )
                print('White Area [%] :', white_ratio)
                print('Black Area [%] :', black_ratio)

        break
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    cv2.imshow('output', frame)
    cv2.imshow('edge', edges)
    cv2.imshow('bin', bin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()