import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

#trimming
xmin, xmax = 240, 400  #100 , 500
ymin, ymax = 180, 240  #100 , 300

cap = cv2.VideoCapture(2)  # 0+cv2.CAP_DSHOW
fontType = cv2.FONT_HERSHEY_COMPLEX

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# カメラFPS設定
cap.set(cv2.CAP_PROP_FPS, 200) 

# カメラ画像の横幅設定  1280pxel
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) 

# カメラ画像の縦幅設定  720pxel
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 

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

def draw(video, point, radius):
    pts = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(video, pts, r, (100, 255, 0), 2)
    cv2.circle(video, pts, 2, (0,0,255), 3)     #中心点を描画

while True :
    #frame取得
    ret , frame = cap.read()

    #GaussianFilter　平滑化
    filter = cv2.GaussianBlur(frame, (5, 5), 1)    

    #Grayscale変換
    gray = cv2.cvtColor(filter[ymin:ymax, xmin:xmax], cv2.COLOR_RGB2GRAY)  #trimmnig
    # gray = cv2.cvtColor(filter, cv2.COLOR_RGB2GRAY)

    "最小外接円"    
    #Canny法　edge検出
    edges = cv2.Canny(gray, 120, 210)  #C1-205では220 , 330
    # edgeを膨張させる(Dilation)  morphology変換
    # edges = cv2.dilate(edges, kernel=np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    for i, cnt in enumerate(contours):
        center, radius = cv2.minEnclosingCircle(cnt)
        if radius < 14 and radius > 8:
            draw(frame, (center[0], center[1]), radius)
            cv2.putText(frame, "iris", (50, 50), fontType, 1, (0, 255, 0),3)
        
            cv2.putText(frame, "blink", (50, 50), fontType, 1, (255, 0, 0),3)
    
    


    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    
    cv2.imshow('output', frame)
    cv2.imshow('edges', edges)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()

# edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel=np.ones((11, 11), np.uint8))
# plt.subplot(121),plt.imshow(frame,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

