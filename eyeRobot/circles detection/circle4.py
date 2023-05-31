import cv2
import numpy as np
from IPython import display
from matplotlib import pyplot as plt

def draw_contours(frame, contours, ax):
    #輪郭の点及び線を画像上に描画する。
    ax.imshow(frame)
    ax.set_axis_off()

    for i, cnt in enumerate(contours):
        # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
        cnt = cnt.squeeze(axis=1)
        # 輪郭の点同士を結ぶ線を描画する。
        ax.add_patch(plt.Polygon(cnt, color="b", fill=None, lw=2))
        # 輪郭の点を描画する。
        ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)


c = cv2.VideoCapture(2)              #カメラ映像の取り込み

c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
c.set(cv2.CAP_PROP_FPS, 70)           # カメラFPSを60FPSに設定
c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

while True:
    ret, frame = c.read()              # フレームを取得
    frame = cv2.resize(frame, dsize=(640, 480))     #ハフ変換用にリサイズ
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  #グレースケール変換
    gray = cv2.GaussianBlur(gray, (33, 33), 1)      #平滑化
    ret, bin = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)   #２値化
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_contours(frame, contours, ax)
    for cnt in contours:
        # 輪郭に外接する円を取得する。
        center, radius = cv2.minEnclosingCircle(cnt)
        # 描画する。
        ax.add_patch(plt.Circle(xy=center, radius=radius, color="g", fill=None, lw=2))

    cv2.imshow('output', frame)             # フレームを画面に表示
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


c.release()
cv2.destroyAllWindows()
