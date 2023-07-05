# FitEllipse

import cv2
import numpy as np

fontType = cv2.FONT_HERSHEY_COMPLEX
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
    ret, img_thr = cv2.threshold(img_gry, p_tile_thr, 255, cv2.THRESH_BINARY)

    return img_thr

# def draw(img, point, radius):
#     npt = (int(point[0]), int(point[1]))
#     r = int(radius)
    #cv2.circle(img, npt, r, (0, 0, 255), 2)
    #cv2.drawMarker(img, npt, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

img = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #グレースケール変換
ptyle = p_tile_threshold(gray, 0.01)
fil = cv2.GaussianBlur(ptyle, (33, 33), 3)      #平滑化
rev = cv2.bitwise_not(fil)
# 輪郭取得
contours, _ = cv2.findContours(rev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
        ellipse = cv2.fitEllipse(cnt)
        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])
        radius = int((ellipse[1][0] + ellipse[1][1])/4)
        cv2.ellipse(img, ellipse, (0, 0, 255), 2)
        

cv2.imwrite('fit_bi.png', img) 