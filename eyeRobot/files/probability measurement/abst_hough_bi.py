# Binary

import numpy as np
import cv2

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


fontType = cv2.FONT_HERSHEY_COMPLEX
img = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #グレースケール変換
ptyle = p_tile_threshold(gray, 0.15)
fil = cv2.GaussianBlur(ptyle, (33, 33), 1)      #平滑化
# HoughCircle
circle = cv2.HoughCircles(fil, cv2.HOUGH_GRADIENT, 1.11, 100, param1=100, param2=47, minRadius=0, maxRadius=0)
circle = np.uint16(np.around(circle))
for i in circle[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),3)     #円周を描画する
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)        #中心点を描画する(単位：[ピクセル]、原点は左上)
    #pts = np.array((int(i[0]), int(i[1])))
    #cv2.putText(img, '{}'.format(pts), (i[0]+5, i[1]+5), fontType, 1, (0, 255, 0), 2)

cv2.imwrite('hough_bi.png', img)

