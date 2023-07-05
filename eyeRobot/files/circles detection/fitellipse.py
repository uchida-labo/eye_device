# 楕円フィッティング

import cv2
import numpy as np

fontType = cv2.FONT_HERSHEY_COMPLEX     #中心点の座標を描画する際のフォント指定
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

c = cv2.VideoCapture(2)              #カメラ映像の取り込み

c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
c.set(cv2.CAP_PROP_FPS, 70)           # カメラFPSを60FPSに設定
c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

def draw(video, point, radius):
    npt = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(video, npt, r, (0, 0, 255), 2)

    cv2.drawMarker(video, npt, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret, frame = c.read()              # フレームを取得
    #frame = cv2.resize(frame, dsize=(640, 480))     #ハフ変換用にリサイズ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #グレースケール変換
    # ２値化
    ptyle = p_tile_threshold(gray, 0.1)
    bi = cv2.GaussianBlur(ptyle, (5, 5), 0)      #平滑化
    rev = cv2.bitwise_not(bi)
    # 輪郭取得
    contours, _ = cv2.findContours(rev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    for i, cnt in enumerate(contours):
        ellipse = cv2.fitEllipse(cnt)
        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])
        radius = int((ellipse[1][0] + ellipse[1][1])/4)
        cv2.ellipse(bi, ellipse, (0, 255, 0), 2)
        draw(frame, (cx, cy), radius)
        
    cv2.imshow('output', frame)

    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 撮影用オブジェクトとウィンドウの解放
c.release()
cv2.destroyAllWindows()