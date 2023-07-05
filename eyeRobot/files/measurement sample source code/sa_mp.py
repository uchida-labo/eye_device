import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw(video, point, radius):
    pts = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(video, pts, r, (100, 255, 0), 2)

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

# Video Reader を作成
cap = cv2.VideoCapture(2)
width = int(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280))
height = int(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720))
fps = cap.set(cv2.CAP_PROP_FPS, 70)

xmin, xmax = 450, 800
ymin, ymax = 200, 300

prev_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if prev_frame is not None:
        # BGR -> grayscale
        prev_gray = cv2.cvtColor(prev_frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
        # 差分を計算
        diff = cv2.absdiff(gray, prev_gray)
        # 閾値処理
        ptile = p_tile_threshold(diff, 0.8)
        #rev = cv2.bitwise_not(ptile)
        #  輪郭抽出
        contours, hierarchy = cv2.findContours(ptile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,offset=(xmin, ymin))
        # 面積でフィルタリング
        contours = list(filter(lambda cnt: cv2.contourArea(cnt) > 450, contours))
        # 輪郭を囲む長方形に変換
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        # 長方形を描画する。
        bgr = frame.copy()
        for x, y, width, height in rects:
            #cv2.rectangle(bgr, (x, y), (x + width, y + height), (255, 0, 255), 2)
            for i, cnt in enumerate(contours):
                center, radius = cv2.minEnclosingCircle(cnt)
                if radius < 20:
                    draw(bgr, (center[0], center[1]), radius)

        cv2.rectangle(bgr, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.imshow('output', bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    prev_frame = frame


cap.release()
cv2.destroyAllWindows()