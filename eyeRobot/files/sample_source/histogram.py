import cv2
import numpy as np
# Trimming area
xmin, xmax = 240, 400  #100 , 500
ymin, ymax = 180, 240  #100 , 300

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


capture = cv2.VideoCapture(2)
width = int(capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400))
height = int(capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400))
fps = capture.set(cv2.CAP_PROP_FPS, 15)


frame_past = None
frame_now = None

while True :
    ret, frame = capture.read()
    if not ret :
        break
    fil = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 250)
    edges = cv2.dilate(edges, kernel=np.ones((5, 5), np.uint8))
    cv2.imshow("Edges", edges)
    
    ptile = p_tile_threshold(fil, 0.08)
    # cv2.imshow("Binary", ptile)
    frame_past = frame_now
    frame_now = edges
    if frame_past is not None and frame_now is not None :
        diff = cv2.absdiff(frame_now, frame_past)
        cv2.imshow("Difference", diff)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imshow("Out", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

capture.release()
cv2.destroyAllWindows()
    


