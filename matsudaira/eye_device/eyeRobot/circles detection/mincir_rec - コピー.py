import cv2
import numpy as np

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

def _detect_iris(eye_img):
        # グレースケール化後、ガウシアンフィルタによる平滑化
        eye_img_gry = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye_img_gau = cv2.GaussianBlur(eye_img_gry, (5, 5), 0)

        # Pタイル法による2値化
        eye_img_thr = p_tile_threshold(eye_img_gau, 0.3)

        cv2.rectangle(eye_img_thr, (0, 0), (eye_img_thr.shape[1] - 1, eye_img_thr.shape[0] - 1), (255, 255, 255), 1)

        # 輪郭抽出
        contours, hierarchy = cv2.findContours(eye_img_thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭から最小外接円により虹彩を求める
        iris = {'center': (0, 0), 'radius': 0}
        for i, cnt in enumerate(contours):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            # 半径が大きすぎる場合、虹彩候補から除外
            if eye_img_thr.shape[0] < radius*0.8:
                # # 虹彩候補の描画
                cv2.circle(eye_img, center, radius, (255, 0, 0))
                continue

            # 最も半径が大きい円を虹彩と認定
            if iris['radius'] < radius:
                iris['center'] = center
                iris['radius'] = radius
                iris['num'] = i

        return iris

c = cv2.VideoCapture(2)              #カメラ映像の取り込み

c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
fps = c.set(cv2.CAP_PROP_FPS, 70)           # カメラFPSを60FPSに設定
width = c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
height = c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定
model = cv2.bgsegm.createBackgroundSubtractorMOG()
roi = np.zeros((height, width), dtype=np.uint8)
cx, cy = width // 2, height // 2
cv2.circle(roi, (cx, cy), 200, 255, -1)


# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret, frame = c.read()              # フレームを取得
    masked_frame = frame.copy()
    masked_frame[roi == 0] = 0
    fgmask = model.apply(masked_frame)
    cv2.circle(frame, (cx, cy), 200, (0, 0, 255), 2)
    cv2.circle(masked_frame, (cx, cy), 200, (0, 0, 255), 2)
    cv2.circle(fgmask, (cx, cy), 200, 255, 2)

    cv2.imshow("fgmask", frame)
    cv2.imshow("frame", fgmask)
    cv2.imshow("masked_frame", masked_frame)

    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 撮影用オブジェクトとウィンドウの解放
c.release()
cv2.destroyAllWindows()