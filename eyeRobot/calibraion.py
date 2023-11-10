import cv2, time, statistics
import numpy as np

cap_cal = cv2.VideoCapture(0)
cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_cal.set(cv2.CAP_PROP_FPS, 30)
cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

avg_dif = None

x_list_dif = []
y_list_dif = []
w_list_dif = []
h_list_dif = []

x_list_eye = []
y_list_eye = []
w_list_eye = []
h_list_eye = []

base_time = time.time()

kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
kernel_hor /= 9

kernel = np.ones((3, 3), np.uint8)

def grayscale_convert(frame):
    """
    取得フレームをグレイスケールに変換

    ・引数
    'frame'：取得フレーム

    ・戻り値
    'gray'：グレイスケール変換後のフレーム
    """
    gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

    return gray

def matrix_append_dif(x, y, w, h):
    """
    フレーム間差分に外接する矩形の座標情報を行列に追加

    ・引数
    'x'：矩形のx座標
    'y'：矩形のy座標
    'w'：矩形の幅
    'h'：矩形の高さ

    ・戻り値
    なし（行列に要素を追加するだけ）
    """
    x_list_dif.append(x)
    y_list_dif.append(y)
    w_list_dif.append(w)
    h_list_dif.append(h)

def matrix_append_msk(x, y, w, h):
    """
    マスク画像により抽出した眼領域に外接する矩形の座標情報を行列に追加

    ・引数
    'x'：矩形のx座標
    'y'：矩形のy座標
    'w'：矩形の幅
    'h'：矩形の高さ

    ・戻り値
    なし（行列に要素を追加するだけ）
    """
    x_list_eye.append(x)
    y_list_eye.append(y)
    w_list_eye.append(w)
    h_list_eye.append(h)


# def recatngle_average(xlist_type, ylist_type, wlist_type, hlist_tye):
#     ave_x = statistics.mean(xlist_type)
#     ave_y = statistics.mean(ylist_type)
#     ave_w = statistics.mean(wlist_type)
#     ave_h = statistics.mean(hlist_tye)

#     xmin_cal = int(ave_x -20)
#     xmax_cal = int(ave_x + ave_w + 20)
#     ymin_cal = int(ave_y - 20)
#     ymax_cal = int(ave_y + ave_h + 20)

def Frame_difference(gray_frame, avg_frame):
    """
    フレーム間差分を計算して瞼の移動領域に外接する矩形を検出

    ・引数
    'gray_frame'：グレイスケール変換後のフレーム
    'avg_frame'：計算用フレーム（比較用）
    """
    cv2.accumulateWeighted(gray_frame, avg_frame, 0.8)
    delta_dif = cv2.absdiff(gray_frame, cv2.convertScaleAbs(avg_frame))
    bin_dif = cv2.threshold(delta_dif, 3, 255, cv2.THRESH_BINARY)[1]
    edges_dif = cv2.Canny(bin_dif, 0, 130)
    contours_dif = cv2.findContours(edges_dif, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i0, cnt0 in enumerate(contours_dif):
        x_dif, y_dif, w_dif, h_dif = cv2.boundingRect(cnt0)
        area_dif = w_dif * h_dif
        if w_dif > h_dif and area_dif > 30000:
            matrix_append_dif(x_dif, y_dif, w_dif, h_dif)

def Mask_process(gray_frame):
    """
    マスク画像により眼領域を抽出し，その領域に外接する矩形を検出

    ・引数
    'gray_frame'：グレイスケール変換後のフレーム
    """
    mask = cv2.inRange(gray_frame, 30, 70)
    pick_msk = cv2.bitwise_and(gray_frame, gray_frame, mask = mask)
    bin_msk = cv2.threshold(pick_msk, 0, 255, cv2.THRESH_BINARY_INV)[1]
    edges_msk = cv2.Canny(bin_msk, 0, 130)
    contours_msk = cv2.findContours(edges_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i1, cnt1 in enumerate(contours_msk):
        x_msk, y_msk, w_msk, h_msk = cv2.boundingRect(cnt1)
        area_msk = w_msk * h_msk
        if w_msk > h_msk and area_msk > 30000:
            matrix_append_msk(x_msk, y_msk, w_msk, h_msk)

def Eyelids_line(gray_frame):
    """
    瞼の線を検出して瞼の傾きを検出

    ・引数
    'gray_frame'：グレイスケール変換後のフレーム
    """
    bin_line = cv2.threshold(gray_frame, 70, 255, cv2.THRESH_BINARY)[1]
    horizon_line = cv2.filter2D(bin_line, -1, kernel = kernel_hor)
    dilation_line = cv2.dilate(horizon_line, kernel = kernel, iterations = 1)
    opening_line = cv2.morphologyEx(dilation_line, cv2.MORPH_OPEN, kernel = kernel)
    closing_line = cv2.morphologyEx(opening_line, cv2.MORPH_CLOSE, kernel = kernel)
    lines = cv2.HoughLinesP(closing_line, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 100, maxLineGap = 10)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            if x1 < 500 and y1 < 200:
                delta_Y = y0 - y1

def Run_time():
    """
    時間計測

    ・戻り値
    'run_time'：プログラム走査時間
    """
    end_time = time.time()
    run_time = end_time - base_time

    return run_time

def main():
    """
    メイン関数
    """
    while True:
        ret, frame_cal = cap_cal.read()
        if not ret:
            break

        gray_cal = grayscale_convert(frame = frame_cal)
        Frame_difference(gray_frame = gray_cal, avg_frame = avg_dif)
        Mask_process(gray_frame = gray_cal)
        Eyelids_line(gray_frame = gray_cal)
        run_time = Run_time()

        if run_time > 20:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap_cal.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
