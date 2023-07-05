import cv2
from function import filter
import matplotlib.pyplot as plt
import time
def judge_square(trim_frame, xmin, xmax, ymin, ymax):
    video_size = (xmax - xmin)*(ymax - ymin)
    white = cv2.countNonZero(trim_frame)
    black = video_size - white
    white_ratio = (white/video_size) * 100
    black_ratio = (black/video_size) * 100

    return white_ratio, black_ratio

fontType = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 15) # FPS setting
width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # width setting  1280pxel
height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # height setting  720pxel

# trimming size setting
xmin, xmax = 240, 400
ymin, ymax = 180, 240

avg = None

white_list = []
val_list = []
val = 0

while True:
    
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break

    gray, bin, edges, contours = filter(frame, xmin, xmax, ymin, ymax)
    
    # 比較用のフレームを取得する
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.6)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    white_ratio, black_ratio = judge_square(thresh, xmin, xmax, ymin, ymax)
    # 画像の閾値に輪郭線を入れる
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    if white_ratio < 22 and white_ratio > 12:
        cv2.putText(frame, 'Blink!', (260, 150), fontType, 1, (0, 0, 255), 3)
        # time.sleep(0.1)
    
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    # 結果を出力
    cv2.imshow("Frame", frame)
    cv2.imshow('thresh', thresh)
    
    print('white[%]', white_ratio)
    print('black[%]', black_ratio)
    # print('list', whiteratio_list)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end = time.time()
    white_list.append(white_ratio)
    val_list.append(val)
    val += 1



plt.plot(val_list, white_list)
plt.show()
cap.release()
cv2.destroyAllWindows()