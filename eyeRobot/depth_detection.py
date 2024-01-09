import cv2, time
import numpy as np

t = time.time()

threshold = 100 #二値化に用いる閾値

# Trimming area
xmin, xmax = 200, 640 #100 , 500 # 640まで 290,440
ymin, ymax = 220, 320  #100 , 300 # 480まで


# Setting of USB camera
cap = cv2.VideoCapture(0)  # 0+cv2.CAP_DSHOW

#　縦線を消すカーネル
kernel = np.array([[0, 0, -1],  
                    [0, 0, 1], 
                    [0, 0, 0]])

#　横線を消すカーネル
kernel2 = np.array([[0, 0, 0], 
                    [1, -1, 0], 
                    [0, 0, 0]])


kernel3 = np.ones((3, 3), np.uint8)

while True :

    #frame取得
    ret, frame = cap.read()
    
    if not ret: break

    copy_frame = frame.copy()

    gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gaussian[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1] #　2値化

    #もしここでエラー吐かれたら上の"cv2.THRESH_BINARY_INV"を"cv2.THRESH_BINARY"に変えて下のcv2.bitwise_notをコメントアウト→57行目にあるbinaryをreverseに変更
    # reverse = cv2.bitwise_not(binary) #　白黒反転
    
    mask = np.zeros(binary.shape, np.uint8) #黒い画面を作る

    #　白色のものを輪郭抽出
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    for i,cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 3000: # 1000~1500くらい
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1) # 面積の小さいものを表示させる

    eyelid = cv2.filter2D(mask, -1, kernel) #　カーネルで横線の輪郭抽出(まぶた)
    pupil = cv2.filter2D(mask, -1, kernel2) #　カーネルで縦線の輪郭抽出(黒目)

    #　上で抽出した横線を縦線抽出の映像にかぶせて
    contours2, hierarchy2 = cv2.findContours(eyelid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i2,cnt2 in enumerate(contours2):
        cv2.drawContours(pupil, [cnt2], -1, 0, thickness=-1) # 対象の塊を黒く塗りつぶす

    dilation_eyelid = cv2.dilate(eyelid, kernel = kernel3, iterations = 1)
    dilation_eyelid_reverse = cv2.bitwise_not(dilation_eyelid)
    iris_vertical = cv2.bitwise_and(pupil, dilation_eyelid_reverse)

    closing = cv2.morphologyEx(iris_vertical, cv2.MORPH_CLOSE, kernel3)

    eyelid_lines = cv2.HoughLinesP(eyelid, rho=1, theta=np.pi/360, threshold=10, minLineLength=30, maxLineGap=5) #　まぶた検出
    
    if eyelid_lines is not None: #　まぶた線　eyelid Eye Detection Area
        for eyelid_line in eyelid_lines:
            x3, y3, x4, y4 = eyelid_line[0]
            xmax_c2 = x4
            ymin_c2 = y3
            detec_area_ymin = y4-30  #Eye Detection Area xの差50 yの差70
            detec_area_ymax = y4+40
            detec_area_xmin = x4-60
            detec_area_xmax = x4-10
            grad = abs((y4-y3)/(x3-x4))
            if 0 < grad < 2:
                cv2.line(copy_frame, ((x3 + xmin), (y3 + ymin)), ((x4 + xmin), (y4 + ymin)), (255, 0, 255), 2) #　まぶたの線
                cv2.rectangle(copy_frame, ((detec_area_xmin + xmin), (detec_area_ymin + ymin)), ((detec_area_xmax + xmin), (detec_area_ymax + ymin)), (0, 255, 255), 2) #　瞳検出場所
                iris_trimmming = closing[detec_area_ymin:detec_area_ymax, detec_area_xmin:detec_area_xmax] #　瞳検出エリアトリミング
                lines = cv2.HoughLinesP(iris_trimmming, rho=1, theta=np.pi/360, threshold=10, minLineLength=10, maxLineGap=5) #　黒目線検出

    else: #まぶたが検出されない限り、検出できたときの検出エリアで虹彩を検出する
        if iris_trimmming is not None:

            iris_trimmming = closing[detec_area_ymin:detec_area_ymax, detec_area_xmin:detec_area_xmax] #　瞳検出エリアトリミング
            lines = cv2.HoughLinesP(iris_trimmming, rho=1, theta=np.pi/360, threshold=10, minLineLength=10, maxLineGap=5) #　黒目線検出
        else:
            continue   

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            differencex = abs(x2-x1)
            differencey = abs(y2-y1)
            xmiddle = int((x1 + x2)/2)
            ymiddle = int((y1 + y2)/2)
            xmax_c = xmiddle + 20 #　白黒比計測エリア
            xmin_c = xmiddle - 20
            ymax_c = ymiddle + 20
            ymin_c = ymiddle - 20

            trim_bin = binary[ymin_c+detec_area_ymin:ymax_c+detec_area_ymin, xmin_c+detec_area_xmin:xmax_c+detec_area_xmin] #　トリミングしたら、左上が座標(0,0)になる

            white_ratio = (cv2.countNonZero(trim_bin) / 1600) * 100

            if (differencex <= 10) and (differencey >= 20): #and (y1 > y2)
                if white_ratio < 75 and white_ratio > 55:
                    cv2.line(frame, (xmin+detec_area_xmin+x1, ymin+detec_area_ymin+y1), (xmin+detec_area_xmin+x2, ymin+detec_area_ymin+y2), (0, 0, 255), 3) #　瞳の線
                    cv2.rectangle(frame, ( xmin+detec_area_xmin+xmin_c, ymin+detec_area_ymin+ymin_c), (xmin+detec_area_xmin+xmax_c, ymin+detec_area_ymin+ymax_c), (0, 255, 0), 2)

                    break

    #　トリミングエリア指定
    frame_trimming = frame[ymin:ymax, xmin:xmax]
    frame_trimming2 = copy_frame[ymin:ymax, xmin:xmax]   

    #　検出範囲表示（青枠）    
    #frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow("frame", frame)

    c = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()