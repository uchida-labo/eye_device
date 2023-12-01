import cv2
import math
import numpy as np
#import openpyxl
#import pandas as pd
import xlwings as xw

#excel起動
#wb = xw.Book()
#mylist = [] #excelに追加するリスト

threshold = 100 #二値化に用いる閾値

# Trimming area
xmin, xmax = 50, 500  #100 , 500 # 640まで
ymin, ymax = 150, 300  #100 , 300 # 480まで

# Setting of USB camera
cap = cv2.VideoCapture(0)  # 0+cv2.CAP_DSHOW
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\VSCode_video\output.mp4', fourcc, fps, (450, 150))
video_bin = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\VSCode_video\output_bin.mp4', fourcc, fps, (450, 150))
video_img_result = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\VSCode_video\img_result.mp4', fourcc, fps, (450, 150))
video_dst = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\VSCode_video\dst.mp4', fourcc, fps, (450, 150))
video_dst3 = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\VSCode_video\dst3.mp4', fourcc, fps, (450, 150))
video_erosion1 = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\VSCode_video\erosion1.mp4', fourcc, fps, (450, 150))

#　縦線を消すカーネル
kernel = np.array([[0, 0, -1],  
                    [0, 0, 1], 
                    [0, 0, 0],])

#　横線を消すカーネル
kernel2 = np.array([[0, 0, 0], 
                    [0, -1, 1], 
                    [0, 0, 0],])


kernel3 = np.ones((3, 3), np.uint8)
#kernel3 = np.ones((5,5),np.uint8)

lines = None # linesの変数を定義
img_result = None

while True :

    #frame取得
    ret, frame = cap.read()
    if not ret:
        break
    
    fil = cv2.GaussianBlur(frame, (5, 5), 1)
    gray_img = cv2.cvtColor(fil[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY) #　2値化
    img_reverse = cv2.bitwise_not(img_thresh) #　白黒反転
    #edges = cv2.Canny(img_thresh, 100, 170)   # Set upper and lower thresholds 100, 170
    
    mask = np.zeros(img_thresh.shape, np.uint8) #黒い画面を作る
    # mask2 = np.zeros(img_thresh.shape, np.uint8) #黒い画面を作る

    #　白色のものを輪郭抽出
    contours, hierarchy = cv2.findContours(img_reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    for i,cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 1500: # 1000~1500くらい
            img_result = cv2.drawContours(mask, [cnt], -1, 255, thickness=-1) # 面積の小さいものを表示させる
    

    dst = cv2.filter2D(img_result, -1, kernel) #　カーネルで横線の輪郭抽出(まぶた)
    dst2 = cv2.filter2D(img_result, -1, kernel2) #　カーネルで縦線の輪郭抽出(黒目)
    #dilation_line2 = cv2.dilate(dst, kernel = kernel3, iterations = 1) #　線を太くする。

    #　上で抽出した横線を縦線抽出の映像にかぶせて
    contours2, hierarchy2 = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i2,cnt2 in enumerate(contours2):
        dst3 = cv2.drawContours(dst2, [cnt2], -1, 0, thickness=-1) # 対象の塊を黒く塗りつぶす

    dilation_line = cv2.dilate(dst3, kernel = kernel3, iterations = 1) #　線を太くする。dst2もdst3も一緒の映像が入ってる
    erosion1 = cv2.erode(dilation_line,kernel3,iterations = 1) #　線を細くする。
    #img_mask = cv2.medianBlur(erosion1, ksize = 3)
    
    
    lines2 = cv2.HoughLinesP(dst, rho=1, theta=np.pi/360, threshold=10, minLineLength=30, maxLineGap=5) #　まぶた検出
    
    
    if lines2 is not None: #　まぶた線　eyelid Eye Detection Area
        for line2 in lines2:
            x3, y3, x4, y4 = line2[0]
            xmax_c2 = x4
            ymin_c2 = y3
            EyeDA_ymin = y4-10  #Eye Detection Area
            EyeDA_ymax = y4+60
            EyeDA_xmin = x4-30
            EyeDA_xmax = x4+20
            #cv2.circle(frame, (x4+xmin, y4+ymin), 15, (255, 255, 255), thickness=-1)
            frame = cv2.line(frame, (x3+xmin, y3+ymin), (x4+xmin, y4+ymin), (255, 0, 255), 3) #　まぶたの線
            cv2.rectangle(frame, ( EyeDA_xmin+xmin, EyeDA_ymin+ymin), (EyeDA_xmax+xmin, EyeDA_ymax+ymin), (0, 255, 255), 2) #　瞳検出場所
            erosion1_trimming = erosion1[EyeDA_ymin:EyeDA_ymax, EyeDA_xmin:EyeDA_xmax] #　瞳検出エリアトリミング
            lines = cv2.HoughLinesP(erosion1_trimming, rho=1, theta=np.pi/360, threshold=10, minLineLength=10, maxLineGap=5) #　黒目線検出
            

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            #tan = (y2-y1)/(x2-x1)
            #theta = int(math.degrees(math.atan(tan)))
            differencex = abs(x2-x1)
            differencey = abs(y2-y1)
            xmiddle = int((x1 + x2)/2)
            ymiddle = int((y1 + y2)/2)
            xmax_c = xmiddle + 20 #　白黒比計測エリア
            xmin_c = xmiddle - 20
            ymax_c = ymiddle + 20
            ymin_c = ymiddle - 20
            image_size = 40*40

            trim_bin = img_thresh[ymin_c+EyeDA_ymin:ymax_c+EyeDA_ymin, xmin_c+EyeDA_xmin:xmax_c+EyeDA_xmin] #　トリミングしたら、左上が座標(0,0)になる

            white = cv2.countNonZero(trim_bin)
            black = image_size - white
            white_ratio = (white/image_size) * 100
            black_ratio = (black/image_size) * 100
            

            if (differencex <= 5) and (differencey >= 10): #and (y1 > y2)
                if white_ratio < 80 and white_ratio > 65:
                    #frame = cv2.line(frame, (x1+xmin, y1+ymin), (x2+xmin, y2+ymin), (0, 0, 255), 3)
                    #cv2.rectangle(frame, ( xmin_c+xmin, ymin_c+ymin), (xmax_c+xmin, ymax_c+ymin), (0, 255, 0), 2)

                    frame = cv2.line(frame, (xmin+EyeDA_xmin+x1, ymin+EyeDA_ymin+y1), (xmin+EyeDA_xmin+x2, ymin+EyeDA_ymin+y2), (0, 0, 255), 3) #　瞳の線
                    #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.rectangle(frame, ( xmin+EyeDA_xmin+xmin_c, ymin+EyeDA_ymin+ymin_c), (xmin+EyeDA_xmin+xmax_c, ymin+EyeDA_ymin+ymax_c), (0, 255, 0), 2)

                    #瞳の検出座標(x)をリストに追加していく
                    #mylist.append(xmiddle)

                    #print(white_ratio)  
                    #print(xmiddle)
                    break

    #　トリミングエリア指定
    frame_trimming = frame[ymin:ymax, xmin:xmax]   
    #　検出映像表示
    cv2.imshow('output', frame_trimming)
    #　検出範囲表示（青枠）    
    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow("img_result", img_result)
    cv2.imshow("dst", dst)
    cv2.imshow("dst3", dst3)
    cv2.imshow("erosion1", erosion1)
    cv2.imshow("dilation_line", dilation_line)
    cv2.imshow("frame", frame)
    
    

    #　映像保存
    video.write(frame_trimming)
    video_bin.write(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)) #2値化映像保存
    video_img_result.write(cv2.cvtColor(img_result, cv2.COLOR_GRAY2BGR))
    video_dst.write(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR))
    video_dst3.write(cv2.cvtColor(dst3, cv2.COLOR_GRAY2BGR))
    video_erosion1.write(cv2.cvtColor(erosion1, cv2.COLOR_GRAY2BGR))
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #xw.Range("B2", transpose=True).value = mylist
        break

cap.release()
video.release()
video_bin.release()
cv2.destroyAllWindows()