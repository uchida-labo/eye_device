import cv2
import math
import numpy as np
#import openpyxl
#import pandas as pd
import xlwings as xw

wb = xw.Book()
mylist = [] #excelに追加するリスト

threshold = 100 #二値化に用いる閾値

# Trimming area
xmin, xmax = 50, 500  #100 , 500
ymin, ymax = 150, 300  #100 , 300


# Setting of USB camera
cap = cv2.VideoCapture(2)  # 0+cv2.CAP_DSHOW
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(R'C:\Users\mkouk\OneDrive\デスクトップ\video\output.mp4', fourcc, fps, (w, h))
video_bin = cv2.VideoWriter(R'C:\Users\mkouk\OneDrive\デスクトップ\video\output_bin.mp4', fourcc, fps, (450, 150))

#右からカメラで見たときの黒目の直線部分の輪郭を強調するフィルター
kernel = np.array([[0, 0, 0], 
                    [0, -1, 1], 
                    [0, 0, 0],])

kernel2 = np.array([[0, 0, 0], 
                    [0, 1, -1], 
                    [0, 0, 0],])


kernel3 = np.ones((3, 3), np.uint8)
#kernel3 = np.ones((5,5),np.uint8)

while True :
    #frame取得
    ret, frame = cap.read()
    if not ret:
        break
    
    fil = cv2.GaussianBlur(frame, (5, 5), 1)
    gray_img = cv2.cvtColor(fil[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(gray_img, 70, 255, cv2.THRESH_BINARY) #オブジェクトimg_blurを閾値threshold = 100で二値化しimg_binaryに代入
    img_reverse = cv2.bitwise_not(img_thresh)
    #edges = cv2.Canny(img_thresh, 100, 170)   # Set upper and lower thresholds 100, 170
    
    mask = np.zeros(img_thresh.shape, np.uint8)
    contours, hierarchy = cv2.findContours(img_reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 600: # 塗りつぶし対象を塊の大きさで判別。
            img_result = cv2.drawContours(mask, [cnt], -1, 255, thickness=-1) # 対象の塊を黒く塗りつぶす
            
    dst = cv2.filter2D(img_result, -1, kernel) #カーネルで輪郭抽出
    #dst2 = cv2.filter2D(img_result, -1, kernel2)
    dilation_line = cv2.dilate(dst, kernel = kernel3, iterations = 1)
    img_mask = cv2.medianBlur(dilation_line, ksize = 5)
    dst2 = cv2.filter2D(dilation_line, -1, kernel2)
    #img_mask2 = cv2.medianBlur(dst, ksize = 3)
    lines = cv2.HoughLinesP(img_mask, rho=1, theta=np.pi/360, threshold=10, minLineLength=10, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            
            #tan = (y2-y1)/(x2-x1)
            #theta = int(math.degrees(math.atan(tan)))
            differencex = abs(x2-x1)
            differencey = abs(y2-y1)
            
            xmiddle = int((x1 + x2)/2)
            ymiddle = int((y1 + y2)/2)
            xmax_c = xmiddle + 20
            xmin_c = xmiddle - 20
            ymax_c = ymiddle + 20
            ymin_c = ymiddle - 20
            image_size = 40*40

            trim_bin = img_thresh[ymin_c:ymax_c, xmin_c:xmax_c]
            white = cv2.countNonZero(trim_bin)
            black = image_size - white
            white_ratio = (white/image_size) * 100
            black_ratio = (black/image_size) * 100
            
            #if white_ratio < 30 and black_ratio > 70:
            if (differencex <= 10) and (differencey >= 20)and (y1 > y2):
                if white_ratio < 95 and white_ratio > 80:
                    frame = cv2.line(frame, (x1+xmin, y1+ymin), (x2+xmin, y2+ymin), (0, 0, 255), 3)
                    cv2.rectangle(frame, ( xmin_c+xmin, ymin_c+ymin), (xmax_c+xmin, ymax_c+ymin), (0, 255, 0), 2)

                    #瞳の検出座標(x)をリストに追加していく
                    mylist.append(xmiddle)

                    #print(white_ratio)    
                    break
            #cv2.line(frame, (x1+xmin, y1+ymin), (x2+xmin, y2+ymin), (0, 0, 255), 3)
            

    #cv2.imshow("rinkaku", edges)
    cv2.imshow("img_result", img_result)
    #cv2.imshow("img_mask", img_mask)
    #cv2.imshow("img_reverse", img_reverse)
    cv2.imshow("dst", dst)
    cv2.imshow("dst2", dst2)
    #cv2.drawContours(edge)
        
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    #cv2.rectangle(frame, ( xmin_c+xmin, ymin_c+ymin), (xmax_c+xmin, ymax_c+ymin), (0, 255, 0), 2)
    cv2.imshow('output', frame)

    #映像保存
    video.write(frame)
    video_bin.write(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        xw.Range("B2", transpose=True).value = mylist
        break

cap.release()
video.release()
video_bin.release()
cv2.destroyAllWindows()