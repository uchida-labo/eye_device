import cv2
import math
import numpy as np
import time
t = time.time()

#fontType = cv2.FONT_HERSHEY_COMPLEX
#threshold = 100 #二値化に用いる閾値

# Trimming area
xmin, xmax = 270, 420  #250 , 400 # 640まで
ymin, ymax = 220, 320  #200 , 300 # 480まで

xrange = xmax - xmin
yrange = ymax - ymin

kernel3 = np.ones((3, 3), np.uint8)

# Setting of USB camera
cap = cv2.VideoCapture(0)  # 0+cv2.CAP_DSHOW
#fps = int(cap.get(cv2.CAP_PROP_FPS))
fps =15
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\kyorisokutei2\output.mp4', fourcc, fps, (xrange, yrange))
video2 = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\kyorisokutei2\output2.mp4', fourcc, fps, (xrange, yrange))
video_bin = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\kyorisokutei2\output_bin.mp4', fourcc, fps, (xrange, yrange))
video_edges = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\kyorisokutei2\edges_bin.mp4', fourcc, fps, (xrange, yrange))
video_reverse = cv2.VideoWriter(R'C:\Users\mkouk\Desktop\kyorisokutei2\reverse_bin.mp4', fourcc, fps, (xrange, yrange))
#　縦線を消すカーネル
kernel = np.array([[0, 0, -1],  
                    [0, 0, 1], 
                    [0, 0, 0],])

edges_trimming = None

ellipseW = None

while True :
    #frame取得
    ret, frame = cap.read()
    ret2, frame2 = cap.read()

    if not ret:
        break
    if not ret2:
        break

    fil = cv2.GaussianBlur(frame, (5, 5), 1)
    gray_img = cv2.cvtColor(fil[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY) #オブジェクトimg_blurを閾値threshold = 100で二値化しimg_binaryに代入
    edges = cv2.Canny(img_thresh, 100, 170)   # Set upper and lower thresholds 100, 170
    #contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (xmin, ymin))
    img_reverse = cv2.bitwise_not(img_thresh) #　白黒反転

    dst = cv2.filter2D(img_reverse, -1, kernel) #　カーネルで横線の輪郭抽出(まぶた)
    dilation_line = cv2.dilate(dst, kernel = kernel3, iterations = 1)
    cv2.imshow('dst', dst)

    lines2 = cv2.HoughLinesP(dilation_line, rho=1, theta=np.pi/360, threshold=10, minLineLength=30, maxLineGap=5) #　まぶた検出
    
    
    if lines2 is not  None: #　まぶた線　eyelid Eye Detection Area
        for line2 in lines2:
            x3, y3, x4, y4 = line2[0]
            xmax_c2 = x4
            ymin_c2 = y3
            EyeDA_ymin = y4-30  #Eye Detection Area
            EyeDA_ymax = y4+40
            EyeDA_xmin = x4-60
            EyeDA_xmax = x4-10
            #cv2.circle(frame, (x4+xmin, y4+ymin), 15, (255, 255, 255), thickness=-1)
            frame2 = cv2.line(frame2, (x3+xmin, y3+ymin), (x4+xmin, y4+ymin), (255, 0, 255), 2) #　まぶたの線
            cv2.rectangle(frame2, ( EyeDA_xmin+xmin, EyeDA_ymin+ymin), (EyeDA_xmax+xmin, EyeDA_ymax+ymin), (0, 255, 255), 2) #　瞳検出場所
            #edges_trimming = edges[EyeDA_ymin:EyeDA_ymax, EyeDA_xmin:EyeDA_xmax] #　瞳検出エリアトリミング
            edges_trimming = edges[EyeDA_ymin:EyeDA_ymax, EyeDA_xmin:EyeDA_xmax] #　瞳検出エリアトリミング

    else: #まぶたが検出されない限り、検出できたときの検出エリアで虹彩を検出する
        if edges_trimming is not None:

            #frame2 = cv2.line(frame2, (x3+xmin, y3+ymin), (x4+xmin, y4+ymin), (255, 0, 255), 2) #　まぶたの線
            #cv2.rectangle(frame2, ( EyeDA_xmin+xmin, EyeDA_ymin+ymin), (EyeDA_xmax+xmin, EyeDA_ymax+ymin), (0, 255, 255), 2) #　瞳検出場所
            edges_trimming = edges[EyeDA_ymin:EyeDA_ymax, EyeDA_xmin:EyeDA_xmax] #　瞳検出エリアトリミング
        else:
            continue
    

    #cv2.imshow("rinkaku", edges)
    cv2.imshow("bin", img_thresh)
    #cv2.drawContours(edge)
    
    #虹彩測定範囲の中で楕円フィッティングをしてる
    contours, hierarchy = cv2.findContours(edges_trimming, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (EyeDA_xmin + xmin,EyeDA_ymin + ymin))
    for i, cnt in enumerate(contours):
        #print(f"contour: {i}, number of points: {len(cnt)}")
        # 楕円フィッティング

        #円描写　#(x,y),radius = cv2.minEnclosingCircle(cnt)
        #cv2.circle(frame,(int(x),int(y)),int(radius),(0,0,255),3)

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # print(ellipse)
            # print(ellipse[0][0])

            
            ellipseW = ellipse[1][0]
            ellipseH = ellipse[1][1]
            angle = int(ellipse[2])
            #ellipseArea = ellipseW*ellipseH
            
            #print(ellipseW, ellipseH)

            #print(angle)
            if ellipseW >= 5 and ellipseW <= 20:
                if ellipseH >= 10 and ellipseH <= 70:
                    if (angle >= 0 and angle <=10) or (angle >= 350 and angle <=360): 
                # 楕円描画
                        cx = int(ellipse[0][0])
                        cy = int(ellipse[0][1])

                        #cv2.putText(resimg, str(i+1), (cx+3,cy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 1,cv2.LINE_AA)
                        #print(ellipseArea)
                        #cv2.imshow('resimg',resimg)
                        xmax_c = cx + 20 #　白黒比計測エリア
                        xmin_c = cx - 20
                        ymax_c = cy + 20
                        ymin_c = cy - 20
                        image_size = 40*40
                        

                        if image_size == 0:
                            break
        
                        trim_bin = img_thresh[ymin_c-ymin:ymax_c-ymin, xmin_c-xmin:xmax_c-xmin]
                        
                        #trim_bin = img_thresh[20:140, 20:340]
                        #cv2.imshow("trim", trim_bin)
                        
                        white = cv2.countNonZero(trim_bin)
                        #black = image_size - white
                        white_ratio = (white/image_size) * 100
                        #print(white_ratio)
                        #black_ratio = (black/image_size) * 100
                        if white_ratio < 90 and white_ratio > 80:
                            frame = cv2.ellipse(frame,ellipse,(0,0,255),2)
                            cv2.rectangle(frame, (xmin_c, ymin_c), (xmax_c, ymax_c), (0, 255, 0), 2)
                        #cv2.drawMarker(frame, (cx,cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    frame_trimming = frame[ymin:ymax, xmin:xmax]
    frame_trimming2 = frame2[ymin:ymax, xmin:xmax]                    
    
    #測定範囲(青枠)
    #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow('output', frame)
    cv2.imshow('output2', frame_trimming2)
    cv2.imshow('edges', edges)
    cv2.imshow('reverse', img_reverse)

    video.write(frame_trimming)
    video2.write(frame_trimming2)
    video_bin.write(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)) #2値化映像保存
    video_edges.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)) 
    video_reverse.write(cv2.cvtColor(img_reverse, cv2.COLOR_GRAY2BGR)) 

    c = time.time()
    if c - t >= 15 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()