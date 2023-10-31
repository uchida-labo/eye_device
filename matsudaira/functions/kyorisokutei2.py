import cv2
import math
import numpy as np

fontType = cv2.FONT_HERSHEY_COMPLEX
threshold = 100 #二値化に用いる閾値

# Trimming area
xmin, xmax = 252, 600  #100 , 500
ymin, ymax = 150, 300  #100 , 300
w_meter = 100
h_meter = 100

# Setting of USB camera
cap = cv2.VideoCapture(2)  # 0+cv2.CAP_DSHOW
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True :
    #frame取得
    ret, frame = cap.read()
    if not ret:
        break
    
    fil = cv2.GaussianBlur(frame, (5, 5), 1)
    gray_img = cv2.cvtColor(fil[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(gray_img, 125, 255, cv2.THRESH_BINARY) #オブジェクトimg_blurを閾値threshold = 100で二値化しimg_binaryに代入
    edges = cv2.Canny(img_thresh, 100, 170)   # Set upper and lower thresholds 100, 170
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (xmin, ymin))


    #cv2.imshow("rinkaku", edges)
    cv2.imshow("2値化画像", img_thresh)
    #cv2.drawContours(edge)
    
    for i, cnt in enumerate(contours):
        #print(f"contour: {i}, number of points: {len(cnt)}")
        # 楕円フィッティング

        #円描写　#(x,y),radius = cv2.minEnclosingCircle(cnt)
        #cv2.circle(frame,(int(x),int(y)),int(radius),(0,0,255),3)

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # print(ellipse)
            # print(ellipse[0][0])

            cx = (ellipse[0][0])
            cy = (ellipse[0][1])
            
            cx = int(cx)
            cy = int(cy)
            ellipseW = (ellipse[1][0])
            ellipseH = (ellipse[1][1])
            angle = ellipse[2]
            ellipseArea = ellipseW*ellipseH
            #print(ellipseW, ellipseH)

            #print(angle)
            if ellipseW >= 5 and ellipseW <= 20:
                if ellipseH >= 20 and ellipseH <= 40:
                    if (angle >= 0 and angle <=10) or (angle >= 350 and angle <=360): 
                # 楕円描画
                        frame = cv2.ellipse(frame,ellipse,(255,0,0),2)
                        cv2.drawMarker(frame, (cx,cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                         #cv2.putText(resimg, str(i+1), (cx+3,cy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 1,cv2.LINE_AA)
                        #print(ellipseArea)
                        #cv2.imshow('resimg',resimg)
                        break
    
        
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imshow('output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()