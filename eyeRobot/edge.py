import cv2
import numpy as np

# Trimming area
xmin, xmax = 320, 403  #100 , 500
ymin, ymax = 50, 235  #100 , 300

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 


kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32)

def draw(video, point, radius):
    pts = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(video, pts, r, (100, 255, 0), 2)
    cv2.circle(video, pts, 2, (0,0,255), 3)     #中心点を描画

def getpoint(xcenter, ycenter):
    wm, hm = 540, 360
    x = int((xcenter * wm) / 1000 * width)
    y = int((ycenter * hm) / 1000 * height)

    return x, y

def judge_eye(eye_frame, center, radius):
    """
    Black-White area decision per measurement area for eye detection

    ・argument
    'frame':Camera image output screen

    ・return
    'white_eye' and 'black_eye':Area percentages of white and black, respectively
    """
    xs = int(center[0] - 320 - radius)
    xl = int(center[0] - 320 + radius)
    ys = int(center[1] - 250 - radius)
    yl = int(center[1] - 250 + radius)
    image_size = int(4 * (radius ** 2))
    # trim_gray = cv2.cvtColor(frame[ys:yl, xs:xl], cv2.COLOR_RGB2GRAY)
    retthresh, trim_bin = cv2.threshold(eye_frame[ys:yl, xs:xl], 65, 255, cv2.THRESH_BINARY)
    white = cv2.countNonZero(trim_bin)
    black = image_size - white
    white_eye = (white/image_size) * 100
    black_eye = (black/image_size) * 100

    return white_eye, black_eye

while True :
    #frame取得
    ret , frame = cap.read()
    # frame = frame[ymin:ymax, xmin:xmax]
    frame1 = cv2.filter2D(frame, -1, kernel)
    filter = cv2.GaussianBlur(frame1, (5, 5), 1)    
    gray = cv2.cvtColor(filter[ymin:ymax, xmin:xmax], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 150, 350)  #C1-205では220 , 330
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    for i, cnt in enumerate(contours):
        center, radius = cv2.minEnclosingCircle(cnt)
        if radius < 26 and radius > 20:
            xs = int(center[0] - xmin - radius)
            xl = int(center[0] - xmin + radius)
            ys = int(center[1] - ymin - radius)
            yl = int(center[1] - ymin + radius)
            
            if xs < 0 or ys < 0 or ((center[0] + radius) > xmax) or ((center[1] + radius) > ymax):
                break

            image_size = int(4 * (radius ** 2))
            eye_frame = gray[ys:yl, xs:xl]
            trim_bin = cv2.threshold(eye_frame, 30, 255, cv2.THRESH_BINARY)[1]
            white = cv2.countNonZero(trim_bin)
            black = image_size - white
            white_eye = (white/image_size) * 100
            black_eye = (black/image_size) * 100

            # if white_eye < 40 and black_eye > 60:
            draw(frame, (center[0], center[1]), radius)
            intarray = np.asarray(center, dtype = int)
            x, y = getpoint(intarray[0], intarray[1])

            xx = (146 - x)//2
            yy = (125 - y)//2
            z = 3
            if xx < 1 and xx > -1 and yy < 1 and yy > -1:
                z =  6            
            # print(( xx , yy ))
            print("white:", white_eye)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    cv2.imshow('output', frame)
    # cv2.imshow('save frame', frame1)
    cv2.imshow("edge", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()