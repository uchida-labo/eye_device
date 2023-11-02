import cv2
import numpy as np
import serial
import time

# Trimming area
xmin, xmax = 252, 335  #100 , 500
ymin, ymax = 250, 435  #100 , 300

xs, xl = 270, 380
ys, yl = 110, 200

cap = cv2.VideoCapture(1)
savevideo = cv2.VideoCapture(0+cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 20) 
width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) 
height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\experiments.mp4', fourcc, 15, (70, 60))


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

def judge(frame, x, y, width, height):
    imagesize = width * height
    trim_frame = cv2.cvtColor(frame[y:y+height, x:x+width], cv2.COLOR_RGB2GRAY)
    ret_a, trim = cv2.threshold(trim_frame, 0, 255, cv2.THRESH_OTSU)
    white = cv2.countNonZero(trim)
    black = imagesize - white
    white_ratio = (white/imagesize) * 100
    black_ratio = (black/imagesize) * 100

    return white_ratio, black_ratio

while True :
    #frame取得
    ret , frame = cap.read()
    ret1, frame1 = savevideo.read()
    frame1 = frame1[ys:yl, xs:xl]
    rgbf = frame    #追記しました

    frame = cv2.filter2D(frame, -1, kernel)

    #GaussianFilter　平滑化
    filter = cv2.GaussianBlur(frame, (5, 5), 1)    

    #Grayscale変換
    gray = cv2.cvtColor(filter[ymin:ymax, xmin:xmax], cv2.COLOR_RGB2GRAY)  #trimmnig
    # gray = cv2.cvtColor(filter, cv2.COLOR_RGB2GRAY)

    "最小外接円"    
    #Canny法　edge検出
    edges = cv2.Canny(gray, 100, 250)  #C1-205では220 , 330
    # edgeを膨張させる(Dilaion)  morphology変換
    # edges = cv2.dilate(edges, kernel=np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))
    for i, cnt in enumerate(contours):
        center, radius = cv2.minEnclosingCircle(cnt)
        xp, yp, w, h = cv2.boundingRect(cnt)
        white_ratio, black_ratio = judge(frame, xp, yp, w, h)
        if white_ratio < 30 and black_ratio > 70:
            if radius < 25 and radius > 15:
                draw(frame, (center[0], center[1]), radius)
                intarray = np.asarray(center, dtype = int)
                x, y = getpoint(intarray[0], intarray[1])
                # (196, 131)
                # #################################################################################
                # BGRBox =  rgbf[y-8:y-5, x+2:x+5]    #写真の元の画像から色見ないといけないけどどれがそうだかわからん
                # b = BGRBox.T[0].flatten().mean()
                # g = BGRBox.T[1].flatten().mean()
                # r = BGRBox.T[2].flatten().mean()
                # print("B: %.2f" % (b))
                # print("G: %.2f" % (g))
                # print("R: %.2f" % (r))
                # ###################################################################################

                xx = (146 - x)//2
                yy = (125 - y)//2
                z = 3
                if xx < 1 and xx > -1 and yy < 1 and yy > -1:
                    z =  6            
                print(( xx , yy ))    
                #######################################################################################
                d = str(xx) + ':' + str(yy) + ';' + str(z) + '/'
                ser = serial.Serial()
                ser.port = "COM4"     #デバイスマネージャでArduinoのポート確認
                ser.baudrate = 9600   #Arduinoと合わせる
                ser.setDTR(False)     #DTRを常にLOWにしReset阻止
                ser.open()            #COMポートを開く
                ser.write(bytes(d, 'utf-8'))          #送りたい内容をバイト列で送信
                ser.close()           #COMポートを閉じる
                time.sleep(5)         #送受信両方の通信のためにディレイが必要。ならもうとりあえずこれでいいや
                if z == 6:
                    break
                #######################################################################################

                    

                # ser = serial.Serial("COM4", 9600, timeout = None)
                # while True:
                #     print('i')      
                #     xy = ser.read() 
                #     if xy == b'\x03':
                #         break
                # ser.close()
                
            
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    
    cv2.imshow('output', frame)
    # cv2.imshow('save frame', frame1)
    cv2.imshow("edge", edges)
    video.write(frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
savevideo.release()
video.release()
cv2.destroyAllWindows()
