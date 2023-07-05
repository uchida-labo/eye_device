import cv2
import numpy as np
import serial
from function import filter, drawcircle_and_getpoint, feature_point

# camera settings
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW) # mac:2  USB camera:0+cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 15) # FPS setting
width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # width setting  1280pxel
height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # height setting  720pxel

# trimming size setting
xmin, xmax = 240, 400
ymin, ymax = 180, 240

# Frame size
wm = 540
hm = 360

while True:
    # get numpy array
    # return value : bool value and image array
    bool, frame = cap.read()

    gray, bin, edges, contours = filter(frame)

    # enumerate contours (get index number and element)
    for i, cnt in enumerate(contours) :
        x_center, y_center, radius, white_ratio, black_ratio = feature_point(cnt, frame)

        # white area ratio < 30 , black area ratio < 70 [%]
        if white_ratio < 30 and black_ratio > 70:

            # 10 < radius < 17 [pixel]
            if radius < 17 and radius > 10:
                # circle drawing process
                out, x, y = drawcircle_and_getpoint(frame, x_center, y_center, radius, wm, hm, width, height)
                # PC⇔Arduino
                # d = str(x) + ':' + str(y) + ';'
                # ser = serial.Serial()
                # ser.port = "COM4"
                # ser.baudrate = 9600
                # ser.setDTR(False)
                # ser.open()
                # ser.write(bytes(d, 'utf-8'))
                # ser.close()

                # acquisition value confirmation
                print('( x , y ) = ', x , y )
                print('White Area [%] :', white_ratio)
                print('Black Area [%] :', black_ratio)

        break

    # rectangle drawing process
    cv2.rectangle(out, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    # displays
    cv2.imshow('output', out)
    cv2.imshow('edge', edges)
    cv2.imshow('bin', bin)

    # Break by pressing "q" key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# freeing shooting objects and windows
cap.release()
cv2.destroyAllWindows()