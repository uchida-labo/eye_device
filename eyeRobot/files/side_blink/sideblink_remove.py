import cv2
import matplotlib.pyplot as plt
from IPython import display
import numpy as np



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

xmin, xmax = 240, 600
ymin, ymax = 0, 200

maskimg = cv2.imread(R'C:\Users\admin\Desktop\data\side_binary\mask.jpg', cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cutframe = frame[ymin:ymax, xmin:xmax]
    dst = cv2.bitwise_or(cutframe, maskimg)
    gau = cv2.GaussianBlur(dst, (5, 5), 1)
    gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

    bin = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    reverse = cv2.bitwise_not(bin)
    edges = cv2.Canny(bin, 0, 130, True)
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (xmin, ymin))[0]

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > h:
            if h < 180 and w > 100:
                # rect_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                fill = cv2.fillPoly(frame, cnt, (255, 255, 0))
    # cv2.imshow('Rectangle frame', rect_frame)
    # cv2.imshow('Fill poly', fill)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    # cv2.imshow('Binary', bin)
    cv2.imshow('Cut frame', dst)
    cv2.imshow('Reverse', reverse)
    # cv2.imshow('Egdes', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
