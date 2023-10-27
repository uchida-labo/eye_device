import cv2
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

def update_threshold(value):
    thresh1 = cv2.getTrackbarPos('Threshold1', 'Threshold Adjust')
    thresh2 = cv2.getTrackbarPos('Threshold2', 'Threshold Adjust')
    thresh3 = cv2.getTrackbarPos('Threshold3', 'Threshold Adjust')
    thresh4 = cv2.getTrackbarPos('Threshold4', 'Threshold Adjust')
    mask = cv2.inRange(gray, thresh1, thresh2)
    pick = cv2.bitwise_and(gray, gray, mask = mask)
    edges_slidebar = (pick, thresh3, thresh4, True)

    cv2.imshow('Threshold Adjust', pick)
    cv2.imshow('Threshold Adjust', edges_slidebar)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

xmin, xmax = 240, 600
ymin, ymax = 50, 350

fig, ax = plt.subplots(figsize = (8, 6))

cv2.namedWindow('Threshold Adjust', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Threshold1', 'Threshold Adjust', 0, 255, update_threshold)
cv2.createTrackbar('Threshold2', 'Threshold Adjust', 0, 255, update_threshold)
cv2.createTrackbar('Threshold3', 'Threshold Adjust', 0, 255, update_threshold)
cv2.createTrackbar('Threshold4', 'Threshold Adjust', 0, 255, update_threshold)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gau = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gau[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    update_threshold(0)

    mask = cv2.inRange(gray, 24, 26)
    img_pickup = cv2.bitwise_and(gray, gray, mask = mask)

    # update_threshold_for_Canny(0)
    edges_pickup = cv2.Canny(img_pickup, 0, 130, True)
    kernel = np.ones((5, 5), np.uint8)
    edges_pickup_closing = cv2.morphologyEx(edges_pickup, cv2.MORPH_CLOSE, kernel = kernel)    
    contours_pickup = cv2.findContours(edges_pickup, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (xmin, ymin))

    bin = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(bin, 0, 130, True)
    edges_closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(edges_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (xmin, ymin))[0]


    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > h:
            if h < 180 and w > 100:
                rect_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                fill = cv2.fillPoly(frame, cnt, (255, 255, 0))
    # cv2.imshow('Rectangle frame', rect_frame)
    # cv2.imshow('Fill poly', fill)
    # cv2.imshow('Pick up binary', edges_pickup)
    # cv2.imshow('Pick up edges closing', edges_pickup_closing)

    # cv2.imshow('Frame', frame)
    # cv2.imshow('Binary', bin)
    # cv2.imshow('Egdes', edges)
    # cv2.imshow('Edges after closing', edges_closing)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
