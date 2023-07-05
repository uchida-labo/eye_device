# MinEnclosingCircle

import cv2
import numpy as np

fontType = cv2.FONT_HERSHEY_COMPLEX

def draw(img, point, radius):
    pts = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(img, pts, r, (0, 0, 255), 2)
    #cv2.drawMarker(img, pts, (0,0,0),markerType=cv2.MARKER_CROSS,markerSize=10, thickness=1)

img = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")
decolor, _ = cv2.decolor(img);
fil = cv2.GaussianBlur(decolor, (33, 33), 4)      #平滑化
contours, _ = cv2.findContours(fil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
        center, radius = cv2.minEnclosingCircle(cnt)
        draw(img, (center[0], center[1]), radius)

cv2.imwrite('mincir_decolor.png', img)
