# FitEllipse

import cv2
import numpy as np

fontType = cv2.FONT_HERSHEY_COMPLEX

def draw(img, point, radius):
    npt = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(img, npt, r, (0, 0, 255), 2)

    #cv2.drawMarker(img, npt, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

img = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")
decolor, _ = cv2.decolor(img);
#fil = cv2.GaussianBlur(decolor, (33, 33), 4)      #平滑化
contours, _ = cv2.findContours(decolor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
        ellipse = cv2.fitEllipse(cnt)
        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])
        radius = int((ellipse[1][0] + ellipse[1][1])/4)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)
        draw(img, (cx, cy), radius)

cv2.imwrite('fit_decolor.png', img) 