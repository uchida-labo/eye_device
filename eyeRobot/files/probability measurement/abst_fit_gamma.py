# FitEllipse

import cv2
import numpy as np

fontType = cv2.FONT_HERSHEY_COMPLEX
gamma22LUT = np.array([pow(x/255.0, 2.2) * 255 for x in range(256)], dtype = 'uint8')
gamma045LUT = np.array([pow(x/255.0, 1.0/2.2) * 255 for x in range(256)], dtype = 'uint8')


def draw(img, point, radius):
    npt = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(img, npt, r, (0, 0, 255), 2)

    #cv2.drawMarker(img, npt, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

img = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")
lut = cv2.LUT(img, gamma22LUT)  #sRGB => linear(approximate value 2.2.)
gray = cv2.cvtColor(lut, cv2.COLOR_RGB2GRAY)  #グレースケール変換
gray = cv2.LUT(gray, gamma045LUT)   # linear => sRGB
fil = cv2.GaussianBlur(gray, (33, 33), 4)      #平滑化
contours, _ = cv2.findContours(fil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
        ellipse = cv2.fitEllipse(cnt)
        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])
        radius = int((ellipse[1][0] + ellipse[1][1])/4)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)
        draw(img, (cx, cy), radius)

cv2.imwrite('fit_gamma.png', img) 