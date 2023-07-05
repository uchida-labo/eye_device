# decolor

import numpy as np
import cv2


fontType = cv2.FONT_HERSHEY_COMPLEX
gamma22LUT = np.array([pow(x/255.0, 2.2) * 255 for x in range(256)], dtype = 'uint8')
gamma045LUT = np.array([pow(x/255.0, 1.0/2.2) * 255 for x in range(256)], dtype = 'uint8')

img = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")
decolor, _ = cv2.decolor(img);
fil = cv2.GaussianBlur(decolor, (33, 33), 1)      #平滑化

# HoughCircle
circle = cv2.HoughCircles(fil, cv2.HOUGH_GRADIENT, 1.08, 100, param1=90, param2=78, minRadius=0, maxRadius=0)
circle = np.uint16(np.around(circle))
for i in circle[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),3)     #円周を描画する
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)        #中心点を描画する(単位：[ピクセル]、原点は左上)
    # pts = np.array((int(i[0]), int(i[1])))
    # cv2.putText(img, '{}'.format(pts), (i[0]+5, i[1]+5), fontType, 1, (0, 255, 0), 2)

cv2.imwrite('hough_decolor.png', img)