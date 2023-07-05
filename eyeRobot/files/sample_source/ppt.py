import cv2
import numpy 

def judge(bin):
    height, width = bin.shape
    img_size = width * height
    white = cv2.countNonZero(bin)
    black = img_size - white
    white_ratio = (white/img_size) * 100
    black_ratio = (black/img_size) * 100
    
    return white_ratio, black_ratio

fontType = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread('/Users/nakanokota/Desktop/photo/sample_6.png')
fil = cv2.GaussianBlur(img, (5, 5), 1)
gray = cv2.cvtColor(fil, cv2.COLOR_RGB2GRAY)
thresh_val, bin = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(gray, 95, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, cnt in enumerate(contours):
    center, radius = cv2.minEnclosingCircle(cnt)
    white_ratio, black_ratio = judge(bin)
    pts = (int(center[0]), int(center[1]))
    r = int(radius)
    if radius < 30 and radius > 20:
        cv2.circle(img, pts, r, (100, 255, 0), 2)
        cv2.circle(img, pts, 2, (0,0,255), 3)
    # cv2.putText(img, radius, (center[0] + 50, center[1]), fontType, 1, (0, 0, 255), 3)
        print('radius :', radius)
        print('white ratio :', white_ratio)

cv2.imwrite('/Users/nakanokota/Desktop/photo/gaussian.png', fil)
cv2.imwrite('/Users/nakanokota/Desktop/photo/gray.png', gray)
cv2.imwrite('/Users/nakanokota/Desktop/photo/bin.png', bin)
cv2.imwrite('/Users/nakanokota/Desktop/photo/edges.png', edges)
cv2.imwrite('/Users/nakanokota/Desktop/photo/sample_6a.png', img)

