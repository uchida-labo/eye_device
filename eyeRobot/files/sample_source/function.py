import cv2
import numpy as np

def drawcircle_and_getpoint(frame, x_center, y_center, radius, wm, hm, width, height):
    center = (int(x_center), int(y_center))
    r = int(radius)
    cv2.circle(frame, center, r, (100, 255, 0), 2)
    cv2.circle(frame, center, 2, (0,0,255), 3)     #中心点を描画
    x = (wm * x_center)/width
    y = (hm * y_center)/height
    
    return frame, x, y

def judge(frame, x_center, y_center, radius):
    xmin0 = x_center - (radius/2)
    ymin0 = y_center - (radius/2)
    xmax0 = x_center + (radius/2)
    ymax0 = y_center + (radius/2)
    squareimagesize = radius * radius
    trim_frame = cv2.cvtColor(frame[ymin0:ymax0, xmin0:xmax0], cv2.COLOR_RGB2GRAY)
    ret_a, trim = cv2.threshold(trim_frame, 65, 255, cv2.THRESH_BINARY)
    white = cv2.countNonZero(trim)
    black = squareimagesize - white
    white_ratio = (white/squareimagesize) * 100
    black_ratio = (black/squareimagesize) * 100

    return white_ratio, black_ratio

# Function of acquisition image processing
def filter(frame, xmin, xmax, ymin, ymax):

    # GaussianFilter processing (smoothing)
    fil = cv2.GaussianBlur(frame, (5, 5), 1)

    # trimming and grayscale conversion
    gray = cv2.cvtColor(fil[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    
    # binarization processing (Otsu's binarization)
    # return value : threshold value and binarized image
    th_val, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    
    # edge detection processing (canny method)
    edges = cv2.Canny(gray, 50, 100)   # C1-205 : 220, 330
    
    # outline extraction processing
    # return value : contours(pixel information) and hierarchy(Hierarchical structure information)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(xmin, ymin))

    return gray, bin, edges, contours

# Functions for feature point extraction and area determination
def feature_point(cnt, frame):
    # minimum circumscribed circle detection process
    # return value : center(center point of the detected circle) and radius(radius of the detected circle) [pixel]
    (x_center, y_center), radius = cv2.minEnclosingCircle(cnt)
    
    # iris determination processing
    # return value : white area ratio and black area ratio [%]
    white_ratio, black_ratio = judge(frame, x_center, y_center, radius)

    return x_center, y_center, radius, white_ratio, black_ratio

