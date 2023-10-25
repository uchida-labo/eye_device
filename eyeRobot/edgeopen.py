import cv2
import numpy as np

img = cv2.imread(R"C:\Users\admin\Desktop\white_remove_data\side.jpg",cv2.IMREAD_UNCHANGED)
img = cv2.GaussianBlur(img, (5, 5), 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 20,40)

kernel = np.ones((3, 3), np.uint8)

result = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)

cv2.imshow('edge', edge)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

