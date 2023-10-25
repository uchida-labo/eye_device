import cv2
import numpy as np

img = cv2.imread(R"C:\Users\admin\Desktop\white_remove_data\side.jpg", cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

low_bound = 20
high_bound = 75

mask = cv2.inRange(img, low_bound, high_bound)
result = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow('Result', result)
cv2.imwrite(R"C:\Users\admin\Desktop\white_remove_data\side_20to75.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()