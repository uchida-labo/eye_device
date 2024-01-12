import cv2

img = cv2.imread('/Users/nakanokota/Desktop/fig2.jpg', cv2.IMREAD_UNCHANGED)
rev = cv2.bitwise_not(img)
edges = cv2.Canny(rev, 0, 130)

cv2.imwrite('/Users/nakanokota/Desktop/edges2.jpg', edges)