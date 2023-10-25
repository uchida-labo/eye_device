import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

xmin, xmax = 240, 600
ymin, ymax = 50, 350

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gaussian[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    # low_bound = 70
    # high_bound = 90
    # mask = cv2.inRange(gray, low_bound, high_bound)
    # pickupimg = cv2.bitwise_and(gray, gray, mask = mask)
    th_val, bin = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(bin, 100, 170)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (xmin, ymin))
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
    
    cv2.drawContours(frame, contours, -1, (255, 255, 0), 4)
    max_contour = max(contours, key = lambda x : cv2.contourArea(x))
    cv2.drawContours(frame, max_contour, -1, (0, 0, 255), 4)

    cv2.imshow('Frame', frame)
    cv2.imshow('Binary', bin)
    # cv2.imshow('Pick up image', pickupimg)
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Contours', cntframe)
    print(area)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
