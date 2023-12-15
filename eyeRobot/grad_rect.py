import cv2
import  numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

kernel_hor = np.array([
    [-1, -2, -1], 
    [0, 0, 0], 
    [1, 2, 1]], dtype = np.float32)
kernel_hor /= 9

kernel_ver = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype = np.float32)

kernel_cal = np.ones((3, 3), np.uint8)
kernel_ver = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gau = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(gray, 20, 70)
    pick = cv2.bitwise_and(gray, gray, mask = mask)
    bin = cv2.threshold(pick, 5, 255, cv2.THRESH_BINARY)[1]
    ver = cv2.filter2D(bin, -1, kernel = kernel_ver)
    hor = cv2.filter2D(bin, -1, kernel = kernel_hor)
    side = cv2.bitwise_and(ver, hor)
    # dil = cv2.dilate(side, kernel = kernel_cal, iterations = 1)
    cls = cv2.morphologyEx(side, cv2.MORPH_CLOSE, kernel = kernel_cal)
    opn = cv2.morphologyEx(cls, cv2.MORPH_OPEN, kernel = kernel_cal)
    
    lines = cv2.HoughLinesP(side, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 50, maxLineGap = 20)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            delta_X = x1 - x0
            delta_Y = y1 - y0
            grad = 10 * (delta_Y / delta_X)
        if grad > -10:
            cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
            print(grad)

    cv2.imshow('Binary', bin)
    cv2.imshow('Frame', frame)
    cv2.imshow('Dilation', side)
    cv2.imshow('Opening', opn)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()