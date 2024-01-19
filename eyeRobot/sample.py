import cv2
import numpy as np

cap = cv2.VideoCapture(0)

kernel_hor = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernel_cal = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret: break

    gau = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)
    bin = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)[1]
    horizon = cv2.filter2D(bin, -1, kernel_hor)
    dilation = cv2.dilate(horizon, kernel_cal, 1)
    lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 130, minLineLength = 130, maxLineGap = 100)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            grad = ((y1 - y0) / (x1 - x0)) * 10
            if grad > -5 and grad < 10 and x1 < 580 and x0 > 300:
                cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
                print('x1 = ', x1)

    cv2.imshow('Frame', frame)
    cv2.imshow('dilation', dilation)
    cv2.imshow('bin', bin)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
