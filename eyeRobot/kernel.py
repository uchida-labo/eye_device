import cv2
import numpy as np

kernel_ver = np.array([
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]], dtype = np.float32)
kernel_ver /= 9

kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
kernel_hor /= 9

kernel = np.ones((3, 3), np.uint8)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


while True:
    ret, frame = capture.read()
    if not ret:
        break

    gau = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)
    bin = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
    # edges = cv2.Canny(bin, 0, 100)
    filhor = cv2.filter2D(bin, -1, kernel = kernel_hor)
    dilation = cv2.dilate(filhor, kernel = kernel, iterations = 1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel = kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel = kernel)

    lines = cv2.HoughLinesP(closing, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 100, maxLineGap = 10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 < 500 and y2 < 200:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 6)
                deltaY = y1 - y2
                print('Delta Y :', deltaY)


    # cv2.imshow('Binary', bin)
    cv2.imshow('Frame', frame)
    cv2.imshow('Horizontal', filhor)
    # cv2.imshow('Edges', edges)
    cv2.imshow('Dilation', dilation)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
