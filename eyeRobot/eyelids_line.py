import cv2
import numpy as np

cap_cal = cv2.VideoCapture(0)
cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_cal.set(cv2.CAP_PROP_FPS, 30)
cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
kernel_hor /= 9

kernel = np.ones((5, 5), np.uint8)

delta_list = []

def line_gradients(list):
    error_low = list[4] - list[0]
    error_high = list[(len(list) - 5)] - list[(len(list) - 1)]

    minerror, maxerror = 0, 0

    if error_low > 0 and abs(error_low) < 4:
        minerror = 3
    
    if error_low == 0:
        minerror = 2
    
    if error_high < 0 and abs(error_high) < 4:
        maxerror = 3

    if error_high == 0:
        maxerror = 2

    return minerror, maxerror

while True:
    ret, frame = cap_cal.read()
    if not ret:
        break

    gau = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

    bin_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
    horizon_line = cv2.filter2D(bin_line, -1, kernel = kernel_hor)
    dilation_line = cv2.dilate(horizon_line, kernel = kernel, iterations = 1)
    closing_line = cv2.morphologyEx(dilation_line, cv2.MORPH_CLOSE, kernel = kernel)
    opening_line = cv2.morphologyEx(closing_line, cv2.MORPH_OPEN, kernel = kernel)
    

    lines = cv2.HoughLinesP(opening_line, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            if x1 < 500 and y1 < 200:
                delta_Y = y1 - y0
                delta_list.append(delta_Y)
                cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 3)

    newlist = sorted(delta_list)

    cv2.imshow('Frame', frame)
    cv2.imshow('Binary', bin_line)
    cv2.imshow('filter', horizon_line)
    cv2.imshow('Dilation', dilation_line)
    cv2.imshow('Closing', closing_line)
    cv2.imshow('Opening', opening_line)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

min_error, max_error = line_gradients(delta_list)
min_grad = delta_list[0]
max_grad = delta_list[(len(delta_list) - 1)]

# print('min0', newlist[0])
# print('min1', newlist[1])
# print('min2', newlist[2])
# print('max2', newlist[(len(newlist)-3)])
# print('max1', newlist[(len(newlist)-2)])
# print('max0', newlist[(len(newlist)-1)])


cap_cal.release()
cv2.destroyAllWindows()

