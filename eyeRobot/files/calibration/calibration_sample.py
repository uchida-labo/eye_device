import cv2, statistics
import numpy as np
import time

cap_cal = cv2.VideoCapture(0)
cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_cal.set(cv2.CAP_PROP_FPS, 30)
cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

avg_dif = None

x_list_dif = []
y_list_dif = []
w_list_dif = []
h_list_dif = []

x_list_eye = []
y_list_eye = []
w_list_eye = []
h_list_eye = []

base_time = time.time()

kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
kernel_hor /= 9

kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame_cal = cap_cal.read()
    if not ret:
        break

    gau_cal = cv2.GaussianBlur(frame_cal, (5, 5), 1)
    gray_cal = cv2.cvtColor(gau_cal, cv2.COLOR_BGR2GRAY)


    # Interframe difference
    if avg_dif is None:
        avg_dif = gray_cal.copy().astype("float")
        continue
    cv2.accumulateWeighted(gray_cal, avg_dif, 0.8)
    delta_dif = cv2.absdiff(gray_cal, cv2.convertScaleAbs(avg_dif))
    bin_dif = cv2.threshold(delta_dif, 3, 255, cv2.THRESH_BINARY)[1]
    edges_dif = cv2.Canny(bin_dif, 0, 130)
    contours_dif = cv2.findContours(edges_dif, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i0, cnt0 in enumerate(contours_dif):
        x_dif, y_dif, w_dif, h_dif = cv2.boundingRect(cnt0)
        area_dif = w_dif * h_dif
        if w_dif > h_dif and area_dif > 30000:
            x_list_dif.append(x_dif)
            y_list_dif.append(y_dif)
            w_list_dif.append(w_dif)
            h_list_dif.append(h_dif)

    
    # Threshold process (by the mask image)
    mask = cv2.inRange(gray_cal, 30, 70)
    pick_msk = cv2.bitwise_and(gray_cal, gray_cal, mask = mask)
    bin_msk = cv2.threshold(pick_msk, 0, 255, cv2.THRESH_BINARY_INV)[1]
    edges_msk = cv2.Canny(bin_msk, 0, 130)
    contours_msk = cv2.findContours(edges_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i1, cnt1 in enumerate(contours_msk):
        x_msk, y_msk, w_msk, h_msk = cv2.boundingRect(cnt1)
        area_msk = w_msk * h_msk
        if w_msk > h_msk and area_msk > 30000:
            x_list_eye.append(x_msk)
            y_list_eye.append(y_msk)
            w_list_eye.append(w_msk)
            h_list_eye.append(h_msk)


    # Eyelids line
    bin_line = cv2.threshold(gray_cal, 70, 255, cv2.THRESH_BINARY)[1]
    horizon_line = cv2.filter2D(bin_line, -1, kernel = kernel_hor)
    dilation_line = cv2.dilate(horizon_line, kernel = kernel, iterations = 1)
    opening_line = cv2.morphologyEx(dilation_line, cv2.MORPH_OPEN, kernel = kernel)
    closing_line = cv2.morphologyEx(opening_line, cv2.MORPH_CLOSE, kernel = kernel)
    lines = cv2.HoughLinesP(closing_line, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 100, maxLineGap = 10)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            if x1 < 500 and y1 < 200:
                delta_Y = y0 - y1

    end_time = time.time()
    run_time = end_time - base_time

    if run_time > 20:
        break

    if cv2.waitKey(1) % 0xFF == ord('q'):
        break

cap_cal.release()
cv2.destroyAllWindows()