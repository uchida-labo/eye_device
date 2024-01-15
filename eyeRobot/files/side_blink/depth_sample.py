import cv2, time
import numpy as np

cap = cv2.VideoCapture(0)

kernel_hor = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]])

kernel_hor_dlt = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]])

kernel_ver = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]])

kernel_ver_dlt = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]])

kernel_cal = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret: break

    gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    binary_rev = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1]
    binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]

    eyelid = cv2.filter2D(binary_rev, -1, kernel_hor)
    eyelid_dlt = cv2.filter2D(binary_rev, -1, kernel_hor_dlt)
    pupil = cv2.filter2D(binary_rev, -1, kernel_ver)
    pupil_dlt = cv2.filter2D(binary_rev, -1, kernel_ver_dlt)

    # eyelid = cv2.dilate(eyelid, kernel_cal, 1)
    # eyelid_dlt = cv2.dilate(eyelid_dlt, kernel_cal, 1)
    # pupil = cv2.dilate(pupil, kernel_cal, 1)
    # pupil_dlt = cv2.dilate(pupil_dlt, kernel_cal, 1)

    # eyelid = cv2.bitwise_not(eyelid)
    eyelid_dlt = cv2.bitwise_not(eyelid_dlt)
    pupil_dlt = cv2.bitwise_not(pupil_dlt)

    andprocess_1 = cv2.bitwise_and(pupil, eyelid_dlt)
    andprocess_2 = cv2.bitwise_or(andprocess_1, pupil)
    andprocess_3 = cv2.bitwise_and(andprocess_2, eyelid_dlt)
    # andprocess_1_dlt = cv2.dilate(andprocess_1, kernel_cal, 1)
    # andprocess_2 = cv2.bitwise_and(andprocess_1, eyelid_dlt)
    # andprocess_2 = cv2.erode(andprocess_2, kernel_cal, 1)



    dilation_eyelid = cv2.bitwise_not(cv2.dilate(eyelid_dlt, kernel_cal, 1))
    iris_vertical = cv2.bitwise_and(pupil, dilation_eyelid)
    iris = cv2.bitwise_and(cv2.dilate(iris_vertical, kernel_cal, 1), eyelid)

    # iris_cls = cv2.morphologyEx(andprocess_2, cv2.MORPH_CLOSE, kernel_cal)
    eyelid_lines = cv2.HoughLinesP(eyelid, rho = 1, theta = np.pi / 360, threshold = 90, minLineLength = 130, maxLineGap = 70)
    if eyelid_lines is not None:
        x0, y0, x1, y1 = eyelid_lines[0][0][0], eyelid_lines[0][0][1], eyelid_lines[0][0][2], eyelid_lines[0][0][3]
        if x1 < 550:
            cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
            iris_vertical_line = cv2.HoughLinesP(andprocess_1, rho = 1, theta = np.pi / 360, threshold = 5, minLineLength = 20, maxLineGap = 200)
            if iris_vertical_line is not None:
                x0_iris, y0_iris, x1_iris, y1_iris = iris_vertical_line[0][0][0], iris_vertical_line[0][0][1], iris_vertical_line[0][0][2], iris_vertical_line[0][0][3]
                if x1_iris > 400 and x1_iris < 500:
                    print('iris line = ', iris_vertical_line[0][0][0])
                    cv2.line(frame, (x0_iris, y0_iris), (x1_iris, y1_iris), (0, 0, 255), 2)



    # cv2.imshow('binary_rev', binary)
    cv2.imshow('eyelid', eyelid)
    cv2.imshow('eyelid dlt', eyelid_dlt)
    cv2.imshow('pupil', pupil)
    cv2.imshow('pupil dlt', pupil_dlt)
    cv2.imshow('and process 1', andprocess_1)
    # cv2.imshow('and process 1 close', andprocess_1_dlt)
    cv2.imshow('and process 2', andprocess_2)
    cv2.imshow('and process 3', andprocess_3)
    # cv2.imshow('iris_vertical', iris_cls)
    cv2.imshow('frame', frame)
    # cv2.imshow('iris', iris)
                    

    # print(eyelid_lines.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()