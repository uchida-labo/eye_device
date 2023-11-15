import cv2
import numpy as np

cap_cal = cv2.VideoCapture(0)
cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_cal.set(cv2.CAP_PROP_FPS, 30)
cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

x_list_dif = []
y_list_dif = []
w_list_dif = []
h_list_dif = []

x_list_eye = []
y_list_eye = []
w_list_eye = []
h_list_eye = []

delta_list = []

avg_dif = None

kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
kernel_hor /= 9

kernel = np.ones((3, 3), np.uint8)

def Frame_coordinates():
    while True:
        ret, frame_cal = cap_cal.read()
        if not ret:
            break

        gaussian = cv2.GaussianBlur(frame_cal, (5, 5), 1)
        gray_cal = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)


        # Frame difference
        if avg_dif is None:
            avg_dif = gray_cal.copy().astype("float")
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

        
        # Mask process
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

        ave_x0_dif = sum(x_dif) / len(x_dif)
        ave_y0_dif = sum(y_dif) / len(y_dif)
        ave_w_dif = sum(w_dif) / len(w_dif)
        ave_h_dif = sum(h_dif) / len(h_dif)

        ave_x0_eye = sum(x_msk) / len(x_msk)
        ave_y0_eye = sum(y_msk) / len(y_msk)
        ave_w_eye = sum(w_msk) / len(w_msk)
        ave_h_eye = sum(h_msk) / len(h_msk)

        if ave_x0_dif > ave_x0_eye:
            xmin_cal = int(ave_x0_eye - 20)
        else:
            xmin_cal = int(ave_x0_dif - 20)
        
        if ave_y0_dif > ave_y0_eye:
            ymin_cal = int(ave_y0_eye - 20)
        else:
            ymin_cal = int(ave_y0_dif - 20)

        if ymin_cal < 0 :
            ymin_cal = 0

        if ave_w_dif > ave_w_eye:
            xmax_cal = int(xmin_cal + ave_w_dif + 40)
        else:
            xmax_cal = int(xmin_cal + ave_w_eye + 40)
        
        if xmax_cal > 640:
            xmax_cal = 640

        if ave_h_dif > ave_h_eye:
            ymax_cal = int(ymin_cal + ave_h_dif + 40)
        else:
            ymax_cal = int(ymin_cal + ave_h_eye + 40)

        cv2.imshow('Frame', frame_cal)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap_cal.release()
    cv2.destroyAllWindows()

    return xmin_cal, xmax_cal, ymin_cal, ymax_cal