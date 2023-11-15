import cv2, time, threading, openpyxl
import numpy as np
from calibration_rect_coordinate import Frame_coordinates

cap_cal = cv2.VideoCapture(0)
cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_cal.set(cv2.CAP_PROP_FPS, 30)
cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
fourcc_capcal = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc_binary = cv2.VideoWriter_fourcc(*'XVID')
video_capcal = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\video_data\Capture_cal.mp4', fourcc_capcal, 30, (640, 360))
video_bin_dif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\video_data\Capture_bindif.avi', fourcc_binary, 30, (640, 360))
video_edge_dif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\video_data\Capture_edgedif.avi', fourcc_binary, 30, (640, 360))
video_bin_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\video_data\Capture_binmsk.avi', fourcc_binary, 30, (640, 360))
video_edge_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\video_data\Capture_edgemsk.avi', fourcc_binary, 30, (640, 360))


x_list_dif = []
y_list_dif = []
w_list_dif = []
h_list_dif = []

x_list_eye = []
y_list_eye = []
w_list_eye = []
h_list_eye = []

delta_list = []

kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
kernel_hor /= 9

kernel = np.ones((3, 3), np.uint8)

basetime = time.time()


def Frame_coordinates(avg_dif):
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

        cv2.imshow('Frame', frame_cal)

        endtime = time.time()
        runtime = endtime - basetime

        if runtime > 20:
            break

        video_capcal.write(frame_cal)
        video_bin_dif.write(cv2.cvtColor(bin_dif, cv2.COLOR_GRAY2BGR))
        video_edge_dif.write(cv2.cvtColor(edges_dif, cv2.COLOR_GRAY2BGR))
        video_bin_msk.write(cv2.cvtColor(bin_msk, cv2.COLOR_GRAY2BGR))
        video_edge_msk.write(cv2.cvtColor(edges_msk, cv2.COLOR_GRAY2BGR))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap_cal.release()
    video_capcal.release()
    video_bin_dif.release()
    video_bin_msk.release()
    video_edge_dif.release()
    video_edge_msk.release()
    cv2.destroyAllWindows()

def average_value(xdif, ydif, wdif, hdif, xmsk, ymsk, wmsk, hmsk):
    ave_x0_dif = sum(xdif) / len(xdif)
    ave_y0_dif = sum(ydif) / len(ydif)
    ave_w_dif = sum(wdif) / len(wdif)
    ave_h_dif = sum(hdif) / len(hdif)

    ave_x0_eye = sum(xmsk) / len(xmsk)
    ave_y0_eye = sum(ymsk) / len(ymsk)
    ave_w_eye = sum(wmsk) / len(wmsk)
    ave_h_eye = sum(hmsk) / len(hmsk)

    if ave_x0_dif > ave_x0_eye:
        xmin = int(ave_x0_eye - 20)
    else:
        xmin = int(ave_x0_dif - 20)
    
    if ave_y0_dif > ave_y0_eye:
        ymin = int(ave_y0_eye - 20)
    else:
        ymin = int(ave_y0_dif - 20)

    if ymin < 0 :
        ymin = 0

    if ave_w_dif > ave_w_eye:
        xmax = int(xmin + ave_w_dif + 40)
    else:
        xmax = int(xmin + ave_w_eye + 40)
    
    if xmax > 640:
        xmax = 640

    if ave_h_dif > ave_h_eye:
        ymax = int(ymin + ave_h_dif + 40)
    else:
        ymax = int(ymin + ave_h_eye + 40)

    xmin_rnd = round(xmin, -1)
    xmax_rnd = round(xmax, -1)
    ymin_rnd = round(ymin, -1)
    ymax_rnd = round(ymax, -1)

    return xmin_rnd, xmax_rnd, ymin_rnd, ymax_rnd



if __name__ == '__main__':
    avg_dif = None
    rect_thread = threading.Thread(target = Frame_coordinates(avg_dif = avg_dif), name = 'Thread 1', daemon = True)
    rect_thread.start()

    rect_thread.join()

    # time.sleep(10)

    xmin_cal, xmax_cal, ymin_cal, ymax_cal = average_value(x_list_dif, y_list_dif, w_list_dif, h_list_dif, x_list_eye, y_list_eye, w_list_eye, h_list_eye)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    fourcc_cap = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc_cut = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_cap = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\video_data\Capture.mp4', fourcc_cap, 30, (640, 360))
    video_cut = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\video_data\Capture_cut.mp4', fourcc_cut, 30, (xmax_cal - xmin_cal, ymax_cal - ymin_cal))

    wb = openpyxl.load_workbook(R'C:\Users\admin\Desktop\data\excel_data\rect_coordinates.xlsx')
    wb.create_sheet('20231115')
    ws = wb['20231115']

    ws["D3"].value = 'x_dif'
    ws["E3"].value = 'y_dif'
    ws["F3"].value = 'w_dif'
    ws["G3"].value = 'h_dif'

    ws["H3"].value = 'x_msk'
    ws["I3"].value = 'y_msk'
    ws["J3"].value = 'w_msk'
    ws["K3"].value = 'h_msk'

    for i1 in range(0, len(x_list_dif)):
        ws.cell(i1 + 4, 4, value = x_list_dif[i1])
        ws.cell(i1 + 4, 5, value = y_list_dif[i1])
        ws.cell(i1 + 4, 6, value = w_list_dif[i1])
        ws.cell(i1 + 4, 7, value = h_list_dif[i1])

    for i2 in range(0, len(x_list_eye)):
        ws.cell(i2 + 4, 8, value = x_list_eye[i2])
        ws.cell(i2 + 4, 9, value = y_list_eye[i2])
        ws.cell(i2 + 4, 10, value = w_list_eye[i2])
        ws.cell(i2 + 4, 11, value = h_list_eye[i2])

    wb.save(R'C:\Users\admin\Desktop\data\excel_data\rect_coordinates.xlsx')
    wb.close()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cutframe = frame[ymin_cal:ymax_cal, xmin_cal:xmax_cal]
        cv2.rectangle(frame, (xmin_cal, ymin_cal), (xmax_cal, ymax_cal), (255, 255, 0), 3)
        
        cv2.imshow('Frame', frame)
        cv2.imshow('Cut frame', cutframe)

        video_cap.write(frame)
        video_cut.write(cutframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_cap.release()
    video_cut.release()
    cv2.destroyAllWindows



