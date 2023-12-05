import cv2, time, threading, openpyxl
import numpy as np


x_list_dif, y_list_dif, w_list_dif, h_list_dif = [], [], [], []
x_list_eye, y_list_eye, w_list_eye, h_list_eye = [], [], [], []
delta_list = []
wbratio_list = []

kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
kernel_hor /= 9

kernel_cal = np.ones((3, 3), np.uint8)

blink_time = 0

def calibration(avg_dif):
    cap_cal = cv2.VideoCapture(0)
    cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_cal.set(cv2.CAP_PROP_FPS, 30)
    cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc_capcal = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video_capcal = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_cal.mp4', fourcc_capcal, 30, (640, 360))

    video_bin_dif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_bindif.mp4', fourcc_capcal, 30, (640, 360))
    video_edge_dif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_edgedif.mp4', fourcc_capcal, 30, (640, 360))
    video_deltadif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_deltadif.mp4', fourcc_capcal, 30, (640, 360))

    video_bin_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_binmsk.mp4', fourcc_capcal, 30, (640, 360))
    video_edge_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_edgemsk.mp4', fourcc_capcal, 30, (640, 360))
    video_pick_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_pick_msk.mp4', fourcc_capcal, 30, (640, 360))

    video_bin_line = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_bin_line.mp4', fourcc_capcal, 30, (640, 360))
    video_hor_line = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_hor_line.mp4', fourcc_capcal, 30, (640, 360))
    video_dil_line = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_dil_line.mp4', fourcc_capcal, 30, (640, 360))
    video_cls_line = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_cls_line.mp4', fourcc_capcal, 30, (640, 360))
    video_opn_line = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\calibration\Capture_opn_line.mp4', fourcc_capcal, 30, (640, 360))


    basetime = time.time()
    while True:
        ret, frame_cal = cap_cal.read()
        if not ret:
            break

        gaussian = cv2.GaussianBlur(frame_cal, (5, 5), 1)
        gray_cal = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)


        # Frame difference
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


        # Mask process
        mask = cv2.inRange(gray_cal, 30, 70)
        pick_msk = cv2.bitwise_and(gray_cal, gray_cal, mask = mask)
        bin_msk = cv2.threshold(pick_msk, 3, 255, cv2.THRESH_BINARY_INV)[1]
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
        

        # Eye lids gradients
        bin_line = cv2.threshold(gray_cal, 70, 255, cv2.THRESH_BINARY)[1]
        horizon_line = cv2.filter2D(bin_line, -1, kernel = kernel_hor)
        dilation_line = cv2.dilate(horizon_line, kernel = kernel_cal, iterations = 1)
        closing_line = cv2.morphologyEx(dilation_line, cv2.MORPH_CLOSE, kernel = kernel_cal)
        opening_line = cv2.morphologyEx(closing_line, cv2.MORPH_OPEN, kernel = kernel_cal)
        lines = cv2.HoughLinesP(opening_line, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 100, maxLineGap = 50)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                if x0 < 500 and y0 < 200:
                    delta_Y = y1 - y0
                    delta_X = x1 - x0
                    Gradient = 10 * (delta_Y / delta_X)
                    if Gradient < 6 and Gradient > 0:
                        delta_list.append(Gradient)

        cv2.imshow('Frame', frame_cal)

        endtime = time.time()
        runtime = endtime - basetime

        if runtime > 15:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        video_capcal.write(frame_cal)
        video_bin_dif.write(cv2.cvtColor(bin_dif, cv2.COLOR_GRAY2BGR))
        video_edge_dif.write(cv2.cvtColor(edges_dif, cv2.COLOR_GRAY2BGR))
        video_deltadif.write(cv2.cvtColor(delta_dif, cv2.COLOR_GRAY2BGR))
        video_bin_msk.write(cv2.cvtColor(bin_msk, cv2.COLOR_GRAY2BGR))
        video_edge_msk.write(cv2.cvtColor(edges_msk, cv2.COLOR_GRAY2BGR))
        video_pick_msk.write(cv2.cvtColor(pick_msk, cv2.COLOR_GRAY2BGR))
        video_bin_line.write(cv2.cvtColor(bin_line, cv2.COLOR_GRAY2BGR))
        video_hor_line.write(cv2.cvtColor(horizon_line, cv2.COLOR_GRAY2BGR))
        video_dil_line.write(cv2.cvtColor(dilation_line, cv2.COLOR_GRAY2BGR))
        video_cls_line.write(cv2.cvtColor(closing_line, cv2.COLOR_GRAY2BGR))
        video_opn_line.write(cv2.cvtColor(opening_line, cv2.COLOR_GRAY2BGR))

    cap_cal.release()
    video_capcal.release()
    video_bin_dif.release()
    video_deltadif.release()
    video_edge_dif.release()
    video_bin_msk.release()
    video_edge_msk.release()
    video_pick_msk.release()
    video_bin_line.release()
    video_hor_line.release()
    video_dil_line.release()
    video_cls_line.release()
    video_opn_line.release()
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

    # if ave_x0_dif > ave_x0_eye:
    #     xmin = int(ave_x0_eye - 10)
    # else:
    #     xmin = int(ave_x0_dif - 10)
    
    # if ave_y0_dif > ave_y0_eye:
    #     ymin = int(ave_y0_eye - 10)
    # else:
    #     ymin = int(ave_y0_dif - 10)

    # if ymin < 0 :
    #     ymin = 0

    # if ave_w_dif > ave_w_eye:
    #     xmax = int(xmin + ave_w_dif + 20)
    # else:
    #     xmax = int(xmin + ave_w_eye + 20)
    
    # if xmax > 640:
    #     xmax = 640

    # if ave_h_dif > ave_h_eye:
    #     ymax = int(ymin + ave_h_dif + 20)
    # else:
    #     ymax = int(ymin + ave_h_eye + 20)

    xmin_rnd = int(ave_x0_dif - 20)
    xmax_rnd = int(xmin_rnd + ave_w_dif + 20)
    ymin_rnd = int(ave_y0_dif - 20)
    ymax_rnd = int(ymin_rnd + ave_h_dif + 20)

    return xmin_rnd, xmax_rnd, ymin_rnd, ymax_rnd

def Excel_write_calibration(x_list_dif, y_list_dif, w_list_dif, h_list_dif, 
                x_list_eye, y_list_eye, w_list_eye, h_list_eye, 
                delta_list, new_delta_list, wbratio_list):
    wb = openpyxl.load_workbook(R'C:\Users\admin\Desktop\data\Watchvideo\excel_data\calibration.xlsx')
    wb.create_sheet('calibration_Watchvideo_1204-4')
    ws = wb['calibration_Watchvideo_1204-4']

    ws["D3"].value = 'x_dif'
    ws["E3"].value = 'y_dif'
    ws["F3"].value = 'w_dif'
    ws["G3"].value = 'h_dif'

    ws["I3"].value = 'x_msk'
    ws["J3"].value = 'y_msk'
    ws["K3"].value = 'w_msk'
    ws["L3"].value = 'h_msk'

    ws["N3"].value = 'delta Y'
    ws["O3"].value = 'delta Y(descending)'

    ws["Q3"].value = 'maximum gradient'
    ws["R3"].value = 'minimum gradient'

    ws["T3"].value = 'White ratio'
    ws["U3"].value = 'White ratio(descending)'

    for i1 in range(0, len(x_list_dif)):
        ws.cell(i1 + 4, 4, value = x_list_dif[i1])
        ws.cell(i1 + 4, 5, value = y_list_dif[i1])
        ws.cell(i1 + 4, 6, value = w_list_dif[i1])
        ws.cell(i1 + 4, 7, value = h_list_dif[i1])

    for i2 in range(0, len(x_list_eye)):
        ws.cell(i2 + 4, 9, value = x_list_eye[i2])
        ws.cell(i2 + 4, 10, value = y_list_eye[i2])
        ws.cell(i2 + 4, 11, value = w_list_eye[i2])
        ws.cell(i2 + 4, 12, value = h_list_eye[i2])

    for i4 in range(0, len(new_delta_list)):
        ws.cell(i4 + 4, 14, value = delta_list[i4])
        ws.cell(i4 + 4, 15, value = new_delta_list[i4])
    
    ws.cell(4, 17, value = new_delta_list[0])
    ws.cell(4, 18, value = new_delta_list[-1])

    new_wbratio_list = sorted(wbratio_list)

    for i4 in range(0, len(wbratio_list)):
        ws.cell(i4 + 4, 20, value = wbratio_list[i4])
        ws.cell(i4 + 4, 21, value = new_wbratio_list[i4])

    wb.save(R'C:\Users\admin\Desktop\data\Watchvideo\excel_data\calibration.xlsx')
    wb.close()

def wb_ratio_calculation(binframe, xmin, xmax, ymin, ymax):
    size = int((xmax - xmin) * (ymax - ymin))
    white_pixel = cv2.countNonZero(binframe)
    white_ratio = (white_pixel / size) * 100

    return white_ratio

def Blink_calibration(avg_blink, xmin, xmax, ymin, ymax):
    cap_blink = cv2.VideoCapture(0)
    cap_blink.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_blink.set(cv2.CAP_PROP_FPS, 30)
    cap_blink.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_blink.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    basetime_blink = time.time()
    blink_time = 0
    while True:
        ret, frame_blink = cap_blink.read()
        if not ret:
            break

        gau_blink = cv2.GaussianBlur(frame_blink[ymin:ymax, xmin:xmax], (5, 5), 1)
        gray_blink = cv2.cvtColor(gau_blink, cv2.COLOR_BGR2GRAY)

        if avg_blink is None:
            avg_blink = gray_blink.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray_blink, avg_blink, 0.8)
        delta_blink = cv2.absdiff(gray_blink, cv2.convertScaleAbs(avg_blink))
        bin_blink = cv2.threshold(delta_blink, 3, 255, cv2.THRESH_BINARY)[1]
        white_ratio = wb_ratio_calculation(binframe = bin_blink, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
        time_diff = time.time() - blink_time

        if time_diff > 0.3:
            blink_time = time.time()
            wbratio_list.append(white_ratio)

        cv2.imshow('Blink', bin_blink)

        endtime_blink = time.time()
        runtime_blink = endtime_blink - basetime_blink

        if runtime_blink > 15:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_blink.release()
    cv2.destroyAllWindows()

def Detection_data_excel(grad_val, grad_time, grad_delta, 
                        dif_val, dif_time, dif_ratio, 
                        grad_dif_val, grad_dif_time, grad_dif_delta, grad_dif_ratio, 
                        dif_grad_val, dif_grad_time, dif_grad_delta, dif_grad_ratio, 
                        timelist, ratiolist, deltalist, gradient_time_list, 
                        x0list_grad, x1list_grad, y0list_grad, y1list_grad):
    wb = openpyxl.load_workbook(R'C:\Users\admin\Desktop\data\Watchvideo\excel_data\detection.xlsx')
    wb.create_sheet('detection_Watchvideo_1204-4')
    wb.create_sheet('detection_all_data_Watchvideo_1204-4')
    wb.create_sheet('gradients coordinates_Watchvideo_1204-4')
    ws_detection = wb['detection_Watchvideo_1204-4']
    ws_alldata = wb['detection_all_data_Watchvideo_1204-4']
    ws_gradcoord = wb['gradients coordinates_Watchvideo_1204-4']

    ws_detection["D2"].value = 'gradient process'
    ws_detection["D3"].value = 'value'
    ws_detection["E3"].value = 'time'
    ws_detection["F3"].value = 'gradient'

    ws_detection["H2"].value = 'ratio process'
    ws_detection["H3"].value = 'value'
    ws_detection["I3"].value = 'time'
    ws_detection["J3"].value = 'ratio'

    ws_detection["L2"].value = 'gradient → ratio process'
    ws_detection["L3"].value = 'value'
    ws_detection["M3"].value = 'time'
    ws_detection["N3"].value = 'gradient'
    ws_detection["O3"].value = 'ratio'

    ws_detection["Q2"].value = 'ratio → gradient process'
    ws_detection["Q3"].value = 'value'
    ws_detection["R3"].value = 'time'
    ws_detection["S3"].value = 'ratio'
    ws_detection["T3"].value = 'gradient'

    ws_alldata["D3"].value = 'time'
    ws_alldata["E3"].value = 'ratio'
    ws_alldata["G3"].value = 'time'
    ws_alldata["H3"].value = 'gradient'

    ws_gradcoord["D3"].value = 'x0'
    ws_gradcoord["E3"].value = 'x1'
    ws_gradcoord["F3"].value = 'y0'
    ws_gradcoord["G3"].value = 'y1'

    for i0 in range(0, len(grad_val)):
        ws_detection.cell(i0 + 4, 4, value = grad_val[i0])
        ws_detection.cell(i0 + 4, 5, value = grad_time[i0])
        ws_detection.cell(i0 + 4, 6, value = grad_delta[i0])

    for i1 in range(0, len(dif_val)):
        ws_detection.cell(i1 + 4, 8, value = dif_val[i1])
        ws_detection.cell(i1 + 4, 9, value = dif_time[i1])
        ws_detection.cell(i1 + 4, 10, value = dif_ratio[i1])

    for i2 in range(0, len(grad_dif_val)):
        ws_detection.cell(i2 + 4, 12, value = grad_dif_val[i2])
        ws_detection.cell(i2 + 4, 13, value = grad_dif_time[i2])
        ws_detection.cell(i2 + 4, 14, value = grad_dif_delta[i2])
        ws_detection.cell(i2 + 4, 15, value = grad_dif_ratio[i2])

    for i3 in range(0, len(dif_grad_val)):
        ws_detection.cell(i3 + 4, 17, value = dif_grad_val[i3])
        ws_detection.cell(i3 + 4, 18, value = dif_grad_time[i3])
        ws_detection.cell(i3 + 4, 19, value = dif_grad_ratio[i3])
        ws_detection.cell(i3 + 4, 20, value = dif_grad_delta[i3])

    for i4 in range(0, len(timelist)):
        ws_alldata.cell(i4 + 4, 4, value = timelist[i4])
    
    for i5 in range(0, len(ratiolist)):
        ws_alldata.cell(i5 + 4, 5, value = ratiolist[i5])

    for i6 in range(0, len(deltalist)):
        ws_alldata.cell(i6 + 4, 7, value = gradient_time_list[i6])
        ws_alldata.cell(i6 + 4, 8, value = deltalist[i6])

    for i7 in range(0, len(x0list_grad)):
        ws_gradcoord.cell(i7 + 4, 4, value = x0list_grad[i7])
        ws_gradcoord.cell(i7 + 4, 5, value = x1list_grad[i7])
        ws_gradcoord.cell(i7 + 4, 6, value = y0list_grad[i7])
        ws_gradcoord.cell(i7 + 4, 7, value = y1list_grad[i7])

    wb.save(R'C:\Users\admin\Desktop\data\Watchvideo\excel_data\detection.xlsx')
    wb.close()

def Detection_data_excel_ver2(time_cal_list, grad_cal_list, 
                            time_detec_list, grad_detec_list, ratio_detec_list, 
                            time_list, grad_list, ratio_list):
    wb = openpyxl.load_workbook(R'C:\Users\admin\Desktop\data\Watchvideo\excel_data\detection.xlsx')
    wb.create_sheet('detection_Watchvideo_1204-4')
    ws = wb['detection_Watchvideo_1204-4']

    ws["D2"].value = 'calculation'
    ws["D3"].value = 'time'
    ws["E3"].value = 'gradient'

    ws["G2"].value = 'detection'
    ws["G3"].value = 'time'
    ws["H3"].value = 'gradient'
    ws["I3"].value = 'ratio'

    ws["K2"].value = 'all data'
    ws["K3"].value = 'time'
    ws["L3"].value = 'gradient'
    ws["M3"].value = 'ratio'

    for i0 in range(0, len(time_cal_list)):
        ws.cell(i0 + 4, 4, value = time_cal_list[i0])
        ws.cell(i0 + 4, 5, value = grad_cal_list[i0])
    
    for i1 in range(0, len(time_detec_list)):
        ws.cell(i1 + 4, 7, value = time_detec_list[i1])
        ws.cell(i1 + 4, 8, value = grad_detec_list[i1])
        ws.cell(i1 + 4, 9, value = ratio_detec_list[i1])

    for i2 in range(0, len(time_list)):
        ws.cell(i2 + 4, 11, value = time_list[i2])
        ws.cell(i2 + 4, 12, value = grad_list[i2])
        ws.cell(i2 + 4, 13, value = ratio_list[i2])

    wb.save(R'C:\Users\admin\Desktop\data\Watchvideo\excel_data\detection.xlsx')
    wb.close()

if __name__ == '__main__':
    avg_dif = None
    avg_blink = None

    calibration(avg_dif = avg_dif)

    xmin_cal, xmax_cal, ymin_cal, ymax_cal = average_value(x_list_dif, y_list_dif, w_list_dif, h_list_dif, x_list_eye, y_list_eye, w_list_eye, h_list_eye)
    new_delta_list = sorted(delta_list, reverse = True)
    max_grad = new_delta_list[0]
    grad_thresh_low = max_grad - 1.5
    grad_thresh_high = max_grad + 0.5

    Blink_calibration(avg_blink = avg_blink, xmin = xmin_cal, xmax = xmax_cal, ymin = ymin_cal, ymax = ymax_cal)
    
    new_wb_list = sorted(wbratio_list, reverse = True)
    max_blink_score = new_wb_list[0]
    score_high = int(max_blink_score + 6)
    score_low = int(max_blink_score - 6)

    Excel_write_calibration(x_list_dif, y_list_dif, w_list_dif, h_list_dif, 
                            x_list_eye, y_list_eye, w_list_eye, h_list_eye, 
                            delta_list, new_delta_list, wbratio_list)
    
    fontType = cv2.FONT_HERSHEY_COMPLEX

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    cutsize_x = xmax_cal - xmin_cal
    cutsize_y = ymax_cal - ymin_cal

    fourcc_cap = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video_cap = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture.mp4', fourcc_cap, 30, (640, 360))
    video_cut = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_cut.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_msk.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_knl = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_knl.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_dil = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_dil.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_cls = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_cls.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_opn = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_opn.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_dlt = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_dlt.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_dlt_bin = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_dlt_bin.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_gau = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_gau.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))
    video_gry = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\Watchvideo\video_data\detection\Capture_gry.mp4', fourcc_cap, 30, (cutsize_x, cutsize_y))

    avg = None

    time_cal_list, grad_cal_list = [], []
    time_detec_list, grad_detec_list, ratio_detec_list = [], [], []
    time_list, grad_list, ratio_list = [], [], []

    time_comparison, grad_comparison, ratio_comparison = [0, 0], [0, 0], [0, 0]

    val = 0
    blinktime, calibrationtime = 0, 0

    basetime = time.time()

    kernel_hor = np.array([
        [1, 2, 1], 
        [0, 0, 0], 
        [-1, -2, -1]], dtype = np.float32)
    kernel_hor /= 9

    kernel_detec = np.ones((5, 5), np.uint8)

    Judge0 = 0
    Judge1 = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cutframe = frame[ymin_cal:ymax_cal, xmin_cal:xmax_cal]
        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)


        bin_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon_line = cv2.filter2D(bin_line, -1, kernel = kernel_hor)
        dilation_line = cv2.dilate(horizon_line, kernel = kernel_detec, iterations = 1)
        closing_line = cv2.morphologyEx(dilation_line, cv2.MORPH_CLOSE, kernel = kernel_detec)
        opening_line = cv2.morphologyEx(closing_line, cv2.MORPH_OPEN, kernel = kernel_detec)
        lines = cv2.HoughLinesP(opening_line, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)
        
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                Gradient = 10 * (delta_Y / delta_X)
                # if Gradient < 5 and Gradient > 0:
                # timediff_cal = time.time() - calibrationtime
                # if timediff_cal > 0.05:
                #     calibrationtime = time.time()
            if Gradient < 10 and Gradient > 0:
                time_cal = time.time() - basetime
                time_cal_list.append(time_cal)
                grad_cal_list.append(Gradient)
                cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 3)

        if avg is None:
            avg = gray.copy().astype("float")
            continue
        cv2.accumulateWeighted(gray, avg, 0.8)
        delta_dif = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        bin_dif = cv2.threshold(delta_dif, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = wb_ratio_calculation(bin_dif, xmin_cal, xmax_cal, ymin_cal, ymax_cal)

        comparison_time = time.time() - basetime
        time_comparison.append(comparison_time)
        grad_comparison.append(Gradient)
        ratio_comparison.append(whiteratio)

        if Gradient > grad_thresh_low and Gradient < grad_thresh_high:
            timediff_detec = time.time() - blinktime
            if timediff_detec > 0.3:
                if whiteratio < score_high and whiteratio > score_low:
                    val += 1
                    blinktime = time.time()
                    time_detec = time.time() - basetime
                    time_detec_list.append(time_detec)
                    grad_detec_list.append(Gradient)
                    ratio_detec_list.append(whiteratio)
                    Judge0 = 1
                
                if Judge0 == 0:
                    indexW = ratio_comparison[-2]
                    indexV = ratio_comparison[-3]

                    if indexW < score_high and indexW > score_low:
                        val += 1
                        blinktime = time.time()
                        time_detec = time.time() - basetime
                        time_detec_list.append(time_detec)
                        grad_detec_list.append(Gradient)
                        ratio_detec_list.append(whiteratio)
                        Judge1 = 1

                    if Judge0 == 0 and Judge1 == 0:
                        if indexV < score_high and indexV > score_low:
                            val += 1
                            blinktime = time.time()
                            time_detec = time.time() - basetime
                            time_detec_list.append(time_detec)
                            grad_detec_list.append(Gradient)
                            ratio_detec_list.append(whiteratio)

# NOT REMOVE
        # if (Gradient > grad_thresh_low and Gradient < grad_thresh_high) and (whiteratio < score_high and whiteratio > score_low):
        #     timediff_detec = time.time() - blinktime
        #     if timediff_detec > 0.3:
        #         val += 1
        #         blinktime = time.time()
        #         time_detec = time.time() - basetime
        #         time_detec_list.append(time_detec)
        #         grad_detec_list.append(Gradient)
        #         ratio_detec_list.append(whiteratio)

        run_time = time.time() - basetime

        time_list.append(run_time)
        grad_list.append(Gradient)
        ratio_list.append(whiteratio)

        cv2.putText(frame, 'Count:', (10, 350), fontType, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(val), (200, 350), fontType, 1, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Cut frame', cutframe)

        Judge0 = 0
        Judge1 = 0

        video_cap.write(frame)
        video_cut.write(cutframe)
        video_msk.write(cv2.cvtColor(bin_line, cv2.COLOR_GRAY2BGR))
        video_knl.write(cv2.cvtColor(horizon_line, cv2.COLOR_GRAY2BGR))
        video_dil.write(cv2.cvtColor(dilation_line, cv2.COLOR_GRAY2BGR))
        video_cls.write(cv2.cvtColor(closing_line, cv2.COLOR_GRAY2BGR))
        video_opn.write(cv2.cvtColor(opening_line, cv2.COLOR_GRAY2BGR))
        video_dlt.write(cv2.cvtColor(delta_dif, cv2.COLOR_GRAY2BGR))
        video_dlt_bin.write(cv2.cvtColor(bin_dif, cv2.COLOR_GRAY2BGR))
        video_gau.write(gaussian)
        video_gry.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if run_time > 900:
            break


    print('Grad thresh low:', grad_thresh_low)
    print('Grad thresh high:', grad_thresh_high)
    print('Ratio thresh low:', score_low)
    print('Ratio thresh high:', score_high)

    Detection_data_excel_ver2(time_cal_list, grad_cal_list, 
                            time_detec_list, grad_detec_list, ratio_detec_list, 
                            time_list, grad_list, ratio_list)
    
    cap.release()
    video_cap.release()
    video_cut.release()
    video_msk.release()
    video_knl.release()
    video_dil.release()
    video_cls.release()
    video_opn.release()
    video_dlt.release()
    video_dlt_bin.release()
    video_gau.release()
    video_gry.release()
    cv2.destroyAllWindows()