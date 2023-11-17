import cv2, time, threading, openpyxl
import numpy as np

cap_cal = cv2.VideoCapture(0)
cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_cal.set(cv2.CAP_PROP_FPS, 30)
cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

fourcc_capcal = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc_binary = cv2.VideoWriter_fourcc(*'XVID')

video_capcal = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_cal.mp4', fourcc_capcal, 30, (640, 360))

video_bin_dif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_bindif.avi', fourcc_binary, 30, (640, 360))
video_edge_dif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_edgedif.avi', fourcc_binary, 30, (640, 360))
video_deltadif = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_deltadif.avi', fourcc_binary, 30, (640, 360))

video_bin_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_binmsk.avi', fourcc_binary, 30, (640, 360))
video_edge_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_edgemsk.avi', fourcc_binary, 30, (640, 360))
video_pick_msk = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_pick_msk.avi', fourcc_binary, 30, (640, 360))

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
                if x1 < 500 and y1 < 200:
                    delta_Y = y1 - y0
                    delta_list.append(delta_Y)
                    # cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 3)

        cv2.imshow('Frame', frame_cal)

        endtime = time.time()
        runtime = endtime - basetime

        if runtime > 15:
            break

        # video_capcal.write(frame_cal)
        # video_bin_dif.write(cv2.cvtColor(bin_dif, cv2.COLOR_GRAY2BGR))
        # video_edge_dif.write(cv2.cvtColor(edges_dif, cv2.COLOR_GRAY2BGR))
        # video_bin_msk.write(cv2.cvtColor(bin_msk, cv2.COLOR_GRAY2BGR))
        # video_edge_msk.write(cv2.cvtColor(edges_msk, cv2.COLOR_GRAY2BGR))
        # video_pick_msk.write(cv2.cvtColor(pick_msk, cv2.COLOR_GRAY2BGR))
        # video_deltadif.write(cv2.cvtColor(delta_dif, cv2.COLOR_GRAY2BGR))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap_cal.release()
    # video_capcal.release()
    # video_bin_dif.release()
    # video_bin_msk.release()
    # video_edge_dif.release()
    # video_edge_msk.release()
    # video_pick_msk.release()
    # video_deltadif.release()
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

def Excel_write(x_list_dif, y_list_dif, w_list_dif, h_list_dif, 
                x_list_eye, y_list_eye, w_list_eye, h_list_eye, 
                delta_list, new_delta_list, wbratio_list):
    wb = openpyxl.load_workbook(R'C:\Users\admin\Desktop\data\calibration\calibration.xlsx')
    wb.create_sheet('20231115')
    ws = wb['20231115']

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
    ws["P3"].value = 'minimum error'
    ws["Q3"].value = 'maximum error'
    ws["R3"].value = 'minimum gradient'
    ws["S3"].value = 'maximum gradient'

    ws["U3"].value = 'White ratio'

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

    for i3 in range(0, len(new_delta_list)):
        ws.cell(i3 + 4, 14, value = delta_list[i3])
        ws.cell(i3 + 4, 15, value = new_delta_list[i3])
    
    ws.cell(4, 16, value = min_error)
    ws.cell(4, 17, value = max_error)
    ws.cell(4, 18, value = min_grad)
    ws.cell(4, 19, value = max_grad)

    new_wbratio_list = sorted(wbratio_list)

    for i4 in range(0, len(wbratio_list)):
        ws.cell(i4 + 4, 21, value = wbratio_list[i4])
        ws.cell(i4 + 4, 22, value = new_wbratio_list[i4])

    wb.save(R'C:\Users\admin\Desktop\data\calibration\calibration.xlsx')
    wb.close()

def wb_ratio_calculation(binframe, xmin, xmax, ymin, ymax):
    size = int((xmax - xmin) * (ymax - ymin))
    white_pixel = cv2.countNonZero(binframe)
    white_ratio = (white_pixel / size) * 100

    return white_ratio

def Blink_calibration(avg_blink, xmin, xmax, ymin, ymax):
    video_blink_cut = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_blink_cut.avi', fourcc_binary, 30, (xmax - xmin, ymax - ymin))
    basetime_blink = time.time()
    while True:
        ret, frame_blink = cap_cal.read()
        if not ret:
            break

        gau_blink = cv2.GaussianBlur(frame_blink, (5, 5), 1)
        gray_blink = cv2.cvtColor(gau_blink, cv2.COLOR_BGR2GRAY)

        if avg_blink is None:
            avg_blink = gray_blink.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray_blink, avg_blink, 0.8)
        delta_blink = cv2.absdiff(gray_blink, cv2.convertScaleAbs(avg_blink))
        bin_blink = cv2.threshold(delta_blink, 3, 255, cv2.THRESH_BINARY)[1]
        white_ratio = wb_ratio_calculation(bin_blink, xmin, xmax, ymin, ymax)
        time_diff = time.time() - blink_time

        if time_diff > 0.3:
            blink_time = time.time()
            wbratio_list.append(white_ratio)

        cv2.imshow('Blink', bin_blink)

        # video_blink_cut.write(cv2.cvtColor(bin_blink, cv2.COLOR_GRAY2BGR))

        endtime_blink = time.time()
        runtime_blink = endtime_blink - basetime_blink

        if runtime_blink > 15:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_cal.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    avg_dif = None
    avg_blink = None
    rect_thread = threading.Thread(target = calibration(avg_dif = avg_dif), name = 'Thread 1', daemon = True)
    rect_thread.start()
    rect_thread.join()

    xmin_cal, xmax_cal, ymin_cal, ymax_cal = average_value(x_list_dif, y_list_dif, w_list_dif, h_list_dif, x_list_eye, y_list_eye, w_list_eye, h_list_eye)
    new_delta_list = sorted(delta_list)
    min_error, max_error = line_gradients(new_delta_list)
    min_grad, max_grad = new_delta_list[0], new_delta_list[(len(new_delta_list) - 1)]
    thresh_gradient = min_grad + min_error
        
    blink_thread = threading.Thread(target = Blink_calibration(avg_blink, xmin_cal, xmax_cal, ymin_cal, ymax_cal), name = 'Thread 2', daemon = True)
    blink_thread.start()
    blink_thread.join()

    blink_score_ave = sum(wbratio_list) / len(wbratio_list)
    score_high = int(blink_score_ave + 5)
    score_low = int(blink_score_ave - 5)

    # Excel_write(x_list_dif, y_list_dif, w_list_dif, h_list_dif, 
    #             x_list_eye, y_list_eye, w_list_eye, h_list_eye, 
    #             delta_list, new_delta_list)
    
    fontType = cv2.FONT_HERSHEY_COMPLEX

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    fourcc_cap = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc_cut = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_cap = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture.mp4', fourcc_cap, 30, (640, 360))
    video_cut = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\calibration\video_data\Capture_cut.mp4', fourcc_cut, 30, (xmax_cal - xmin_cal, ymax_cal - ymin_cal))

    avg = None

    white_list0, time_list0, blink_list0, detectime_list0, detecratio_list0 = [], [], [], [], []
    white_list1, time_list1, blink_list1, detectime_list1, detecratio_list1 = [], [], [], [], []
    grad_list, detec_grad_list, grad_dif_list = [], []

    val_dif, val_grad = 0, 0

    basetime = time.time()
    blinktime = 0

    kernel_hor = np.array([
        [1, 2, 1], 
        [0, 0, 0], 
        [-1, -2, -1]], dtype = np.float32)
    kernel_hor /= 9

    kernel_detec = np.ones((5, 5), np.uint8)


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
                x0, x1, y0, y1 = line[0]
                grad_Y = y1 - y0
                grad_list.append(grad_Y)
                if grad_Y < thresh_gradient and grad_Y > (min_grad - 1):
                    grad_list.append(grad_Y)


        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        bin = cv2.threshold(delta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = wb_ratio_calculation(bin, xmin_cal, xmax_cal, ymin_cal, ymax_cal)

        if whiteratio < score_high and whiteratio > score_low:
            timediff = time.time() - blinktime
            if timediff > 0.3:
                val_dif += 1
                blinktime = time.time()
                detection_time = blinktime - basetime
                blink_list0.append(val_dif)
                detectime_list0.append(detection_time)
                detecratio_list0.append(whiteratio)
                if grad_Y < thresh_gradient and grad_Y > (min_grad - 1):
                    grad_dif_list.append()


        


        cv2.putText(frame, 'Blink count : ', (50, 100), fontType, 1, (0, 0, 255), 3)
        cv2.putText(frame, str(val_dif), (150, 100), fontType, 1, (0, 0, 255), 3)


        cv2.rectangle(frame, (xmin_cal, ymin_cal), (xmax_cal, ymax_cal), (255, 255, 0), 3)
        
        cv2.imshow('Frame', frame)
        cv2.imshow('Cut frame', cutframe)

        # video_cap.write(frame)
        # video_cut.write(cutframe)

        end_time = time.time()
        run_time = end_time - basetime

        white_list0.append(whiteratio)
        time_list0.append(run_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_cap.release()
    video_cut.release()
    cv2.destroyAllWindows()



