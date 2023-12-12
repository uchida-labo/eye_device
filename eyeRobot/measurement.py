import cv2, time, openpyxl, os
import numpy as np

xlist_framedetec, ylist_framedetec, wlist_framedetec, hlist_framedetec = [], [], [], []
ratiolist_calculation, gradientlist_calculation = [], []
comparison_time, comparison_gradient, comparison_ratio = [0, 0], [0, 0], [0, 0]
timelist_detec, gradientlist_detec, ratiolist_detec = [], [], []
timelist, gradientlist, ratiolist = [], [], []

date_number_path = 'Readbook_1211_1'

def Frame_detect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video_savepath_framedetec = 'C:\\Users\\admin\\Desktop\\measurement_data\\frame_detection\\' + date_number_path
    os.makedirs(video_savepath_framedetec)

    video_capture = cv2.VideoWriter(video_savepath_framedetec + '\\capture.mp4', fourcc, 30, (640, 360))
    video_gaussian = cv2.VideoWriter(video_savepath_framedetec + '\\gaussian.mp4', fourcc, 30, (640, 360))
    video_gray = cv2.VideoWriter(video_savepath_framedetec + '\\gray.mp4', fourcc, 30, (640, 360))
    video_framedeleta = cv2.VideoWriter(video_savepath_framedetec + '\\framedelta.mp4', fourcc, 30, (640, 360))
    video_binaryary = cv2.VideoWriter(video_savepath_framedetec + '\\binaryary.mp4', fourcc, 30, (640, 360))
    video_edges = cv2.VideoWriter(video_savepath_framedetec + '\\edges.mp4', fourcc, 30, (640, 360))

    base_time = time.time()

    avg = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        gau = cv2.GaussianBlur(frame, (5, 5), 1)
        gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.Canny(binary, 0, 130)
        contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if y > 50 and area > 25000:
                xlist_framedetec.append(x)
                ylist_framedetec.append(y)
                wlist_framedetec.append(w)
                hlist_framedetec.append(h)
                cv2.rectangle(edges, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Edges', edges)

        run_time = time.time() - base_time

        if run_time > 15: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        video_capture.write(frame)
        video_gaussian.write(gau)
        video_gray.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        video_framedeleta.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        video_binaryary.write(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        video_edges.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    average_x = sum(xlist_framedetec) / len(xlist_framedetec)
    average_y = sum(ylist_framedetec) / len(ylist_framedetec)
    average_w = sum(wlist_framedetec) / len(wlist_framedetec)
    average_h = sum(hlist_framedetec) / len(hlist_framedetec)

    xmin = int(average_x - 20)
    xmax = int(xmin + average_w + 60)
    ymin = int(average_y - 100)
    ymax = int(ymin + average_h + 40)

    cap.release()
    video_capture.release()
    video_gaussian.release()
    video_gray.release()
    video_framedeleta.release()
    video_binaryary.release()
    video_edges.release()
    cv2.destroyAllWindows()

    return xmin, xmax, ymin, ymax

def Ratio_calculation(binframe, xmin, xmax, ymin, ymax):
    size = int((xmax - xmin) * (ymax - ymin))
    white_pixel = cv2.countNonZero(binframe)
    white_ratio = (white_pixel / size) * 100

    return white_ratio

def Thresh_calculation(xmin, xmax, ymin, ymax):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    cut_width = xmax - xmin
    cut_height = ymax - ymin

    video_savepath_threshdetec = 'C:\\Users\\admin\\Desktop\\measurement_data\\thresh_detection\\' + date_number_path
    os.makedirs(video_savepath_threshdetec)

    video_capture = cv2.VideoWriter(video_savepath_threshdetec + '\\capture.mp4', fourcc, 30, (640, 360))
    video_cutframe = cv2.VideoWriter(video_savepath_threshdetec + '\\cutframe.mp4', fourcc, 30, (cut_width, cut_height))
    video_gaussian = cv2.VideoWriter(video_savepath_threshdetec + '\\gaussian.mp4', fourcc, 30, (cut_width, cut_height))
    video_gray = cv2.VideoWriter(video_savepath_threshdetec + '\\gray.mp4', fourcc, 30, (cut_width, cut_height))
    video_framedelta = cv2.VideoWriter(video_savepath_threshdetec + '\\framedelta.mp4', fourcc, 30, (cut_width, cut_height))
    video_binary_framedelta = cv2.VideoWriter(video_savepath_threshdetec + '\\binary_framedelta.mp4', fourcc, 30, (cut_width, cut_height))
    video_binary_line = cv2.VideoWriter(video_savepath_threshdetec + '\\binary_line.mp4', fourcc, 30, (cut_width, cut_height))
    video_horizon = cv2.VideoWriter(video_savepath_threshdetec + '\\horizon.mp4', fourcc, 30, (cut_width, cut_height))
    video_dilation = cv2.VideoWriter(video_savepath_threshdetec + '\\dilation.mp4', fourcc, 30, (cut_width, cut_height))
    video_closing = cv2.VideoWriter(video_savepath_threshdetec + '\\closing.mp4', fourcc, 30, (cut_width, cut_height))
    video_opening = cv2.VideoWriter(video_savepath_threshdetec + '\\opening.mp4', fourcc, 30, (cut_width, cut_height))

    base_time = time.time()
    blink_time = 0

    avg = None

    kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
    kernel_hor /= 9

    kernel_calculation = np.ones((3, 3), np.uint8)


    while True:
        ret, frame = cap.read()
        if not ret: break

        cut_frame = frame[ymin:ymax, xmin:xmax]
        gau = cv2.GaussianBlur(cut_frame, (5, 5), 1)
        gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = Ratio_calculation(binary_fd, xmin, xmax, ymin, ymax)
        time_diff = time.time() - blink_time

        if time_diff > 0.2:
            blink_time = time.time()
            ratiolist_calculation.append(whiteratio)

        binary_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_line, -1, kernel = kernel_hor)
        dilation = cv2.dilate(horizon, kernel = kernel_calculation, iterations = 1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel = kernel_calculation)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel = kernel_calculation)
        lines = cv2.HoughLinesP(opening, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                Gradient = 10 * (delta_Y / delta_X)
                if Gradient < 6 and Gradient > 0 and x1 < xmax:
                    gradientlist_calculation.append(Gradient)
                    cv2.line(cut_frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

        cv2.imshow('Blink', binary_fd)
        cv2.imshow('Cut frame', cut_frame)

        run_time = time.time() - base_time

        if run_time > 15: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        video_capture.write(frame)
        video_cutframe.write(cut_frame)
        video_gaussian.write(gau)
        video_gray.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        video_framedelta.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        video_binary_framedelta.write(cv2.cvtColor(binary_fd, cv2.COLOR_GRAY2BGR))
        video_binary_line.write(cv2.cvtColor(binary_line, cv2.COLOR_GRAY2BGR))
        video_horizon.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        video_dilation.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        video_closing.write(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR))
        video_opening.write(cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR))
        
    cap.release()
    video_capture.release()
    video_cutframe.release()
    video_gaussian.release()
    video_gray.release()
    video_framedelta.release()
    video_binary_framedelta.release()
    video_binary_line.release()
    video_horizon.release()
    video_dilation.release()
    video_closing.release()
    video_opening.release()
    cv2.destroyAllWindows()

    sorted_ratiolist = sorted(ratiolist_calculation, reverse = True)
    max_ratio = sorted_ratiolist[0]
    thresh_ratio_high = int(max_ratio + 7)
    thresh_ratio_low = int(max_ratio - 7)

    sorted_gradientlist = sorted(gradientlist_calculation, reverse = True)
    max_gradient = sorted_gradientlist[0]
    thresh_gradient_high = max_gradient + 1.5
    thresh_gradient_low = max_gradient - 1.5

    return thresh_ratio_high, thresh_ratio_low, thresh_gradient_high, thresh_gradient_low

def Excel_data_entry():
    save_path = 'C:\\Users\\admin\\Desktop\\measurement_data\\excel_data\\' + date_number_path + '.xlsx'
    wb = openpyxl.Workbook()

    sheetname_framedetection = 'frame_detaction'
    sheetname_threshdetection = 'thresh_detection'
    sheetname_measurement = 'measurement'

    wb.create_sheet(sheetname_framedetection)
    wb.create_sheet(sheetname_threshdetection)
    wb.create_sheet(sheetname_measurement)

    ws_fd = wb[sheetname_framedetection]
    ws_td = wb[sheetname_threshdetection]
    ws_msr = wb[sheetname_measurement]

    ws_fd['B2'] = 'Frame detection'
    ws_fd['D3'] = 'x'
    ws_fd['E3'] = 'y'
    ws_fd['F3'] = 'w'
    ws_fd['G3'] = 'h'
    ws_fd['I3'] = 'xmin'
    ws_fd['I4'] = 'xmax'
    ws_fd['I5'] = 'ymin'
    ws_fd['I6'] = 'ymax'

    ws_td['B2'] = 'Thresh detection'
    ws_td['D3'] = 'gradient'
    ws_td['E3'] = 'gradient (descending)'
    ws_td['G3'] = 'deviation of gradient (positive)'
    ws_td['H3'] = 1.5
    ws_td['G4'] = 'deviation of gradient (negative)'
    ws_td['H4'] = 1.5
    ws_td['G5'] = 'max gradient'
    ws_td['G6'] = 'low thresh of gradient'
    ws_td['G7'] = 'high thresh of gradient'
    ws_td['J3'] = 'ratio'
    ws_td['K3'] = 'ratio (descending)'
    ws_td['M3'] = 'deviation of ratio'
    ws_td['N3'] = 7
    ws_td['M4'] = 'max ratio'
    ws_td['M5'] = 'low thresh of ratio'
    ws_td['M6'] = 'high thresh of ratiot'

    ws_msr['B2'] = 'measurement'
    ws_msr['D2'] = 'detection data'
    ws_msr['D3'] = 'time'
    ws_msr['E3'] = 'gradient'
    ws_msr['F3'] = 'ratio'
    ws_msr['H2'] = 'comparison data'
    ws_msr['H3'] = 'time'
    ws_msr['I3'] = 'gradient'
    ws_msr['J3'] = 'ratio'
    ws_msr['L2'] = 'all data'
    ws_msr['L3'] = 'time'
    ws_msr['M3'] = 'gradient'
    ws_msr['N3'] = 'ratio'


    average_x = sum(xlist_framedetec) / len(xlist_framedetec)
    average_y = sum(ylist_framedetec) / len(ylist_framedetec)
    average_w = sum(wlist_framedetec) / len(wlist_framedetec)
    average_h = sum(hlist_framedetec) / len(hlist_framedetec)

    xmin = int(average_x - 20)
    xmax = int(xmin + average_w + 60)
    ymin = int(average_y - 60)
    ymax = int(ymin + average_h + 40)

    sorted_ratiolist = sorted(ratiolist_calculation, reverse = True)
    max_ratio = sorted_ratiolist[0]
    thresh_ratio_high = int(max_ratio + 7)
    thresh_ratio_low = int(max_ratio - 7)

    sorted_gradientlist = sorted(gradientlist_calculation, reverse = True)
    max_gradient = sorted_gradientlist[0]
    thresh_gradient_high = max_gradient + 1.5
    thresh_gradient_low = max_gradient - 1.5

    for i0 in range(0, len(xlist_framedetec)):
        ws_fd.cell(i0 + 4, 4, value = xlist_framedetec[i0])
        ws_fd.cell(i0 + 4, 5, value = ylist_framedetec[i0])
        ws_fd.cell(i0 + 4, 6, value = wlist_framedetec[i0])
        ws_fd.cell(i0 + 4, 7, value = hlist_framedetec[i0])
    
    ws_fd.cell(3, 10, value = xmin)
    ws_fd.cell(4, 10, value = xmax)
    ws_fd.cell(5, 10, value = ymin)
    ws_fd.cell(6, 10, value = ymax)

    for i1 in range(0, len(gradientlist_calculation)):
        ws_td.cell(i1 + 4, 4, value = gradientlist_calculation[i1])
        ws_td.cell(i1 + 4, 5, value = sorted_gradientlist[i1])

    for i2 in range(0, len(ratiolist_calculation)):
        ws_td.cell(i2 + 4, 10, value = ratiolist_calculation[i2])
        ws_td.cell(i2 + 4, 11, value = sorted_ratiolist[i2])
    
    ws_td.cell(5, 8, value = max_gradient)
    ws_td.cell(6, 8, value = thresh_gradient_low)
    ws_td.cell(7, 8, value = thresh_gradient_high)
    ws_td.cell(4, 14, value = max_ratio)
    ws_td.cell(5, 14, value = thresh_ratio_low)
    ws_td.cell(6, 14, value = thresh_ratio_high)

    for i3 in range(0, len(timelist_detec)):
        ws_msr.cell(i3 + 4, 4, value = timelist_detec[i3])
        ws_msr.cell(i3 + 4, 5, value = gradientlist_detec[i3])
        ws_msr.cell(i3 + 4, 6, value = ratiolist_detec[i3])

    for i4 in range(0, len(comparison_time)):
        ws_msr.cell(i4 + 4, 8, value = comparison_time[i4])
        ws_msr.cell(i4 + 4, 9, value = comparison_gradient[i4])
        ws_msr.cell(i4 + 4, 10, value = comparison_ratio[i4])

    for i5 in range(0, len(timelist)):
        ws_msr.cell(i5 + 4, 12, value = timelist[i5])
        ws_msr.cell(i5 + 4, 13, value = gradientlist[i5])
        ws_msr.cell(i5 + 4, 14, value = ratiolist[i5])

    wb.save(save_path)
    wb.close()

def main():
    xmin, xmax, ymin, ymax = Frame_detect()
    th_ratio_high, th_ratio_low, th_grad_high, th_grad_low = Thresh_calculation(xmin, xmax, ymin, ymax)

    fonttype = cv2.FONT_HERSHEY_COMPLEX

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    width = xmax - xmin
    height = ymax - ymin

    video_savepath_main = 'C:\\Users\\admin\\Desktop\\measurement_data\\measurement\\' + date_number_path
    os.makedirs(video_savepath_main)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    frame_save = cv2.VideoWriter(video_savepath_main + '\\frame.mp4', fourcc, 30, (640, 360))
    cutframe_save = cv2.VideoWriter(video_savepath_main + '\\cutframe.mp4', fourcc, 30, (width, height))
    gau_save = cv2.VideoWriter(video_savepath_main + '\\gaussian.mp4', fourcc, 30, (width, height))
    gray_save = cv2.VideoWriter(video_savepath_main + '\\gray.mp4', fourcc, 30, (width, height))
    binline_save = cv2.VideoWriter(video_savepath_main + '\\bin_line.mp4', fourcc, 30, (width, height))
    horizon_save = cv2.VideoWriter(video_savepath_main + '\\horizon.mp4', fourcc, 30, (width, height))
    dilation_save = cv2.VideoWriter(video_savepath_main + '\\dilation.mp4', fourcc, 30, (width, height))
    closing_save = cv2.VideoWriter(video_savepath_main + '\\closing.mp4', fourcc, 30, (width, height))
    opening_save = cv2.VideoWriter(video_savepath_main + '\\openig.mp4', fourcc, 30, (width, height))
    framedelta_save = cv2.VideoWriter(video_savepath_main + '\\framedelta.mp4', fourcc, 30, (width, height))
    binfd_save = cv2.VideoWriter(video_savepath_main + '\\bin_fd.mp4', fourcc, 30, (width, height))

    avg = None

    kernel_hor = np.array([
        [1, 2, 1], 
        [0, 0, 0], 
        [-1, -2, -1]], dtype = np.float32)
    kernel_hor /= 9

    kernel_detec = np.ones((5, 5), np.uint8)

    base_time = time.time()
    blink_time = 0

    val = 0

    while True:
        judge0, judge1 = 0, 0

        ret, frame = cap.read()
        if not ret: break

        cut_frame = frame[ymin:ymax, xmin:xmax]
        gaussian = cv2.GaussianBlur(cut_frame, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)


        bin_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(bin_line, -1, kernel = kernel_hor)
        dilation = cv2.dilate(horizon, kernel = kernel_detec, iterations = 1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel = kernel_detec)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel = kernel_detec)
        lines = cv2.HoughLinesP(opening, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                Gradient = 10 * (delta_Y / delta_X)
                if Gradient < 10 and Gradient > -5:
                    cv2.line(cut_frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        bin_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = Ratio_calculation(bin_fd, xmin, xmax, ymin, ymax)

        cmp_time = time.time() - base_time
        comparison_time.append(cmp_time)
        comparison_gradient.append(Gradient)
        comparison_ratio.append(whiteratio)

        if Gradient < th_grad_high and Gradient > th_grad_low:
            time_diff = time.time() - blink_time
            if time_diff > 0.2:
                if whiteratio < th_ratio_high and whiteratio > th_ratio_low:
                    val += 1
                    blink_time = time.time()
                    detec_time = time.time() - base_time
                    timelist_detec.append(detec_time)
                    gradientlist_detec.append(Gradient)
                    ratiolist_detec.append(whiteratio)
                    judge0 = 1

                if judge0 == 0:
                    indexW = comparison_ratio[-2]
                    indexV = comparison_ratio[-3]

                    if indexW < th_ratio_high and indexW > th_ratio_low:
                        val += 1
                        blink_time = time.time()
                        detec_time = time.time() - base_time
                        timelist_detec.append(detec_time)
                        gradientlist_detec.append(Gradient)
                        ratiolist_detec.append(indexW)
                        judge1 = 1

                    if judge0 == 0 and judge1 == 0:
                        if indexV < th_ratio_high and indexV > th_ratio_low:
                            val += 1
                            blink_time = time.time()
                            detec_time = time.time() - base_time
                            timelist_detec.append(detec_time)
                            gradientlist_detec.append(Gradient)
                            ratiolist_detec.append(indexV)

        run_time = time.time() - base_time

        timelist.append(run_time)
        gradientlist.append(Gradient)
        ratiolist.append(whiteratio)

        cv2.putText(frame, 'Count:', (10, 350), fonttype, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(val), (150, 350), fonttype, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        frame_save.write(frame)
        cutframe_save.write(cut_frame)
        gau_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        binline_save.write(cv2.cvtColor(bin_line, cv2.COLOR_GRAY2BGR))
        horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        closing_save.write(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR))
        opening_save.write(cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binfd_save.write(cv2.cvtColor(bin_fd, cv2.COLOR_GRAY2BGR))

        if run_time > 900: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print('Gradient thresh low:', th_grad_low)
    print('Gradient thresh high:', th_grad_high)
    print('Ratio thresh low:', th_ratio_low)
    print('Ratio thresh high:', th_ratio_high)

    Excel_data_entry()

    cap.release()
    cutframe_save.release()
    gau_save.release()
    gray_save.release()
    binline_save.release()
    horizon_save.release()
    dilation_save.release()
    closing_save.release()
    opening_save.release()
    framedelta_save.release()
    binfd_save.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()