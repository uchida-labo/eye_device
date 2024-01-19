import cv2, openpyxl, os, time, math, statistics, threading
import numpy as np

# List difinition
xlist_FAD, ylist_FAD, wlist_FAD, hlist_FAD, timelist_FAD = [], [], [], [], []
x0list_TAD, y0list_TAD, x1list_TAD, y1list_TAD, timelist_TAD, gradlist_TAD, ratiolist_TAD, degreelist_TAD, radianlist_TAD, stopblinktime_TAD, stopblinkdegree_TAD = [], [], [], [], [], [], [], [], [], [], []
x0list_main, y0list_main, x1list_main, y1list_main, timelist_main, gradlist_main, degreelist_main, radianlist_main, ratiolist_main = [], [], [], [], [], [], [], [], []
comparison_time, comparison_gradient, comparison_degree, comparison_ratio = [0, 0], [0, 0], [0, 0], [0, 0]
vallist_detec_main, timelist_detec_main, gradlist_detec_main, degreelist_detec_main, radianlist_detec_main, ratiolist_detec_main = [], [], [], [], [], []
stopblink_time_main, stopblink_degree_main, noblinktimelist_main, intervallist_main = [], [], [], []
noblinktimelist_detect, tolerancetimelist_detect = [], []
timelist_threshcalculation_interval20, ratiolist_threshcalculation_interval20, gradlist_threshcalculation_interval20, degreelist_threshcalculation_interval20 = [], [], [], []
timelist_threshcalculation_interval40, ratiolist_threshcalculation_interval40, gradlist_threshcalculation_interval40, degreelist_threshcalculation_interval40 = [], [], [], []
starttimerotaionlist_interval20, maxlist_ratio_interval20, maxlist_grad_interval20, maxlist_degree_interval20, degreedifflist_interval20 = [], [], [], [], []
starttimerotaionlist_interval40, maxlist_ratio_interval40, maxlist_grad_interval40, maxlist_degree_interval40, degreedifflist_interval40 = [], [], [], [], []

date = time.strftime('%m%d_%H%M%S')

# --Frame Auto Detection Process--

def ExcelEntry_FAD(savepath):
    excelpath_FAD = savepath + '\\index_FAD.xlsx'
    wb = openpyxl.Workbook()

    wb.create_sheet(date)
    ws = wb[date]

    dispersion_xmin = 20
    dispersion_xmax = 50
    dispersion_ymin = 80
    dispersion_ymax = 80

    ws['C3'] = 'time'
    ws['D3'] = 'x'
    ws['E3'] = 'y'
    ws['F3'] = 'w'
    ws['G3'] = 'h'
    ws['I3'] = 'dispersion:xmin(' + str(dispersion_xmin) + '), xmax(' + str(dispersion_xmax) + '), ymin(' + str(dispersion_ymin) + '), ymax(' + str(dispersion_ymax) + ')'
    ws['I4'] = 'xmin'
    ws['I5'] = 'xmax'
    ws['I6'] = 'ymin'
    ws['I7'] = 'ymax'

    average_x = sum(xlist_FAD) / len(xlist_FAD)
    average_y = sum(ylist_FAD) / len(ylist_FAD)
    average_w = sum(wlist_FAD) / len(wlist_FAD)
    average_h = sum(hlist_FAD) / len(hlist_FAD)

    xmin = int(average_x - dispersion_xmin)
    xmax = int(xmin + average_w + dispersion_xmax)
    ymin = int(average_y - dispersion_ymin)
    ymax = int(ymin + average_h + dispersion_ymax)

    for i in range(0, len(timelist_FAD)):
        ws.cell(i + 4, 3, timelist_FAD[i])
        ws.cell(i + 4, 4, xlist_FAD[i])
        ws.cell(i + 4, 5, ylist_FAD[i])
        ws.cell(i + 4, 6, wlist_FAD[i])
        ws.cell(i + 4, 7, hlist_FAD[i])

    ws.cell(4, 10, xmin)
    ws.cell(5, 10, xmax)
    ws.cell(6, 10, ymin)
    ws.cell(7, 10, ymax)

    wb.save(excelpath_FAD)
    wb.close()

    return xmin, xmax, ymin, ymax

def FrameAutoDetect():
    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    savepath = 'C:\\Users\\admin\\Desktop\\blink_data\\FrameAutoDetect\\' + date
    os.makedirs(savepath)
    frame_save = cv2.VideoWriter(savepath + '\\frame.mp4', fourcc, 30, (640, 480))
    gaussian_save = cv2.VideoWriter(savepath + '\\gaussian.mp4', fourcc, 30, (640, 480))
    gray_save = cv2.VideoWriter(savepath + '\\gray.mp4', fourcc, 30, (640, 480))
    framedelta_save = cv2.VideoWriter(savepath + '\\framedelta.mp4', fourcc, 30, (640, 480))
    binary_save = cv2.VideoWriter(savepath + '\\binary.mp4', fourcc, 30, (640, 480))
    edges_save = cv2.VideoWriter(savepath + '\\edges.mp4', fourcc, 30, (640, 480))

    # Parmeter setting
    basetime = time.time()
    avg = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

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
                xlist_FAD.append(x)
                ylist_FAD.append(y)
                wlist_FAD.append(w)
                hlist_FAD.append(h)
                comparison_time = time.time() - basetime
                timelist_FAD.append(comparison_time)
                cv2.rectangle(edges, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        cv2.imshow('frame (frame detection)', frame)
        cv2.imshow('edges', edges)

        frame_save.write(frame)
        gaussian_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binary_save.write(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        edges_save.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

        runtime = time.time() - basetime

        if runtime > 15: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    frame_save.release()
    gaussian_save.release()
    gray_save.release()
    framedelta_save.release()
    binary_save.release()
    edges_save.release()
    cap.release()
    cv2.destroyAllWindows()

    xmin, xmax, ymin, ymax = ExcelEntry_FAD(savepath)

    return xmin, xmax, ymin, ymax

# --------------------------------


# --Threshold Auto Detection Process--

def ExcelEntry_TAD(savepath):
    dispersion_grad = 1.5
    sorted_gradlist = sorted(gradlist_TAD, reverse = True)
    thresh_grad_high = sorted_gradlist[0] + dispersion_grad
    thresh_grad_low = sorted_gradlist[0] - dispersion_grad

    dispersion_ratio = 7
    sorted_ratiolist = sorted(ratiolist_TAD, reverse = True)
    thresh_ratio_high = int(sorted_ratiolist[0] + dispersion_ratio)
    thresh_ratio_low = int(sorted_ratiolist[0] - dispersion_ratio)

    average_degree = sum(stopblinkdegree_TAD) / len(stopblinkdegree_TAD)

    excelpath_TAD = savepath + '\\index_TAD.xlsx'
    wb = openpyxl.Workbook()

    wb.create_sheet(date)
    ws = wb[date]

    ws['D6'] = 'x0'
    ws['E6'] = 'y0'
    ws['F6'] = 'x1'
    ws['G6'] = 'y1'

    ws['I6'] = 'time'
    ws['J6'] = 'gradient'
    ws['K6'] = 'radian'
    ws['L6'] = 'degree'
    ws['M6'] = 'whiteratio'

    ws['J1'] = 'dispersion grad : ' + str(dispersion_grad)
    ws['J2'] = 'max gradient : ' + str(sorted_gradlist[0])
    ws['J3'] = 'grad thresh high : ' + str(thresh_grad_high)
    ws['J4'] = 'grad thresh low : ' + str(thresh_grad_low)

    ws['M1'] = 'dispersion ratio : ' + str(dispersion_ratio)
    ws['M2'] = 'max ratio : ' + str(sorted_ratiolist[0])
    ws['M3'] = 'ratio thresh high : ' + str(thresh_ratio_high)
    ws['M4'] = 'ratio thresh low : ' + str(thresh_ratio_low)

    ws['O6'] = 'time'
    ws['P6'] = 'stop blink degree'

    ws['P4'] = 'average degree (stop blink) : ' + str(average_degree)

    for i0 in range(0, len(x0list_TAD)):
        ws.cell(i0 + 7, 4, x0list_TAD[i0])
        ws.cell(i0 + 7, 5, y0list_TAD[i0])
        ws.cell(i0 + 7, 6, x1list_TAD[i0])
        ws.cell(i0 + 7, 7, y1list_TAD[i0])
        ws.cell(i0 + 7, 9, timelist_TAD[i0])
        ws.cell(i0 + 7, 10, gradlist_TAD[i0])
        ws.cell(i0 + 7, 11, radianlist_TAD[i0])
        ws.cell(i0 + 7, 12, degreelist_TAD[i0])
    
    for i1 in range(0, len(ratiolist_TAD)):
        ws.cell(i1 + 7, 13, ratiolist_TAD[i1])
    
    for i2 in range(0, len(stopblinktime_TAD)):
        ws.cell(i2 + 7, 15, stopblinktime_TAD[i2])
        ws.cell(i2 + 7, 16, stopblinkdegree_TAD[i2])

    wb.save(excelpath_TAD)
    wb.close()

    return thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, average_degree

def EyelidParameterCalculation(x0, y0, x1, y1):
    parallel = x1 - x0
    perpendicular = y1 - y0
    gradient = (perpendicular / parallel) * 10
    oblique = math.sqrt(parallel ** 2 + perpendicular ** 2)
    radian = np.arccos(parallel / oblique)
    degree = np.rad2deg(radian)

    return gradient, degree, radian

def ThreshAutoDetection(xmin, xmax, ymin, ymax):
    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = xmax - xmin
    height = ymax - ymin
    savepath = 'C:\\Users\\admin\\Desktop\\blink_data\\ThreshAutoDetection\\' + date
    os.makedirs(savepath)
    frame_save = cv2.VideoWriter(savepath + '\\frame.mp4', fourcc, 30, (640, 480))
    copyframe_save = cv2.VideoWriter(savepath + '\\copyframe.mp4', fourcc, 30, (640, 480))
    cutframe_save = cv2.VideoWriter(savepath + '\\cutframe.mp4', fourcc, 30, (width, height))
    cutframecopy_save = cv2.VideoWriter(savepath + '\\cutframe_copy.mp4', fourcc, 30, (width, height))
    gaussian_save = cv2.VideoWriter(savepath + '\\gaussian.mp4', fourcc, 30, (width, height))
    gray_save = cv2.VideoWriter(savepath + '\\gray.mp4', fourcc, 30, (width, height))
    binary_eyelid_save = cv2.VideoWriter(savepath + '\\binary_eyelid.mp4', fourcc, 30, (width, height))
    horizon_save = cv2.VideoWriter(savepath + '\\horizon.mp4', fourcc, 30, (width, height))
    dilation_save = cv2.VideoWriter(savepath + '\\dilation.mp4', fourcc, 30, (width, height))
    framedelta_save = cv2.VideoWriter(savepath + '\\framedelta.mp4', fourcc, 30, (width, height))
    binaryframedelta_save = cv2.VideoWriter(savepath + '\\binary_framedelta.mp4', fourcc, 30, (width, height))

    # Parameter setting
    basetime = time.time()
    blinktime = 0
    avg = None
    kernel_hor = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_cal = np.ones((3, 3), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break

        copyframe = frame.copy()
        cutframe_copy = copyframe[ymin:ymax, xmin:xmax]
        cutframe = frame[ymin:ymax, xmin:xmax]

        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_framedelta = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100
        timediff = time.time() - blinktime
        if timediff > 0.2:
            blinktime = time.time()
            ratiolist_TAD.append(whiteratio)

        binary_eyelid = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_eyelid, -1, kernel_hor)
        dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
        lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 130, minLineLength = 130, maxLineGap = 100)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                gradient, degree, radian = EyelidParameterCalculation(x0, y0, x1, y1)
                if y0 > y1:
                    degree = degree * (-1)


            if gradient< 10 and gradient > -5 and x1 < (xmax - 20):
                x0list_TAD.append(x0)
                y0list_TAD.append(y0)
                x1list_TAD.append(x1)
                y1list_TAD.append(y1)
                gradlist_TAD.append(gradient)
                degreelist_TAD.append(degree)
                radianlist_TAD.append(radian)
                comparison_time = time.time() - basetime
                timelist_TAD.append(comparison_time)
                cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)

                if whiteratio < 5:
                    stopblinktime = time.time() - basetime
                    stopblinktime_TAD.append(stopblinktime)
                    stopblinkdegree_TAD.append(degree)

        cv2.rectangle(copyframe, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('Binary', binary_framedelta)
        cv2.imshow('cut frame', cutframe)

        frame_save.write(frame)
        copyframe_save.write(copyframe)
        cutframe_save.write(cutframe)
        cutframecopy_save.write(cutframe_copy)
        gaussian_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        binary_eyelid_save.write(cv2.cvtColor(binary_eyelid, cv2.COLOR_GRAY2BGR))
        horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binaryframedelta_save.write(cv2.cvtColor(binary_framedelta, cv2.COLOR_GRAY2BGR))

        runtime = time.time() - basetime

        if runtime > 15: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    frame_save.release()
    copyframe_save.release()
    cutframe_save.release()
    cutframecopy_save.release()
    gaussian_save.release()
    gray_save.release()
    binary_eyelid_save.release()
    horizon_save.release()
    dilation_save.release()
    framedelta_save.release()
    binaryframedelta_save.release()
    cap.release()
    cv2.destroyAllWindows()

    thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, average_degree = ExcelEntry_TAD(savepath)

    return thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, average_degree

# ------------------------------------


# --main loop--

def ExcelEntry_main(savepath):
    excelpath_main = savepath + '\\index_main.xlsx'
    sheetname_rotation = 'rotation'
    sheetname_alldata = 'all data'

    wb = openpyxl.Workbook()
    wb.create_sheet(date)
    wb.create_sheet(sheetname_rotation)
    wb.create_sheet(sheetname_alldata)

    ws = wb[date]
    ws_r = wb[sheetname_rotation]
    ws_a = wb[sheetname_alldata]

    average_stopblink_degree = sum(stopblink_degree_main) / len(stopblink_degree_main)

    ws['D2'] = 'eyelid lines'
    ws['D3'] = 'x0'
    ws['E3'] = 'y0'
    ws['F3'] = 'x1'
    ws['G3'] = 'y1'

    ws['I2'] = 'calculation'
    ws['I3'] = 'time'
    ws['J3'] = 'gradient'
    ws['K3'] = 'degree'
    ws['L3'] = 'radian'
    ws['M3'] = 'whiteratio'

    ws['O2'] = 'average stop blink degree : ' + str(average_stopblink_degree)
    ws['O3'] = 'time'
    ws['P3'] = 'stop blink degree'

    ws['S2'] = 'detections'
    ws['S3'] = 'count'
    ws['T3'] = 'time'
    ws['U3'] = 'gradient'
    ws['V3'] = 'degree'
    ws['W3'] = 'radian'
    ws['X3'] = 'whiteratio'

    ws['Z2'] = 'no blinks'
    ws['Z3'] = 'time'
    ws['AA3'] = 'interval'

    ws['AC3'] = 'time'
    ws['AD3'] = 'tolerance time'

    ws_r['D2'] = 'thresh calculation for interval 20'
    ws_r['D3'] = 'time'
    ws_r['E4'] = 'whiteratio'
    ws_r['F3'] = 'gradient'
    ws_r['G3'] = 'degree'

    ws_r['I2'] = 'thresh calculation for interval 40'
    ws_r['I3'] = 'time'
    ws_r['J3'] = 'whiteratio'
    ws_r['K3'] = 'gradient'
    ws_r['L3'] = 'degree'

    ws_r['N1'] = 'dispersion = ratio(+4, -10), grad(+2, -1)'
    ws_r['N2'] = 'max values in interval > 20'
    ws_r['N3'] = 'time'
    ws_r['O3'] = 'whiteratio'
    ws_r['P3'] = 'gradient'
    ws_r['Q3'] = 'degree'
    ws_r['R3'] = 'degree diff'

    ws_r['T2'] = 'max values in interval > 40'
    ws_r['T3'] = 'time'
    ws_r['U3'] = 'whiteratio'
    ws_r['V3'] = 'gradient'
    ws_r['W3'] = 'degree'
    ws_r['X3'] = 'degree diff'

    ws_a['D3'] = 'time'
    ws_a['E3'] = 'whiteratio'
    ws_a['F3'] = 'gradient'
    ws_a['G3'] = 'degree'


    for i0 in range(0, len(x0list_main)):
        ws.cell(i0 + 4, 4, x0list_main[i0])
        ws.cell(i0 + 4, 5, y0list_main[i0])
        ws.cell(i0 + 4, 6, x1list_main[i0])
        ws.cell(i0 + 4, 7, y1list_main[i0])
        ws.cell(i0 + 4, 9, timelist_main[i0])
        ws.cell(i0 + 4, 10, gradlist_main[i0])
        ws.cell(i0 + 4, 11, degreelist_main[i0])
        ws.cell(i0 + 4, 12, radianlist_main[i0])
        ws.cell(i0 + 4, 13, ratiolist_main[i0])

    for i1 in range(0, len(stopblink_time_main)):
        ws.cell(i1 + 4, 15, stopblink_time_main[i1])
        ws.cell(i1 + 4, 16, stopblink_degree_main[i1])

    for i2 in range(0, len(vallist_detec_main)):
        ws.cell(i2 + 4, 19, vallist_detec_main[i2])
        ws.cell(i2 + 4, 20, timelist_detec_main[i2])
        ws.cell(i2 + 4, 21, gradlist_detec_main[i2])
        ws.cell(i2 + 4, 22, degreelist_detec_main[i2])
        ws.cell(i2 + 4, 23, radianlist_detec_main[i2])
        ws.cell(i2 + 4, 24, ratiolist_detec_main[i2])

    for i3 in range(0, len(noblinktimelist_main)):
        ws.cell(i3 + 4, 26, noblinktimelist_main[i3])
        ws.cell(i3 + 4, 27, intervallist_main[i3])

    for i4 in range(0, len(tolerancetimelist_detect)):
        ws.cell(i4 + 4, 29, noblinktimelist_detect[i4])
        ws.cell(i4 + 4, 30, tolerancetimelist_detect[i4])

    for i5 in range(0, len(timelist_threshcalculation_interval20)):
        ws_r.cell(i5 + 4, 4, timelist_threshcalculation_interval20[i5])
        ws_r.cell(i5 + 4, 5, ratiolist_threshcalculation_interval20[i5])
        ws_r.cell(i5 + 4, 6, gradlist_threshcalculation_interval20[i5])
        ws_r.cell(i5 + 4, 7, degreelist_threshcalculation_interval20[i5])

    for i6 in range(0, len(timelist_threshcalculation_interval40)):
        ws_r.cell(i6 + 4, 9, timelist_threshcalculation_interval20[i6])
        ws_r.cell(i6 + 4, 10, ratiolist_threshcalculation_interval20[i6])
        ws_r.cell(i6 + 4, 11, gradlist_threshcalculation_interval20[i6])
        ws_r.cell(i6 + 4, 12, degreelist_threshcalculation_interval20[i6])

    for i7 in range(0, len(starttimerotaionlist_interval20)):
        ws_r.cell(i7 + 4, 14, starttimerotaionlist_interval20[i7])
        ws_r.cell(i7 + 4, 15, maxlist_ratio_interval20[i7])
        ws_r.cell(i7 + 4, 16, maxlist_grad_interval20[i7])
        ws_r.cell(i7 + 4, 17, maxlist_degree_interval20[i7])
        ws_r.cell(i7 + 4, 18, degreedifflist_interval20[i7])

    for i8 in range(0, len(starttimerotaionlist_interval40)):
        ws_r.cell(i8 + 4, 20, starttimerotaionlist_interval40[i8])
        ws_r.cell(i8 + 4, 21, maxlist_ratio_interval40[i8])
        ws_r.cell(i8 + 4, 22, maxlist_grad_interval40[i8])
        ws_r.cell(i8 + 4, 23, maxlist_degree_interval40[i8])
        ws_r.cell(i8 + 4, 24, degreedifflist_interval40[i8])

    for i9 in range(0, len(comparison_time)):
        ws_a.cell(i9 + 4, 4, comparison_time[i9])
        ws_a.cell(i9 + 4, 5, comparison_ratio[i9])
        ws_a.cell(i9 + 4, 6, comparison_gradient[i9])
        ws_a.cell(i9 + 4, 7, comparison_degree[i9])

    wb.save(excelpath_main)
    wb.close()

def ListAppendForMain(val ,detectiontime, gradient, whiteratio, degree, radian):
    vallist_detec_main.append(val)
    timelist_detec_main.append(detectiontime)
    gradlist_detec_main.append(gradient)
    ratiolist_detec_main.append(whiteratio)
    degreelist_detec_main.append(degree)
    radianlist_detec_main.append(radian)

def main():
    date = time.strftime('%m%d_%H%M%S')
    xmin, xmax, ymin, ymax = FrameAutoDetect()
    grad_high, grad_low, ratio_high, ratio_low, ave_degree_TAD = ThreshAutoDetection(xmin, xmax, ymin, ymax)

    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = xmax - xmin
    height = ymax - ymin
    savepath = 'C:\\Users\\admin\\Desktop\\blink_data\\main_detection\\' + date
    os.makedirs(savepath)
    frame_save = cv2.VideoWriter(savepath + '\\frame.mp4', fourcc, 30, (640, 480))
    copyframe_save = cv2.VideoWriter(savepath + '\\copyframe.mp4', fourcc, 30, (640, 480))
    cutframe_save = cv2.VideoWriter(savepath + '\\cutframe.mp4', fourcc, 30, (width, height))
    gaussian_save = cv2.VideoWriter(savepath + '\\gaussian.mp4', fourcc, 30, (width, height))
    gray_save = cv2.VideoWriter(savepath + '\\gray.mp4', fourcc, 30, (width, height))
    binaryeyelid_save = cv2.VideoWriter(savepath + '\\binary_eyelid.mp4', fourcc, 30, (width, height))
    horizon_save = cv2.VideoWriter(savepath + '\\horizon.mp4', fourcc, 30, (width, height))
    dilation_save = cv2.VideoWriter(savepath + '\\dilation.mp4', fourcc, 30, (width, height))
    framedelta_save = cv2.VideoWriter(savepath + '\\framedelta.mp4', fourcc, 30, (width, height))
    binaryframedelta_save = cv2.VideoWriter(savepath + '\\binary_framedelta.mp4', fourcc, 30, (width, height))

    # Parameter setting
    basetime = time.time()
    blinktime = time.time()
    val = 0
    interval = 0
    avg = None
    rotation_flag = 1
    degree_diff = 0
    kernel_hor = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_cal = np.ones((3, 3), np.uint8)
    fonttype = cv2.FONT_HERSHEY_COMPLEX

    # List difinition

    while True:
        ret, frame = cap.read()
        if not ret: break

        if degree_diff > 0:
            rotation_flag = 1

        rotation_matrix = cv2.getRotationMatrix2D((320, 240), degree_diff, 1.0)
        frame = cv2.warpAffine(frame, rotation_matrix, (640, 480))

        copyframe = frame.copy()
        cutframe = frame[ymin:ymax, xmin:xmax]
        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_framedelta = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100
        

        binary_eyelid = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_eyelid, -1, kernel_hor)
        dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
        lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 130, minLineLength = 130, maxLineGap = 100)

        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                gradient, degree, radian = EyelidParameterCalculation(x0, y0, x1, y1)
                if y0 > y1:
                    degree = degree * (-1)

            x0list_main.append(x0)
            y0list_main.append(y0)
            x1list_main.append(x1)
            y1list_main.append(y1)
            gradlist_main.append(gradient)
            caltime = time.time() - basetime
            timelist_main.append(caltime)
            degreelist_main.append(degree)
            radianlist_main.append(radian)
            ratiolist_main.append(whiteratio)
            cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)

            if whiteratio < 5:
                stopblinktime = time.time() - basetime
                stopblink_time_main.append(stopblinktime)
                stopblink_degree_main.append(degree)

        else:
            comparison_time.append(False)
            comparison_gradient.append(False)
            comparison_degree.append(False)
            comparison_ratio.append(False)


        comp_time = time.time() - basetime
        comparison_time.append(comp_time)
        comparison_gradient.append(gradient)
        comparison_degree.append(degree)
        comparison_ratio.append(whiteratio)

        if gradient < grad_high and gradient > grad_low:
            timediff = time.time() - blinktime
            if timediff > 0.2:
                if whiteratio < ratio_high and whiteratio > ratio_low:
                    val += 1
                    blinktime = time.time()
                    detectiontime = time.time() - basetime
                    ListAppendForMain(val, detectiontime, gradient, whiteratio, degree, radian)
                    interval = 0

                else:
                    noblinktime = time.time() - basetime
                    interval = time.time() - blinktime
                    noblinktimelist_main.append(noblinktime)
                    intervallist_main.append(interval)
                    indexW = comparison_ratio[-2]
                    indexV = comparison_ratio[-3]

                    if indexW < ratio_high and indexW > ratio_low:
                        val += 1
                        blinktime = time.time()
                        detectiontime = time.time() - basetime
                        ListAppendForMain(val, detectiontime, gradient, whiteratio, degree, radian)
                        interval = 0

                    else:
                        noblinktime = time.time() - basetime
                        interval = time.time() - blinktime
                        noblinktimelist_main.append(noblinktime)
                        intervallist_main.append(interval)

                        if indexV < ratio_high and indexV > ratio_low:
                            val += 1
                            blinktime = time.time()
                            detectiontime = time.time() - basetime
                            ListAppendForMain(val, detectiontime, gradient, whiteratio, degree, radian)
                            interval = 0

                        else:
                            noblinktime = time.time() - basetime
                            interval = time.time() - blinktime
                            noblinktimelist_main.append(noblinktime)
                            intervallist_main.append(interval)

        interval = time.time() - blinktime

        if interval > 2:
            past_ave_ratio = sum(comparison_ratio[-3:]) / 3
            timelist_threshcalculation_interval20.append(time.time() - basetime)
            ratiolist_threshcalculation_interval20.append(whiteratio)
            gradlist_threshcalculation_interval20.append(gradient)
            degreelist_threshcalculation_interval20.append(degree)
            if past_ave_ratio < 5 or (gradient < grad_low and whiteratio < ratio_low) :
                noblinktime = time.time() - basetime
                interval = time.time() - blinktime
                noblinktimelist_detect.append(noblinktime)
                tolerancetimelist_detect.append(interval)

            if interval > 20 and rotation_flag == 1:
                timelist_threshcalculation_interval40.append(time.time() - basetime)
                ratiolist_threshcalculation_interval40.append(whiteratio)
                gradlist_threshcalculation_interval40.append(gradient)
                degreelist_threshcalculation_interval40.append(degree)

                ave_degree_main = sum(stopblink_degree_main) / len(stopblink_degree_main)
                degree_diff_cal = ave_degree_main - ave_degree_TAD
                print('Loop in (interval > 20)')

                if abs(degree_diff_cal) > 0:

                    print('Loop in (rotation_degree > 0)')

                    degree_diff = degree_diff_cal

                    max_ratio_interval20 = max(ratiolist_threshcalculation_interval20)
                    max_grad_interval20 = max(gradlist_threshcalculation_interval20)
                    max_degree_interval20 = max(degreelist_threshcalculation_interval20)
                    ratio_high, ratio_low = (max_ratio_interval20 + 4), (max_ratio_interval20 - 10)
                    grad_high, grad_low = (max_grad_interval20 + 2), (max_grad_interval20 - 1)

                    starttimerotaionlist_interval20.append(time.time() - basetime)
                    maxlist_ratio_interval20.append(max_ratio_interval20)
                    maxlist_grad_interval20.append(max_grad_interval20)
                    maxlist_degree_interval20.append(max_degree_interval20)
                    degreedifflist_interval20.append(degree_diff)

                    rotation_flag = 0

                    print('Rotation complete(interval > 20), rotation angle : ', degree_diff)

            if interval > 40 and rotation_flag == 0:

                ave_degree_main_increase = sum(stopblink_degree_main) / len(stopblink_degree_main)
                rotation_degree_increase_cal = ave_degree_main_increase - ave_degree_TAD
                print('Loop in (interval > 40)')
                if abs(rotation_degree_increase_cal) > 0:

                    print('Loop in (degree increase)')

                    rotation_degree_increase = rotation_degree_increase_cal

                    max_ratio_interval40 = max(ratiolist_threshcalculation_interval40)
                    max_grad_interval40 = max(gradlist_threshcalculation_interval40)
                    max_degree_interval40 = max(degreelist_threshcalculation_interval40)
                    ratio_high, ratio_low = (max_ratio_interval40 + 7), (max_ratio_interval40 - 7)
                    grad_high, grad_low = (max_grad_interval40 + 1.5), (max_grad_interval40 - 1.5)
                    
                    starttimerotaionlist_interval40.append(time.time() - basetime)
                    maxlist_ratio_interval40.append(max_ratio_interval40)
                    maxlist_grad_interval40.append(max_grad_interval40)
                    maxlist_degree_interval40.append(max_degree_interval40)
                    degreedifflist_interval40.append(rotation_degree_increase)

                    rotation_flag = 2

                    print('Rotation complete(interval > 40), rotation angle : ', rotation_degree_increase)
                # print('rotation start : ', starttime_rotation)

        runtime = time.time() - basetime

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, 'count : ', (10, 350), fonttype, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(val), (150, 350), fonttype, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'time : ', (10, 400), fonttype, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(interval), (150, 400), fonttype, 1, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('cutframe', cutframe)

        frame_save.write(frame)
        copyframe_save.write(copyframe)
        cutframe_save.write(cutframe)
        gaussian_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        binaryeyelid_save.write(cv2.cvtColor(binary_eyelid, cv2.COLOR_GRAY2BGR))
        horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binaryframedelta_save.write(cv2.cvtColor(binary_framedelta, cv2.COLOR_GRAY2BGR))

        if runtime > 900: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    copyframe_save.release()
    cutframe_save.release()
    gaussian_save.release()
    gray_save.release()
    binaryeyelid_save.release()
    horizon_save.release()
    dilation_save.release()
    framedelta_save.release()
    binaryframedelta_save.release()
    cv2.destroyAllWindows()

    ExcelEntry_main(savepath)

# -------------

if __name__ == '__main__':
    main()