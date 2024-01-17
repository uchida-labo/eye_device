import cv2, openpyxl, os, time, math, statistics, threading
import numpy as np
from queue import Queue

# --Frame auto detect process--

def ExcelEntry_FAD(path, version, xlist, ylist, wlist, hlist, timelist):
    excelpath_FAD = 'C:\\Users\\admin\\blink_data\\' + path + '\\FrameAutoDetect\\framecoordinate.xlsx'
    wb = openpyxl.Workbook()

    wb.create_sheet(version)
    ws = wb[version]

    dispersion_xmin = 20
    dispersion_xmax = 60
    dispersion_ymin = 80
    dispersion_ymax = 60

    ws['B2'] = 'version : ' + version
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

    average_x = sum(xlist) / len(xlist)
    average_y = sum(ylist) / len(ylist)
    average_w = sum(wlist) / len(wlist)
    average_h = sum(hlist) / len(hlist)

    xmin = int(average_x - dispersion_xmin)
    xmax = int(xmin + average_w + dispersion_xmax)
    ymin = int(average_y - dispersion_ymin)
    ymax = int(ymin + average_h + dispersion_ymax)

    for i in range(0, len(timelist)):
        ws.cell(i + 4, 3, timelist[i])
        ws.cell(i + 4, 4, xlist[i])
        ws.cell(i + 4, 5, ylist[i])
        ws.cell(i + 4, 6, wlist[i])
        ws.cell(i + 4, 7, hlist[i])

    ws.cell(4, 10, xmin)
    ws.cell(5, 10, xmax)
    ws.cell(6, 10, ymin)
    ws.cell(7, 10, ymax)

    wb.save(excelpath_FAD)
    wb.close()

    return xmin, xmax, ymin, ymax

def FrameAutoDetect(path, version, rotation_status):
    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    savepath = 'C:\\Users\\admin\\blink_data\\' + path + '\\FrameAutoDetect' + version
    if rotation_status is not None:
        savepath = savepath + '_rotation'
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

    # List difinitioin
    xlist_FAD, ylist_FAD, wlist_FAD, hlist_FAD, timelist_FAD = [], [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret: break

        if rotation_status is not None:
            frame = cv2.warpAffine(frame, rotation_status, (640, 480))

        gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)

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

    xmin, xmax, ymin, ymax = ExcelEntry_FAD(path, version, xlist_FAD, ylist_FAD, wlist_FAD, hlist_FAD, timelist_FAD)

    return xmin, xmax, ymin, ymax

# -----------------------------



# --Threshold auto detect process--

def EyelidParameterCalculation(x0list, y0list, x1list, y1list):
    parallel = x1list[-1] - x0list[-1]
    perpendicular = y1list[-1] - y0list[-1]
    gradient = (perpendicular / parallel) * 10
    oblique = math.sqrt(parallel ** 2 + perpendicular ** 2)
    radian = np.arccos(parallel / oblique)
    degree = int(np.rad2deg(radian))

    return gradient, degree, radian

def ExcelEntry_TAD(path, version, x0list, y0list, x1list, y1list, timelist, gradlist, ratiolist, degreelist, radianlist, stopblinkdegree):
    dispersion_grad = 1.5
    sorted_gradlist = sorted(gradlist, reverse = True)
    thresh_grad_high = sorted_gradlist[0] + dispersion_grad
    thresh_grad_low = sorted_gradlist[0] - dispersion_grad

    dispersion_ratio = 7
    sorted_ratiolist = sorted(ratiolist, reverse = True)
    thresh_ratio_high = int(sorted_ratiolist[0] + dispersion_ratio)
    thresh_ratio_low = int(sorted_ratiolist[0] - dispersion_ratio)

    average_degree = sum(stopblinkdegree) / len(stopblinkdegree)

    excelpath_TAD = 'C:\\Users\\admin\\blink_data\\' + path + '\\ThreshAutoDetect\\index.xlsx'
    wb = openpyxl.Workbook()

    wb.create_sheet(version)
    ws = wb[version]
    ws['B2'] = 'version : ' + version
    ws['D5'] = 'x0'
    ws['E5'] = 'y0'
    ws['F5'] = 'x1'
    ws['G5'] = 'y1'

    ws['I5'] = 'time'
    ws['J5'] = 'gradient'
    ws['K5'] = 'radian'
    ws['L5'] = 'degree'
    ws['M5'] = 'whiteratio'

    ws['J1'] = 'dispersion grad : ' + str(dispersion_grad)
    ws['J2'] = 'max gradient : ' + str(sorted_gradlist[0])
    ws['J3'] = 'grad thresh high : ' + str(thresh_grad_high)
    ws['J4'] = 'grad thresh low : ' + str(thresh_grad_low)

    ws['M1'] = 'dispersion ratio : ' + str(dispersion_ratio)
    ws['M2'] = 'max ratio : ' + str(sorted_ratiolist[0])
    ws['M3'] = 'ratio thresh high : ' + str(thresh_ratio_high)
    ws['M4'] = 'ratio thresh low : ' + str(thresh_ratio_low)

    ws['O5'] = 'stop blink degree'

    ws['O4'] = 'average degree (stop blink) : ' + average_degree

    for i0 in range(0, len(x0list)):
        ws.cell(i0 + 6, 4, x0list[i0])
        ws.cell(i0 + 6, 5, y0list[i0])
        ws.cell(i0 + 6, 6, x1list[i0])
        ws.cell(i0 + 6, 7, y1list[i0])
        ws.cell(i0 + 6, 9, timelist[i0])
        ws.cell(i0 + 6, 10, gradlist[i0])
        ws.cell(i0 + 6, 11, radianlist[i0])
        ws.cell(i0 + 6, 12, degreelist[i0])
        ws.cell(i0 + 6, 13, ratiolist[i0])
    
    for i2 in range(0, len(stopblinkdegree)):
        ws.cell(i2 + 6, 15, stopblinkdegree[i2])

    wb.save(excelpath_TAD)
    wb.close()

    return thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, average_degree

def ThreshAutoDetection(path, version, xmin, xmax, ymin, ymax, rotation_status):
    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = xmax - xmin
    height = ymax - ymin
    savepath = 'C:\\Users\\admin\\Desktop\\blink_data\\' + path + '\\ThreshDetection' + version
    if rotation_status is not None:
        savepath = savepath + '_rotation'
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
    avg = None
    kernel_hor = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_cal = np.ones((3, 3), np.uint8)

    # List difinition
    x0list_TAD, y0list_TAD, x1list_TAD, y1list_TAD, timelist_TAD, gradlist_TAD, ratiolist_TAD, degreelist_TAD, radianlist_TAD, stopblink_degree = [], [], [], [], [], [], [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret: break

        if rotation_status is not None:
            frame = cv2.warpAffine(frame, rotation_status, (640, 480))

        copyframe = frame.copy()
        cutframe_copy = copyframe[ymin:ymax, xmin:xmax]
        cutframe = frame[ymin:ymax, xmin:xmax]

        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        binary_eyelid = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_eyelid, -1, kernel_hor)
        dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
        lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 30)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_framedelta = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100

        if lines is not None:
            x0, y0, x1, y1 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3]
            delta_Y = y1 - y0
            delta_X = x1 - x0
            grad = 10 * (delta_Y / delta_X)
            if grad > -10:
                x0list_TAD.append(x0)
                y0list_TAD.append(y0)
                x1list_TAD.append(x1)
                y1list_TAD.append(y1)
                gradient, degree, radian = EyelidParameterCalculation(x0list_TAD, y0list_TAD, x1list_TAD, y1list_TAD)
                gradlist_TAD.append(gradient)
                degreelist_TAD.append(degree)
                radianlist_TAD.append(radian)
                ratiolist_TAD.append(whiteratio)
                comparison_time = time.time() - basetime
                timelist_TAD.append(comparison_time)
                cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)
                if whiteratio == 0:
                    stopblink_degree.append(degree)

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

    thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, average_degree = ExcelEntry_TAD(path, version, x0list_TAD, y0list_TAD, x1list_TAD, y1list_TAD, timelist_TAD, gradlist_TAD, ratiolist_TAD, degreelist_TAD, radianlist_TAD, stopblink_degree)

    return thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, average_degree

# ---------------------------------



# --Blink tolerance time measurement thread--

def ExcelEntry_BTM(path, version, x0list, y0list, x1list, y1list, timelist, gradlist, ratiolist, degreelist, radianlist, tolerancetimelist, endtimelist):
    excelpath_BTM = 'C:\\Users\\admin\\blink_data\\' + path + '\\BlinkToleranceMeasurement\\index.xlsx'
    wb = openpyxl.Workbook()
    wb.create_sheet(version)
    ws = wb[version]

    ws['B2'] = 'version : ' + version
    ws['D3'] = 'x0'
    ws['E3'] = 'y0'
    ws['F3'] = 'x1'
    ws['G3'] = 'y1'

    ws['I3'] = 'time'
    ws['J3'] = 'gradient'
    ws['K3'] = 'degree'
    ws['L3'] = 'radian'
    ws['M3'] = 'whiteratio'

    ws['O3'] = 'endtime'
    ws['P3'] = 'tolerance time'

    for i0 in range(0, len(x0list)):
        ws.cell(i0 + 4, 4, x0list[i0])
        ws.cell(i0 + 4, 5, y0list[i0])
        ws.cell(i0 + 4, 6, x1list[i0])
        ws.cell(i0 + 4, 7, y1list[i0])

    for i1 in range(0, len(timelist)):
        ws.cell(i1 + 4, 9, timelist[i1])
        ws.cell(i1 + 4, 10, gradlist[i1])
        ws.cell(i1 + 4, 11, degreelist[i1])
        ws.cell(i1 + 4, 12, radianlist[i1])
        ws.cell(i1 + 4, 13, ratiolist[i1])

    for i2 in range(0, len(0, endtimelist)):
        ws.cell(i2 + 4, 15, endtimelist[i2])
        ws.cell(i2 + 4, 16, tolerancetimelist[i2])

    wb.save(excelpath_BTM)
    wb.close()

def ToleranceThread(path, version, xmin, xmax, ymin, ymax, interval, thresh_grad_low, starttime, rotation_status, queue_midway, queue_finish):
    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = xmax - xmin
    height = ymax - ymin
    savepath =  'C:\\Users\\admin\\Desktop\\blink_data\\' + path + '\\ToleranceMeasurement' + version
    if rotation_status is not None:
        savepath = savepath + '_rotation'
    os.makedirs(savepath)
    frame_save = cv2.VideoWriter(savepath + '\\frame.mp4', fourcc, 30, (640, 480))
    cutframe_save = cv2.VideoWriter(savepath + '\\cutframe.mp4', fourcc, 30, (width, height))
    gaussian_save = cv2.VideoWriter(savepath + '\\gaussian.mp4', fourcc, 30, (width, height))
    gray_save = cv2.VideoWriter(savepath + '\\gray.mp4', fourcc, 30, (width, height))
    binaryeyelid_save = cv2.VideoWriter(savepath + '\\binary_eyelid.mp4', fourcc, 30, (width, height))
    horizon_save = cv2.VideoWriter(savepath + '\\horizon.mp4', fourcc, 30, (width, height))
    dilation_save = cv2.VideoWriter(savepath + '\\dilation.mp4', fourcc, 30, (width, height))
    framedelta_save = cv2.VideoWriter(savepath + '\\framedelta.mp4', fourcc, 30, (width, height))
    binaryframedelta_save = cv2.VideoWriter(savepath + '\\binary_framedelta.mp4', fourcc, 30, (width, height))

    # Paramete setting
    basetime = time.time()
    avg = None
    kernel_hor = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_cal = np.ones((3, 3), np.uint8)
    fonttype = cv2.FONT_HERSHEY_COMPLEX

    # List difinition
    x0list_TM, y0list_TM, x1list_TM, y1list_TM, timelist_TM, gradlist_TM, ratiolist_TM, degreelist_TM, radianlist_TM, tolerancetimelist_TM, endtimelist_TM = [], [], [], [], [], [], [], [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret: break

        if rotation_status is not None:
            frame = cv2.warpAffine(frame, rotation_status, (640, 480))

        copyframe = frame.copy()
        cutframe = copyframe[ymin:ymax, xmin:xmax]
        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        binary_eyelid = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_eyelid, -1, kernel_hor)
        dilation = cv2.dilate(horizon, kernel_cal, 1)
        lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_framedelta = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100

        if lines is not None:
            x0, y0, x1, y1 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3]
            x0list_TM.append(x0)
            y0list_TM.append(y0)
            x1list_TM.append(x1)
            y1list_TM.append(y1)
            gradient, degree, radian = EyelidParameterCalculation(x0list_TM, y0list_TM, x1list_TM, y1list_TM)
            if gradient > -10:
                gradlist_TM.append(gradient)
                degreelist_TM.append(degree)
                radianlist_TM.append(radian)
                comparisontime = time.time() - basetime
                timelist_TM.append(comparisontime)
                ratiolist_TM.append(whiteratio)

            if whiteratio > 10 and gradient > thresh_grad_low:
                endtime = time.time() - starttime
                tolerancetime = time.time() - basetime + interval
                queue_finish.put(tolerancetime)
                endtimelist_TM.append(endtime)
                tolerancetimelist_TM.append(tolerancetime)
                break

            else:
                tolerancetime = time.time() - basetime + interval
                queue_midway.put(tolerancetime)

        cv2.putText(frame, 'Tolerance time[s] : ', (10, 400), fonttype, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(tolerancetime), (150, 400), fonttype, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        cv2.imshow('Tolerance blink', frame)

        frame_save.write(frame)
        cutframe_save.write(cutframe)
        gaussian_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        binaryeyelid_save.write(cv2.cvtColor(binary_eyelid, cv2.COLOR_GRAY2BGR))
        horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binaryframedelta_save.write(cv2.cvtColor(binary_framedelta, cv2.COLOR_GRAY2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    frame_save.release()
    gaussian_save.release()
    gray_save.release()
    binaryeyelid_save.release()
    horizon_save.release()
    dilation_save.release()
    framedelta_save.release()
    binaryframedelta_save.release()
    cv2.destroyAllWindows()

    ExcelEntry_BTM(path, version, x0list_TM, y0list_TM, x1list_TM, y1list_TM, timelist_TM, gradlist_TM, ratiolist_TM, degreelist_TM, radianlist_TM, tolerancetimelist_TM)

# -------------------------------------------



# --Rotation frame detection thread--
    
def ExcelEntry_RFD(path, version, rotation_degree, starttime, x0list, y0list, x1list, y1list, gradlist, ratiolist, degreelist, radianlist, timelist, vallist_detec, timelist_detec, gradlist_detec, ratiolist_detec, degreelist_detec, radianlist_detec, noblinktimelist, intervallist):
    excelpath_RFD = 'C:\\Users\\admin\\blink_data\\' + path + '\\RotationFrameDetection\\index.xlsx'
    wb = openpyxl.Workbook()
    wb.create_sheet(version)
    ws = wb[version]

    ws['A1'] = 'rotation degree : ' + str(rotation_degree) + '[°]'
    ws['E1'] = 'start time : ' + str(starttime)

    ws['B2'] = 'version : ' + version
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

    ws['O2'] = 'detections'
    ws['O3'] = 'count'
    ws['P3'] = 'time'
    ws['Q3'] = 'gradient'
    ws['R3'] = 'degree'
    ws['S3'] = 'radian'
    ws['T3'] = 'whiteratio'

    ws['V2'] = 'no blinks'
    ws['V3'] = 'time'
    ws['W3'] = 'interval'

    for i0 in range(0, len(x0list)):
        ws.cell(i0 + 4, 4, x0list[i0])
        ws.cell(i0 + 4, 5, y0list[i0])
        ws.cell(i0 + 4, 6, x1list[i0])
        ws.cell(i0 + 4, 7, y1list[i0])
        ws.cell(i0 + 4, 9, timelist[i0])
        ws.cell(i0 + 4, 10, gradlist[i0])
        ws.cell(i0 + 4, 11, degreelist[i0])
        ws.cell(i0 + 4, 12, radianlist[i0])
        ws.cell(i0 + 4, 13, ratiolist[i0])

    for i1 in range(0, len(vallist_detec)):
        ws.cell(i1 + 4, 15, vallist_detec[i1])
        ws.cell(i1 + 4, 16, timelist_detec[i1])
        ws.cell(i1 + 4, 17, gradlist_detec[i1])
        ws.cell(i1 + 4, 18, degreelist_detec[i1])
        ws.cell(i1 + 4, 19, radianlist_detec[i1])
        ws.cell(i1 + 4, 20, ratiolist_detec[i1])

    for i2 in range(0, len(noblinktimelist)):
        ws.cell(i2 + 4, 22, noblinktimelist[i2])
        ws.cell(i2 + 4, 22, intervallist[i2])

    wb.save(excelpath_RFD)
    wb.close()

def RotationFrameDetection(path, version, xmin, xmax, ymin, ymax, rotation_degree, starttime):

    # Rotation matrix calculation
    rotation_matrix = cv2.getRotationMatrix2D((320, 240), rotation_degree, 1.0)
    xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation = FrameAutoDetect(path, version, rotation_matrix)
    th_grad_high_rotation, th_grad_low_rotation, th_ratio_high_rotaion, th_ratio_low_rotation, average_degree_rotation = ThreshAutoDetection(path, version, xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation, rotation_matrix)

    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = xmax - xmin
    height = ymax - ymin
    savepath = 'C:\\Users\\admin\\Desktop\\blink_data\\' + path + '\\RotationFrameDetection' + version
    os.makedirs(savepath)
    rotationframe_save = cv2.VideoWriter(savepath + '\\frame.mp4', fourcc, 30, (640, 480))
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
    avg = None
    kernel_hor = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_cal = np.ones((3, 3), np.uint8)
    fonttype = cv2.FONT_HERSHEY_COMPLEX

    # List difinition
    x0list_RFD, y0list_RFD, x1list_RFD, y1list_RFD, gradlist_RFD, ratiolist_RFD, timelist_RFD, degreelist_RFD, radianlist_RFD = [], [], [], [], [], [], [], [], []
    vallist_detec_RFD, timelist_detec_RFD, gradlist_detec_RFD, ratiolist_detec_RFD, degreelist_detec_RFD, radianlist_detec_RFD = [], [], [], [], [], []
    noblinktimelist_RFD, intervallist_RFD = [], []

    while True:
        ret, frame = cap.read()
        if not ret: break

        rotationframe = cv2.warpAffine(frame, rotation_matrix, (640, 480))
        copyframe = rotationframe.copy()
        cutframe = rotationframe[ymin_rotation:ymax_rotation, xmin_rotation:xmax_rotation]
        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        binary_eyelid = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_eyelid, -1, kernel_hor)
        dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
        lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 30)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_framedelta = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100

        if lines is not None:
            x0, y0, x1, y1 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3]
            delta_Y = y1 - y0
            delta_X = x1 - x0
            grad = 10 * (delta_Y / delta_X)
            if grad > -10 and grad < 8:
                x0list_RFD.append(x0)
                y0list_RFD.append(y0)
                x1list_RFD.append(x1)
                y1list_RFD.append(y1)
                gradient, degree, radian = EyelidParameterCalculation(x0list_RFD, y0list_RFD, x1list_RFD, y1list_RFD)
                gradlist_RFD.append(gradient)
                ratiolist_RFD.append(whiteratio)
                comparison_time = time.time() - basetime
                timelist_RFD.append(comparison_time)
                degreelist_RFD.append(degree)
                radianlist_RFD.append(radian)
                cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)

        if gradient < th_grad_high_rotation and gradient > th_grad_low_rotation:
            timediff = time.time() - blinktime
            if timediff > 0.2:
                if whiteratio < th_ratio_high_rotaion and whiteratio > th_ratio_low_rotation:
                    val += 1
                    blinktime = time.time()
                    detectiontime = starttime + (time.time() - basetime)
                    vallist_detec_RFD.append(val)
                    timelist_detec_RFD.append(detectiontime)
                    gradlist_detec_RFD.append(gradient)
                    ratiolist_detec_RFD.append(whiteratio)
                    degreelist_detec_RFD.append(degree)
                    radianlist_detec_RFD.append(radian)
                    interval = 0

                else: 
                    noblinktime = starttime + (time.time() - basetime)
                    interval = time.time() - blinktime
                    noblinktimelist_RFD.append(noblinktime)
                    intervallist_RFD.append(interval)
                    indexW = ratiolist_RFD[-2]
                    indexV = ratiolist_RFD[-3]

                    if indexW < th_ratio_high_rotaion and indexW > th_ratio_low_rotation:
                        val += 1
                        blinktime = time.time()
                        detectiontime = starttime + (time.time() - basetime)
                        vallist_detec_RFD.append(val)
                        timelist_detec_RFD.append(detectiontime)
                        gradlist_detec_RFD.append(gradient)
                        ratiolist_detec_RFD.append(whiteratio)
                        degreelist_detec_RFD.append(degree)
                        radianlist_detec_RFD.append(radian)
                        interval = 0

                    else:
                        noblinktime = starttime + (time.time() - basetime)
                        interval = time.time() - blinktime
                        noblinktimelist_RFD.append(noblinktime)
                        intervallist_RFD.append(interval)

                        if indexV < th_ratio_high_rotaion and indexV > th_ratio_low_rotation:
                            val += 1
                            blinktime = time.time()
                            detectiontime = starttime + (time.time() - basetime)
                            vallist_detec_RFD.append(val)
                            timelist_detec_RFD.append(detectiontime)
                            gradlist_detec_RFD.append(gradient)
                            ratiolist_detec_RFD.append(whiteratio)
                            degreelist_detec_RFD.append(degree)
                            radianlist_detec_RFD.append(radian)
                            interval = 0

                        else:
                            noblinktime = starttime + (time.time() - basetime)
                            interval = time.time() - blinktime
                            noblinktimelist_RFD.append(noblinktime)
                            intervallist_RFD.append(interval)

        runtime = starttime + (time.time() - basetime)

        cv2.rectangle(copyframe, (xmin_rotation, ymin_rotation), (xmax_rotation, ymax_rotation), (0, 255, 0), 2)
        cv2.putText(rotationframe, 'Count:', (10, 350), fonttype, 1, (0, 0, 255), 2)
        cv2.putText(rotationframe, str(val), (150, 350), fonttype, 1, (0, 0, 255), 2)

        cv2.imshow('Frame (rotation)', rotationframe)

        rotationframe_save.write(rotationframe)
        copyframe_save.write(copyframe)
        cutframe_save.write(cutframe)
        gaussian_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        binaryeyelid_save.write(cv2.cvtColor(binary_eyelid, cv2.COLOR_GRAY2BGR))
        horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binaryframedelta_save.write(cv2.cvtColor(binary_framedelta, cv2.COLOR_GRAY2BGR))

        if interval > 60: break

        if runtime > 900: break
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    rotationframe_save.release()
    copyframe_save.release()
    cutframe_save.release()
    gaussian_save.release()
    gray_save.release()
    binaryeyelid_save.release()
    horizon_save.release()
    dilation_save.release()
    framedelta_save.release()
    binaryframedelta_save.release()

    ExcelEntry_RFD(path, version, rotation_degree, starttime, x0list_RFD, y0list_RFD, x1list_RFD, y1list_RFD, gradlist_RFD, ratiolist_RFD, degreelist_RFD, radianlist_RFD, timelist_RFD, vallist_detec_RFD, timelist_detec_RFD, gradlist_detec_RFD, ratiolist_detec_RFD, degreelist_detec_RFD, radianlist_detec_RFD, noblinktimelist_RFD, intervallist_RFD)

# -----------------------------------



# --main loop--
    
def ExcelEntry_main(path, version, x0list, y0list, x1list, y1list, timelist, gradlist, degreelist, radianlist, ratiolist, vallist_detec, timelist_detec, gradlist_detec, degreelist_detec, radianlist_detec, ratiolist_detec, noblinktimelist, intervallist, stopblinkdegree, tolerancetimelist):
    excelpath_main = 'C:\\Users\\admin\\blink_data\\' + path + '\\main_detection\\index.xlsx'
    wb = openpyxl.Workbook()
    wb.create_sheet(version)
    ws = wb[version]

    average_stopblink_degree = sum(stopblinkdegree) / len(stopblinkdegree)

    ws['B2'] = 'version : ' + version
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
    ws['O3'] = 'stop blink degree'

    ws['Q2'] = 'detections'
    ws['R3'] = 'count'
    ws['S3'] = 'time'
    ws['T3'] = 'gradient'
    ws['U3'] = 'degree'
    ws['V3'] = 'radian'
    ws['W3'] = 'whiteratio'

    ws['Y2'] = 'no blinks'
    ws['Z3'] = 'time'
    ws['AA3'] = 'interval'

    ws['AC3'] = 'tolerance time'

    for i0 in range(0, len(x0list)):
        ws.cell(i0 + 4, 4, x0list[i0])
        ws.cell(i0 + 4, 5, y0list[i0])
        ws.cell(i0 + 4, 6, x1list[i0])
        ws.cell(i0 + 4, 7, y1list[i0])
        ws.cell(i0 + 4, 9, timelist[i0])
        ws.cell(i0 + 4, 10, gradlist[i0])
        ws.cell(i0 + 4, 11, degreelist[i0])
        ws.cell(i0 + 4, 12, radianlist[i0])
        ws.cell(i0 + 4, 13, ratiolist[i0])

    for i1 in range(0, len(stopblinkdegree)):
        ws.cell(i1 + 4, 15, stopblinkdegree[i1])

    for i2 in range(0, len(vallist_detec)):
        ws.cell(i2 + 4, 17, vallist_detec[i2])
        ws.cell(i2 + 4, 18, timelist_detec[i2])
        ws.cell(i2 + 4, 19, gradlist_detec[i2])
        ws.cell(i2 + 4, 20, degreelist_detec[i2])
        ws.cell(i2 + 4, 21, radianlist_detec[i2])
        ws.cell(i2 + 4, 22, ratiolist_detec[i2])

    for i3 in range(0, len(noblinktimelist)):
        ws.cell(i3 + 4, 24, noblinktimelist[i3])
        ws.cell(i3 + 4, 25, intervallist[i3])

    for i4 in range(0, len(tolerancetimelist)):
        ws.cell(i4 + 4, 27, tolerancetimelist[i4])

    wb.save(excelpath_main)
    wb.close()

def main():
    date = time.strftime('%m%d')
    times = time.strftime('%H%M%S')
    xmin, xmax, ymin, ymax = FrameAutoDetect(date, times, None)
    grad_high, grad_low, ratio_high, ratio_low, ave_degree_TAD = ThreshAutoDetection(date, times, xmin, xmax, ymin, ymax, None)

    cap = cv2.VideoCapture(0)

    # Video save setting
    fourcc = cv2.VideoWriter_foucc('m', 'p', '4', 'v')
    width = xmax - xmin
    height = ymax - ymin
    savepath = 'C:\\Users\\admin\\Desktop\\blink_data\\' + date + '\\main_detection' + times
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
    avg = None
    kernel_hor = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_cal = np.ones((3, 3), np.uint8)
    fonttype = cv2.FONT_HERSHEY_COMPLEX

    # List difinition
    x0list_main, y0list_main, x1list_main, y1list_main, gradlist_main, ratiolist_main, timelist_main, degreelist_main, radianlist_main = [], [], [], [], [], [], [], [], []
    vallist_detec_main, timelist_detec_main, gradlist_detec_main, ratiolist_detec_main, degreelist_detec_main, radianlist_detec_main = [], [], [], [], [], []
    noblinktimelist_main, intervallist_main = [], []
    stopblink_degree_main = []
    tolerancetimelist_main = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        copyframe = frame.copy()
        cutframe = frame[ymin:ymax, xmin:xmax]
        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        binary_eyelid = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_eyelid, -1, kernel_hor)
        dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
        lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 30)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_framedelta = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100

        if lines is not None:
            x0, y0, x1, y1 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3]
            delta_Y = y1 - y0
            delta_X = x1 - x0
            grad = 10 * (delta_Y / delta_X)
            if grad > -10 and grad < 8:
                x0list_main.append(x0)
                y0list_main.append(y0)
                x1list_main.append(x1)
                y1list_main.append(y1)
                gradient, degree, radian = EyelidParameterCalculation(x0list_main, y0list_main, x1list_main, y1list_main)
                gradlist_main.append(gradient)
                ratiolist_main.append(whiteratio)
                comparison_time = time.time() - basetime
                timelist_main.append(comparison_time)
                degreelist_main.append(degree)
                radianlist_main.append(radian)
                cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)
                if whiteratio == 0:
                    stopblink_degree_main.append(degree)

        if gradient < grad_high and gradient > grad_low:
            timediff = time.time() - blinktime
            if timediff > 0.2:
                if whiteratio < ratio_high and whiteratio > ratio_low:
                    val += 1
                    blinktime = time.time()
                    detectiontime = time.time() - basetime
                    vallist_detec_main.append(val)
                    timelist_detec_main.append(detectiontime)
                    gradlist_detec_main.append(gradient)
                    ratiolist_detec_main.append(whiteratio)
                    degreelist_detec_main.append(degree)
                    radianlist_detec_main.append(radian)
                    interval = 0

                else:
                    noblinktime = time.time() - basetime
                    interval = time.time() - blinktime
                    noblinktimelist_main.append(noblinktime)
                    intervallist_main.append(interval)
                    indexW = ratiolist_main[-2]
                    indexV = ratiolist_main[-3]

                    if indexW < ratio_high and indexW > ratio_low:
                        val += 1
                        blinktime = time.time()
                        detectiontime = time.time() - basetime
                        vallist_detec_main.append(val)
                        timelist_detec_main.append(detectiontime)
                        gradlist_detec_main.append(gradient)
                        ratiolist_detec_main.append(whiteratio)
                        degreelist_detec_main.append(degree)
                        radianlist_detec_main.append(radian)
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
                            vallist_detec_main.append(val)
                            timelist_detec_main.append(detectiontime)
                            gradlist_detec_main.append(gradient)
                            ratiolist_detec_main.append(whiteratio)
                            degreelist_detec_main.append(degree)
                            radianlist_detec_main.append(radian)
                            interval = 0

                        else:
                            noblinktime = time.time() - basetime
                            interval = time.time() - blinktime
                            noblinktimelist_main.append(noblinktime)
                            intervallist_main.append(interval)
        if interval > 2:
            past_ave_ratio = sum(ratiolist_main[-5:]) / 5
            if past_ave_ratio < 5:
                version = time.strftime('%H%M%S')
                starttime = time.time()
                midway_queue = Queue()
                finish_queue = Queue()
                timer_thread = threading.Thread(target = ToleranceThread, name = 'Tolerance measurement', args = (date, version, xmin, xmax, ymin, ymax, interval, grad_low, starttime, None, finish_queue))
                timer_thread.start()

                midway_tolerancetime = midway_queue.get()
                cv2.putText(frame, 'tolerance time : ', (10, 200), fonttype, 1, (255, 255, 0), 2)
                cv2.putText(frame, str(midway_tolerancetime), (160, 200), fonttype, 1, (255, 255, 0), 2)

                timer_thread.join()

                finish_tolerancetime = finish_queue.get()
                
                tolerancetimelist_main.append(finish_tolerancetime)

                print('Tolerance time : ', finish_tolerancetime)

            elif interval > 30:
                version = time.strftime('%H%M%S')
                starttime = time.time()
                ave_degree_main = sum(stopblink_degree_main) / len(stopblink_degree_main)
                rotation_degree = ave_degree_main - ave_degree_TAD
                rotation_thread = threading.Thread(target = RotationFrameDetection, name = 'Rotaion frame measurement', args = (date, version, xmin, xmax, ymin, ymax, rotation_degree, starttime))
                rotation_thread.start()

                rotation_thread.join()

                break

        runtime = time.time() - basetime

        cv2.rectangle(copyframe, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, 'count : ', (10, 350), (0, 0, 255), 2)
        cv2.putText(frame, str(val), (150, 350), (0, 0, 255), 2)

        cv2.imshow('Frame', frame)

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

# -------------
    
if __name__ == '__main__':
    main()