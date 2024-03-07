from collections.abc import Callable, Iterable, Mapping
from typing import Any
import cv2, openpyxl, os, time, math, statistics, threading
import numpy as np

date_number_name = time.strftime('%m%d_%H-%M-%S')

xlist_FD, ylist_FD, wlist_FD, hlist_FD, timelist_FD = [], [], [], [], []
x0list_TD, y0list_TD, x1list_TD, y1list_TD, timelist_TD, gradlist_TD, ratiolist_TD, degreelist_TD, radianlist_TD = [], [], [], [], [], [], [], [], []
x0list_MD, y0list_MD, x1list_MD, y1list_MD, timelist_MD, gradlist_MD, ratiolist_MD, degreelist_MD, radianlist_MD = [], [], [], [], [], [], [], [], []
timelist_detection, gradlist_detection, ratiolist_detection, degreelist_detection, radianlist_detection = [], [], [], [], []
intervaltime_list, interval_list = [], []
x0list_BT, y0list_BT, x1list_BT, y1list_BT, gradlist_BT, ratiolist_BT, degreelist_BT, radianlist_BT, timelist_BT, tolerancetime_list = [], [], [], [], [], [], [], [], [], []

kernel_hor = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]])

kernel_cal = np.ones((3, 3), np.uint8)

def FrameCoordinatePointSet(x_list, y_list, w_list, h_list):
    average_x = sum(x_list) / len(x_list)
    average_y = sum(y_list) / len(y_list)
    average_w = sum(w_list) / len(w_list)
    average_h = sum(h_list) / len(h_list)

    xmin = int(average_x - 20)
    xmax = int(xmin + average_w + 60)
    ymin = int(average_y - 80)
    ymax = int(ymin + average_h + 60)

    return xmin, xmax, ymin, ymax

def FrameDetection():
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video_save_path = 'C:\\Users\\admin\\Desktop\\blink_data\\frame_detection\\' + date_number_name
    os.makedirs(video_save_path)

    frame_save = cv2.VideoWriter(video_save_path + '\\frame.mp4', fourcc, 30, (640, 480))
    gaussian_save = cv2.VideoWriter(video_save_path + '\\gaussian.mp4', fourcc, 30, (640, 480))
    gray_save = cv2.VideoWriter(video_save_path + '\\gray.mp4', fourcc, 30, (640, 480))
    framedelta_save = cv2.VideoWriter(video_save_path + '\\framedelta.mp4', fourcc, 30, (640, 480))
    binary_save = cv2.VideoWriter(video_save_path + '\\binary.mp4', fourcc, 30, (640, 480))
    edges_save = cv2.VideoWriter(video_save_path + '\\edges.mp4', fourcc, 30, (640, 480))

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
                xlist_FD.append(x)
                ylist_FD.append(y)
                wlist_FD.append(w)
                hlist_FD.append(h)
                comparison_time = time.time() - basetime
                timelist_FD.append(comparison_time)
                cv2.rectangle(edges, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

        cv2.imshow('frame', frame)
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

    xmin, xmax, ymin, ymax = FrameCoordinatePointSet(xlist_FD, ylist_FD, wlist_FD, hlist_FD)

    return xmin, xmax, ymin, ymax

def EyelidAngleCalculation(x0list, y0list, x1list, y1list):
    parallel = x1list[-1] - x0list[-1]
    perpendicular = y1list[-1] - y0list[-1]
    oblique = math.sqrt(parallel ** 2 + perpendicular ** 2)
    radian = np.arccos(parallel / oblique)
    degree = int(np.rad2deg(radian))

    return degree, radian

def ThresholdParameterSet(gradient_list, whiteratio_list):
    dispersion_grad = 1.5
    sorted_gradlist = sorted(gradient_list, reverse = True)
    thresh_grad_high = sorted_gradlist[0] + dispersion_grad
    thresh_grad_low = sorted_gradlist[0] - dispersion_grad

    dispersion_ratio = 7
    sorted_ratiolist = sorted(whiteratio_list, reverse = True)
    thresh_ratio_high = int(sorted_ratiolist[0] + dispersion_ratio)
    thresh_ratio_low = int(sorted_ratiolist[0] - dispersion_ratio)

    mode_degree = statistics.mode(degreelist_TD)
    
    return thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, mode_degree

def ThreshDetection(xmin, xmax, ymin, ymax):
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    width = xmax - xmin
    height = ymax - ymin

    video_save_path = 'C:\\Users\\admin\\Desktop\\blink_data\\thresh_detection\\' + date_number_name
    os.makedirs(video_save_path)

    frame_save = cv2.VideoWriter(video_save_path + '\\frame.mp4', fourcc, 30, (640, 480))
    copyframe_save = cv2.VideoWriter(video_save_path + '\\copyframe.mp4', fourcc, 30, (640, 480))
    cutframe_save = cv2.VideoWriter(video_save_path + '\\cutframe.mp4', fourcc, 30, (width, height))
    gaussian_save = cv2.VideoWriter(video_save_path + '\\gaussian.mp4', fourcc, 30, (width, height))
    gray_save = cv2.VideoWriter(video_save_path + '\\gray.mp4', fourcc, 30, (width, height))
    binary_eyelid_save = cv2.VideoWriter(video_save_path + '\\binary_eyelid.mp4', fourcc, 30, (width, height))
    horizon_save = cv2.VideoWriter(video_save_path + '\\horizon.mp4', fourcc, 30, (width, height))
    dilation_save = cv2.VideoWriter(video_save_path + '\\dilation.mp4', fourcc, 30, (width, height))
    framedelta_save = cv2.VideoWriter(video_save_path + '\\framedelta.mp4', fourcc, 30, (width, height))
    binary_framedelta_save = cv2.VideoWriter(video_save_path + '\\binary_framedelta.mp4', fourcc, 30, (width, height))

    basetime = time.time()

    avg = None

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
        whiteratio = (cv2.countNonZero(binary_framedelta) / ((xmax - xmin) * (ymax - ymin))) * 100

        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                gradient = 10 * (delta_Y / delta_X)
                if gradient > -10:
                    x0list_TD.append(x0)
                    y0list_TD.append(y0)
                    x1list_TD.append(x1)
                    y1list_TD.append(y1)
                    gradlist_TD.append(gradient)
                    ratiolist_TD.append(whiteratio)
                    comparison_time = time.time() - basetime
                    timelist_TD.append(comparison_time)
                    cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)
                    if whiteratio == 0:
                        degree = EyelidAngleCalculation(x0list_TD, y0list_TD, x1list_TD, y1list_TD)[0]
                        degreelist_TD.append(degree)

        cv2.rectangle(copyframe, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('Binary', binary_framedelta)
        cv2.imshow('cut frame', cutframe)

        frame_save.write(frame)
        copyframe_save.write(copyframe)
        cutframe_save.write(cutframe)
        gaussian_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        binary_eyelid_save.write(cv2.cvtColor(binary_eyelid, cv2.COLOR_GRAY2BGR))
        horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binary_framedelta_save.write(cv2.cvtColor(binary_framedelta, cv2.COLOR_GRAY2BGR))

        runtime = time.time() - basetime

        if runtime > 15: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    frame_save.release()
    copyframe_save.release()
    cutframe_save.release()
    gaussian_save.release()
    gray_save.release()
    binary_eyelid_save.release()
    horizon_save.release()
    dilation_save.release()
    framedelta_save.release()
    binary_framedelta_save.release()
    cap.release()
    cv2.destroyAllWindows()

    thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, mode_degree = ThresholdParameterSet()

    return thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, mode_degree

def RotationFramePoint(xmin, xmax, ymin, ymax, radian):
    plot1_x = xmin * math.cos(radian) + ymin * math.sin(radian)
    plot1_y = 


    xmin_rotation = xmin * math.cos(radian) + ymin * math.sin(radian)
    xmax_rotation = xmax * math.cos(radian) + ymax * math.sin(radian)
    ymin_rotation = ymin * math.cos(radian) - xmin * math.sin(radian)
    ymax_rotation = ymax * math.cos(radian) - xmax * math.sin(radian)

    return xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation

class ToleranceDetection(threading.Thread):
    def __init__(self, xmin, xmax, ymin, ymax, interval, th_grad_low):
        super(ToleranceDetection, self).__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.interval = interval
        self.th_grad_low = th_grad_low
        self.tolerancetime = None

    def BlinkToleranceDetectionThread(self):
        cap = cv2.VideoCapture(0)
        avg = None
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        basetime = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break

            cutframe = frame[self.ymin:self.ymax, self.xmin:self.xmax]
            gau = cv2.GaussianBlur(cutframe, (5, 5), 1)
            gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

            bin_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
            horizon = cv2.filter2D(bin_line, -1, kernel_hor)
            dilation = cv2.dilate(horizon, kernel_cal, 1)
            lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)

            if avg is None:
                avg = gray.copy().astype("float")
                continue

            cv2.accumulateWeighted(gray, avg, 0.8)
            framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
            bin_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]

            if lines is not None:
                x0, y0, x1, y1 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3]
                x0list_BT.append(x0)
                y0list_BT.append(y0)
                x1list_BT.append(x1)
                y1list_BT.append(y1)
                grad = ((y1 - y0) / (x1 - x0)) * 10
                degree, radian = EyelidAngleCalculation(x0list_BT, y0list_BT, x1list_BT, y1list_BT)
                whiteratio = (cv2.countNonZero(bin_fd) / (width * height)) * 100
                runtime = time.time() - basetime
                gradlist_BT.append(grad)
                ratiolist_BT.append(whiteratio)
                degreelist_BT.append(degree)
                radianlist_BT.append(radian)
                timelist_BT.append(runtime)

                if whiteratio > 10 and grad > self.th_grad_low:
                    self.tolerancetime = time.time() - basetime + self.interval
                    tolerancetime_list.append(tolerancetime_list)
                    break
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


def ListAppendForMD(detectiontime, gradient, whiteratio, degree, radian):
    timelist_detection.append(detectiontime)
    gradlist_detection.append(gradient)
    ratiolist_detection.append(whiteratio)
    degreelist_detection.append(degree)
    radianlist_detection.append(radian)

def ListAppendForInterval(time, interval):
    intervaltime_list.append(time)
    interval_list.append(interval)

class Rotation(threading.Thread):
    def __init__(self, xmin, xmax, ymin, ymax, mode_degree_TD, mode_degree_MD):
        super(Rotation, self).__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.mode_degree_TD = mode_degree_TD
        self.mode_degree_MD = mode_degree_MD
        self.xmin_rotation = None
        self.xmax_rotation = None
        self.ymin_rotation = None
        self.ymax_rotation = None
        self.th_grad_high = None
        self.th_grad_low = None
        self.th_ratio_high = None
        self.th_ratio_low = None
        self.th_degree_high = None
        self.th_degree_low = None
    
    def RotationFrameThreshDetaction(self):
        delta_degree = int(self.mode_degree_TD - self.mode_degree_MD)
        self.xmin_rotation, self.xmax_rotation, self.ymin_rotation, self.ymax_rotation = RotationFramePoint(self.xmin, self.xmax, self.ymin, self.ymax, delta_degree)

        cap = cv2.VideoCapture(0)

        basetime = time.time()
        blinktime = time.time()
        avg = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            cutframe = frame[self.ymin_rotation:self.ymax_rotation, self.xmin_rotation:self.xmax_rotation]
            gau = cv2.GaussianBlur(cutframe, (5, 5), 1)
            gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

            if avg is None:
                avg = gray.copy().astype("float")
                continue

            cv2.accumulateWeighted(gray, avg, 0.8)
            framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
            bin_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
            whiteratio = (cv2.countNonZero(bin_fd) / (width * height)) * 100





def main():
    xmin, xmax, ymin, ymax = FrameDetection()
    thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, mode_degree = ThreshDetection(xmin, xmax, ymin, ymax)

    fonttype = cv2.FONT_HERSHEY_COMPLEX

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    width = xmax - xmin
    height = ymax - ymin

    video_save_path = 'C:\\Users\\admin\\Desktop\\blink_data\\measurement\\' + date_number_name
    os.makedirs(video_save_path)

    avg = None

    basetime = time.time()
    blinktime = time.time()

    val = 0

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

        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                gradient = 10 * (delta_Y / delta_X)
                if gradient > -10 and gradient < 8:
                    x0list_MD.append(x0)
                    y0list_MD.append(y0)
                    x1list_MD.append(x1)
                    y1list_MD.append(y1)
            gradlist_MD.append(gradient)
            whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100
            ratiolist_MD.append(whiteratio)
            comparison_time = time.time() - basetime
            timelist_MD.append(comparison_time)
            degree, radian = EyelidAngleCalculation(x0list_MD, y0list_MD, x1list_MD, y1list_MD)
            degreelist_MD.append(degree)
            radianlist_MD.append(radian)
            cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)

# if whiteratio == 0:
#     dispersion_degree = 2
#     thresh_degree_high = degree + dispersion_degree
#     thresh_degree_low = degree + dispersion_degree
#     if degree > thresh_degree_high or degree < thresh_degree_low:
#         version_update = time.strftime('%m%d_%H-%M-%S')
#         remaining_time = 900 - (time.time() - basetime)
#         xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation = RotationFramePoint(xmin, xmax, ymin, ymax, radian)
#         new_threshold_calculation_thread = threading.Thread(target = ThreshDetection, args = (xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation), name = 'Thread1')


        if gradient < thresh_grad_high and gradient > thresh_grad_low:
            timediff = time.time() - blinktime
            if timediff > 0.2:
                if whiteratio < thresh_ratio_high and whiteratio > thresh_ratio_low:
                    val += 1
                    blinktime = time.time()
                    detectiontime = time.time() - basetime
                    ListAppendForMD(detectiontime, gradient, whiteratio, degree, radian)
                    interval = 0

                else: 
                    interval_time = time.time() - basetime
                    interval = time.time() - blinktime
                    ListAppendForInterval(interval_time, interval)
                    indexW = ratiolist_MD[-2]
                    indexV = ratiolist_MD[-3]

                    if indexW < thresh_ratio_high and indexW > thresh_ratio_low:
                        val += 1
                        blinktime = time.time()
                        detectiontime = time.time() - basetime
                        ListAppendForMD(detectiontime, gradient, whiteratio, degree, radian)
                        interval = 0

                    else:
                        interval_time = time.time() - basetime
                        interval = time.time() - blinktime
                        ListAppendForInterval(interval_time, interval)

                        if indexV < thresh_ratio_high and indexV > thresh_ratio_low:
                            val += 1
                            blinktime = time.time()
                            detectiontime = time.time() - basetime
                            ListAppendForMD(detectiontime, gradient, whiteratio, degree, radian)
                            interval = 0

                        else:
                            interval_time = time.time() - basetime
                            interval = time.time() - blinktime
                            ListAppendForInterval(interval_time, interval)

        runtime = time.time() - basetime
        
        if interval > 2:
            average_ratio_past = (ratiolist_MD[-1] + ratiolist_MD[-2] + ratiolist_MD[-3]) / 3
            delta_grad_V = abs(gradlist_MD[-3] - gradlist_MD[-2])
            delta_grad_W = abs(gradlist_MD[-2] - gradlist_MD[-1])
            if average_ratio_past < 5 and delta_grad_V < 0.08 and delta_grad_W < 0.08:
                version = str(runtime)
                tolerancethread = ToleranceDetection(xmin, xmax, ymin, ymax, interval, thresh_grad_low)
                tolerancethread.start()

                tolerancethread.join()

                tolerancetime = tolerancethread.tolerancetime

            elif interval > 15:
                version = time.strftime('%H%M%S')
                remaining_time = 900 - (time.time() - basetime)
                modevalue_degree = statistics.mode(degreelist_MD[-400:])

                version_update = time.strftime('%m%d_%H-%M-%S')
                remaining_time = 900 - (time.time() - basetime)
                xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation = RotationFramePoint(xmin, xmax, ymin, ymax, radian)
                new_threshold_calculation_thread = threading.Thread(target = ThreshDetection, args = (xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation), name = 'Thread1')









if __name__ == '__main__':
    main()