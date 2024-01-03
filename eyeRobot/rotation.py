import cv2, openpyxl, os, time, math, statistics
import numpy as np

date_number_name = ''

xlist, ylist, wlist, hlist = [], [], [], []
x0list_TD, y0list_TD, x1list_TD, y1list_TD, gradlist_TD, ratiolist_TD, degreelist_TD = [], [], [], [], [], [], []
x0list_MD, y0list_MD, x1list_MD, y1list_MD, gradlist_MD, ratiolist_MD, degreelist_MD = [], [], [], [], [], [], []

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
                xlist.append(x)
                ylist.append(y)
                wlist.append(w)
                hlist.append(h)
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

    xmin, xmax, ymin, ymax = FrameCoordinatePointSet(xlist, ylist, wlist, hlist)

    return xmin, xmax, ymin, ymax

def EyelidAngleCalculation(x0list, y0list, x1list, y1list):
    parallel = x1list[-1] - x0list[-1]
    perpendicular = y1list[-1] - y0list[-1]
    oblique = math.sqrt(parallel ** 2 + perpendicular ** 2)
    radian = np.arccos(parallel / oblique)
    degree = int(np.rad2deg(radian))

    return degree, radian

def ThresholdParameterSet():
    dispersion_grad = 1.5
    sorted_gradlist = sorted(gradlist_TD, reverse = True)
    thresh_grad_high = sorted_gradlist[0] + dispersion_grad
    thresh_grad_low = sorted_gradlist[0] - dispersion_grad

    dispersion_ratio = 7
    sorted_ratiolist = sorted(ratiolist_TD, reverse = True)
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
    xmin_rotation = xmin * math.cos(radian) - ymin * math.sin(radian)
    xmax_rotation = xmax * math.cos(radian) - ymax * math.sin(radian)
    ymin_rotation = xmin * math.sin(radian) + ymin * math.cos(radian)
    ymax_rotation = xmax * math.sin(radian) + ymax * math.cos(radian)

    return xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation

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
    blinktime = 0

    val = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        copyframe = frame.copy()
        cutframe = frame[ymin:ymax, xmin:xmax]
        gaussian = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        binary_eyelid = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)[1]
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
                if gradient > -10:
                    x0list_MD.append(x0)
                    y0list_MD.append(y0)
                    x1list_MD.append(x1)
                    y1list_MD.append(y1)
                    gradlist_MD.append(gradient)
                    whiteratio = (cv2.countNonZero(binary_framedelta) / (width * height)) * 100
                    ratiolist_MD.append(whiteratio)
                    cv2.line(cutframe, (x0, y0), (x1, y1), (255, 255, 0), 2)
                    if whiteratio == 0:
                        degree, radian = EyelidAngleCalculation(x0list_MD, y0list_MD, x1list_MD, y1list_MD)
                        dispersion_degree = 2
                        thresh_degree_high = degree + dispersion_degree
                        thresh_degree_low = degree + dispersion_degree
                        if degree > thresh_degree_high or degree < thresh_degree_low:
                            xmin_rotation, xmax_rotation, ymin_rotation, ymax_rotation = RotationFramePoint(xmin, xmax, ymin, ymax, radian)






