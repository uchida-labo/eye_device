import cv2, openpyxl, os, time, math, statistics, threading
import numpy as np
import calculation_functions as cal

# 測定範囲の自動決定用配列の定義
xlist_FD, ylist_FD, wlist_FD, hlist_FD, timelist_FD = [], [], [], [], []

# 閾値の自動決定用配列の定義
x0list_TD, y0list_TD, x1list_TD, y1list_TD, gradlist_TD, degreelist_TD, radianlist_TD, ratiolist_TD, timelist_TD = [], [], [], [], [], [], [], [], []


def FrameDetection(date_number_name):
    """
    測定範囲を自動決定する関数
    ・15秒間瞼が移動する領域に外接する矩形を検出
    ・矩形の中で瞼に関連すると思われる領域の座標情報を配列に追加
    ・座標情報から平均値を算出してそれを測定範囲とし，その座標を出力
    
    注意事項
    ・関数定義前に配列を定義
    """
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

    xmin, xmax, ymin, ymax = cal.FrameCoordinatePointSet(xlist_FD, ylist_FD, wlist_FD, hlist_FD)

    return xmin, xmax, ymin, ymax

def ThreshDetection(date_number_name, xmin, xmax, ymin, ymax):
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

    kernel_hor = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]])
    
    kernel_cal = np.ones((3, 3), np.uint8)

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
            x0, y0, x1, y1 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3]
            
            gradient, degree, radian = cal.EyelidParameterCalculation()
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

    thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, mode_degree = ThresholdParameterSet(gradlist_TD, ratiolist_TD, degreelist_TD)

    return thresh_grad_high, thresh_grad_low, thresh_ratio_high, thresh_ratio_low, mode_degree
