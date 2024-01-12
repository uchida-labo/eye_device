import cv2, time, threading, openpyxl
import numpy as np

def Calibration():
    """
    測定を行う前に閾値と測定範囲を自動的に決定する関数
    「Frame」というフレームが出力されてから15秒間測定を行い、測定が終了するとフレームが閉じる
    途中で中断する場合は「q」を入力すると終了する
    フレームが閉じてから測定結果を計算する

    ・返り値（戻り値）
    測定範囲の座標情報（左上の座標点と右下の座標点）
    瞼の傾きを判断するときの閾値（上と下の閾値が出力→a_high > x > a_low）
    """
    cap_cal = cv2.VideoCapture(0)
    cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_cal.set(cv2.CAP_PROP_FPS, 30)
    cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    xlist_fd, ylist_fd, wlist_fd, hlist_fd = [], [], [], []
    xlist_pick, ylist_pick, wlist_pick, hlist_pick = [], [], [], []
    gradientlist = []

    base_time = time.time()

    avg_cal = None

    kernel_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0], 
    [-1, -2, -1]], dtype = np.float32)
    kernel_hor /= 9

    kernel_cal = np.ones((3, 3), np.uint8)

    while True:
        ret, frame_cal = cap_cal.read()
        if not ret:
            break

        gau_cal = cv2.GaussianBlur(frame_cal, (5, 5), 1)
        gray_cal = cv2.cvtColor(gau_cal, cv2.COLOR_BGR2GRAY)

        if avg_cal is None:
            avg_cal = gray_cal.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray_cal, avg_cal, 0.8)
        framedelta = cv2.absdiff(gray_cal, cv2.convertScaleAbs(avg_cal))
        binary_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        edges_fd = cv2.Canny(binary_fd, 0, 130)
        contours_fd = cv2.findContours(edges_fd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        for i0, cnt0 in enumerate(contours_fd):
            x_fd, y_fd, w_fd, h_fd = cv2.boundingRect(cnt0)
            area_fd = w_fd * h_fd
            if w_fd > h_fd and area_fd > 30000:
                xlist_fd.append(x_fd)
                ylist_fd.append(y_fd)
                wlist_fd.append(w_fd)
                hlist_fd.append(h_fd)

        mask = cv2.inRange(gray_cal, 30, 70)
        pick = cv2.bitwise_and(gray_cal, gray_cal, mask = mask)
        bin_pick = cv2.threshold(pick, 3, 255, cv2.THRESH_BINARY_INV)[1]
        edges_pick = cv2.Canny(bin_pick, 0, 130)
        contours_pick = cv2.findContours(edges_pick, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        for i1, cnt1 in enumerate(contours_pick):
            x_pick, y_pick, w_pick, h_pick = cv2.boundingRect(cnt1)
            area_pick = w_pick * h_pick
            if w_pick > h_pick and area_pick > 30000:
                xlist_pick.append(x_pick)
                ylist_pick.append(y_pick)
                wlist_pick.append(w_pick)
                hlist_pick.append(h_pick)

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
                    gradient = 10 * (delta_Y / delta_X)
                    if gradient < 6 and gradient > 0:
                        gradientlist.append(gradient)

        cv2.imshow('Frame', frame_cal)

        end_time = time.time()
        run_time = end_time - base_time


        if run_time > 15:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_cal.release()
    cv2.destroyAllWindows()

    avex_fd = sum(xlist_fd) / len(xlist_fd)
    avey_fd = sum(ylist_fd) / len(ylist_fd)
    avew_fd = sum(wlist_fd) / len(wlist_fd)
    aveh_fd = sum(hlist_fd) / len(hlist_fd)

    avex_pick = sum(xlist_pick) / len(xlist_pick)
    avey_pick = sum(ylist_pick) / len(ylist_pick)
    avew_pick = sum(wlist_pick) / len(wlist_pick)
    aveh_pick = sum(hlist_pick) / len(hlist_pick)

    if avex_fd > avex_pick:
        xmin = int(avex_pick - 10)
    else:
        xmin = int(avex_fd - 10)
    
    if avey_fd > avey_pick:
        ymin = int(avey_pick - 10)
    else:
        ymin = int(avey_fd - 10)

    if ymin < 0:
        ymin = 0

    if avew_fd > avew_pick:
        xmax = int(xmin + avew_fd + 20)
    else:
        xmax = int(xmin + avew_pick + 20)
    
    if xmax > 640:
        xmax = 640
    
    if aveh_fd > aveh_pick:
        ymax = int(ymin + aveh_fd + 20)
    else:
        ymax = int(ymin + aveh_pick + 20)
    
    xmin_rnd = round(xmin, -1)
    xmax_rnd = round(xmax, -1)
    ymin_rnd = round(ymin, -1)
    ymax_rnd = round(ymax, -1)

    gradientlist_sorted = sorted(gradientlist, reverse = True)
    gradient_max = gradientlist_sorted[0]
    thresh_gradient_high = gradient_max - 1.5
    thresh_gradient_low = gradient_max + 0.5

    return xmin_rnd, xmax_rnd, ymin_rnd, ymax_rnd, thresh_gradient_high, thresh_gradient_low

def Blink_calibration(xmin, xmax, ymin, ymax):
    """
    「Calibration()」で算出した測定範囲内で行われた瞬きを数値化して閾値を自動決定する関数
    先ほどと同様に「Blink」というフレームが出力されてから15秒間測定を行う
    測定終了後はフレームが閉じて閾値を決定する計算を行う
    
    ・返り値（戻り値）
    瞬きかどうかの判断に必要な閾値（上と下の閾値が出力→a_high > x > a_low）
    """
    cap_blink = cv2.VideoCapture(0)
    cap_blink.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_blink.set(cv2.CAP_PROP_FPS, 30)
    cap_blink.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_blink.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    base_time_blink = time.time()
    comparison_time = 0

    whiteratiolist = []

    avg = None

    while True:
        ret, frame_blink = cap_blink.read()
        if not ret:
            break

        gaussian_blink = cv2.GaussianBlur(frame_blink, (5, 5), 1)
        gray_blink = cv2.cvtColor(gaussian_blink, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray_blink.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray_blink, avg, 0.8)
        framedelta_blink = cv2.absdiff(gray_blink, cv2.convertScaleAbs(avg))
        bin_blink = cv2.threshold(framedelta_blink, 3, 255, cv2.THRESH_BINARY)[1]
        framesize = int((xmax - xmin) * (ymax - ymin))
        white_pixel = cv2.countNonZero(bin_blink)
        whiteratio = (white_pixel / framesize) * 100

        time_diff = time.time() - comparison_time

        if time_diff > 0.3:
            comparison_time = time.time()
            whiteratiolist.append(whiteratio)
        
        cv2.imshow('Blink frame', bin_blink)

        end_time_blink = time.time()
        run_time_blink = end_time_blink - base_time_blink

        if run_time_blink > 15:
            break

        if cv2.waitKey(1) % 0xFF == ord('q'):
            break

    cap_blink.release()
    cv2.destroyAllWindows()

    whiteratiolist_sorted = sorted(whiteratiolist, reverse = True)
    whiteratio_max = whiteratiolist_sorted[0]
    thresh_whiteratio_high = int(whiteratio_max + 6)
    thresh_whiteratio_low = int(whiteratio_max - 6)

    return thresh_whiteratio_high, thresh_whiteratio_low

def main_thread():

    xmin, xmax, ymin, ymax, thresh_grad_high, thresh_grad_low = Calibration()

    thresh_ratio_high, thresh_ratio_low = Blink_calibration(xmin, xmax, ymin, ymax)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fontType = cv2.FONT_HERSHEY_COMPLEX

    avg = None

    kernel_hor = np.array([
        [1, 2, 1], 
        [0, 0, 0], 
        [-1, -2, -1]], dtype = np.float32)
    kernel_hor /= 9

    kernel_detec = np.ones((5, 5), np.uint8)

    comparisonlist_time, comparisonlist_gradient, comparisonlist_ratio = [], [], []

    base_time = time.time()
    blink_time = 0

    val = 0

    Judge0, Judge1 = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ret1, cutframe = cap.read()
        if not ret1:
            break

        cutframe = cutframe[ymin:ymax, xmin:xmax]
        gaussian = cv2.GaussianBlur(frame[ymin:ymax, xmin:xmax], (5, 5), 1)
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
                delta_Y = y1 - y0
                delta_X = x1 - x0
                gradient = 10 * (delta_Y / delta_X)

        if avg is None:
            avg = gray.copy().astype("float")
            continue
        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        bin_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        framesize = int((xmax - xmin) * (ymax - ymin))
        white_pixel = cv2.countNonZero(bin_fd)
        whiteratio = (white_pixel / framesize) * 100

        comparison_time = time.time() - base_time
        comparisonlist_time.append(comparison_time)
        comparisonlist_gradient.append(gradient)
        comparisonlist_ratio.append(whiteratio)

        if gradient > thresh_grad_low and gradient < thresh_grad_high:
            time_diff = time.time() - blink_time
            if time_diff > 0.3:
                if whiteratio > thresh_ratio_low and whiteratio < thresh_ratio_high:
                    val += 1
                    blink_time = time.time()
                    time_detec = time.time() - base_time
                    Judge0 = 1
                
                if Judge0 == 0 :
                    indexW = comparisonlist_ratio[-2]
                    indexV = comparisonlist_ratio[-3]

                    if indexW > thresh_ratio_low and indexW < thresh_ratio_high:
                        val += 1
                        blink_time = time.time()
                        time_detec = time.time() - base_time
                        Judge1 = 1

                    if Judge0 == 0 and Judge1 == 0:
                        if indexV > thresh_ratio_low and indexV < thresh_ratio_high:
                            val += 1
                            blink_time = time.time()
                            time_detec = time.time() - base_time
        
        now_time = time.time()
        detectime_diff = now_time - time_detec
        if detectime_diff > 10:
            calibration_thread = threading.Thread(target = Calibration)

            calibration_thread.start()

            calibration_thread.join()

            xmin_new, xmax_new, ymin_new, ymax_new = calibration_thread.result
            blink_calibration_thread = threading.Thread()



