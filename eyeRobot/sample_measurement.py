import cv2, time, openpyxl, os
import numpy as np
import measurement as m

xlist_framedetec, ylist_framedetec, wlist_framedetec, hlist_framedetec = [], [], [], []

def Frame_detect(date_number_path):
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # video_savepath_framedetec = 'C:\\Users\\admin\\Desktop\\frame_detection\\' + date_number_path
    # os.makedirs(video_savepath_framedetec)

    # video_capture = cv2.VideoWriter(video_savepath_framedetec + '\\capture.mp4', fourcc, 30, (640, 360))
    # video_gaussian = cv2.VideoWriter(video_savepath_framedetec + '\\gaussian.mp4', fourcc, 30, (640, 360))
    # video_gray = cv2.VideoWriter(video_savepath_framedetec + '\\gray.mp4', fourcc, 30, (640, 360))
    # video_framedeleta = cv2.VideoWriter(video_savepath_framedetec + '\\framedelta.mp4', fourcc, 30, (640, 360))
    # video_binaryary = cv2.VideoWriter(video_savepath_framedetec + '\\binaryary.mp4', fourcc, 30, (640, 360))
    # video_edges = cv2.VideoWriter(video_savepath_framedetec + '\\edges.mp4', fourcc, 30, (640, 360))

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

        # video_capture.write(frame)
        # video_gaussian.write(gau)
        # video_gray.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        # video_framedeleta.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        # video_binaryary.write(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        # video_edges.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    average_x = sum(xlist_framedetec) / len(xlist_framedetec)
    average_y = sum(ylist_framedetec) / len(ylist_framedetec)
    average_w = sum(wlist_framedetec) / len(wlist_framedetec)
    average_h = sum(hlist_framedetec) / len(hlist_framedetec)

    xmin = int(average_x - 20)
    xmax = int(xmin + average_w + 60)
    ymin = int(average_y - 80)
    ymax = int(ymin + average_h + 60)

    cap.release()
    # video_capture.release()
    # video_gaussian.release()
    # video_gray.release()
    # video_framedeleta.release()
    # video_binaryary.release()
    # video_edges.release()
    cv2.destroyAllWindows()

    return xmin, xmax, ymin, ymax

def main(xmin, xmax, ymin, ymax):
    cap = cv2.VideoCapture(0)

    kernel_hor = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]])

    kernel_hor_dlt = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]])

    kernel_ver = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])

    kernel_ver_dlt = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]])

    kernel_cal = np.ones((3, 3), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break

        cutframe = frame[ymin:ymax, xmin:xmax]

        gau = cv2.GaussianBlur(cutframe, (5, 5), 1)
        gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1]

        eyelid = cv2.bitwise_not(cv2.filter2D(binary, -1, kernel_hor_dlt))
        pupil = cv2.filter2D(binary, -1, kernel_ver)
        andprocess = cv2.dilate(cv2.bitwise_and(pupil, eyelid), kernel_cal, 1)

        horizon = cv2.filter2D(binary, -1, kernel_hor)
        eyelid_lines = cv2.HoughLinesP(horizon, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)
        if eyelid_lines is not None:
            x0, y0, x1, y1 = eyelid_lines[0][0][0], eyelid_lines[0][0][1], eyelid_lines[0][0][2], eyelid_lines[0][0][3]
            cv2.line(cutframe, (x0, y0), (x1, y1), (2155, 255, 0), 2)
            iris_line = cv2.HoughLinesP(andprocess, rho = 1, theta = np.pi / 360, threshold = 10, minLineLength = 20, maxLineGap = 40)
            if iris_line is not None:
                x0_iris, y0_iris, x1_iris, y1_iris = iris_line[0][0][0], iris_line[0][0][1], iris_line[0][0][2], iris_line[0][0][3]
                if x0_iris > ((xmax - xmin) / 2) and ((xmax - x1_iris) > 30) and y1 < y1_iris and (x1 + 10) > x1_iris:
                    cv2.line(cutframe, (x0_iris, y0_iris), (x1_iris, y1_iris), (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('cut frame', cutframe)
        cv2.imshow('and process', andprocess)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = '0115-5'
    xmin, xmax, ymin, ymax = Frame_detect(path)
    main(xmin, xmax, ymin, ymax)