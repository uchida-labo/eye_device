import cv2, openpyxl, os, time, math, statistics, threading
import numpy as np

# --List difinition--

# frame detection
xlist, ylist, wlist, hlist, timelist = [], [], [], [], []


# -------------------

class FrameDetection:
    def __init__(self, path):
        super(FrameDetection, self).__init__()
        self.path = path
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def frmedetect(self):
        cap = cv2.VideoCapture(0)

        # Video save setting
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_save_path = 'C:\\Users\\admin\\Desktop\\blink_data\\' + self.path + '\\frame_detection'
        os.makedirs(video_save_path)
        frame_save = cv2.VideoWriter(video_save_path + '\\frame.mp4', fourcc, 30, (640, 480))
        gaussian_save = cv2.VideoWriter(video_save_path + '\\gaussian.mp4', fourcc, 30, (640, 480))
        gray_save = cv2.VideoWriter(video_save_path + '\\gray.mp4', fourcc, 30, (640, 480))
        framedelta_save = cv2.VideoWriter(video_save_path + '\\framedelta.mp4', fourcc, 30, (640, 480))
        binary_save = cv2.VideoWriter(video_save_path + '\\binary.mp4', fourcc, 30, (640, 480))
        edges_save = cv2.VideoWriter(video_save_path + '\\edges.mp4', fourcc, 30, (640, 480))

        # Parameter setting
        basetime = time.time()
        avg = None

        # List difinition
        # xlist, ylist, wlist, hlist, timelist = [], [], [], [], []

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
            binary = cv2.threshold(framedelta, 70, 255, cv2.THRESH_BINARY)[1]
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
                    t = time.time() - basetime
                    timelist.append(t)
                    cv2.rectangle(edges, (x, y), ((x + w), (y + h)), 255, 2)

            cv2.imshow('Frame', frame)
            cv2.imshow('Edges', edges)

            runtime = time.time() - basetime
            if runtime > 15: break

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_save.write(frame)
        gaussian_save.write(gaussian)
        gray_save.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        framedelta_save.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        binary_save.write(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        edges_save.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        
        cap.release()
        frame_save.release()
        gaussian_save.release()
        gray_save.release()
        framedelta_save.release()
        binary_save.release()
        edges_save.release()
        cv2.destroyAllWindows()

        average_x = sum(xlist) / len(xlist)
        average_y = sum(ylist) / len(ylist)
        average_w = sum(wlist) / len(wlist)
        average_h = sum(hlist) / len(hlist)

        self.xmin = int(average_x - 20)
        self.xmax = int(self.xmin + average_w + 60)
        self.ymin = int(average_y - 80)
        self.ymax = int(self.ymin + average_h + 60)

class TreshDetection:
    def __init__(self, xmin, xmax, ymin, ymax):
        super(TreshDetection, self).__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.grad_high = None
        self.grad_low = None
        self.ratio_high = None
        self.ratio_low = None
        self.degree_mode = None



