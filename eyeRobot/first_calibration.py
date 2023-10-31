import cv2, time

def calibration():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.videoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    xmin_cal, xmax_cal = 240, 600
    ymin_cal, ymax_cal = 0, 200

    avg_cal = None

    while True:
        ret, frame_cal = capture.read()
        if not ret:
            break

        gaussian_cal = cv2.GaussianBlur(frame_cal[ymin_cal:ymax_cal, xmin_cal:xmax_cal], (5, 5), 1)
        gray_cal = cv2.cvtColor(gaussian_cal, cv2.COLOR_BGR2GRAY)

        if avg_cal is None:
            avg_cal = gray_cal.copy().astype("float")
            continue

        binary_cal = cv2.threshold(gray_cal, 70, 255, cv2.THRESH_BINARY)[1]
