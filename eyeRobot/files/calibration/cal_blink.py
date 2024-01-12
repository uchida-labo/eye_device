import cv2, statistics
import numpy as np

cap_cal = cv2.VideoCapture(0)
cap_cal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_cal.set(cv2.CAP_PROP_FPS, 30)
cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

avg = None

xlist = []
ylist = []
wlist = []
hlist = []

while True:
    ret, frame_cal = cap_cal.read()
    if not ret:
        break

    gau = cv2.GaussianBlur(frame_cal, (5, 5), 1)
    gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

    if avg is None:
        avg = gray.copy().astype("float")

    cv2.accumulateWeighted(gray, avg, 0.8)
    delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    bin = cv2.threshold(delta, 3, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(bin, 0, 130)
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if w > h and area > 30000:
            xlist.append(x)
            ylist.append(y)
            wlist.append(w)
            hlist.append(h)
            avex = int(statistics.mean(xlist))
            avey = int(statistics.mean(ylist))
            avew = int(statistics.mean(wlist))
            aveh = int(statistics.mean(hlist))
            cv2.rectangle(frame_cal, (avex, avey), (avex + avew, avey + aveh), (255, 255, 0), 3)

    cv2.imshow('frame', frame_cal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_cal.release()
cv2.destroyAllWindows()
