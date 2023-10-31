import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

xmin, xmax = 240, 600
ymin, ymax = 0, 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gaussian = cv2.GaussianBlur(frame[ymin:ymax, xmin:xmax], (5, 5), 1)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    bin = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow('Frame', frame)
    cv2.imshow('Binarization', bin)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()