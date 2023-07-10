import cv2

cap1 = cv2.VideoCapture(0+cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\experiments.mp4', fourcc, 15, (70, 60))

xs, xl = 270, 380
ys, yl = 110, 200

while True:
    # ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    frame1 = frame1[ys:yl, xs:xl]

    cv2.imshow("save frame", frame1)

    video.write(frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
video.release()
cv2.destroyAllWindows()
