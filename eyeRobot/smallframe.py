import cv2

cap1 = cv2.VideoCapture(0+cv2.CAP_DSHOW)
cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(R'C:\Users\admin\Desktop\data\sample.mp4', fourcc, 15, (70, 60))

fps = int(cap1.get(cv2.CAP_PROP_FPS))
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# xmin, xmax = 320, 403  #100 , 500
# ymin, ymax = 250, 435  #100 , 300

xs, xl = 270, 380
ys, yl = 110, 200
print('WIDTH:', width)
print("HEIGHT:", height)
print("FPS", fps)

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    # frame = frame[ymin:ymax, xmin:xmax]
    # frame1 = frame1[ys:yl, xs:xl]
    # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imshow("frame", frame1)
    # cv2.imshow("save frame", frame1)

    # video.write(frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
video.release()
cv2.destroyAllWindows()
