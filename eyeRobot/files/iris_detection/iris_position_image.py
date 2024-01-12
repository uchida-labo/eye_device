import cv2

cap = cv2.VideoCapture(1)

xmin, xmax = 180, 320 #190, 310
ymin, ymax = 170, 260 #180, 250

xmin_s, xmax_s = 190, 310
ymin_s, ymax_s = 180, 250

while True:
    ret, frame = cap.read()
    if not ret: break

    cutframe = frame[ymin_s:ymax_s, xmin_s:xmax_s]
    cutframe_copy = cutframe.copy()
    gau = cv2.GaussianBlur(frame[ymin:ymax, xmin:xmax], (5, 5), 1)
    gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)
    bin = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1]
    edges = cv2.Canny(gray, 170, 220)
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    for i, cnt in enumerate(contours):
        center, radius = cv2.minEnclosingCircle(cnt)
        x_center = int(center[0])
        y_center = int(center[1])
        radius = int(radius)
        if radius > 12 and radius < 20:
            iris_size = 4 * (radius ** 2)
            iris_bin = bin[(y_center - radius):(y_center + radius), (x_center - radius):(x_center + radius)]
            cv2.rectangle(cutframe_copy, ((x_center - radius - 10), (y_center - radius - 10)), ((x_center + radius - 10), (y_center + radius - 10)), (255, 255, 0), 2)
            if iris_size > 0:
                whiteratio = (cv2.countNonZero(iris_bin) / iris_size) * 100
                if whiteratio > 0:
                    print('White ratio: ', whiteratio)

            cv2.circle(frame, ((x_center + 180), (y_center + 170)), radius, (100, 255, 0), 2)
            cv2.circle(frame, ((x_center + 180), (y_center + 170)), 2, (0, 0, 255), 2)
            
            # print('radius : ', radius)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('Cut frame', cutframe)
    cv2.imshow('Cut frame copy', cutframe_copy)
    cv2.imshow('Edges', edges)
    cv2.imshow('Binary', bin)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()