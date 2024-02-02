import cv2

cap = cv2.VideoCapture(0)

savepath = 'C:\\Users\\admin\\Desktop\\data\\iris_fig\\'

xmin, xmax = 240, 460
ymin, ymax = 190, 350

width = xmax - xmin
height = ymax - ymin

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# copycut_save = cv2.VideoWriter(savepath + 'copycut.mp4', fourcc, 30, (width, height))
# gray_save = cv2.VideoWriter(savepath + 'gray.mp4', fourcc, 30, (width, height))
# edges_save = cv2.VideoWriter(savepath + 'edges.mp4', fourcc, 30, (width, height))
# cutframe_save = cv2.VideoWriter(savepath + 'cutframe.mp4', fourcc, 30, (width, height))

while True:
    ret, frame = cap.read()
    if not ret: break

    cutframe = frame[ymin:ymax, xmin:xmax]
    copycut_input = cutframe.copy()
    copycut_rect = cutframe.copy()
    copycut_rect_circle = cutframe.copy()
    gray = cv2.cvtColor(cutframe, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(gray, 110, 200)
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i, cnt in enumerate(contours):
        center, radius = cv2.minEnclosingCircle(cnt)
        if radius > 40 and radius < 50:
            xmin_iris = int(center[0]) - int(radius)
            xmax_iris = int(center[0]) + int(radius)
            ymin_iris = int(center[1]) - int(radius)
            ymax_iris = int(center[1]) + int(radius)
            cv2.circle(cutframe, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
            cv2.circle(cutframe, (int(center[0]), int(center[1])), 2, (0, 0, 255), 2)
            cv2.circle(copycut_rect_circle, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
            cv2.circle(copycut_rect_circle, (int(center[0]), int(center[1])), 2, (0, 0, 255), 2)
            cv2.rectangle(copycut_rect, (xmin_iris, ymin_iris), (xmax_iris, ymax_iris), (255, 255, 0), 2)
            cv2.rectangle(copycut_rect_circle, (xmin_iris, ymin_iris), (xmax_iris, ymax_iris), (255, 255, 0), 2)
            cv2.imwrite(savepath + 'input\\' + str(i) + '.jpg', copycut_input)
            cv2.imwrite(savepath + 'gray\\' + str(i) + '.jpg', gray)
            cv2.imwrite(savepath + 'edges\\' + str(i) + '.jpg', edges)
            cv2.imwrite(savepath + 'cutframe\\' + str(i) + '.jpg', cutframe)
            cv2.imwrite(savepath + 'copycut_rect\\' + str(i) + '.jpg', copycut_rect)
            cv2.imwrite(savepath + 'copycut_rect_circle\\' + str(i) + '.jpg', copycut_rect_circle)
            if xmin_iris >= 0 and ymin_iris >= 0:
                cv2.imwrite(savepath + 'binary\\' + str(i) + '.jpg', binary[ymin_iris:ymax_iris, xmin_iris:xmax_iris])


    cv2.rectangle(frame, (xmin - 5, ymin - 5), (xmax + 5, ymax + 5), (255, 0, 0), 2)

    # cv2.imshow('frame', frame)
    cv2.imshow('cut frame', cutframe)
    cv2.imshow('edges', edges)
    cv2.imshow('copy cut input', copycut_input)
    cv2.imshow('copy cut rect', copycut_rect)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()