import cv2, time, statistics

def judge(trim_frame, xmin, xmax, ymin, ymax):
    video_size = ( xmax - xmin ) * ( ymax - ymin )
    white = cv2.countNonZero(trim_frame)
    black = video_size - white
    white_blink = ( white / video_size ) * 100
    black_blink = ( black / video_size ) * 100

    return white_blink, black_blink



capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# xmin_cal, xmax_cal = 240, 600
# ymin_cal, ymax_cal = 0, 200

avg_dif = None

x_list = []
y_list = []
w_list = []
h_list = []

while True:
    ret, frame_cal = capture.read()
    if not ret:
        break

    gaussian_cal = cv2.GaussianBlur(frame_cal, (5, 5), 1)
    gray_cal = cv2.cvtColor(gaussian_cal, cv2.COLOR_BGR2GRAY)


    # Interframe difference
    if avg_dif is None:
        avg_dif = gray_cal.copy().astype("float")
        continue
    cv2.accumulateWeighted(gray_cal, avg_dif, 0.8)
    frameDelta_dif = cv2.absdiff(gray_cal, cv2.convertScaleAbs(avg_dif))
    binary_dif = cv2.threshold(frameDelta_dif, 3, 255, cv2.THRESH_BINARY)[1]
    edges_diff = cv2.Canny(binary_dif, 0, 130)
    # whiteratio_dif = judge(binary_dif, xmin_cal, xmax_cal, ymin_cal, ymax_cal)[0]
    contours_diff = cv2.findContours(edges_diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i1, cnt1 in enumerate(contours_diff):
        x_diff, y_diff, w_diff, h_diff = cv2.boundingRect(cnt1)
        area_diff = w_diff * h_diff
        if w_diff > h_diff and area_diff > 30000:
            cv2.rectangle(edges_diff, (x_diff, y_diff), (x_diff + w_diff, y_diff + h_diff), (255, 255, 0), 2)
            print(area_diff)

    # Mask image processing
    mask = cv2.inRange(gray_cal, 30, 70)
    pick_msk = cv2.bitwise_and(gray_cal, gray_cal, mask = mask)
    binary_msk = cv2.threshold(pick_msk, 0, 255, cv2.THRESH_BINARY_INV)[1]
    reverse_msk = cv2.bitwise_not(binary_msk)
    white_msk = cv2.countNonZero(reverse_msk)

    # Edge detection and Rectangle area 
    edges_rect = cv2.Canny(binary_msk, 0, 130)
    contours_rect = cv2.findContours(edges_rect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i2, cnt2 in enumerate(contours_rect):
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt2)
        area_edges = w_rect * h_rect
        if w_rect > h_rect and area_edges > 30000:
            cv2.rectangle(edges_rect, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (255, 255, 0), 2)
            x_list.append(x_rect)
            y_list.append(y_rect)
            w_list.append(w_rect)
            h_list.append(h_rect)
            # print(area)


    cv2.imshow('Frame', frame_cal)
    # cv2.imshow('Frame difference', binary_dif)
    # cv2.imshow('Mask image reverse', reverse_msk)
    cv2.imshow('Edges', edges_rect)
    cv2.imshow('Edges frame difference', edges_diff)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ave_x = statistics.mean(x_list)
ave_y = statistics.mean(y_list)
ave_w = statistics.mean(w_list)
ave_h = statistics.mean(h_list)

xmin_cal = int(ave_x - 20)
xmax_cal = int(ave_x + ave_w + 20)
ymin_cal = int(ave_y - 20)
ymax_cal = int(ave_y + ave_h + 20)

# print(xmin_cal, xmax_cal, ymin_cal, ymax_cal)

capture.release()
cv2.destroyAllWindows()


