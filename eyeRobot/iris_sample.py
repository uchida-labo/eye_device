import cv2

fontType = cv2.FONT_HERSHEY_COMPLEX

# Trimming area
xmin, xmax = 252, 335  #100 , 500
ymin, ymax = 250, 435  #100 , 300
w_meter = 100
h_meter = 100

# Setting of USB camera
cap = cv2.VideoCapture(1)  # 0+cv2.CAP_DSHOW
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True :
    #frame取得
    ret, frame = cap.read()
    if not ret:
        break
    
    fil = cv2.GaussianBlur(frame, (5, 5), 1)
    gray_img = cv2.cvtColor(fil[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 170)   # Set upper and lower thresholds
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (xmin, ymin))

    for i, cnt in enumerate(contours):
        center, radius = cv2.minEnclosingCircle(cnt)
        x = int(center[0])
        y = int(center[1])
        r = int(radius)
        ymin_c = y - r
        ymax_c = y + r
        xmin_c = x - r
        xmax_c = x + r
        image_size = 4 * (r ** 2)

        if image_size == 0:
             break
        
        trim_bin = cv2.threshold(gray_img[ymin_c:ymax_c, xmin_c:xmax_c], 55, 255, cv2.THRESH_BINARY)[1]
        white = cv2.countNonZero(trim_bin)
        black = image_size - white
        white_ratio = (white/image_size) * 100
        black_ratio = (black/image_size) * 100
        if white_ratio < 30 and black_ratio > 70:
            if radius < 20 and radius > 14:
                    cv2.circle(frame, (x, y), r, (100, 255, 0), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                    cx = (w_meter * x)/w
                    cy = (h_meter * y)/h
        break

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



