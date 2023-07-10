import cv2
from functions import image_processing, data_acquisition, judge_area

fontType = cv2.FONT_HERSHEY_COMPLEX

# Trimming area
xmin, xmax = 252, 335  #100 , 500
ymin, ymax = 250, 435  #100 , 300
w_meter = 100
h_meter = 100

# Setting of USB camera
cap = cv2.VideoCapture(1)  # 0+cv2.CAP_DSHOW
fps, width, height = judge_area.video_parameter(cap)

while True :
    #frame取得
    ret, frame = cap.read()
    if not ret:
        break

    gray, bin, edges, contours = image_processing.img_process(frame, xmin, xmax, ymin, ymax)

    for i, cnt in enumerate(contours) :
        center, radius = cv2.minEnclosingCircle(cnt)
        white_ratio, black_ratio = judge_area.judge_eye(frame, center, radius)
        if white_ratio < 30 and black_ratio > 70:
            if radius < 20 and radius > 14:
                frame, cir_x, cir_y = data_acquisition.draw_and_output(frame, center[0], center[1], radius, w_meter, h_meter, width, height)
                print('( x , y ) = ', cir_x , cir_y )
                print('White Area [%] :', white_ratio)
                print('Black Area [%] :', black_ratio)
                cv2.putText(frame, str(radius), (440, 440), fontType, 1, (0, 0, 255), 3)
        break
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    cv2.imshow('output', frame)
    cv2.imshow('edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()