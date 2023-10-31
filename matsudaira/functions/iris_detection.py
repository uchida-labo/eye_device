import cv2
#import image_processing , data_acquisition , judge_area

def draw_and_output(frame, x_center, y_center, radius, w_meter, h_meter, width, height):
    """
    Drawing circles and obtaining coordinates

    ・argument
    'frame':Frame for drawing
    'center', 'radius':Center coordinates and radius of the detection circle
    'meter':Camera coverage (in [mm])
    'width', 'height':Frame size

    ・return
    'frame':Frame with circles drawn
    'x', 'y':Coordinates of detection circle (in [mm])
    """

    center = (int(x_center), int(y_center))
    r = int(radius)
    cv2.circle(frame, center, r, (100, 255, 0), 2)
    cv2.circle(frame, center, 2, (0, 0, 255), 3)
    x = (w_meter * x_center)/width
    y = (h_meter * y_center)/height

    return frame, x, y


def img_process(frame,trim_xmin, trim_xmax, trim_ymin, trim_ymax):
    """
    Image processing

    ・argument
    'frame':Camera image output screen
    'trim_x' and 'trim_y':Coordinates for specifying the cut area

    ・return
    'gray_img':Video after grayscale conversion
    'bin_img':Video after binarization
    'edges':Edges detected by the canny method
    'contours':Contour points extracted from 'edges'
    """
    gau_filter = cv2.GaussianBlur(frame, (5, 5), 1)
    gray_img = cv2.cvtColor(gau_filter[trim_ymin:trim_ymax, trim_xmin:trim_xmax], cv2.COLOR_BGR2GRAY)
    th_val, bin_img = cv2.threshold(gray_img, 55, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_img, 100, 170)   # Set upper and lower thresholds
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (trim_xmin, trim_ymin))

    return gray_img, bin_img, edges, contours


def video_parameter(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return fps, w, h

def judge_eye(frame, center, radius):
    """
    Black-White area decision per measurement area　for eye detection

    ・argument
    'frame':Camera image output screen

    ・return
    'white_eye' and 'black_eye':Area percentages of white and black, respectively
    """
    x = int(center[0])
    y = int(center[1])
    r = int(radius)
    image_size = 4 * (r ** 2)
    trim_gray = cv2.cvtColor(frame[(y-r):(y+r), (x-r):(x+r)], cv2.COLOR_RGB2GRAY)
    trim_bin = cv2.threshold(trim_gray, 55, 255, cv2.THRESH_BINARY)[1]
    white = cv2.countNonZero(trim_bin)
    black = image_size - white
    white_eye = (white/image_size) * 100
    black_eye = (black/image_size) * 100

    return white_eye, black_eye


def judge_blink(trim_frame, xmin, xmax, ymin, ymax):
    """
    Black-White area decision per measurement area　for blink detection

    ・argument
    'trim_frame':Binarized cropping camera image
    'x', 'y', 'w', and 'h':Vertex of the smallest bounding rectangle relative to the black eye

    ・return
    'white_eye' and 'black_eye':Area percentages of white and black, respectively
    """
    video_size = (xmax - xmin)*(ymax - ymin)
    white = cv2.countNonZero(trim_frame)
    black = video_size - white
    white_blink = (white/video_size) * 100
    black_blink = (black/video_size) * 100

    return white_blink, black_blink


fontType = cv2.FONT_HERSHEY_COMPLEX

# Trimming area
xmin, xmax = 220, 420  #100 , 500
ymin, ymax = 180, 240  #100 , 300
w_meter = 100
h_meter = 100

# Setting of USB camera
cap = cv2.VideoCapture(1)  # 0+cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 15) # カメラFPS設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # カメラ画像の横幅設定  1280pxel
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # カメラ画像の縦幅設定  720pxel

fps, width, height = video_parameter(cap)

while True :
    #frame取得
    ret, frame = cap.read()
    if not ret:
        break

    gray, bin, edges, contours = img_process(frame, xmin, xmax, ymin, ymax)

    for i, cnt in enumerate(contours) :
        center, radius = cv2.minEnclosingCircle(cnt)
        white_ratio, black_ratio = judge_eye(frame, center, radius)
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