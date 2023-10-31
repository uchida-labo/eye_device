import cv2

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