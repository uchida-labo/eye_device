import cv2

def videocapture():
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 15) # FPS setting
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # width setting  1280pxel
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # height setting  720pxel
    
    while True:
        bool, frame = cap.read()
        cv2.imshow('SETTING (drag the eye area)', frame)

        

        