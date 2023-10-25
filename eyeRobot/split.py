import cv2, time
import numpy as np
import matplotlib.pyplot as plt

# camera setting
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30) # FPS setting
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # width setting  1280pxel
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # height setting  720pxel

# trimming size setting
xmin, xmax = 240,600
ymin, ymax = 50, 350

# parameter setting
rows = 10
cols = 12
chunks = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cut_frame = frame[ymin:ymax, xmin:xmax]

    if len(chunks) < 120:
        for row_frame in np.array_split(cut_frame, rows, axis = 0):
            for chunk in np.array_split(cut_frame, cols, axis = 1):
                chunks.append(chunk)
        
    cv2.imshow('split', cut_frame)
    # print('chunks:', len(chunks))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('chunk:', chunks[0])
cap.release()
cv2.destroyAllWindows()