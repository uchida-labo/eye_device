import cv2, time, openpyxl
from functions import camera_param, image_processing, judge_area, entrydata_forexcel
import matplotlib.pyplot as plt

# Camera setting
cap = cv2.VideoCapture(0)
cap, fps, width, height, fourcc, video = camera_param.cam_set(cap)

# Trimming size setting
xmin, xmax = 270, 430
ymin, ymax = 100, 160

# Excel sheet setting
sheetname = ''
wbpath = ''
blink_list = []
detectime_list =[]
detecwhite_list = []
time_list = []
white_list = []

# Parameter setting
fontType = cv2.FONT_HERSHEY_COMPLEX
avg = None
val = 0
basetime = time.time()
blink_time = 0










wb = entrydata_forexcel.sheet_setting(sheetname, wbpath, blink_list, detectime_list, detecwhite_list)

