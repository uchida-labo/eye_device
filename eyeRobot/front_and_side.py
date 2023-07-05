import cv2, time, openpyxl
from functions import image_processing, judge_area
import matplotlib.pyplot as plt

# Camera setting (for front)
front_cap = cv2.VideoCapture(1)
front_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
front_cap.set(cv2.CAP_PROP_FPS, 30)
front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
fourcc_front = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps_front = int(front_cap.get(cv2.CAP_PROP_FPS))
width_front = int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_front = int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

side_cap = cv2.VideoCapture(0)
side_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
side_cap.set(cv2.CAP_PROP_FPS, 30)
side_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
side_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
fourcc_side = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps_side = int(side_cap.get(cv2.CAP_PROP_FPS))
width_side = int(side_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_side = int(side_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Parameter setting
fontType = cv2.FONT_HERSHEY_COMPLEX   # font setting
avg_front = None   # computation frame(for front)
avg_side = None    # computation frame(for side)
xmin_front, xmax_front = 270,430
ymin_front, ymax_front = 100, 160
xmin_side, xmax_side = 220, 600
ymin_side, ymax_side = 0, 300
val_front, val_side = 0, 0
base_time = time.time()
blinktime_front, blinktime_side = 0, 0

# For graph drawing
whitelist_front = []
whitelist_side = []
timelist = []

# For data confirmation
blist_front = []
blist_side =[]
dtimelist_front = []
dtimelist_side = []
dwlist_front = []
dwlist_side = []

# Excel sheet setting
sheetname_front = 'PC_front'   # Editing section
sheetname_side = 'PC_side'     # Editing section
wb = openpyxl.load_workbook(R'F:\M2卒研\data\front_and_side\result.xlsx')
wb.create_sheet(sheetname_front)
wb.create_sheet(sheetname_side)
ws_front = wb[sheetname_front]
ws_side = wb[sheetname_side]

ws_front["B2"].value = sheetname_front
ws_front["D3"].value = "Detections"
ws_front["E3"].value = "time[s]"
ws_front["F3"].value = "white ratio[%]"
ws_front["G3"].value = "run time[s]"
ws_front["H3"].value = "white ratio[%]"

ws_side["B2"].value = sheetname_side
ws_side["D3"].value = "Detections"
ws_side["E3"].value = "time[s]"
ws_side["F3"].value = "white ratio[%]"
ws_side["G3"].value = "run time[s]"
ws_side["H3"].value = "white ratio[%]"

video_front = cv2.VideoWriter(R'F:\\M2卒研\\data\\front_and_side\\' + sheetname_front + '.mp4', fourcc_front, fps_front, (width_front, height_front))
video_side = cv2.VideoWriter(R'F:\\M2卒研\\data\\front_and_side\\' + sheetname_side + '.mp4', fourcc_side, fps_side, (width_side, height_side))

while True:
    ret1, frame_front = front_cap.read()
    ret2, frame_side = side_cap.read()

    if not ret1:
        break

    if not ret2:
        break

    gray_front = image_processing.img_process(frame_front, xmin_front, xmax_front, ymin_front, ymax_front)[0]
    gray_side = image_processing.img_process(frame_side, xmin_side, xmax_side, ymin_side, ymax_side)[0]


    if avg_front is None:
        avg_front = gray_front.copy().astype("float")
        continue

    if avg_side is None:
        avg_side = gray_side.copy().astype("float")
        continue

    cv2.accumulateWeighted(gray_front, avg_front, 0.8)
    cv2.accumulateWeighted(gray_side, avg_side, 0.8)
    Delta_front = cv2.absdiff(gray_front, cv2.convertScaleAbs(avg_front))
    Delta_side = cv2.absdiff(gray_side, cv2.convertScaleAbs(avg_side))
    thr_front = cv2.threshold(Delta_front, 3, 255, cv2.THRESH_BINARY)[1]
    thr_side = cv2.threshold(Delta_side, 3, 255, cv2.THRESH_BINARY)[1]
    whtratio_front = judge_area.judge_blink(thr_front, xmin_front, xmax_front, ymin_front, ymax_front)[0]
    whtratio_side = judge_area.judge_blink(thr_side, xmin_side, xmax_side, ymin_side, ymax_side)[0]

    if 8 < whtratio_front and whtratio_front < 16:
        timediff_front = time.time() - blinktime_front
        if timediff_front > 0.3:
            val_front += 1
            blinktime_front = time.time()
            dtime_front = blinktime_front - base_time
            cv2.putText(frame_front, 'Blink!', (330, 90), fontType, 2, (0, 0, 255), 3)
            blist_front.append(val_front)
            dtimelist_front.append(dtime_front)
            dwlist_front.append(whtratio_front)
    
    if 23 < whtratio_side and whtratio_side < 35:
        timediff_side = time.time() - blinktime_side
        if timediff_side > 0.3:
            val_side += 1
            blinktime_side = time.time()
            dtime_side = blinktime_side - base_time
            cv2.putText(frame_side, 'Blink!', (300, 280), fontType, 2, (0, 0, 255), 3)
            blist_side.append(val_side)
            dtimelist_side.append(dtime_side)
            dwlist_side.append(whtratio_side)

    cv2.putText(frame_front, str(val_front), (270, 90), fontType, 1, (0, 0, 255), 3)
    cv2.putText(frame_side, str(val_side), (240, 280), fontType, 1, (0, 0, 255), 3)
    cv2.rectangle(frame_front, (xmin_front, ymin_front), (xmax_front, ymax_front), (255, 0, 0), 2)
    cv2.rectangle(frame_side, (xmin_side, ymin_side), (xmax_side, ymax_side), (255, 0, 0), 2)

    cv2.imshow("Front", frame_front)
    cv2.imshow("Side", frame_side)

    end_time = time.time()
    run_time = end_time - base_time

    whitelist_front.append(whtratio_front)
    whitelist_side.append(whtratio_side)
    timelist.append(run_time)

    video_front.write(frame_front)
    video_side.write(frame_side)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for i1 in range(0, len(blist_front)):
    ws_front.cell(i1 + 4, 4, value = blist_front[i1])
    ws_front.cell(i1 + 4, 5, value = dtimelist_front[i1])
    ws_front.cell(i1 + 4, 6, value = dwlist_front[i1])

for i2 in range(0, len(blist_side)):
    ws_side.cell(i2 + 4, 4, value = blist_side[i2])
    ws_side.cell(i2 + 4, 5, value = dtimelist_side[i2])
    ws_side.cell(i2 + 4, 6, value = dwlist_side[i2])

for i3 in range(0, len(timelist)):
    ws_front.cell(i3 + 4, 7, value = timelist[i3])
    ws_side.cell(i3 + 4, 7, value = timelist[i3])
    ws_front.cell(i3 + 4, 8, value = whitelist_front[i3])
    ws_side.cell(i3 + 4, 8, value = whitelist_side[i3])

sheetname = 'PC'

wb.save(R'F:\\M2卒研\\data\\front_and_side\\' + sheetname + '.xlsx')
wb.close()
plt.plot(timelist, whitelist_front)
plt.plot(timelist, whitelist_side)
plt.show()
video_front.release()
video_side.release()
front_cap.release()
side_cap.release()
cv2.destroyAllWindows()

