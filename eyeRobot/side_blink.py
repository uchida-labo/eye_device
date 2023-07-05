import cv2, time, openpyxl
from functions import image_processing, judge_area 
import matplotlib.pyplot as plt

# camera setting
cap1 = cv2.VideoCapture(0) # mac:2  USB camera:0+cv2.CAP_DSHOW
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap1.set(cv2.CAP_PROP_FPS, 30) # FPS setting
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 540) # width setting  1280pxel
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # height setting  720pxel

# trimming size setting
xmin, xmax = 220,600
ymin, ymax = 0, 300

# excel sheet setting
# sheetname = 'Normal'   # ← Editing section
# wb = openpyxl.load_workbook(R'F:\M2卒研\data\side_blink\data_sheet.xlsx')
# wb.create_sheet(sheetname)
# ws = wb[sheetname]
# ws["B2"].value = sheetname
# ws["D3"].value = "Detections"
# ws["E3"].value = "time[s]"
# ws["F3"].value = "white ratio[%]"
# ws["G3"].value = "run time[s]"
# ws["H3"].value = "white ratio[%]"
# ws["I3"].value = "Up looking"
# ws["J3"].value = "Down looking"

# parameter setting
fontType = cv2.FONT_HERSHEY_COMPLEX   # font setting
fps = int(cap1.get(cv2.CAP_PROP_FPS))
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# video = cv2.VideoWriter(R'F:\\M2卒研\\data\\side_blink\\' + sheetname + '.mp4', fourcc, fps, (width, height))

avg = None   # computation frame

# For graph drawing
white_list = []
time_list = []

# For data confirmation
blink_list = []
detectime_list = []
detecwhite_list = []

# For count confirmation
up_list = []
down_list = []

# Definition
val = 0
up_counter = 0
down_counter = 0


base_time = time.time()
blink_time = 0

while True:
    ret, frame = cap1.read()
    if not ret:
        break

    gray, bin, edges, contours = image_processing.img_process(frame, xmin, xmax, ymin, ymax)
    
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    cv2.accumulateWeighted(gray, avg, 0.8)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    white_ratio = judge_area.judge_blink(thresh, xmin, xmax, ymin, ymax)[0]

    if 23 < white_ratio and white_ratio < 35:
        time_diff = time.time() - blink_time
        if time_diff > 0.3:
            val += 1
            blink_time = time.time()
            detection_time = blink_time - base_time
            cv2.putText(frame, 'Blink!', (300, 280), fontType, 1, (0, 0, 255), 3)
            blink_list.append(val)
            detectime_list.append(detection_time)
            detecwhite_list.append(white_ratio)

    cv2.putText(frame, str(val), (240, 280), fontType, 1, (0, 0, 255), 3)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Bin", thresh)

    print('white[%]', white_ratio)
    
    end_time = time.time()
    run_time = end_time - base_time
    
    white_list.append(white_ratio)
    time_list.append(run_time)

    # video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# for i in range(0, len(blink_list)):
#     ws.cell(i + 4, 4, value = blink_list[i])
#     ws.cell(i + 4, 5, value = detectime_list[i])
#     ws.cell(i + 4, 6, value = detecwhite_list[i])

# for i1 in range(0, len(time_list)):
#     ws.cell(i1 + 4, 7, value = time_list[i1])
#     ws.cell(i1 + 4, 8, value = white_list[i1])

# wb.save(R'F:\M2卒研\data\side_blink\ ' + sheetname + '.xlsx')
# wb.close()
plt.plot(time_list, white_list)
plt.show()
# video.release()
cap1.release()
cv2.destroyAllWindows()



