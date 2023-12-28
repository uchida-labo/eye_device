import cv2, openpyxl, os, time
import numpy as np
from measurement import Frame_detect

date_number = '1228_1'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

video_path = 'C:\\Users\\admin\\Desktop\\data\\rotation_data\\video_data\\' + date_number
os.makedirs(video_path)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_save = cv2.VideoWriter(video_path + '\\frame.mp4', fourcc, 30, (640, 360))
binline_save = cv2.VideoWriter(video_path + '\\bin_line.mp4', fourcc, 30, (640, 360))
horizon_save = cv2.VideoWriter(video_path + '\\horizon.mp4', fourcc, 30, (640, 360))
dilation_save = cv2.VideoWriter(video_path + '\\dilation.mp4', fourcc, 30, (640, 360))
binframedelta_save = cv2.VideoWriter(video_path + '\\bin_framedelta.mp4', fourcc, 30, (640, 360))
edges_save = cv2.VideoWriter(video_path + '\\edges.mp4', fourcc, 30, (640, 360))
copyframe_save = cv2.VideoWriter(video_path + '\\copy_frame.mp4', fourcc, 30, (640, 360))

avg = None

kernel_hor = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]])

kernel_cal = np.ones((3, 3), np.uint8)

x0list, y0list, x1list, y1list = [], [], [], []
gradlist = [0, 0]
whiteratiolist = [0, 0]
xlist, ylist, wlist, hlist = [], [], [], []
xlist_framedetec, ylist_framedetec, wlist_framedetec, hlist_framedetec = [], [], [], []

def Frame_detect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video_savepath_framedetec = 'C:\\Users\\admin\\Desktop\\measurement_data\\frame_detection\\' + date_number
    os.makedirs(video_savepath_framedetec)

    video_capture = cv2.VideoWriter(video_savepath_framedetec + '\\capture.mp4', fourcc, 30, (640, 360))
    video_gaussian = cv2.VideoWriter(video_savepath_framedetec + '\\gaussian.mp4', fourcc, 30, (640, 360))
    video_gray = cv2.VideoWriter(video_savepath_framedetec + '\\gray.mp4', fourcc, 30, (640, 360))
    video_framedeleta = cv2.VideoWriter(video_savepath_framedetec + '\\framedelta.mp4', fourcc, 30, (640, 360))
    video_binaryary = cv2.VideoWriter(video_savepath_framedetec + '\\binaryary.mp4', fourcc, 30, (640, 360))
    video_edges = cv2.VideoWriter(video_savepath_framedetec + '\\edges.mp4', fourcc, 30, (640, 360))

    base_time = time.time()

    avg = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        gau = cv2.GaussianBlur(frame, (5, 5), 1)
        gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.Canny(binary, 0, 130)
        contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if y > 50 and area > 25000:
                xlist_framedetec.append(x)
                ylist_framedetec.append(y)
                wlist_framedetec.append(w)
                hlist_framedetec.append(h)
                cv2.rectangle(edges, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Edges', edges)

        run_time = time.time() - base_time

        if run_time > 15: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        video_capture.write(frame)
        video_gaussian.write(gau)
        video_gray.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        video_framedeleta.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        video_binaryary.write(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        video_edges.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    average_x = sum(xlist_framedetec) / len(xlist_framedetec)
    average_y = sum(ylist_framedetec) / len(ylist_framedetec)
    average_w = sum(wlist_framedetec) / len(wlist_framedetec)
    average_h = sum(hlist_framedetec) / len(hlist_framedetec)

    xmin = int(average_x - 20)
    xmax = int(xmin + average_w + 60)
    ymin = int(average_y - 80)
    ymax = int(ymin + average_h + 60)

    cap.release()
    video_capture.release()
    video_gaussian.release()
    video_gray.release()
    video_framedeleta.release()
    video_binaryary.release()
    video_edges.release()
    cv2.destroyAllWindows()

    return xmin, xmax, ymin, ymax

def excel_entry():
    path = 'C:\\Users\\admin\\Desktop\\data\\rotation_data\\excel_data\\' + date_number + '.xlsx'
    wb = openpyxl.Workbook()

    linesheetname = 'lines'
    measurementdatasheetname = 'data'
    rectdatasheetname = 'rect'

    wb.create_sheet(linesheetname)
    wb.create_sheet(measurementdatasheetname)
    wb.create_sheet(rectdatasheetname)

    ws_line = wb[linesheetname]
    ws_md = wb[measurementdatasheetname]
    ws_rd = wb[rectdatasheetname]

    ws_line['C2'] = 'average'
    ws_line['D3'] = 'x0'
    ws_line['E3'] = 'y0'
    ws_line['F3'] = 'x1'
    ws_line['G3'] = 'y1'
    ws_line['J2'] = 'sorted'
    ws_line['J3'] = 'x0'
    ws_line['K3'] = 'y0'
    ws_line['L3'] = 'x1'
    ws_line['M3'] = 'y1'

    ws_md['D3'] = 'gradient'
    ws_md['E3'] = 'white ratio'

    ws_rd['D3'] = 'x'
    ws_rd['E3'] = 'y'
    ws_rd['F3'] = 'w'
    ws_rd['G3'] = 'h'

    x0list_sorted = sorted(x0list, reverse = True)
    y0list_sorted = sorted(y0list, reverse = True)
    x1list_sorted = sorted(x1list, reverse = True)
    y1list_sorted = sorted(y1list, reverse = True)

    for i0 in range(0, len(x0list)):
        ws_line.cell(i0 + 4, 4, value = x0list[i0])
        ws_line.cell(i0 + 4, 5, value = y0list[i0])
        ws_line.cell(i0 + 4, 6, value = x1list[i0])
        ws_line.cell(i0 + 4, 7, value = y1list[i0])
        ws_line.cell(i0 + 4, 10, value = x0list_sorted[i0])
        ws_line.cell(i0 + 4, 11, value = y0list_sorted[i0])
        ws_line.cell(i0 + 4, 12, value = x1list_sorted[i0])
        ws_line.cell(i0 + 4, 13, value = y1list_sorted[i0])
        ws_md.cell(i0 + 4, 4, value = gradlist[i0])

    for i1 in range(0, len(whiteratiolist)):
        ws_md.cell(i1 + 4, 5, value = whiteratiolist[i1])

    for i2 in range(0, len(xlist)):
        ws_rd.cell(i2 + 4, 4, value = xlist[i2])
        ws_rd.cell(i2 + 4, 5, value = ylist[i2])
        ws_rd.cell(i2 + 4, 6, value = wlist[i2])
        ws_rd.cell(i2 + 4, 7, value = hlist[i2])

    avex0 = sum(x0list) / len(x0list)
    avey0 = sum(y0list) / len(y0list)
    avex1 = sum(x1list) / len(x1list)
    avey1 = sum(y1list) / len(y1list)

    ws_line.cell(2, 4, value = avex0)
    ws_line.cell(2, 5, value = avey0)
    ws_line.cell(2, 6, value = avex1)
    ws_line.cell(2, 7, value = avey1)

    wb.save(path)
    wb.close()

# def frame_rotation(xmin, xmax, ymin, ymax):

while True:
    ret, frame = cap.read()
    if not ret: break

    copyframe = frame.copy()

    gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    bin_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
    horizon = cv2.filter2D(bin_line, -1, kernel_hor)
    dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
    lines = cv2.HoughLinesP(horizon, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 30)

    cv2.accumulateWeighted(gray, avg, 0.8)
    framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    bin_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
    
    edges = cv2.Canny(bin_fd, 0, 130)
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if lines is not None:
        row_columm = lines.shape
        for line in lines:
            x0, y0, x1, y1 = line[0]
            delta_Y = y1 - y0
            delta_X = x1 - x0
            gradient = (delta_Y / delta_X) * 10
            if gradient > -10:
                x0list.append(x0)
                y0list.append(y0)
                x1list.append(x1)
                y1list.append(y1)
                gradlist.append(gradient)
                whiteratio = (cv2.countNonZero(bin_fd) / (640 * 360)) * 100
                whiteratiolist.append(whiteratio)
                cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)


    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area  = w * h
        xlist.append(x)
        ylist.append(y)
        wlist.append(w)
        hlist.append(h)
        cv2.rectangle(copyframe, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

    # deltagrad_indexW = gradlist[-1] - gradlist[-2]
    # deltagrad_indexV = gradlist[-2] - gradlist[-3]
    # deltaratio_indexW = whiteratiolist[-1] - whiteratiolist[-2]
    # deltaratio_indexV = whiteratiolist[-2] - whiteratiolist[-3]

    cv2.imshow('frame', frame)
    cv2.imshow('copy frame', copyframe)
    cv2.imshow('binary lines', bin_line)
    cv2.imshow('binary frame delta', bin_fd)
    cv2.imshow('edges', edges)
    cv2.imshow('horizon', horizon)
    cv2.imshow('dilation', dilation)
    # print('(row, columm)=', row_columm)]
    print('line number', row_columm[0])
    # print('lines = ', lines)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

    frame_save.write(frame)
    binline_save.write(cv2.cvtColor(bin_line, cv2.COLOR_GRAY2BGR))
    horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
    dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
    binframedelta_save.write(cv2.cvtColor(bin_fd, cv2.COLOR_GRAY2BGR))
    edges_save.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    copyframe_save.write(copyframe)

excel_entry()

cap.release()
frame_save.release()
binline_save.release()
horizon_save.release()
dilation_save.release()
binframedelta_save.release()
edges_save.release()
copyframe_save.release()
cv2.destroyAllWindows()