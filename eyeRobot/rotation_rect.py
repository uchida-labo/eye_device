import cv2, openpyxl, os, time, math
import numpy as np

date_number = '1228_1'

xlist_framedetec, ylist_framedetec, wlist_framedetec, hlist_framedetec = [], [], [], []
x0list_tc, y0list_tc, x1list_tc, y1list_tc = [], [], [], []
gradientlist_calculation, ratiolist_calculation, degreelist_calculation = [], [], []
x0list_md, y0list_md, x1list_md, y1list_md = [], [], [], []
gradlist_md, whiteratiolist_md, degreelist_md = [], [], []

def Frame_detect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video_savepath_framedetec = 'C:\\Users\\admin\\Desktop\\measurement_data\\rotaion_date\\frame_detection\\' + date_number
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

def Thresh_calculation(xmin, xmax, ymin, ymax):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    cut_width = xmax - xmin
    cut_height = ymax - ymin

    video_savepath_threshdetec = 'C:\\Users\\admin\\Desktop\\measurement_data\\rotation_date\\thresh_detection\\' + date_number
    os.makedirs(video_savepath_threshdetec)

    video_capture = cv2.VideoWriter(video_savepath_threshdetec + '\\capture.mp4', fourcc, 30, (640, 360))
    video_cutframe = cv2.VideoWriter(video_savepath_threshdetec + '\\cutframe.mp4', fourcc, 30, (cut_width, cut_height))
    video_gaussian = cv2.VideoWriter(video_savepath_threshdetec + '\\gaussian.mp4', fourcc, 30, (cut_width, cut_height))
    video_gray = cv2.VideoWriter(video_savepath_threshdetec + '\\gray.mp4', fourcc, 30, (cut_width, cut_height))
    video_framedelta = cv2.VideoWriter(video_savepath_threshdetec + '\\framedelta.mp4', fourcc, 30, (cut_width, cut_height))
    video_binary_framedelta = cv2.VideoWriter(video_savepath_threshdetec + '\\binary_framedelta.mp4', fourcc, 30, (cut_width, cut_height))
    video_binary_line = cv2.VideoWriter(video_savepath_threshdetec + '\\binary_line.mp4', fourcc, 30, (cut_width, cut_height))
    video_horizon = cv2.VideoWriter(video_savepath_threshdetec + '\\horizon.mp4', fourcc, 30, (cut_width, cut_height))
    video_dilation = cv2.VideoWriter(video_savepath_threshdetec + '\\dilation.mp4', fourcc, 30, (cut_width, cut_height))
    video_closing = cv2.VideoWriter(video_savepath_threshdetec + '\\closing.mp4', fourcc, 30, (cut_width, cut_height))
    video_opening = cv2.VideoWriter(video_savepath_threshdetec + '\\opening.mp4', fourcc, 30, (cut_width, cut_height))

    base_time = time.time()
    blink_time = 0

    avg = None

    kernel_hor = np.array([
    [1, 1, 1], 
    [0, 0, 0], 
    [-1, -1, -1]])

    kernel_calculation = np.ones((3, 3), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break

        cut_frame = frame[ymin:ymax, xmin:xmax]
        gau = cv2.GaussianBlur(cut_frame, (5, 5), 1)
        gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        binary_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        

        binary_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(binary_line, -1, kernel = kernel_hor)
        dilation = cv2.dilate(horizon, kernel = kernel_calculation, iterations = 1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel = kernel_calculation)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel = kernel_calculation)
        lines = cv2.HoughLinesP(opening, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                Gradient = 10 * (delta_Y / delta_X)
                if Gradient > -10:
                    x0list_tc.append(x0)
                    y0list_tc.append(y0)
                    x1list_tc.append(x1)
                    y1list_tc.append(y1)
                    gradientlist_calculation.append(Gradient)
                    whiteratio = (cv2.countNonZero(binary_fd) / ((xmax - xmin) * (ymax - ymin))) * 100
                    ratiolist_calculation.append(whiteratio)
                    cv2.line(cut_frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
                    if whiteratio == 0:
                        B = x1list_tc[-1] - x0list_tc[-1]
                        C = y1list_tc[-1] - y0list_tc[-1]
                        A = math.sqrt(B ** 2 + C ** 2)
                        radian = np.arccos(B / A)
                        degree = np.rad2deg(radian)
                        degreelist_calculation.append(degree)

        cv2.imshow('Blink', binary_fd)
        cv2.imshow('Cut frame', cut_frame)

        run_time = time.time() - base_time

        if run_time > 15: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        video_capture.write(frame)
        video_cutframe.write(cut_frame)
        video_gaussian.write(gau)
        video_gray.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        video_framedelta.write(cv2.cvtColor(framedelta, cv2.COLOR_GRAY2BGR))
        video_binary_framedelta.write(cv2.cvtColor(binary_fd, cv2.COLOR_GRAY2BGR))
        video_binary_line.write(cv2.cvtColor(binary_line, cv2.COLOR_GRAY2BGR))
        video_horizon.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
        video_dilation.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
        video_closing.write(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR))
        video_opening.write(cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR))
        
    cap.release()
    video_capture.release()
    video_cutframe.release()
    video_gaussian.release()
    video_gray.release()
    video_framedelta.release()
    video_binary_framedelta.release()
    video_binary_line.release()
    video_horizon.release()
    video_dilation.release()
    video_closing.release()
    video_opening.release()
    cv2.destroyAllWindows()

    sorted_ratiolist = sorted(ratiolist_calculation, reverse = True)
    max_ratio = sorted_ratiolist[0]
    thresh_ratio_high = int(max_ratio + 7)
    thresh_ratio_low = int(max_ratio - 7)

    sorted_gradientlist = sorted(gradientlist_calculation, reverse = True)
    max_gradient = sorted_gradientlist[0]
    thresh_gradient_high = max_gradient + 1.5
    thresh_gradient_low = max_gradient - 1.5

    return thresh_ratio_high, thresh_ratio_low, thresh_gradient_high, thresh_gradient_low

def excel_entry():
    path = 'C:\\Users\\admin\\Desktop\\measurement_data\\rotation_data\\excel_data\\' + date_number + '.xlsx'
    wb = openpyxl.Workbook()

    framedetec_sheetname = 'frame_detection'
    threshcal_sheetname = 'thresh_calculation'
    measurement_sheetname = 'measurement'

    wb.create_sheet(framedetec_sheetname)
    wb.create_sheet(threshcal_sheetname)
    wb.create_sheet(measurement_sheetname)

    ws_fd = wb[framedetec_sheetname]
    ws_tc = wb[threshcal_sheetname]
    ws_md = wb[measurement_sheetname]

    ws_fd['B2'] = 'Frame detection'
    ws_fd['D3'] = 'x'
    ws_fd['E3'] = 'y'
    ws_fd['F3'] = 'w'
    ws_fd['G3'] = 'h'
    ws_fd['I3'] = 'xmin'
    ws_fd['I4'] = 'xmax'
    ws_fd['I5'] = 'ymin'
    ws_fd['I6'] = 'ymax'

    ws_tc['B2'] = 'Thresh detection'
    ws_tc['D3'] = 'x0'
    ws_tc['E3'] = 'y0'
    ws_tc['F3'] = ''


    # x0list_sorted = sorted(x0list, reverse = True)
    # y0list_sorted = sorted(y0list, reverse = True)
    # x1list_sorted = sorted(x1list, reverse = True)
    # y1list_sorted = sorted(y1list, reverse = True)

    # avex0 = sum(x0list) / len(x0list)
    # avey0 = sum(y0list) / len(y0list)
    # avex1 = sum(x1list) / len(x1list)
    # avey1 = sum(y1list) / len(y1list)

    ws_fd.cell(2, 4, value = avex0)
    ws_fd.cell(2, 5, value = avey0)
    ws_fd.cell(2, 6, value = avex1)
    ws_fd.cell(2, 7, value = avey1)

    wb.save(path)
    wb.close()

# def frame_rotation(xmin, xmax, ymin, ymax):
#     B = x1list[-1] - x0list[-1]
#     C = y1list[-1] - y0list[-1]
#     A = math.sqrt(B ** 2 + C ** 2)
#     radian = np.arccos(B / A)
#     degree = np.rad2deg(radian)

def main_sample():
    xmin, xmax, ymin, ymax = Frame_detect()
    ratio_high, ratio_low, grad_high, grad_low = Thresh_calculation(xmin, xmax, ymin, ymax)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    avg = None

    kernel_hor = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]])
    
    kernel_cal = np.ones((3, 3), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break

        copyframe = frame.copy()

        gaussian = cv2.GaussianBlur(frame, (3, 3), 1)
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        bin_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        horizon = cv2.filter2D(bin_line, -1, kernel_hor)
        dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
        lines = cv2.HoughLinesP(dilation, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 30)

        cv2.accumulateWeighted(gray, avg, 0.8)
        framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        bin_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]

        if lines is not None:
            row_columm = lines.shape
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                gradient = (delta_Y / delta_X) * 10
                if gradient > -10:
                    x0list_md.append(x0)
                    y0list_md.append(y0)
                    x1list_md.append(x1)
                    y1list_md.append(y1)
                    gradlist_md.append(gradient)
                    whiteratio = (cv2.countNonZero(bin_fd) / (640 * 360)) * 100
                    whiteratiolist_md.append(whiteratio)
                    cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
                    if whiteratio == 0:
                        B = x1list_md[-1] - x0list_md[-1]
                        C = y1list_md[-1] - y0list_md[-1]
                        A = math.sqrt(B ** 2 + C ** 2)
                        radian = np.arccos(B / A)
                        degree = np.rad2deg(radian)
                        degreelist_md.append(degree)

        




# def main():
#     xmin, xmax, ymin, ymax = Frame_detect()
#     ratio_high, ratio_low, grad_high, grad_low = Thresh_calculation(xmin, xmax, ymin, ymax)

#     fonttype = cv2.FONT_HERSHEY_COMPLEX

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#     width = xmax - xmin
#     height = ymax - ymin

#     video_path = 'C:\\Users\\admin\\Desktop\\data\\rotation_data\\video_data\\' + date_number
#     os.makedirs(video_path)

#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

#     frame_save = cv2.VideoWriter(video_path + '\\frame.mp4', fourcc, 30, (640, 360))
#     binline_save = cv2.VideoWriter(video_path + '\\bin_line.mp4', fourcc, 30, (640, 360))
#     horizon_save = cv2.VideoWriter(video_path + '\\horizon.mp4', fourcc, 30, (640, 360))
#     dilation_save = cv2.VideoWriter(video_path + '\\dilation.mp4', fourcc, 30, (640, 360))
#     binframedelta_save = cv2.VideoWriter(video_path + '\\bin_framedelta.mp4', fourcc, 30, (640, 360))
#     edges_save = cv2.VideoWriter(video_path + '\\edges.mp4', fourcc, 30, (640, 360))
#     copyframe_save = cv2.VideoWriter(video_path + '\\copy_frame.mp4', fourcc, 30, (640, 360))

#     avg = None

#     kernel_hor = np.array([
#         [1, 1, 1],
#         [0, 0, 0],
#         [-1, -1, -1]])

#     kernel_cal = np.ones((3, 3), np.uint8)


#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         copyframe = frame.copy()

#         gaussian = cv2.GaussianBlur(frame, (5, 5), 1)
#         gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

#         if avg is None:
#             avg = gray.copy().astype("float")
#             continue

#         bin_line = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
#         horizon = cv2.filter2D(bin_line, -1, kernel_hor)
#         dilation = cv2.dilate(horizon, kernel_cal, iterations = 1)
#         lines = cv2.HoughLinesP(horizon, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 30)

#         cv2.accumulateWeighted(gray, avg, 0.8)
#         framedelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
#         bin_fd = cv2.threshold(framedelta, 3, 255, cv2.THRESH_BINARY)[1]
        
#         edges = cv2.Canny(bin_fd, 0, 130)
#         contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
#         if lines is not None:
#             row_columm = lines.shape
#             for line in lines:
#                 x0, y0, x1, y1 = line[0]
#                 delta_Y = y1 - y0
#                 delta_X = x1 - x0
#                 gradient = (delta_Y / delta_X) * 10
#                 if gradient > -10:
#                     x0list.append(x0)
#                     y0list.append(y0)
#                     x1list.append(x1)
#                     y1list.append(y1)
#                     gradlist.append(gradient)
#                     whiteratio = (cv2.countNonZero(bin_fd) / (640 * 360)) * 100
#                     whiteratiolist.append(whiteratio)
#                     cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

#         cv2.imshow('frame', frame)
#         cv2.imshow('copy frame', copyframe)
#         cv2.imshow('binary lines', bin_line)
#         cv2.imshow('binary frame delta', bin_fd)
#         cv2.imshow('edges', edges)
#         cv2.imshow('horizon', horizon)
#         cv2.imshow('dilation', dilation)
#         # print('(row, columm)=', row_columm)]
#         print('line number', row_columm[0])
#         # print('lines = ', lines)

#         if cv2.waitKey(1) & 0xFF == ord('q'): break

#         frame_save.write(frame)
#         binline_save.write(cv2.cvtColor(bin_line, cv2.COLOR_GRAY2BGR))
#         horizon_save.write(cv2.cvtColor(horizon, cv2.COLOR_GRAY2BGR))
#         dilation_save.write(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR))
#         binframedelta_save.write(cv2.cvtColor(bin_fd, cv2.COLOR_GRAY2BGR))
#         edges_save.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
#         copyframe_save.write(copyframe)

#     excel_entry()

#     cap.release()
#     frame_save.release()
#     binline_save.release()
#     horizon_save.release()
#     dilation_save.release()
#     binframedelta_save.release()
#     edges_save.release()
#     copyframe_save.release()
#     cv2.destroyAllWindows()

