import cv2, time, openpyxl
import numpy as np

kernel_hor = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]])

grad_list, x0list, x0list_after, gradlist_after = [], [], [], []

date_and_number = '1225-3'

def exceldata_entry():
    path = 'C:\\Users\\admin\\Desktop\\data\\gradient_data\\grad.xlsx'
    wb = openpyxl.Workbook()

    wb.create_sheet(date_and_number)

    ws = wb[date_and_number]


    ws['D3'] = 'gradient'
    ws['E3'] = 'x0'
    ws['F3'] = 'x0 sorted'
    ws['I3'] = 'gradient'
    ws['J3'] = 'x0'

    ws['L3'] = 'average x0'

    newlist = sorted(x0list, reverse = True)

    for i0 in range(0, len(grad_list)):
        ws.cell(i0 + 4, 4, value = grad_list[i0])
        ws.cell(i0 + 4, 5, value = x0list[i0])
        ws.cell(i0 + 4, 6, value = newlist[i0])

    for i1 in range(0, len(gradlist_after)):
        ws.cell(i1 + 4, 9, value = gradlist_after[i1])
        ws.cell(i1 + 4, 10, value = x0list_after[i1])

    ave_x0 = sum(x0list) / len(x0list)
    ws.cell(4, 12, value = ave_x0)

    wb.save(path)
    wb.close()

def grad_rect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


    while True:
        ret, frame = cap.read()
        if not ret: break

        gau = cv2.GaussianBlur(frame, (5, 5), 1)
        gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

        bin = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
        hor = cv2.filter2D(bin, -1, kernel = kernel_hor)
        lines = cv2.HoughLinesP(hor, rho = 1, theta = np.pi / 360, threshold = 100, minLineLength = 130, maxLineGap = 70)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                delta_Y = y1 - y0
                delta_X = x1 - x0
                grad = 10 * (delta_Y / delta_X)
                x0list.append(x0)
                grad_list.append(grad)
                cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
                if grad < 6 and grad > 0 and x0 < 450:
                    x0list_after.append(x0)
                    gradlist_after.append(grad)
                    cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 1)


        cv2.imshow('Horizon', hor)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    exceldata_entry()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    grad_rect()