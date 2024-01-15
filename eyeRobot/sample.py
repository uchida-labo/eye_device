import cv2, math
import numpy as np

cap = cv2.VideoCapture(0)

xmin, xmax = 100, 300
ymin, ymax = 200, 400

width = 200
height = 200

topleft = (300,50)
topright = (500, 50)
bottomleft = (300, 250)
bottomright = (500, 250)

# pts1 = np.float32([topleft, topright, bottomleft, bottomright])
# pts2 = np.float32([[[100, 100], [500, 100], [0, height], [width, height]]])

while True:
    ret, frame = cap.read()
    if not ret: break

    originpts = np.array([topleft, bottomleft, bottomright, topright], dtype = np.float32)
    center = (originpts[0] + originpts[2]) / 2

    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), 30, 1.0)
    rotatedroi = cv2.warpAffine(frame, rotation_matrix, (640, 480))

    mask = np.zeros_like(frame, dtype=np.uint8)
    # cv2.fillConvexPoly(mask, originpts.astype(np.int32), (0, 255, 0))
    # cv2.rectangle(mask, topleft, bottomright, (255, 0, 0), 2)
    result = cv2.bitwise_and(mask, frame)
    result += rotatedroi
    cutresult = result[50:250, 300:500]






    # rotatedpts = cv2.transform(np.array([originpts]), rotation_matrix)[0]

    # dstpts = np.array([[0, 0], [0, 200], [200, 200], [200, 0]], dtype = np.float32)
    # matrix = cv2.getPerspectiveTransform(rotatedpts, dstpts)
    # result = cv2.warpPerspective(frame, matrix, (width + 100, height + 100))

    # cutframe = frame[ymin:ymax, xmin:xmax]

    # tplx = 100 * math.cos(math.radians(30)) - 300 * math.sin(math.radians(30))
    # tply = 100 * math.sin(math.radians(30)) + 300 * math.cos(math.radians(30))

    # tprx = 300 * math.cos(math.radians(30)) - 300 * math.sin(math.radians(30))
    # tpry = 300 * math.sin(math.radians(30)) + 300 * math.cos(math.radians(30))

    # btlx = 100 * math.cos(math.radians(30)) - 400 * math.sin(math.radians(30))
    # btly = 100 * math.sin(math.radians(30)) + 400 * math.cos(math.radians(30))

    # btrx = 300 * math.cos(math.radians(30)) - 400 * math.sin(math.radians(30))
    # btry = 300 * math.sin(math.radians(30)) + 400 * math.cos(math.radians(30))

    # pts1 = np.float32([topleft, topright, bottomleft, bottomright])
    # pts2 = np.float32([[tplx, tply], [btlx, btly], [tprx, tpry], [btrx, btry]])

    # origin_size = np.float32([[0, 0], [0, 200], [200, 0], [200, 200]])

    # center = (pts1[0] + pts1[3]) / 2 

    # matrix = cv2.getPerspectiveTransform(pts2, origin_size)
    # result = cv2.warpPerspective(frame, matrix, (width + 100, height + 100))

    # # rotation_matrix = cv2.getRotationMatrix2D((100, 100), 30, 1)
    # # rotated_result = cv2.warpAffine(result, rotation_matrix, (width, height))

    cv2.imshow('result', result)
    # # cv2.imshow('rotated result', rotated_result)
    cv2.imshow('frame', frame)
    cv2.imshow('cut result', cutresult)
    # cv2.imshow('cutframe', cutframe)


    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()