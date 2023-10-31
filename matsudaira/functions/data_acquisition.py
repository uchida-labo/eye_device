import cv2



def draw_and_output(frame, x_center, y_center, radius, w_meter, h_meter, width, height):
    """
    Drawing circles and obtaining coordinates

    ・argument
    'frame':Frame for drawing
    'center', 'radius':Center coordinates and radius of the detection circle
    'meter':Camera coverage (in [mm])
    'width', 'height':Frame size

    ・return
    'frame':Frame with circles drawn
    'x', 'y':Coordinates of detection circle (in [mm])
    """

    center = (int(x_center), int(y_center))
    r = int(radius)
    cv2.circle(frame, center, r, (100, 255, 0), 2)
    cv2.circle(frame, center, 2, (0, 0, 255), 3)
    x = (w_meter * x_center)/width
    y = (h_meter * y_center)/height

    return frame, x, y