import cv2

def img_process(frame,trim_xmin, trim_xmax, trim_ymin, trim_ymax):
    """
    Image processing

    ・argument
    'frame':Camera image output screen
    'trim_x' and 'trim_y':Coordinates for specifying the cut area

    ・return
    'gray_img':Video after grayscale conversion
    'bin_img':Video after binarization
    'edges':Edges detected by the canny method
    'contours':Contour points extracted from 'edges'
    """
    gau_filter = cv2.GaussianBlur(frame, (5, 5), 1)
    gray_img = cv2.cvtColor(gau_filter[trim_ymin:trim_ymax, trim_xmin:trim_xmax], cv2.COLOR_BGR2GRAY)
    th_val, bin_img = cv2.threshold(gray_img, 55, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_img, 100, 170,)   # Set upper and lower thresholds
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (trim_xmin, trim_ymin))

    return gray_img, bin_img, edges, contours

