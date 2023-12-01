import cv2
import numpy as np

def calc_black_whiteArea(bw_image):
    image_size = bw_image.size
    whitePixels = cv2.countNonZero(bw_image)
    blackPixels = bw_image.size - whitePixels

    whiteAreaRatio = (whitePixels/image_size)*100#[%]
    blackAreaRatio = (blackPixels/image_size)*100#[%]

    print("White Area [%] : ", whiteAreaRatio)
    print("Black Area [%] : ", blackAreaRatio)
    print("image size:", image_size)


if __name__ == "__main__":
    # read input image
    image = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")

    # convert grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # black white
    ret, bw_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

    # save image
    cv2.imwrite("/Users/nakanokota/Desktop/black_white.jpg", bw_image)

    # calculation black and white area
    calc_black_whiteArea(bw_image)