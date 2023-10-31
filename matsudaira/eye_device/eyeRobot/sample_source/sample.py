import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt


# def imshow(img):
#     """ndarray 配列をインラインで Notebook 上に表示する。
#     """
#     ret, encoded = cv2.imencode("/Users/nakanokota/Documents/sample.jpg", img)
#     display(Image(encoded))


# 画像を読み込む。
img = cv2.imread("/Users/nakanokota/Documents/sample.jpg")

# グレースケールに変換する。
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ハフ変換で円検出する。
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=10, param1=80, param2=100
)

# 検出結果を描画する。
if circles is not None:
    for cx, cy, r in circles.squeeze(axis=0).astype(int):
        # 円の円周を描画する。
        cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
        # 円の中心を描画する。
        cv2.circle(img, (cx, cy), 2, (0, 255, 0), 2)
cv2.imwrite('sample_after_sample.png', img)