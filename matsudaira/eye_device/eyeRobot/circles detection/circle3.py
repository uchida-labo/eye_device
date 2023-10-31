import cv2
import numpy as np
from IPython import display
from matplotlib import pyplot as plt


# def imshow(img, format=".jpg", **kwargs):
#     """ndarray 配列をインラインで Notebook 上に表示する。
#     """
#     img = cv2.imencode(format, img)[1]
#     img = display.Image(img, **kwargs)
#     display.display(img)


def draw_contours(img, contours, ax):
    """輪郭の点及び線を画像上に描画する。
    """
    ax.imshow(img)
    ax.set_axis_off()

    for i, cnt in enumerate(contours):
        # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
        cnt = cnt.squeeze(axis=1)
        # 輪郭の点同士を結ぶ線を描画する。
        ax.add_patch(plt.Polygon(cnt, color="b", fill=None, lw=2))
        # 輪郭の点を描画する。
        ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)
        # 輪郭の番号を描画する。
        # ax.text(cnt[0][0], cnt[0][1], i, color="r", size="20", bbox=dict(fc="w"))

# 画像を読み込む。
img = cv2.imread("/Users/nakanokota/Documents/sample.jpg")

# グレースケールに変換する。
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2値化する
ret, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

# 輪郭を抽出する。
contours, hierarchy = cv2.findContours(
    bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

fig, ax = plt.subplots(figsize=(8, 8))
draw_contours(img, contours, ax)
for cnt in contours:
    # 輪郭に外接する円を取得する。
    center, radius = cv2.minEnclosingCircle(cnt)
    # 描画する。
    ax.add_patch(plt.Circle(xy=center, radius=radius, color="g", fill=None, lw=2))

plt.show()

cv2.imwrite('sample_cir.png', img)