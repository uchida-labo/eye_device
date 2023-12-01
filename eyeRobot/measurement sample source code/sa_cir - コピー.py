import cv2
import numpy as np

fontType = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread("/Users/nakanokota/Documents/sa_cira.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #グレースケール変換
gray = cv2.GaussianBlur(gray, (33, 33), 1)      #平滑化
#gray = cv2.bilateralFilter(gray, 9, 75, 75)

#ハフ変換による円検出
circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.08, 70, param1=100, param2=65, minRadius=0, maxRadius=270)

#np.float型から整数値に丸めて、さらに16ビット情報にキャスト
circle = np.uint16(np.around(circle))



for i in circle[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),3)     #円周を描画する
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)        #中心点を描画する(単位：[ピクセル]、原点は左上)
    # pts = []
    # pts.append((int(i[0]), int(i[1])))
    pts = np.array((int(i[0]), int(i[1])))
    print(pts)
    cv2.putText(img, '{}'.format(pts), (i[0]+5, i[1]+5), fontType, 1, (0, 255, 0), 2)

cv2.imwrite('sample_aftera.png', img)      # フレームを画面に表示

