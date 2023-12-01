# gratscale変換
# https://docs.opencv.org/4.1.0/de/d25/imgproc_color_conversions.html

import numpy as np 
import cv2

fontType = cv2.FONT_HERSHEY_COMPLEX     #中心点の座標を描画する際のフォント指定
c = cv2.VideoCapture(2)              #カメラ映像の取り込み

c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
c.set(cv2.CAP_PROP_FPS, 70)           # カメラFPSを60FPSに設定
c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret, frame = c.read()              # フレームを取得
    frame = cv2.resize(frame, dsize=(640, 480))     #ハフ変換用にリサイズ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #グレースケール変換
    # ２値化
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bi = cv2.GaussianBlur(bw, (33, 33), 1)      #平滑化
    #ハフ変換による円検出
    circle = cv2.HoughCircles(bi, cv2.HOUGH_GRADIENT, 1.15, bw.shape[0]/3, param1=55, param2=25, minRadius=0, maxRadius=50)

    if circle is not None:
        #np.float型から整数値に丸め、さらに16ビット情報にキャスト
        circle = np.uint16(np.around(circle))

        for i in circle[0,:]:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)  #円周を描画
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)     #中心点を描画
            pts = np.array((int(i[0]), int(i[1])))
            correction = np.array((int(i[0]), -1500))
            pts_cor = [x-y for (x, y) in zip(pts, correction)]
            print(pts, pts_cor)                  #座標の表示(x[mm], y[mm])
            #円の中心点から右に5[pixel]、上に5[pixel]の場所に補正前の中心点の座標描画
            cv2.putText(frame, '{}'.format(pts), (i[0]+5, i[1]+50), fontType, 1, (0, 255, 0),3)
            #円の中心点から右に50[pixel]、上に50[pixel]の場所に補正後の中心点の座標を描画
            cv2.putText(frame, '{}'.format(pts_cor), (i[0]+5, i[1]+5), fontType, 1, (0, 255, 0),3)
            
            # pts = []        #空の配列を作成
            # pts.append((float(i[0]), float(i[1])))  #中心点のxy座標を空の配列ptsに格納
            #中心点から右に5[pixel]、上に5[pixel]離れた場所に中心点の座標を描画
            # cv2.putText(frame, ','.join(map(str, pts)), (i[0]+5, i[1]+50), fontType, 1, (0, 255, 0),3)
            # cv2.putText(frame, ','.join(map(str, pts_cor)), (i[0]+5, i[1]+5), fontType, 1, (0, 255, 0),3)
    cv2.imshow('output', frame)             # フレームを画面に表示

    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 撮影用オブジェクトとウィンドウの解放
c.release()
cv2.destroyAllWindows()