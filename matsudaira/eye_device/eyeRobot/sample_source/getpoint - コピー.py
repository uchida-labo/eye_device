import numpy as np 
import cv2

fontType = cv2.FONT_HERSHEY_COMPLEX     #中心点の座標を描画する際のフォント指定
c = cv2.VideoCapture(0+cv2.CAP_DSHOW)              #カメラ映像の取り込み

c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
c.set(cv2.CAP_PROP_FPS, 200)           # カメラFPSを60FPSに設定
c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

def p_tile_threshold(img_gry, per):
    """
    Pタイル法による2値化処理
    :param img_gry: 2値化対象のグレースケール画像
    :param per: 2値化対象が画像で占める割合
    :return img_thr: 2値化した画像
    """

    # ヒストグラム取得
    img_hist = cv2.calcHist([img_gry], [0], None, [256], [0, 256])

    # 2値化対象が画像で占める割合から画素数を計算
    all_pic = img_gry.shape[0] * img_gry.shape[1]
    pic_per = all_pic * per

    # Pタイル法による2値化のしきい値計算
    p_tile_thr = 0
    pic_sum = 0

    # 現在の輝度と輝度の合計(高い値順に足す)の計算
    for hist in img_hist:
        pic_sum += hist

        # 輝度の合計が定めた割合を超えた場合処理終了
        if pic_sum > pic_per:
            break

        p_tile_thr += 1

    # Pタイル法によって取得したしきい値で2値化処理
    ret, img_thr = cv2.threshold(img_gry, p_tile_thr, 200, cv2.THRESH_BINARY)

    return img_thr

# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret, frame = c.read()              # フレームを取得
    #frame = cv2.resize(frame, dsize=(640, 480))     #ハフ変換用にリサイズ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #グレースケール変換
    ptile = p_tile_threshold(gray, 0.18)
    bi = cv2.GaussianBlur(ptile, (33, 33), 2)      #平滑化
    #ハフ変換による円検出
    circle = cv2.HoughCircles(bi, cv2.HOUGH_GRADIENT, 1, bi.shape[0]/3, param1=80, param2=39, minRadius=0, maxRadius=0)

    if circle is not None:
        #np.float型から整数値に丸め、さらに16ビット情報にキャスト
        circle = np.uint16(np.around(circle))

        for i in circle[0,:]:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)  #円周を描画
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)     #中心点を描画
            # pts = np.array((int(i[0]), int(i[1])))
            # correction = np.array((int(i[0]), -1500))
            # pts_cor = [x-y for (x, y) in zip(pts, correction)]
            # print(pts, pts_cor)                  #座標の表示(x[mm], y[mm])
            #円の中心点から右に5[pixel]、上に5[pixel]の場所に補正前の中心点の座標描画
            #cv2.putText(frame, '{}'.format(pts), (i[0]+5, i[1]+50), fontType, 1, (0, 255, 0),3)
            #円の中心点から右に50[pixel]、上に50[pixel]の場所に補正後の中心点の座標を描画
            #cv2.putText(frame, '{}'.format(pts_cor), (i[0]+5, i[1]+5), fontType, 1, (0, 255, 0),3)
            
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