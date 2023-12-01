import cv2
 
c = cv2.VideoCapture(0+cv2.CAP_DSHOW)               
c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
c.set(cv2.CAP_PROP_FPS, 200)           # カメラFPSを60FPSに設定
c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret, frame = c.read()              # フレームを取得
    cv2.imshow('camera', frame)             # フレームを画面に表示
 
    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 撮影用オブジェクトとウィンドウの解放
c.release()
cv2.destroyAllWindows()