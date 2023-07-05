import cv2
 
c = cv2.VideoCapture(0+cv2.CAP_DSHOW)               
w, h= c.get(cv2.CAP_PROP_FRAME_WIDTH), c.get(cv2.CAP_PROP_FRAME_HEIGHT)#w:640.0px+h:480.0px
print('w:'+str(w)+'px+h:'+str(h)+'px')
c.set(cv2.CAP_PROP_FRAME_WIDTH, w/2)
c.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)
w, h= c.get(cv2.CAP_PROP_FRAME_WIDTH), c.get(cv2.CAP_PROP_FRAME_HEIGHT)#w:320.0px+h:240.0px
print('w:'+str(w)+'px+h:'+str(h)+'px')

# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret, frame = c.read()              # フレームを取得
    cv2.imshow('output', frame)             # フレームを画面に表示

    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 撮影用オブジェクトとウィンドウの解放
c.release()
cv2.destroyAllWindows()