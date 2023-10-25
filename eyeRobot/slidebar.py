import cv2

# トラックバーのコールバック関数
def update_brightness(x):
    # スライドバーの値を取得
    brightness = x / 100.0
    
    # 画像の明るさを調整
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # 画像を表示
    cv2.imshow('Adjust Brightness', adjusted_image)

# 画像を読み込む
image = cv2.imread(R"C:\Users\admin\Desktop\white_remove_data\side.jpg",cv2.IMREAD_UNCHANGED)

# ウィンドウを作成
cv2.namedWindow('Adjust Brightness')

# トラックバーを作成
cv2.createTrackbar('Brightness', 'Adjust Brightness', 100, 200, update_brightness)

# 初期の明るさを設定
update_brightness(100)

# キーイベントを待つ
cv2.waitKey(0)

# ウィンドウを閉じる
cv2.destroyAllWindows()
