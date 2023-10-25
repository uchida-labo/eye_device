import cv2
import matplotlib.pyplot as plt 

def get_histogram(img):
    if len(img.shape) == 3:
        channels = 3
    else:
        channels = 1

    histogram = []
    for ch in range(channels):
        hist_ch = cv2.calcHist([img], [ch], None, [256], [0,256])
        histogram.append(hist_ch[:,0])
    
    return histogram

def draw_histogram(hist):
    ch = len(hist)

    if (ch == 1):
        colors = ["black"]
        label = ["Gray"]
    else:
        colors = ["blue", "green", "red"]
        label = ["B", "G", "R"]

    x = range(256)
    for col in range(ch):
        y = hist[col]
        plt.plot(x, y, color = colors[col], label = label[col])

    plt.xlabel('Pixel')
    plt.ylabel('Frequency')
    plt.xticks()
    plt.legend(loc=2)
    plt.show()

img = cv2.imread(R"C:\Users\admin\Desktop\white_remove_data\side.jpg",cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", gray)
cv2.imwrite(R"C:\Users\admin\Desktop\white_remove_data\side_gray.jpg", gray)

plt.rcParams["font.size"] = 15
plt.rcParams["figure.figsize"] = (8, 6)
hist = get_histogram(gray)
draw_histogram(hist)