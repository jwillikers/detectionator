import cv2

img = cv2.imread("/home/jordan/vlcsnap-2024-06-15-11h27m49s864.png")

# 4608x2592
# 576x324 -> 1920x1080
# 1920 / 576 = 3.3333
# 1080 / 324 = 3.3333

x1 = 69.1488191485405
y1 = 293.4028208255768
x2 = 194.98350620269775
y2 = 158.028906583786

x_crop = (4608 % 1920) // 2
buffer = 5
# x_crop = 0

cv2.rectangle(
    img,
    (int(x1 * 3.333333) + x_crop + buffer, int(y1 * 3.333333) + buffer),
    (int(x2 * 3.333333) + x_crop + buffer, int(y2 * 3.333333) + buffer),
    (0, 255, 0, 0),
)

cv2.imshow("bounding_box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
