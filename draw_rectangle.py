import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# im = Image.open("/home/jordan/vlcsnap-2024-06-13-18h57m26s400.png")
im = Image.open("/home/jordan/vlcsnap-2024-06-13-22h03m18s521.png")

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

# Create a Rectangle patch
# rect = patches.Rectangle((98, 79), 33, 76, linewidth=1, edgecolor="r", facecolor="none")
# rect = patches.Rectangle(
#     (29 * 2 - 5, 79 * 2 - 5),
#     55 * 2 - 5,
#     82 * 2 - 5,
#     linewidth=1,
#     edgecolor="r",
#     facecolor="none",
# )

# 576x324 -> 1920x1080
# 1920 / 576 = 3.3333
# 1080 / 324 = 3.3333
rect = patches.Rectangle(
    (226 * 3.333333, 42 * 3.333333),
    31 * 3.333333,
    40 * 3.333333,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
