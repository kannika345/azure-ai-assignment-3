import tensorflow as tf
from tensorflow.keras import datasets
import cv2
import matplotlib.pyplot as plt

print("--- IMAGE PROCESSING DEMO STARTED ---")

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Pick one sample image
img = x_test[0]

# Show original
plt.subplot(1,4,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

# 2. Convert to OpenCV format
img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 3. Apply blur
blur = cv2.GaussianBlur(img_cv, (5,5), 0)
plt.subplot(1,4,2)
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title("Blur")
plt.axis("off")

# 4. Apply sharpen
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])
sharpen = cv2.filter2D(img_cv, -1, kernel)
plt.subplot(1,4,3)
plt.imshow(cv2.cvtColor(sharpen, cv2.COLOR_BGR2RGB))
plt.title("Sharpen")
plt.axis("off")

# 5. Apply edge detection
edges = cv2.Canny(img_cv, 100, 200)
plt.subplot(1,4,4)
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.axis("off")

plt.show()
