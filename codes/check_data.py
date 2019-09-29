import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("D:\Coding\PycharmProjects\Image Processing Project\Data - DnCNN-S\Train400\\test_023.png", cv2.IMREAD_GRAYSCALE)

image = np.array(image, dtype = np.uint8)
plt.subplot(221)
plt.title('Ground Truth')
plt.imshow(image, cmap = 'gray')

normalized_image = image / 255
plt.subplot(222)
plt.title('Normalized Image')
plt.imshow(normalized_image, cmap = 'gray')

noisy_image = image + np.random.normal(0, 25, image.shape)
plt.subplot(223)
plt.title('Nosiy Image')
plt.imshow(noisy_image, cmap = 'gray')

normalized_noisy_image = normalized_image + np.random.normal(0, 25 / 255, image.shape)
plt.subplot(224)
plt.title('Normalized Noisy Image')
plt.imshow(normalized_noisy_image, cmap = 'gray')

plt.show()