import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn đến file ảnh vệ tinh
file_path = 'IMG/PTAnh.jpg'

# Mở ảnh với Rasterio và đọc băng đầu tiên
with rasterio.open(file_path) as src:
    band = src.read(1)  # Đọc băng 1 (hoặc băng nào mà bạn muốn xử lý)

# Áp dụng bộ lọc Sobel
# Tính đạo hàm theo hướng x và y
sobel_x = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)  # Sobel theo trục x
sobel_y = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)  # Sobel theo trục y

# Tính độ lớn gradient để kết hợp hai hướng
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

# Hiển thị kết quả
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(band, cmap='gray')
plt.title("Ảnh gốc")

plt.subplot(1, 3, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title("Sobel theo trục X")

plt.subplot(1, 3, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel theo trục Y")

plt.figure()
plt.imshow(sobel_combined, cmap='gray')
plt.title("Kết quả Sobel kết hợp")
plt.colorbar()
plt.show()
