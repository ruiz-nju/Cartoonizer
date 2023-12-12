import numpy as np
import cv2
from collections import defaultdict
from scipy import stats
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def build_centroids(histogram):
    max_size = 80  # 质心数组的最大尺寸
    centroids = np.array([128])  # 初始的质心数组

    while True:
        while True:
            groups = defaultdict(list)  # 质心-{灰度值} 字典
            for i in range(len(histogram)):
                if histogram[i] == 0:  # 具有i的灰度值的像素数量为0，不作处理
                    continue

                distances = np.abs(centroids - i)  # 求出质心与当前灰度值的差异
                nearest_centroid_index = np.argmin(distances)  # 找出距离当前灰度值最近的质心
                groups[nearest_centroid_index].append(i)  # 将该灰度值加入最近质心所相对的灰度值List中

            new_centroids = np.array(centroids)

            for index, grayscale_value in groups.items():
                if new_centroids[index] == 0:  # 质心的灰度值为0
                    continue

                new_centroids[index] = int(
                    np.sum(grayscale_value * histogram[grayscale_value])
                    / np.sum(histogram[grayscale_value])  # 计算质心的平均位置
                )

            if np.sum(new_centroids - centroids) == 0:  # 更新后的质心数组与原质心数组相同，代表迭代完成
                break
            centroids = new_centroids

        new_centroids = set()
        for i, grayscale_value in groups.items():
            if len(grayscale_value) < max_size:
                new_centroids.add(centroids[i])
                continue
            z, p = stats.normaltest(histogram[grayscale_value])  # 正态性检验
            if p < 0.001:  # 数据分布非正态
                if i == 0:
                    left = 0
                else:
                    left = centroids[i - 1]

                if i == len(centroids) - 1:
                    right = len(histogram) - 1
                else:
                    right = centroids[i + 1]

                delta = right - left  # 计算左右质心的距离
                if delta >= 3:
                    c1 = (centroids[i] + left) / 2
                    c2 = (centroids[i] + right) / 2
                    new_centroids.add(c1)
                    new_centroids.add(c2)
                else:
                    new_centroids.add(centroids[i])
            else:
                new_centroids.add(centroids[i])

        if len(new_centroids) == len(centroids):
            break
        else:
            centroids = np.array(sorted(new_centroids))

    return centroids


def cartoonize(image):
    output_image = np.array(image)
    height, width, channels = output_image.shape

    # 边缘检测
    edges = cv2.Canny(output_image, 100, 200)

    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2HSV)

    # 双边滤波器
    for i in range(channels):
        output_image[:, :, i] = cv2.bilateralFilter(output_image[:, :, i], 9, 75, 75)

    # 统计灰度值 创建直方图
    hists = []
    for i in range(channels):
        histogram, _ = np.histogram(output_image[:, :, i], bins=np.arange(256 + 1))
        hists.append(histogram)

    # 构建质心数组
    centroids = []
    for histogram in hists:
        centroids.append(build_centroids(histogram))

    # 先降维至二维 处理结束后转化回三维
    output_image = output_image.reshape((-1, channels))
    for i in range(channels):
        channel = output_image[:, i]
        nearest_centroid_index = np.argmin(
            np.abs(channel[:, np.newaxis] - centroids[i]), axis=1
        )
        output_image[:, i] = centroids[i][nearest_centroid_index]
    output_image = output_image.reshape((height, width, channels))
    output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2RGB)

    # 轮廓处理
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_image, contours, -1, (0, 0, 0), thickness=1)

    return output_image


def set_background(window, path):
    # 加载并调整图像大小以适应窗口
    bg_image = Image.open(path)
    bg_image = bg_image.resize((1500, 750), Image.LANCZOS)

    bg_image = ImageTk.PhotoImage(bg_image)

    bg_label = tk.Label(window, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_label.image = bg_image


root = tk.Tk()
root.title("cartoonizer")
root.geometry("1500x750")
root.resizable(width=False, height=False)

set_background(root, "background.png")


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = cv2.imread(file_path)
        display_image(original_image, original_image_label)

        cartoon_image = cartoonize(original_image)
        display_image(cartoon_image, cartoonized_image_label)
        cv2.imwrite("output.jpg", cartoon_image)


def display_image(image, image_label, max_size=500):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = img.size

    scale = min(max_size / width, max_size / height)

    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img


original_image_label = tk.Label(root)
original_image_label.pack(side=tk.LEFT, padx=10, pady=20)

cartoonized_image_label = tk.Label(root)
cartoonized_image_label.pack(side=tk.RIGHT, padx=10, pady=20)


# 添加按钮样式
button_style = {
    "font": ("楷体", 20),
    "bg": "#4a7abc",
    "fg": "white",
    "activebackground": "#6b9edf",
    "padx": 10,
    "pady": 5,
}

open_button = tk.Button(root, text="打开图像", command=open_image, **button_style)
open_button.pack(pady=10)  # 增加按钮周围的间距

image_label = tk.Label(root)
image_label.pack(pady=20)  # 增加间隙

root.mainloop()
