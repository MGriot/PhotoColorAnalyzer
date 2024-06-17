import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = None
        self.load_image()

    def load_image(self):
        if not os.path.exists(self.image_path):
            print(f"The image file {self.image_path} does not exist.")
            return

        self.img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.img is None:
            print(f"The image file {self.image_path} could not be opened.")
            return

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGBA)

    def gaussian_blur(self, kernel_size=7, sigma=5):
        if self.img is None:
            print("No image loaded.")
            return

        img_blur = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), sigma)
        return img_blur

    def simple_blur(self, kernel_size=7):
        if self.img is None:
            print("No image loaded.")
            return

        img_blur = cv2.blur(self.img, (kernel_size, kernel_size))
        return img_blur

    def median_blur(self, kernel_size=7):
        if self.img is None:
            print("No image loaded.")
            return

        img_blur = cv2.medianBlur(self.img, kernel_size)
        return img_blur

    def bilateral_filter(self, d=15, sigma_color=75, sigma_space=75):
        if self.img is None:
            print("No image loaded.")
            return

        img_blur = cv2.bilateralFilter(self.img, d, sigma_color, sigma_space)
        return img_blur

    def show_image(self, img, title="Image"):
        plt.imshow(img)
        plt.title(title)
        plt.show()
