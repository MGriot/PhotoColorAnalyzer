from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import AutoMinorLocator
import webcolors


class ImageCluster:
    def __init__(self, image_path, n_clusters=5, remove_low_alpha=True):
        self.image_path = image_path
        self.n_clusters = n_clusters
        self.remove_low_alpha = remove_low_alpha
        self.img = Image.open(self.image_path).convert("RGBA")
        self.img_array = np.array(self.img)
        self.data = self.img_array.reshape(-1, 4)
        self.clustered_img = None
        self.labels = None
        self.center_colors = None
        self.hex_colors = None
        self.rgb_colors = None
        self.unique_clusters = None
        self.counts = None
        self.total_pixels = None
        self.cluster_percentages = None
        self.counts_dict = None
        self.ordered_colors = None
        self.color_names = None

    def filter_alpha(self):
        if self.remove_low_alpha:
            mask = self.data[:, 3] >= 125
            self.data = self.data[mask]
        else:
            mask = np.ones(self.data.shape[0], dtype=bool)
        return mask

    def cluster(self):
        mask = self.filter_alpha()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.labels = kmeans.fit_predict(self.data)
        self.center_colors = kmeans.cluster_centers_
        self.clustered_img = np.zeros_like(self.img_array)
        labels_full = np.full(mask.shape[0], -1)
        labels_full[mask] = self.labels
        for i in range(self.img_array.shape[0]):
            for j in range(self.img_array.shape[1]):
                if labels_full[i * self.img_array.shape[1] + j] != -1:
                    self.clustered_img[i, j] = self.center_colors[
                        labels_full[i * self.img_array.shape[1] + j]
                    ]
                else:
                    self.clustered_img[i, j] = [
                        255,
                        255,
                        255,
                        0,
                    ]  # white or transparent
        self.unique_clusters, self.counts = np.unique(self.labels, return_counts=True)
        self.counts_dict = dict(zip(self.unique_clusters, self.counts))
        self.ordered_colors = [self.center_colors[i] for i in self.counts_dict.keys()]
        self.total_pixels = np.prod(self.img_array.shape[:2])
        self.cluster_percentages = (
            self.counts / self.total_pixels
        ) * 100  # Percentages

    def RGB_HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def RGBA_HEX(self, color):
        return "#{:02x}{:02x}{:02x}{:02x}".format(
            int(color[0]), int(color[1]), int(color[2]), int(color[3])
        )

    def HEX_COLORS(self):
        self.hex_colors = [
            self.RGBA_HEX(self.ordered_colors[i]) for i in self.counts_dict.keys()
        ]

    def RGB__COLORS(self):
        return [self.ordered_colors[i] for i in self.counts_dict.keys()]

    def get_color_names(self):
        self.color_names = [
            self.closest_color(
                tuple(map(int, webcolors.hex_to_rgb("#" + color[1:7])[0:3]))
            )
            for color in self.hex_colors
        ]

    def closest_color(self, requested_color):
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

    def calculate_brightness(self, color):
        # Converti il colore da hex a RGB
        color = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
        # Calcola la luminosità come la media dei valori RGB
        return sum(color) / (3 * 255)

    def plot_images(self):
        plt.subplot(1, 2, 1)
        plt.imshow(self.img_array)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(self.clustered_img)
        plt.title("Clustered Image ({} clusters)".format(self.n_clusters))
        plt.axis("off")
        plt.tight_layout()

    def plot_cluster_percentages(self):
        bars = plt.bar(
            x=self.unique_clusters,
            height=self.cluster_percentages,
            color=(self.center_colors / 255),
            edgecolor="black",
        )
        plt.bar_label(
            bars, [f"{perc:.1f}%" for perc in self.cluster_percentages]
        )  # Format percentages with 1 decimal place
        plt.xlabel("Cluster")
        plt.ylabel("Percentage (%)")
        plt.title("Percentage of Pixels in each Cluster")
        plt.tight_layout()

    def plot_pie_chart(self):
        self.HEX_COLORS()
        self.get_color_names()
        labels = [
            f"{name}\n{hex}" for name, hex in zip(self.color_names, self.hex_colors)
        ]
        patches, texts, autotexts = plt.pie(
            self.counts,
            labels=labels,
            colors=self.hex_colors,
            autopct="%1.1f%%",
            startangle=45,
            counterclock=False,
        )
        for i in range(len(self.hex_colors)):
            brightness = self.calculate_brightness(self.hex_colors[i][1:])
            if brightness < 0.5:
                autotexts[i].set_color("white")
        plt.tight_layout()

    def plot_bar_chart(self):
        counts_dict = dict(
            sorted(self.counts_dict.items(), key=lambda item: item[1], reverse=True)
        )  # Sort in descending order
        color_order = [
            self.hex_colors[i] for i in counts_dict.keys()
        ]  # Order colors in the same way as counts
        bottom = 0
        for i, color in zip(counts_dict.keys(), color_order):
            height = counts_dict[i] * 100 / sum(counts_dict.values())
            plt.bar(
                x="color",
                height=height,
                color=color,
                bottom=bottom,
            )
            brightness = self.calculate_brightness(color[1:])
            text_color = "white" if brightness < 0.5 else "black"
            plt.text(
                "color",
                bottom + height / 2,
                f"{counts_dict[i]}",
                ha="center",
                va="center",
                color=text_color,
            )
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            bottom += height
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().axes.xaxis.set_visible(False)
        plt.tight_layout()
