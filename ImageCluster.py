from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import webcolors


class ImageCluster:
    def __init__(self, image_path, n_clusters=5, remove_low_alpha=True):
        # Inizializzazione della classe con il percorso dell'immagine, il numero di cluster e un flag per rimuovere i pixel con bassa trasparenza
        self.image_path = image_path
        self.n_clusters = n_clusters
        self.remove_low_alpha = remove_low_alpha

        # Apertura dell'immagine e conversione in formato RGBA
        self.img = Image.open(self.image_path).convert("RGBA")

        # Conversione dell'immagine in un array numpy e ridimensionamento in un array 2D
        self.img_array = np.array(self.img)
        self.data = self.img_array.reshape(-1, 4)

        # Inizializzazione di variabili che verranno utilizzate in seguito
        self.clustered_img = None
        self.labels = None
        self.center_colors = None
        self.hex_colors = None
        self.rgb_colors = None
        self.unique_clusters = None
        self.counts = None
        self.total_pixels = None
        self.cluster_percentages = None
        self.clusters_percentages_colors = None
        self.counts_dict = None
        self.ordered_colors = None
        self.color_names = None

        self.ordered_clusters = None
        self.ordered_percentages = None
        self.ordered_colors = None
        self.ordered_counts = None

    def filter_alpha(self):
        if self.remove_low_alpha:
            mask = self.data[:, 3] >= 125
            self.data = self.data[mask]
        else:
            mask = np.ones(self.data.shape[0], dtype=bool)
        return mask

    def cluster(self):
        # Applicazione del filtro alpha prima di eseguire il clustering
        mask = self.filter_alpha()

        # Esecuzione dell'algoritmo K-Means sull'array 2D
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.labels = kmeans.fit_predict(self.data)

        # Salvataggio dei colori dei centri dei cluster
        self.center_colors = kmeans.cluster_centers_

        # Sostituzione dell'immagine originale con l'immagine clusterizzata
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

        # Calcolo del numero di pixel in ogni cluster e della percentuale di pixel che ogni cluster rappresenta nell'immagine totale
        self.unique_clusters, self.counts = np.unique(self.labels, return_counts=True)
        self.counts_dict = dict(zip(self.unique_clusters, self.counts))
        self.ordered_colors = [self.center_colors[i] for i in self.counts_dict.keys()]
        self.total_pixels = np.prod(self.img_array.shape[:2])
        self.cluster_percentages = (self.counts / len(self.data)) * 100  # Percentages
        # Creazione di una lista di tuple, dove ogni tupla contiene il cluster, la percentuale, il conteggio e il colore del centro del cluster
        self.clusters_percentages_counts_colors = list(
            zip(
                self.unique_clusters,
                self.cluster_percentages,
                self.counts,
                self.center_colors,
            )
        )
        # Ordinamento della lista in base alla percentuale, in ordine decrescente
        self.clusters_percentages_counts_colors.sort(key=lambda x: x[1], reverse=True)
        # Separazione dei cluster, delle percentuali, dei conteggi e dei colori in quattro liste separate
        (
            self.ordered_clusters,
            self.ordered_percentages,
            self.ordered_counts,
            self.ordered_colors,
        ) = zip(*self.clusters_percentages_counts_colors)

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def rgba_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}{:02x}".format(
            int(color[0]), int(color[1]), int(color[2]), int(color[3])
        )

    def hex_to_rgb(self, hex_color):
        # Rimuove il carattere '#' se presente
        hex_color = hex_color.lstrip("#")
        # Converte l'esadecimale in RGB
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def HEX_COLORS(self):
        self.hex_colors = [
            self.rgba_to_hex(self.ordered_colors[i]) for i in self.counts_dict.keys()
        ]

    def RGB_COLORS(self):
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
        # Converti il colore da hex a RGB prima
        # color = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
        # Calcola la luminosità come la media dei valori RGB
        return sum(color) / (3 * 255)

    def plot_original_image(self):
        plt.imshow(self.img_array)
        plt.title("Original Image")
        plt.axis("off")
        plt.tight_layout()

    def plot_clustered_image(self):
        plt.imshow(self.clustered_img)
        plt.title("Clustered Image ({} clusters)".format(self.n_clusters))
        plt.axis("off")
        plt.tight_layout()

    def plot_clustered_image_grid(self):
        # Crea una griglia vuota con le stesse dimensioni dell'immagine clusterizzata
        grid = np.zeros((self.img_array.shape[0], self.img_array.shape[1]))

        # Crea un array di etichette con la stessa forma dell'immagine originale
        labels_full = np.full((self.img_array.shape[0] * self.img_array.shape[1]), -1)

        # Crea un indice numerico che corrisponda ai pixel con un valore alfa alto in labels_full
        if self.remove_low_alpha:
            alpha_high_index = np.where(self.img_array.reshape(-1, 4)[:, 3] >= 125)[0]
        else:
            alpha_high_index = np.where(self.img_array.reshape(-1, 4)[:, 3])[0]

        # Riempie l'array di etichette con le etichette dei cluster solo per i pixel con un valore alfa alto
        labels_full[alpha_high_index] = self.labels

        # Ridimensiona l'array di etichette per corrispondere alla forma dell'immagine originale
        labels_full = labels_full.reshape(
            self.img_array.shape[0], self.img_array.shape[1]
        )

        # Per ogni cluster, riempi le celle corrispondenti della griglia con l'indice del cluster
        for i, cluster in enumerate(self.unique_clusters):
            grid[labels_full == cluster] = i

        # Visualizza la griglia
        plt.imshow(grid, cmap="nipy_spectral")
        plt.title("Clustered Image Grid ({} clusters)".format(self.n_clusters))
        plt.axis("off")
        plt.colorbar(label="Cluster Index")
        plt.tight_layout()

    def plot_images(self):
        plt.subplot(1, 2, 1)
        self.plot_original_image()
        plt.subplot(1, 2, 2)
        self.plot_clustered_image()
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
        # Imposta l'asse x per avere solo numeri interi
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

    def plot_pie_chart(self):
        # Normalizza i colori ordinati
        normalized_colors = [(color / 255).tolist() for color in self.ordered_colors]

        # Crea il grafico a torta
        patches, texts, autotexts = plt.pie(
            self.ordered_counts,
            labels=self.ordered_counts,
            colors=normalized_colors,
            autopct="%1.1f%%",
            startangle=140,
        )
        # Colora i labels in base alla luminosità del colore
        for i in range(len(self.ordered_colors)):
            brightness = self.calculate_brightness(self.ordered_colors[i])
            if brightness < 0.75:
                autotexts[i].set_color("white")

        plt.axis("equal")  # Assicura che il grafico sia disegnato come un cerchio.
        plt.tight_layout()

    def plot_bar_chart(self):
        bottom = 0
        for i, color, percentage in zip(
            self.ordered_clusters, self.ordered_colors, self.ordered_percentages
        ):
            plt.bar(
                x="color",
                height=percentage,
                color=np.array(color) / 255,
                bottom=bottom,
            )
            brightness = self.calculate_brightness(color)
            text_color = "white" if brightness < 0.75 else "black"
            plt.text(
                "color",
                bottom + percentage / 2,
                f"{self.counts_dict[i]}",
                ha="center",
                va="center",
                color=text_color,
            )
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            bottom += percentage
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().axes.xaxis.set_visible(False)
        plt.tight_layout()
