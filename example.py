from ImageProcessor import ImageProcessor
from ImageCluster import ImageCluster

# Uso della classe ImageProcessor
processor = ImageProcessor(
    r"C:\Users\Admin\Documents\GitHub\PhotoColorAnalyzer\img\test2.png"
)

blurred_img = processor.gaussian_blur(kernel_size=11)

from PIL import Image
import matplotlib.pyplot as plt


# uso della classe ImageCluster
# Converti l'immagine di OpenCV in un'immagine PIL
img_blur_pil = Image.fromarray(blurred_img)

# Ora puoi passare img_blur_pil alla classe ImageCluster
img_cluster = ImageCluster(img_blur_pil)
img_cluster.remove_transparent()
img_cluster.plot_original_image()
plt.show()

# color=np.array([[48, 43, 38],[121, 115, 98]])
# img_cluster.cluster(initial_clusters=color)
img_cluster.cluster(n_clusters=4, merge_similar=True, threshold=34)

img_cluster.extract_cluster_info()
img_cluster.plot_clustered_image()
plt.show()

img_cluster.plot_clustered_image_high_contrast()
plt.show()

img_cluster.plot_cluster_pie()
plt.show()

# img_cluster.save_plots()

# img_cluster.plot_cluster_bar()
# plt.show()

# img_cluster.plot_cumulative_barchart()
# plt.show()

# np.savetxt("array.csv", img_cluster.labels_img, delimiter=",")
