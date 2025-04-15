import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import cv2

import re

def natural_sort_key(filename):
    """
    Generates a sorting key that handles numbers correctly.
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', filename)]

directory_path = r"/work2/noaa/nos-surge/felicioc/BeringSea/P09/cu_fig_coup_a_b/"
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
files.sort(key=natural_sort_key)

out_dir = r"/work2/noaa/nos-surge/felicioc/BeringSea/P09/fig_coup_a_b_zoom/"

for fidx, ff in enumerate(files):
    img1 = mpimg.imread(f'/work2/noaa/nos-surge/felicioc/BeringSea/P09/cu_fig_coup_a_b/{ff}')
    img2 = mpimg.imread(f'/work2/noaa/nos-surge/felicioc/BeringSea/P09/hs_fig_coup_a_b/{ff}')
    img3 = mpimg.imread(f'/work2/noaa/nos-surge/felicioc/BeringSea/P09/cu_fig_coup_a_b_zoom/{ff}')
    img4 = mpimg.imread(f'/work2/noaa/nos-surge/felicioc/BeringSea/P09/hs_fig_coup_a_b_zoom/{ff}')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Display images in each subplot
    axes[0, 0].imshow(img1)
    axes[0, 0].axis('off')  # Turn off axis

    axes[0, 1].imshow(img2)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(img3)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img4)
    axes[1, 1].axis('off')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Decrease spacing between subplots
    plt.tight_layout(rect=[0.01, 0.00, 1, 1],pad=0.0)

    plt.savefig(out_dir+f'{ff}', dpi=150)
    plt.close(fig)



images = [img for img in os.listdir(out_dir) if img.endswith(".jpeg")]
images.sort(key=natural_sort_key)

if not images:
    print("No images found in the folder.")

frame = cv2.imread(os.path.join(out_dir, images[0]))
height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("movie_R09coup.mp4", fourcc, 20, (width, height))

for image in images:
    frame = cv2.imread(os.path.join(out_dir, image))
    video.write(frame)

video.release()
print("Video created successfully!")