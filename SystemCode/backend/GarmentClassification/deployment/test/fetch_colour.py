import numpy as np
import binascii
from PIL import Image
import scipy.cluster
from skimage import color

from .colour_dictionary import colors_dict


def detect_colour(imgpath):
    NUM_CLUSTERS = 2
    im = Image.open(imgpath)
    im = im.crop((92, 92, 160, 160))
    im = im.resize((150, 150))
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)
    counts, bins = scipy.histogram(vecs, len(codes))

    index_max = scipy.argmax(counts)
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')

    peaked_color = colour

    # Get a list of color values in hex string format
    hex_rgb_colors = list(colors_dict.keys())

    r = [int(hex[0:2], 16) for hex in hex_rgb_colors]  # List of red elements.
    g = [int(hex[2:4], 16) for hex in hex_rgb_colors]  # List of green elements.
    b = [int(hex[4:6], 16) for hex in hex_rgb_colors]  # List of blue elements.

    r = np.asarray(r, np.uint8)  # Convert r from list to array (of uint8 elements)
    g = np.asarray(g, np.uint8)  # Convert g from list to array
    b = np.asarray(b, np.uint8)  # Convert b from list to array

    rgb = np.dstack((r, g, b))  # Stack r,g,b across third dimention - create to 3D array (of R,G,B elements).

    # Convert from sRGB color spave to LAB color space
    lab = color.rgb2lab(rgb)

    # Convert peaked color from sRGB color spave to LAB color space
    peaked_rgb = np.asarray([int(peaked_color[1:3], 16), int(peaked_color[3:5], 16), int(peaked_color[5:7], 16)], np.uint8)
    peaked_rgb = np.dstack((peaked_rgb[0], peaked_rgb[1], peaked_rgb[2]))
    peaked_lab = color.rgb2lab(peaked_rgb)

    # Compute Euclidean distance from peaked_lab to each element of lab
    lab_dist = ((lab[:, :, 0] - peaked_lab[:, :, 0]) ** 2 + (lab[:, :, 1] - peaked_lab[:, :, 1]) ** 2 + (
            lab[:, :, 2] - peaked_lab[:, :, 2]) ** 2) ** 0.5

    # Get the index of the minimum distance
    min_index = lab_dist.argmin()

    # Get hex string of the color with the minimum Euclidean distance (minimum distance in LAB color space)
    peaked_closest_hex = hex_rgb_colors[min_index]

    # Get color name from the dictionary
    peaked_color_name = colors_dict[peaked_closest_hex]
    return {'colour': colour, 'peaked_color_name': peaked_color_name}