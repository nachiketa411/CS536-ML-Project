# import cv2
from PIL._imaging import display
from skimage import io, exposure
from skimage.transform import resize
import numpy as np
from patchify import patchify
from PIL import Image
# import matplotlib.pyplot as plt


def read_image(img_path):
    return io.imread(img_path)


def display_image(image):
    io.imshow(image)
    io.show()


def resize_image(image):
    return resize(image, (128, 128), anti_aliasing=True)


def create_patches(image, square_patch_size):
    np_img = np.asarray(image)
    return patchify(np_img, (square_patch_size, square_patch_size, 3), step=square_patch_size)


def create_horizontal_patch(patches, image_size, patch_size):
    output_patch = np.empty((image_size // patch_size, patch_size, image_size, 3))
    for row in range(patches.shape[0]):
        strip = patches[row]
        patch_hor_list = []
        for col in range(patches.shape[1]):
            patch_hor_list.append(strip[col, 0])
        output_patch[row] = np.hstack(patch_hor_list)
    return output_patch


def create_vertical_patch(patches, image_size, patch_size):
    output_patch = np.empty((image_size // patch_size, image_size, patch_size, 3))
    for col in range(patches.shape[1]):
        strip = patches[:, col]
        patch_vert_list = []
        for row in range(patches.shape[0]):
            patch_vert_list.append(strip[row, 0])
        output_patch[col] = np.vstack(patch_vert_list)
    return output_patch


root_path = "MVTec"
label_path = f"{root_path}/bottle"
train_path = f"{label_path}/train"
images_path = f"{train_path}/good"
img = io.imread(f'{images_path}/000.png')
# path = "MVTec/bottle/train/good/"
# img_name = f'{path}/000.png'
# img = read_image(path)
# print(im)
print('Image Shape          : ', img.shape)
# display_image(img)

resized_img = resize_image(img)
print('Resized Image Shape  : ', resized_img.shape)
# display_image(resized_img)

patch_size = 16
patches = create_patches(resized_img, patch_size)
print('Patch Shape          : ', patches.shape)

patch_img_arr=patches[3, 0, 0]
print('Patch Image Shape    : ', patch_img_arr.shape)
# patch_img=Image.fromarray(patch_img_arr)
# display_image(patch_img_arr)
# display(patch_img)

horizontal_strips = create_horizontal_patch(patches, 128, patch_size)

vertical_strips = create_vertical_patch(patches, 128, patch_size)

# display_image(horizontal_strips[2])
display_image(vertical_strips[2])