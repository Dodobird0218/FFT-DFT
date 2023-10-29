from DFT import dft2D, idft2D, shift2D
import numpy as np
import cv2
import click


def ideal_circle_mask(image_size, radius):
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    mask = np.zeros(image_size)
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance <= radius:
                mask[x, y] = 1
    return mask


def ideal_low_pass(img, D0):
    input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (512, 512))
    cv2.imshow("origin", input_image)
    dft_img = dft2D(input_image)
    shift_img = shift2D(dft_img)
    P, Q = input_image.shape
    mask = ideal_circle_mask([P, Q], D0)
    f = shift_img * mask
    cv2.imshow("f", np.uint8(f))
    ishift_img = shift2D(f)
    iimg = idft2D(ishift_img)
    cv2.imshow("LPF img", np.uint8(iimg))
    cv2.waitKey(0)


def ideal_high_pass(img, D0):
    input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (512, 512))
    cv2.imshow("origin", input_image)
    dft_img = dft2D(input_image)
    shift_img = shift2D(dft_img)
    P, Q = input_image.shape
    mask = ideal_circle_mask([P, Q], D0)
    f = (shift_img) * (1 - mask)
    cv2.imshow("f", np.uint8(f))
    ishift_img = np.fft.ifftshift(f)
    iimg = idft2D(ishift_img)
    cv2.imshow("HPF img", np.uint8(iimg))
    cv2.waitKey(0)


def bandpass_filter(img, D0, D1):
    input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (512, 512))
    cv2.imshow("origin", input_image)
    dft_img = dft2D(input_image)
    shift_img = shift2D(dft_img)
    P, Q = input_image.shape
    mask1 = ideal_circle_mask([P, Q], D0)
    mask2 = ideal_circle_mask([P, Q], D1)
    f = (shift_img * mask1) + ((shift_img) * (1 - mask2))
    cv2.imshow("f", np.uint8(f))
    ishift_img = np.fft.ifftshift(f)
    iimg = idft2D(ishift_img)
    cv2.imshow("HPF img", np.uint8(iimg))
    cv2.waitKey(0)


@click.command()
@click.option(
    "-l",
    "--low",
    is_flag=True,
    help="ideal low-pass filter",
)
@click.option(
    "-h",
    "--high",
    is_flag=True,
    help="ideal high-pass filter",
)
@click.option(
    "-s",
    "--source",
    default="Lenna.png",
    type=str,
    help="image's name",
)
@click.argument("radius1", default=50)
@click.argument("radius2", default=50)
def CLI(low, high, source, radius1, radius2):
    if (low == True) and (high == False):
        ideal_low_pass(source, radius1)
    elif (low == False) and (high == True):
        ideal_high_pass(source, radius1)
    elif (low == True) and (high == True):
        if radius1 >= radius2:
            print("low_pass radius should shorter than high")
        else:
            bandpass_filter(source, radius1, radius2)


if __name__ == "__main__":
    CLI()
