import numpy as np
from numba import jit
import time
import cv2
import warnings
import click

warnings.filterwarnings("ignore")


def spent_time(func):
    def wrap(image):
        start = time.time()
        result = func(image)
        ST = time.time() - start
        print(
            f"\033[33;40mYou spend ",
            ST,
            f" Seconds on {func}",
            "\033[0m",
        )
        return result

    return wrap


@jit
def dft1D(X):
    x = len(X)
    result = np.zeros(x, dtype=complex)
    for i in range(x):
        for j in range(x):
            result[i] += X[j] * np.exp(-2j * np.pi * i * j / x)
    return result


@jit
@spent_time
def dft2D(img):
    M, N = img.shape
    dft2D_result = np.zeros((M, N), dtype=complex)
    for m in range(M):
        dft2D_result[m, :] = dft1D(img[m, :])
    for n in range(N):
        dft2D_result[:, n] = dft1D(dft2D_result[:, n])
    return dft2D_result


@jit
def idft1D(X):
    x = len(X)
    result = np.zeros(x, dtype=complex)
    for i in range(x):
        for j in range(x):
            result[i] += X[j] * np.exp(2j * np.pi * i * j / x) / x
    return result


@jit
@spent_time
def idft2D(img):
    M, N = img.shape
    DFT2D_result = np.zeros((M, N), dtype=complex)
    for m in range(M):
        DFT2D_result[m, :] = idft1D(img[m, :])
    for n in range(N):
        DFT2D_result[:, n] = idft1D(DFT2D_result[:, n])
    return DFT2D_result


@jit
def shift1D(x):
    N = len(x)
    mid = N // 2
    result = np.zeros(N, dtype=complex)
    for i in range(mid):
        result[i] = x[i + mid]
    for i in range(mid, N):
        result[i] = x[i - mid]

    return result


@jit
@spent_time
def shift2D(image):
    M, N = image.shape
    shift_image = np.zeros((M, N), dtype=complex)

    for m in range(M):
        shift_image[m, :] = shift1D(image[m, :])

    for n in range(N):
        shift_image[:, n] = shift1D(shift_image[:, n])

    return shift_image


def fast_fft(img):
    input_image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (512, 512))
    cv2.imshow("origin", input_image)
    dft_img = dft2D(input_image)
    cv2.imshow("dft image", np.uint8(dft_img))
    dft_enlarge = 15 * np.log(1 + np.abs(dft_img))
    cv2.imshow("enlarge dft image", np.uint8(dft_enlarge))
    shift_img = shift2D(dft_img)
    cv2.imshow("shift image", np.uint8(shift_img))
    dft_shift_enlarge = 15 * np.log(1 + np.abs(dft_img))
    cv2.imshow("enlarge shift image", np.uint8(dft_shift_enlarge))
    shiftback_img = shift2D(shift_img)
    cv2.imshow("shiftback image", np.uint8(shiftback_img))
    idft_img = idft2D(shiftback_img)
    cv2.imshow("shift image", np.uint8(idft_img))
    cv2.waitKey(0)


def only_fft(img):
    input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (512, 512))
    cv2.imshow("origin", input_image)
    dft_img = dft2D(input_image)
    cv2.imshow("dft image", np.uint8(dft_img))
    dft_enlarge = 15 * np.log(1 + np.abs(dft_img))
    cv2.imshow("enlarge image", np.uint8(dft_enlarge))
    shift_img = shift2D(dft_img)
    cv2.imshow("shift image", np.uint8(shift_img))
    cv2.waitKey(0)


@click.command()
@click.option(
    "-f",
    "--fast",
    default="None",
    type=str,
    help="Fourier transform the image",
)
@click.option(
    "-o",
    "--only",
    default="None",
    type=str,
    help="Only do transform",
)
def CLI(fast, only):
    if (fast == "None") and (only == "None"):
        print("Please enter the image name")
    elif (fast != "None") and (only == "None"):
        fast_fft(fast)
    elif (fast == "None") and (only != "None"):
        only_fft(only)


if __name__ == "__main__":
    CLI()
