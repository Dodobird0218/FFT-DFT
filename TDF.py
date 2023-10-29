import cv2
import numpy as np
import os


def img_smoothing(img, kernelsize: list, imgname: str):
    size = len(kernelsize)

    for i in range(size):
        n = kernelsize[i]

        # mean filter
        kernel = np.ones((n, n), dtype=np.float32) / (n * n)
        meanimg = cv2.filter2D(img, -1, kernel)
        filename = "Mean of %s %s^2.jpg" % (imgname, n)
        cv2.imwrite("./test/" + filename, meanimg)

        # Gaussian filter
        gaussian_blur = cv2.GaussianBlur(img, (n, n), 10)
        filename1 = "Gaussian of %s %s^2.jpg" % (imgname, n)
        cv2.imwrite("./test/" + filename1, gaussian_blur)

        # median filter
        median_blur = cv2.medianBlur(img, n)
        filename2 = "Median of %s %s^2.jpg" % (imgname, n)
        cv2.imwrite("./test/" + filename2, median_blur)


def img_edge_detecting(img, kernelsize: list, imgname: str):
    cv2.setUseOptimized(True)

    size = len(kernelsize)
    a = cv2.CV_8U
    b = cv2.CV_16S
    c = [a, b]

    text1 = "CV_8U"
    text2 = "CV_16S"
    t = [text1, text2]
    for i in range(size):
        n = kernelsize[i]

        for j in range(2):
            x = cv2.Sobel(img, c[j], 1, 0, ksize=n)
            y = cv2.Sobel(img, c[j], 0, 1, ksize=n)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

            canny = cv2.Canny(img, c[j], 50, 150, apertureSize=n)

            filename = "Sombel_x of %s %s^2 %s.jpg" % (imgname, n, t[j])
            filename1 = "Sombel_y of %s %s^2 %s.jpg" % (imgname, n, t[j])
            filename2 = "Sombel_x+y of %s %s^2 %s.jpg" % (imgname, n, t[j])
            filename3 = "Canny of %s %s^2 %s.jpg" % (imgname, n, t[j])

            cv2.imwrite("./test/" + filename, absX)
            cv2.imwrite("./test/" + filename1, absY)
            cv2.imwrite("./test/" + filename2, dst)
            cv2.imwrite("./test/" + filename3, canny)


def combine(img, kernelsize: list, imgname: str):
    cv2.setUseOptimized(True)

    for i in range(2):
        n = kernelsize[i]

        # mean filter
        kernel = np.ones((5, 5), dtype=np.float32) / (5 * 5)
        meanimg = cv2.filter2D(img, -1, kernel)
        # Sobel
        x = cv2.Sobel(meanimg, cv2.CV_16S, 1, 0, ksize=n)
        y = cv2.Sobel(meanimg, cv2.CV_16S, 0, 1, ksize=n)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # Canny
        canny = cv2.Canny(meanimg, cv2.CV_16S, 50, 150, apertureSize=n)

        filename = "Mean 5^2  %s x+y %s^2.jpg" % (imgname, n)
        filename1 = "Mean 5^2  %s Canny %s^2.jpg" % (imgname, n)
        cv2.imwrite("./test/" + filename, dst)
        cv2.imwrite("./test/" + filename1, canny)

        # Gaussian
        gaussian_blur = cv2.GaussianBlur(img, (n, n), 10)
        # Sobel
        x1 = cv2.Sobel(gaussian_blur, cv2.CV_16S, 1, 0, ksize=n)
        y1 = cv2.Sobel(gaussian_blur, cv2.CV_16S, 0, 1, ksize=n)
        absX1 = cv2.convertScaleAbs(x1)
        absY1 = cv2.convertScaleAbs(y1)
        dst1 = cv2.addWeighted(absX1, 0.5, absY1, 0.5, 0)
        # Canny
        canny1 = cv2.Canny(gaussian_blur, cv2.CV_16S, 50, 150, apertureSize=n)

        filename2 = "Gaussian 5^2  %s x+y %s^2.jpg" % (imgname, n)
        filename3 = "Gaussian 5^2  %s Canny %s^2.jpg" % (imgname, n)

        cv2.imwrite("./test/" + filename2, dst1)
        cv2.imwrite("./test/" + filename3, canny1)


# if __name__ == "__main__":

#     for i,(filter_func, filter_name) in enumerate(zip(filter_func_ls,filter_name_ls)) 
        
if __name__ == "__main__":
    os.mkdir("test")

    Lenna = cv2.imread("./TDF_picture/Lenna.png", cv2.IMREAD_GRAYSCALE)
    Lamborghini = cv2.imread("./TDF_picture/Lambo.jpeg", cv2.IMREAD_GRAYSCALE)
    Guitar = cv2.imread("./TDF_picture/Guitar.png", cv2.IMREAD_GRAYSCALE)

    size0 = [3, 7, 15]
    size1 = [3, 5, 7]
    size2 = [3, 5]

    img_smoothing(Lenna, size0, "Lenna")
    img_smoothing(Lamborghini, size0, "Lamborghini")
    img_smoothing(Guitar, size0, "Guitar")

    img_edge_detecting(Lenna, size1, "Lenna")
    img_edge_detecting(Lamborghini, size1, "Lamborghini")
    img_edge_detecting(Guitar, size1, "Guitar")

    combine(Lenna, size2, "Lenna")
    combine(Lamborghini, size2, "Lamborghini")
    combine(Guitar, size2, "Guitar")
