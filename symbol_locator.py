import cv2
import imutils
import os
import sys


def write_contour_images(image, dest, name_prefix=''):
    # image = cv2.resize(image, (1000, 460))
    image = cv2.resize(image, None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    counter = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if 35 < cv2.contourArea(c):
            thresh = img_thresh[y:y + h, x:x + w]
            (tH, tW) = thresh.shape
            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            # determine padding
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            cv2.imwrite('%s/%s%s.jpg' % (dest, name_prefix, counter), padded)
            counter += 1


if __name__ == "__main__":
    data_dir = sys.argv[1]
    for subdir, _, files in os.walk(data_dir):
        counter = 0
        for file in files:
            image = cv2.imread('%s/%s' % (subdir, file))
            write_contour_images(image, subdir, '%s' % counter)
            counter += 1
