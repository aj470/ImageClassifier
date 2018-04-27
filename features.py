import math

from images import Datum

# dimensions in characters
FACE_WIDTH = 60
FACE_HEIGHT = 70

DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28


def no_extract(image):
    # default no extraction
    return image


def basic_feature_extractor(image):
    # just converts gray pixels to black
    newImage = []
    image = image.data()
    for line in image:
        newLine = []
        for pixel in line:
            if pixel > 0.5:
                newLine.append(1)
            else:
                newLine.append(0)
        newImage.append(newLine)
    return Datum(newImage)


def digit_extractor1(image):
    # only makes gray pixels black if they are in a cluster of 4 or more
    newImage = []
    image = image.data()

    for y in range(len(image)):
        newLine = []
        line = image[y]
        for x in range(len(line)):
            pixel = line[x]
            if pixel == 0:
                newLine.append(0)
            elif pixel == 1:
                newLine.append(1)
            elif check_surrounding(image, x, y) >= 4:
                newLine.append(1)
            else:
                newLine.append(0)
        newImage.append(newLine)
    return Datum(newImage)


def check_surrounding(image, x, y, checklist=None):
    total = 1

    if checklist and (x, y) in checklist:
        return 0

    if not checklist:
        checklist = set()

    checklist.add((x, y))

    if x > 0:
        # check left
        if image[y][x-1] == 0.5:
            total += check_surrounding(image, x-1, y, checklist)
    if x < DIGIT_WIDTH-1:
        # check right
        if image[y][x+1] == 0.5:
            total += check_surrounding(image, x+1, y, checklist)
    if y > 0:
        # check upwards
        if image[y-1][x] == 0.5:
            total += check_surrounding(image, x, y-1, checklist)
    if y < DIGIT_HEIGHT-1:
        # check down
        if image[y+1][x] == 0.5:
            total += check_surrounding(image, x, y+1, checklist)
    return total


def shu_edge_thinning(image):
    # sobel edge detection
    sobel = []
    for y in range(image.height()):
        line = []
        for x in range(image.width()):
                ex = image.get_pixel(x+1, y-1) + 2*image.get_pixel(x+1, y) + image.get_pixel(x+1, y+1) \
                    - image.get_pixel(x-1, y-1) - 2*image.get_pixel(x-1, y) - image.get_pixel(x-1, y+1)
                ey = image.get_pixel(x - 1, y + 1) + 2 * image.get_pixel(x, y + 1) + image.get_pixel(x + 1, y + 1) \
                    - image.get_pixel(x - 1, y - 1) - 2 * image.get_pixel(x, y - 1) - image.get_pixel(x + 1, y - 1)
                line.append(math.sqrt(ex**2 + ey**2))
        sobel.append(line)

    sobel = Datum(sobel)

    # thinning using shu-edge thinning
    thinned = []
    for y in range(image.height()):
        line = []
        for x in range(image.width()):
            ex = sobel.get_pixel(x-1, y) + sobel.get_pixel(x, y) + sobel.get_pixel(x+1, y)
            ey = 0
            if sobel.get_pixel(x, y) >= sobel.get_pixel(x, y-1) and sobel.get_pixel(x, y) >= sobel.get_pixel(x, y+1):
                ey = sobel.get_pixel(x, y-1) + sobel.get_pixel(x, y) + sobel.get_pixel(x, y+1)

            en = math.sqrt(ex**2 + ey**2)
            if en < sobel.get_pixel(x-1, y) or sobel.get_pixel(x, y) < sobel.get_pixel(x+1, y):
                ex = 0
                en = math.sqrt(ex**2 + ey**2)

            line.append(min(en, 1))
        thinned.append(line)
    return Datum(thinned)
