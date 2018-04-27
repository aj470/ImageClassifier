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


def sobel_edge_detect(image):
    output = []
    gradient = []
    # sobel edge detection
    for y in range(1, image.height() - 1):
        outline = []
        gradline = []
        for x in range(1, image.width() - 1):
            ex = 0
            ey = 0
            # top left
            p = image.get_pixel(x - 1, y - 1)
            ex -= p
            ey -= p
            # bottom left
            p = image.get_pixel(x - 1, y + 1)
            ex -= p
            ey += p
            # top right
            p = image.get_pixel(x + 1, y - 1)
            ex += p
            ey -= p
            # bottom right
            p = image.get_pixel(x + 1, y + 1)
            ex += p
            ey += p
            # left and right
            p = image.get_pixel(x - 1, y)
            ex -= 2 * p
            p = image.get_pixel(x + 1, y)
            ex += 2 * p
            # top and bottom
            p = image.get_pixel(x, y - 1)
            ey -= 2 * p
            p = image.get_pixel(x, y + 1)
            ey += 2 * p

            ee = math.sqrt(ex**2 + ey**2)
            # normalize length
            ee = int(ee / 4328 * 255)
            ed = int(math.degrees(math.atan2(ey, ex)))
            if ed < 0:
                # get the inverse angle
                ed = 180 + ed
            # round to one of 4 directions
            if -22.5 <= ed < 22.5:
                ed = 0
            elif 22.5 <= ed < 67.5:
                ed = 45
            elif 67.5 <= ed < 112.5:
                ed = 90
            elif 112.5 <= ed <= 157.5:
                ed = 135
            else:
                ed = 0

            outline.append(ee)
            gradline.append(ed)
        output.append(outline)
        gradient.append(gradline)

    return output, gradient


def shu_edge_thinning(image):
    sobel, gradient = sobel_edge_detect(image)
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

            line.append(min(en, 255))
        thinned.append(line)
    return Datum(thinned)


def non_maximum_suppression(image):
    sobel, gradient = sobel_edge_detect(image)

    for y in range(1, image.height()-3):
        for x in range(1, image.width()-3):
            direction = gradient[y][x]
            pixel = image.get_pixel(x, y)
            if direction == 0:
                # check left and right
                if pixel > image.get_pixel(x-1, y) and pixel > image.get_pixel(x+1, y):
                    sobel[y][x] = 255
                else:
                    sobel[y][x] = 0
            elif direction == 45:
                # check bottom left and top right
                if pixel > image.get_pixel(x-1, y+1) and pixel > image.get_pixel(x+1, y-1):
                    sobel[y][x] = 255
                else:
                    sobel[y][x] = 0
            elif direction == 90:
                # check up and down
                if pixel > image.get_pixel(x, y+1) and pixel > image.get_pixel(x, y-1):
                    sobel[y][x] = 255
                else:
                    sobel[y][x] = 0
            else:
                # check top left and bottom right
                if pixel > image.get_pixel(x+1, y+1) and pixel > image.get_pixel(x-1, y-1):
                    sobel[y][x] = 255
                else:
                    sobel[y][x] = 0
    return Datum(sobel)
