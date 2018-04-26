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
