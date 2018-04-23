

class Datum:
    def __init__(self, data):
        self._data = list(map(lambda x: list(map(ascii2float, x)), data))  # convert chars to float values
        self._value = 0

    def data(self):
        return self._data.copy()

    def get_pixel(self, x, y):
        return self._data[y][x]

    def print(self):
        for line in self._data:
            for pixel in line:
                if pixel == 0:
                    print(" ", end="")
                elif pixel == 0.5:
                    print("+", end="")
                else:
                    print("#", end="")
            print("")

    def associated_value(self):
        return self._value

    def set_associated_value(self, value):
        self._value = value


def ascii2float(char):
    if char == ' ':
        return 0
    elif char == '+':
        return 0.5
    elif char == '#':
        return 1
    else:
        raise ValueError("Invalid character in data file")


def load_images(filename, height):
    """returns a list of Datum objects, each being one image in the file"""
    images = []
    lines = []

    with open(filename) as file:
        for line in file:
            lines.append(line.strip())
            if len(lines) == height:
                images.append(Datum(lines))
                lines = []

    if len(lines) > 0:
        print("***WARNING*** in file '%s': possible invalid height for images. %s lines remaining when height is %s." %
              (filename, len(lines), height))

    return images
