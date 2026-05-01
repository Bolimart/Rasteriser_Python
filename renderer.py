import numpy as np
from objects import *

# =====[ Utilities ]=====

def swap (x0, x1):
    return x1, x0

def lerp(i0, d0, i1, d1):
    if i0 == i1:
        return [d0]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i0, i1 + 1):
        values.append(round(d))
        d = d + a
    return values

def draw_line(line, color, image: np.array):
    y1, y0 = line.P1[1], line.P0[1]
    x1, x0 = line.P1[0], line.P0[0]
    
    # The line is horizontal-ish
    if abs(x1 - x0) > abs(y1 - y0):
        print("Horizontalish line")
        # Make sure P0 < P1
        if x0 > x1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)
        ys = lerp(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            image[ys[x - x0], x] = color
            
    # The line is vertical-ish
    else:
        print("Verticalish line")
        # Make sure P0 < P1
        if y0 > y1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)
        xs = lerp(y0, x0, y1, x1)
        for y in range(y0, y1 + 1):
            image[y, xs[y - y0]] = color    
    

# =====[ Render ]=====

def render(viewport, width, height, debug=False):
    P1, P2 = Point2D(0, 0), Point2D(10, 39)
    P3, P4 = Point2D(0, 0), Point2D(39, 10)
    image = np.zeros([height, width, 3])

    
    draw_line(Line(P1, P2), np.array([1, 0, 0]), image)
    draw_line(Line(P2, P4), np.array([0, 1, 0]), image)
    draw_line(Line(P3, P4), np.array([0, 0, 1]), image)
    
    return image


if __name__ == '__main__':
    from matplotlib import pyplot

    pyplot.imsave("image.png", render(None, 40, 40))