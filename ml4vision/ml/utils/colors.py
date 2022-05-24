import numpy as np

def get_colors(n=256, float=False):
    colors = []

    for i in range(n):
        r, g, b = 0, 0, 0

        for j in range(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        colors.append([r, g, b])

    colors = np.array(colors)
    if float:
        colors = colors / 255

    return colors
    
