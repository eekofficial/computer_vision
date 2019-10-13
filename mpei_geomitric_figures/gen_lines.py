import  sys
import numpy as np
import matplotlib.pyplot as plt

def line(n, img_rc, width, label):
    imgs = np.zeros((n, img_rc, img_rc), dtype = 'uint8')
    imgs_labels = np.tile(label, n)
    for i in range(n):
        x = 0.
        dx = 0.3
        a = np.random.uniform(0.2, 2)
        b = np.random.uniform(-5, 5)
        while x < img_rc - 1:
            x += dx
            y = a * x + b + np.random.choice([-width, 0, width])
            ix = int(x)
            iy = int(y)
            if iy >= 0 and iy <= img_rc - 1:
                clr = 255
                imgs[i, ix, iy] = clr
    return imgs, imgs_labels

def dot_line(n, img_rc, width, length, label):
    imgs = np.zeros((n, img_rc, img_rc), dtype = 'uint8')
    imgs_labels = np.tile(label, n)
    for i in range(n):
        x = 0.
        dx = 0.3
        delta = 0
        a = np.random.uniform(0.2, 2)
        b = np.random.uniform(-5, 5)
        while x < img_rc - 1:
            x += dx
            delta += dx
            y = a * x + b + np.random.choice([-width, 0, width])
            ix = int(x)
            iy = int(y)
            if iy >= 0 and iy <= img_rc - 1:
                clr = 255
                imgs[i, ix, iy] = clr
                if delta > length:
                    delta = 0
                    x += 3
    for i in imgs_labels:
        i = label
    return imgs, imgs_labels