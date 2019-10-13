import numpy as np
import matplotlib.pyplot as plt
import math

def exp_more_0(n, img_rc, width, label):
    imgs = np.zeros((n, img_rc, img_rc), dtype='uint8')
    imgs_labels = np.tile(label, n)
    for i in range(n):
        a = np.random.uniform(0, 0.15)
        x = 0
        dx = 0.1
        while x <= img_rc - 1:
            x += dx
            y = int(math.exp(a * x) + np.random.choice([-width, 0, width]))
            ix = int(x)
            iy = int(y)
            if iy >= 0 and iy <= img_rc - 1:
                clr = 255
                imgs[i, img_rc - 1 - iy, ix] = clr
    return imgs, imgs_labels

def exp(n, img_rc, width, label):
    imgs = np.zeros((n, img_rc, img_rc), dtype='uint8')
    imgs_labels = np.tile(label, n)
    for i in range(n):
        a = np.random.uniform(-0.15, 0.15)
        x = -int(img_rc/2)
        dx = 0.1
        while x <= img_rc - int(img_rc/2) - 1:
            x += dx
            y = int(math.exp(a * x) + np.random.choice([-width, 0, width]))
            ix = int(x) + int(img_rc/2)
            iy = int(y) + int(img_rc/2)
            if iy >= 0 and iy <= img_rc - 1:
                clr = 255
                imgs[i, img_rc - 1 - iy, ix] = clr
    return imgs, imgs_labels
