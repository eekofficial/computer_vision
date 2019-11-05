import gen_exps
import gen_lines
import gen_lns
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from PIL import Image

def saveData(fn, fn2, x, y):
    with open(fn, 'wb') as write_binary:
        write_binary.writelines(x)
    with open(fn2, 'wb') as write_binary:
        write_binary.writelines(y)

def loadData(fn, fn2):
    with open(fn, 'rb') as read_binary:
        data = np.fromfile(read_binary, dtype = np.uint8)
    with open(fn2, 'rb') as read_binary:
        labels = np.fromfile(read_binary, dtype = np.uint64)
    return data, labels

def show_images(image, label, n):
    label_i = 0
    img_num = 1
    for i in range(6):
        j = 0
        while j < 10:
            #print(i, j, label_i)
            rand_i = np.random.randint(0, label.shape[0])
            if label[rand_i] == label_i:
                plt.subplot(6, 10, img_num)
                print(img_num)
                img = image[rand_i]
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.subplots_adjust(hspace=0.5)
                j += 1
                img_num += 1
        label_i += 1
    plt.show()


