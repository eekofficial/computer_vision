import gen_exps
import gen_lines
import gen_lns
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from PIL import Image

N = 600
IMG_RC = 64
WIDTH = 1
SHOW_N = 15
LENGTH_DOT_LINE = 3
SHOW_IMGS = True

def show_images(image, n):
    for i in range(n):
        plt.subplot(1, n, i + 1)
        img = image[i]
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(hspace=0.5)
    plt.show()

exp_more_0_x_train, exp_more_0_y_train = gen_exps.exp_more_0(N, IMG_RC, WIDTH, 0)
exp_x_train, exp_y_train = gen_exps.exp(N, IMG_RC, WIDTH, 1)
line_x_train, line_y_train = gen_lines.line(N, IMG_RC, WIDTH, 2)
dot_line_x_train, dot_line_y_train = gen_lines.dot_line(N, IMG_RC, WIDTH, LENGTH_DOT_LINE, 3)
ln_more_0_x_train, ln_more_0_y_train = gen_lns.ln_more_0(N, IMG_RC, WIDTH, 4)
ln_x_train, ln_y_train = gen_lns.ln(N, IMG_RC, WIDTH, 5)


if SHOW_IMGS:
    show_images(exp_more_0_x_train, SHOW_N)
    show_images(exp_x_train, SHOW_N)
    show_images(line_x_train, SHOW_N)
    show_images(dot_line_x_train, SHOW_N)
    show_images(ln_more_0_x_train, SHOW_N)
    show_images(ln_x_train, SHOW_N)

from keras.utils.np_utils import to_categorical
num_pixels = IMG_RC * IMG_RC
x = np.vstack((exp_more_0_x_train, exp_x_train, line_x_train, dot_line_x_train, ln_more_0_x_train, ln_x_train))
y = np.vstack((exp_more_0_y_train, exp_y_train, line_y_train, dot_line_y_train, ln_more_0_y_train, ln_y_train))
y = y.reshape(x.shape[0], )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
y_train = to_categorical(y_train, 6) # One-hot encode the labels
y_test = to_categorical(y_test, 6) # One-hot encode the labels
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') / 255
num_classes = y_test.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=3)

model.evaluate(x_test,  y_test, verbose=2)