import gen_exps
import gen_lines
import gen_lns
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils.np_utils import to_categorical
import savedata
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


N = 600
IMG_RC = 64
WIDTH = 1
SHOW_N = 15
LENGTH_DOT_LINE = 3
SHOW_IMGS = False
SAVE_IMGS = False

path = '/Users/eek/Desktop/Data/'

if SAVE_IMGS:
    exp_more_0_x_train, exp_more_0_y_train = gen_exps.exp_more_0(N, IMG_RC, WIDTH, 0)
    exp_x_train, exp_y_train = gen_exps.exp(N, IMG_RC, WIDTH, 1)
    line_x_train, line_y_train = gen_lines.line(N, IMG_RC, WIDTH, 2)
    dot_line_x_train, dot_line_y_train = gen_lines.dot_line(N, IMG_RC, WIDTH, LENGTH_DOT_LINE, 3)
    ln_more_0_x_train, ln_more_0_y_train = gen_lns.ln_more_0(N, IMG_RC, WIDTH, 4)
    ln_x_train, ln_y_train = gen_lns.ln(N, IMG_RC, WIDTH, 5)
    x = np.vstack((exp_more_0_x_train, exp_x_train, line_x_train, dot_line_x_train, ln_more_0_x_train, ln_x_train))
    y = np.vstack((exp_more_0_y_train, exp_y_train, line_y_train, dot_line_y_train, ln_more_0_y_train, ln_y_train))
    y = y.reshape(x.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    savedata.saveData(path + 'x_train.bin', path + 'y_train.bin', x_train, y_train)
    savedata.saveData(path + 'x_test.bin', path + 'y_test.bin', x_test, y_test)

#загрузка изображений
x_train, y_train = savedata.loadData(path + 'x_train.bin', path + 'y_train.bin')
x_test, y_test = savedata.loadData(path + 'x_test.bin', path + 'y_test.bin')

num_pixels = IMG_RC * IMG_RC




x_train = x_train.reshape(int(x_train.shape[0]/num_pixels), num_pixels).astype('float32') / 255
x_test = x_test.reshape(int(x_test.shape[0]/num_pixels), num_pixels).astype('float32') / 255

if SHOW_IMGS:
    savedata.show_images(x_train.reshape(x_train.shape[0], IMG_RC, IMG_RC), y_train, 60)


#x_train = x_train.reshape(x_train.shape[0], IMG_RC, IMG_RC, 1)
#x_test = x_test.reshape(x_test.shape[0], IMG_RC, IMG_RC, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 6)  # One-hot encode the labels
y_test = to_categorical(y_test, 6)
num_classes = y_test.shape[1]



model = keras.Sequential()
model.add(keras.layers.Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Обучаем сеть
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, verbose=2)


model.evaluate(x_test, y_test, verbose=2)

n = x_test.shape[0]
y_pred = model.predict(x_test)
classes_y_pred = []
classes_y_test = []
for m in y_pred:
    classes_y_pred.append(np.argmax(m))
for m in y_test:
    classes_y_test.append(np.argmax(m))
nNotClassified = 0
i_notClassified = []
for i in range(n):
    if classes_y_pred[i] != classes_y_test[i]:
        nNotClassified += 1
        i_notClassified.append(i)
nClassified = n - nNotClassified
acc = round(100.0 * nClassified / x_test.shape[0], 4)
print(nClassified)
print(nNotClassified)
print(acc)
label_i = 0
img_num = 1
x_test = x_test.reshape(x_test.shape[0], IMG_RC, IMG_RC)
names = []
names.append('Экспонента > 0')
names.append('Экспонента')
names.append('Линия')
names.append('Пунктирная линия')
names.append('Логарифм > 0')
names.append('Логарифм')

for i in range(15):
    plt.subplot(5, 3, i + 1)
    img = x_test[i_notClassified[i]]
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
plt.show()


