# -- coding:utf-8 --
from models import c3d_model
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from schedules import onetenth_4_8_12
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import matplotlib
matplotlib.use('AGG')



def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

def process_line(lines,img_path,train=True):
    num = len(lines)
    aa = np.zeros((num,16,128,171,3),dtype='float32')
    bb = np.zeros((num,16,128,171,3),dtype='float32')
    aa_reverse = np.zeros((num,16,128,171,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)
        for j in range(16):
            img = imgs[symbol + j]
            frame = cv2.imread(img_path+path + '/' + img)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(171,128))
            if train:
                aa[i][j][:][:][:] = frame.astype(np.float32)
                rimage = cv2.flip(frame, 1).astype(np.float32)
                aa_reverse[i][j][:][:][:] = rimage
            else:
                image = cv2.resize(frame, (171, 128)).astype(np.float32)
                bb[i][j][:][:][:] = image
        labels[i] = label
    if train:
        return aa,aa_reverse,labels
    else:
        return bb,labels


def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs


def train_data_crop(inputs,inputs_r):
    inputs = inputs[:,:,8:120,30:142,:]
    inputs_r = inputs_r[:,:,8:120,30:142,:]
    return inputs,inputs_r


##一批一批生成训练数据
def generator_train_batch(train_txt,batch_size,num_classes,img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train,x_train_r,x_labels = process_line(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            x_r = preprocess(x_train_r)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x,x_r = train_data_crop(x,x_r)
            yield x, y
            yield x_r, y


##一批一批生成训练数据
def generator_val_batch(val_txt,batch_size,num_classes,img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test,y_labels = process_line(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            test_data = x[:,:, 8:120, 30:142, :]
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield test_data, y


def main():
    img_path = '/home/tianz/datsets/ucfimgs/'
    train_file = 'train_list.txt'
    test_file = 'test_list.txt'
    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()
    f2.close()
    val_samples = len(lines)

    num_classes = 101
    batch_size = 16
    epochs = 16

    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    history = model.fit_generator(generator_train_batch(train_file, batch_size, num_classes,img_path),
                                  steps_per_epoch=train_samples // batch_size *2,
                                  epochs=epochs,
                                  callbacks=[onetenth_4_8_12(lr)],
                                  validation_data=generator_val_batch(test_file,
                                        batch_size,num_classes,img_path),
                                  validation_steps=val_samples // batch_size,
                                  verbose=1)
    if not os.path.exists('results/'):
        os.mkdir('results/')
    plot_history(history, 'results/')
    save_history(history, 'results/')
    model.save_weights('results/weights_c3d.h5')


if __name__ == '__main__':
    main()