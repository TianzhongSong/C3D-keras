# coding=utf8
from models import c3d_model
from keras.optimizers import SGD
import numpy as np
import cv2


def main():
    with open('./ucfTrainTestlist/classInd.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights('./results/weights_c3d.h5', by_name=True)

    # read video
    video = './videos/v_Biking_g05_c02.avi'
    cap = cv2.VideoCapture(video)

    clip = []
    while True:
        ret, frame = cap.read()
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:,:,8:120,30:142,:]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                pred = model.predict(inputs)
                label = np.argmax(pred[0])
                cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                clip.pop(0)
            cv2.imshow('result', frame)
            cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()