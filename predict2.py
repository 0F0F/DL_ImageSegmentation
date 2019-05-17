import numpy as np
import tensorflow as tf
import cv2

cap = cv2.VideoCapture(0)

# from load_data import read_data
from model import model

tf.set_random_seed(21)
x_1, x_2, y, global_step, learning_rate, RB1_upsample2, RB2_upsample2, RB3_upsample2, phase, ya = model()
global_accuracy = 0
epoch_start = 0

# PARAMS
_BATCH_SIZE = 1
_SAVE_PATH = 'C:/Users/user/Desktop/Data/new_data_fulldata-0'
# _SAVE_PATH = 'C:/Users/user/Desktop/save'

saver = tf.train.Saver()
sess = tf.Session()

try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def main():
    test_x_1 = np.zeros(shape=[1, 240, 320, 3], dtype=np.uint8)
    test_x_2 = np.zeros(shape=[1, 240, 320, 3], dtype=np.uint8)
    testimage = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
    predicted_class = np.zeros(shape=[1, 240, 320, 2], dtype=np.int32)
    cnt = True

    while (True):


        ret, img_color = cap.read()
        test_x_1[0, :, :, :] = cv2.resize(img_color, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        #  print(cnt)
        if (cnt == True):
            cnt = False
            test_x_2[0, :, :, :] = cv2.resize(img_color, dsize=(320, 240), interpolation=cv2.INTER_AREA)


        predicted_class[0:1] = sess.run(
            [y],
            feed_dict={x_1: test_x_1, x_2: test_x_2, learning_rate: 0, phase: False}
        )
        # print(np.shape(predicted_class))
        # predicted_class = np.absolute(predicted_class)+predicted_class
        # predicted_class = 255*predicted_class
        predicted_class[predicted_class < 0.05] = 0
        predicted_class[predicted_class >= 0.05] = 255

        testimage[:, :, 0] = predicted_class[0 ,:, :, 0]
        testimage[:, :, 1] = predicted_class[0, :, :, 0]
        testimage[:, :, 2] = predicted_class[0, :, :, 0]

        # testimage[testimage < 1] = 0
        # testimage[testimage >= 1] = test_x_1[testimage >= 1]
        # testimage[testimage >= 1] = 255
        cv2.imshow("original", test_x_1[0])
        cv2.imshow("trans", testimage)
        if cv2.waitKey(1) % 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

sess.close()