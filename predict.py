import numpy as np
import tensorflow as tf
import cv2

from load_data_p import read_data
from model import model


test_x_1, test_x_2, test_y_ = read_data('C:/Users/user/Desktop/Data/predict_data', train=True)

tf.set_random_seed(21)
x_1, x_2, y_, y_320, y, global_step, learning_rate, RB1_upsample2, RB2_upsample2, RB3_upsample2, phase, ya, dr_rate = model()
global_accuracy = 0
epoch_start = 0

# PARAMS
_BATCH_SIZE = 1
_SAVE_PATH = 'C:/Users/user/Desktop/save'

saver = tf.train.Saver()
sess = tf.Session()


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    print(last_chk_path)
    exit(-1)
    last_chk_path = _SAVE_PATH
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def main():
    i = 0
    predicted_class = np.zeros(shape=[len(test_x_1), 240, 320, 2], dtype=np.float32)
    testimage = np.zeros(shape=[240, 320, 3], dtype=np.uint32)
    while i < len(test_x_1):
        j=i+1
        batch_xs_1 = test_x_1[i:j]
        batch_xs_2 = test_x_2[i:j]
        batch_ys_ = test_y_[i:j]
        predicted_class[i:j]= sess.run(
            [ya],
            feed_dict={x_1: batch_xs_1, x_2 : batch_xs_2, y_: batch_ys_, learning_rate: 0, phase: False, dr_rate : 1.0}
        )

        testimage = predicted_class[i, :, :, 0]
        print(testimage)
        print('#')
        testimage[testimage < 0.5] = 0
        testimage[testimage >= 0.5] = 255
        cv2.imwrite('C:/Users/user/Desktop/save/result_' + str(j) + '.bmp', np.uint8(testimage))
        i = j

   #correct_prediction = np.equal(np.cast(np.round(predicted_class), np.uint8), np.cast(test_y_, np.uint8))
 #accuracy = np.mean(np.cast(correct_prediction, tf.float32)) * 100
    print()
    #print("Accuracy on Test-Set: {0:.2f}%)".format(accuracy))



if __name__ == "__main__":
    main()


sess.close()
