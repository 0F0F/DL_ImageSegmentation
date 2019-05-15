import numpy as np
import tensorflow as tf
import cv2
from time import time
import math
import os

from constants import *

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from model import model, lr
from load_data import read_data
from load_data import read_new_data


tf.set_random_seed(RANDOM_SEED)
x_1, x_2, y_, y_320, y, global_step, learning_rate, RB1_upsample2, RB2_upsample2, RB3_upsample2, phase, ya, dr_rate = model()
global_accuracy = 0
epoch_start = 0


with tf.device(DEVICE):
    train_list_all = read_new_data(LOAD_PATH,True, train_test_ration = TRAIN_RATION)
    test_list_all =  read_new_data(LOAD_PATH,False, train_test_ration = TEST_RATION)


#cost and optimizer
label2ch = tf.concat([y_320, 1-y_320], -1)


cost_0 = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=label2ch) #2ch
cost_1 = tf.nn.softmax_cross_entropy_with_logits(logits=RB3_upsample2, labels=label2ch)
cost_2 = tf.nn.softmax_cross_entropy_with_logits(logits=tf.image.resize_images(RB2_upsample2, size=[240, 320]), labels=label2ch)
cost_3 = tf.nn.softmax_cross_entropy_with_logits(logits=tf.image.resize_images(RB1_upsample2, size=[240, 320]), labels=label2ch)

cost = 0.9*(tf.reduce_mean(cost_0))+0.08*(tf.reduce_mean(cost_1))+0.01*(tf.reduce_mean(cost_2))+0.01*(tf.reduce_mean(cost_3))

vars = tf.trainable_variables()
vars_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars)

original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=1.00).minimize(cost+vars_reg)


yabs = tf.abs(y)
test = tf.div(tf.add(y, yabs), 2)
test = tf.round(test)

test_ = tf.nn.relu(y)

#accuracy
correct_prediction = tf.equal(tf.cast(test_, tf.bool), tf.cast(label2ch, tf.bool))
#correct_prediction = tf.equal(tf.round(test), tf.cast(label2ch, tf.bool))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100


# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session() #config=tf.ConfigProto(log_device_placement=True)
train_writer = tf.summary.FileWriter(SAVE_PATH, sess.graph)

sess.run(tf.global_variables_initializer())

def train(epoch):
    global epoch_start
    epoch_start = time()
    batch_size = int(len(train_list_all)/BATCH_SIZE) #25000 / 6
    i_global = 0

    for s in range(batch_size):
        train_x_1, train_x_2, train_y_ = read_data(train_list_all, s)

        start_time = time()
        i_global, _, batch_loss, batch_acc, yy, yyy = sess.run(
            [global_step, optimizer, cost, accuracy, ya, y],
            feed_dict={x_1: train_x_1, x_2:train_x_2, y_: train_y_, learning_rate: lr(epoch), phase: True, dr_rate : 1})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec - lr: {:.8f}"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, BATCH_SIZE / duration, lr(epoch)))
            print("saving training result.. :" + SAVE_PATH + '/train/epoch_' + str(epoch)+'_s_'+str(s) + '.bmp')
            cv2.imwrite(SAVE_PATH + '/train/epoch_' + str(epoch)+'_s_'+str(s) + '.bmp', np.uint8(yy[0,:,:,0] * 255))

    test_and_save(i_global, epoch)

def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start
    batch_size = int(len(test_list_all) / BATCH_SIZE)  # 25000 / 6

    accsum = 0
    for s in range(batch_size):
        test_x_1, test_x_2, train_y_ = read_data(test_list_all, s)
        predicted_class, predicted_class_, acc, right = sess.run(
            [ya, y, accuracy, y_320], #ya는 softmax취한것, y는 output자체
            feed_dict={x_1: test_x_1, x_2: test_x_2, y_: train_y_, learning_rate: lr(epoch), phase: False, dr_rate:1}
        )
        accsum+=acc

    if(epoch==0) :
        cv2.imwrite(SAVE_PATH + '/rightimage' +  '.bmp', np.uint8(right[5, :, :, 0])*255)

    print(predicted_class_[5, :, :, 0])
    testimage = predicted_class[5, :, :, 0]
    testimage[testimage < THRESHOLD] = 0
    testimage[testimage >= THRESHOLD] = 255
    cv2.imwrite(SAVE_PATH +'/save/epoch_'+str(epoch)+'.bmp', np.uint8(testimage))

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}%  - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format((epoch+1), accsum/batch_size, int(hours), int(minutes), seconds))

    if global_accuracy != 0  and global_accuracy < (accsum/batch_size):

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=accsum/batch_size),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format((accsum/batch_size), global_accuracy))
        global_accuracy = (accsum/batch_size)

    elif global_accuracy == 0:
        global_accuracy = (accsum/batch_size)

    print("###########################################################################################################")


def main():
    train_start = time()
    for i in range(EPOCH):
        print("\nEpoch: {}/{}\n".format((i+1), EPOCH))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))


if __name__ == "__main__":
    main()

sess.close()



