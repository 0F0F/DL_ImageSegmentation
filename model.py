import numpy as np
import tensorflow as tf
from time import time
import math

'''
def upsample_2x(a):
    a_shape = tf.shape(a)  # 1x3x3x2 a_shape=[1,3,3,2]

    b = tf.reshape(a, [a_shape[0], a_shape[1], a_shape[2], a_shape[3]])
    c = tf.tile(b, [1, 1, 1, 2])
    d = tf.reshape(c, [a_shape[0], a_shape[1], a_shape[2] * 2, a_shape[3]])
    e = tf.transpose(d, perm=[0, 2, 1, 3])
    f = tf.tile(e, [1, 1, 1, 2])
    h = tf.reshape(f, [a_shape[0], a_shape[1] * 2, a_shape[2] * 2, a_shape[3]])
    i = tf.transpose(h, perm=[0, 2, 1, 3])
    return i

'''


image_width = 320
image_height = 240
channel = 3
def model():
    x_1 = tf.placeholder(tf.float32, shape=[None, image_height, image_width, channel])  # current frame data[0]
    x_2 = tf.placeholder(tf.float32, shape=[None, image_height, image_width, channel])  # reference frame data[1]
    # 앞에 none은 배치사이즈(이미지갯수, 모르니까 none), 뒤에꺼는 데이터사이즈
    y_ = tf.placeholder(tf.float32, shape=[None, 240, 320])
    y__= tf.expand_dims(y_,3)
    y_320 = tf.image.resize_bilinear(y__, size=[240, 320])

    # dr_rate = tf.placeholder("float")

    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    # Weight
    W_hidden1 = tf.get_variable("W_hidden1", shape=[3, 3, 3, 16],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden1_1 = tf.get_variable("W_hidden1_1", shape=[5, 5, 3, 16],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden1_2 = tf.get_variable("W_hidden1_2", shape=[9, 9, 3, 16],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden2 = tf.get_variable("W_hidden2", shape=[3, 3, 16, 32],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden3 = tf.get_variable("W_hidden3", shape=[3, 3, 32, 32],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden3_1 = tf.get_variable("W_hidden3_1", shape=[5, 5, 16, 32],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden4_0 = tf.get_variable("W_hidden4_0", shape=[3, 3, 64, 32],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden4_1 = tf.get_variable("W_hidden4_1", shape=[5, 5, 32, 32],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden4_2 = tf.get_variable("W_hidden4_2", shape=[9, 9, 16, 32],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W_hidden5 = tf.get_variable("W_hidden5", shape=[1, 1, 96, 32],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    # Current frame
    conv_1 = tf.nn.conv2d(x_1, W_hidden1, strides=[1, 1, 1, 1], padding='SAME')
    #conv_1 = tf.nn.dropout(conv_1, dr_rate)  # dropout
    conv_1_1 = tf.nn.conv2d(x_1, W_hidden1_1, strides=[1, 1, 1, 1], padding='SAME')
    #conv_1_1 = tf.nn.dropout(conv_1_1, dr_rate)  # dropout
    conv_1_2 = tf.nn.conv2d(x_1, W_hidden1_2, strides=[1, 1, 1, 1], padding='SAME')
    #conv_1_2 = tf.nn.dropout(conv_1_2, dr_rate)  # dropout

    conv_2 = tf.nn.conv2d(conv_1, W_hidden2, strides=[1, 2, 2, 1], padding='SAME')
    #conv_2 = tf.nn.dropout(conv_2, dr_rate)  # dropout

    conv_3_0 = tf.nn.conv2d(conv_2, W_hidden3, strides=[1, 2, 2, 1], padding='SAME')
    #conv_3_0 = tf.nn.dropout(conv_3_0, dr_rate)  # dropout
    conv_3_1 = tf.nn.conv2d(conv_1_1, W_hidden3_1, strides=[1, 4, 4, 1], padding='SAME')
    #conv_3_1 = tf.nn.dropout(conv_3_1, dr_rate)  # dropout
    conv_3 = tf.concat([conv_3_0, conv_3_1], -1)

    conv_4_0 = tf.nn.conv2d(conv_3, W_hidden4_0, strides=[1, 2, 2, 1], padding='SAME')
    #conv_4_0 = tf.nn.dropout(conv_4_0, dr_rate)  # dropout
    conv_4_1 = tf.nn.conv2d(conv_3_1, W_hidden4_1, strides=[1, 2, 2, 1], padding='SAME')
    #conv_4_1 = tf.nn.dropout(conv_4_1, dr_rate)  # dropout
    conv_4_2 = tf.nn.conv2d(conv_1_2, W_hidden4_2, strides=[1, 8, 8, 1], padding='SAME')
    #conv_4_2 = tf.nn.dropout(conv_4_2, dr_rate)  # dropout
    conv_4 = tf.concat([conv_4_0, conv_4_1, conv_4_2], -1)

    conv_5 = tf.nn.conv2d(conv_4, W_hidden5, strides=[1, 1, 1, 1], padding='SAME')
    #conv_5 = tf.nn.dropout(conv_5, dr_rate)  # dropout

    # reference frame
    r_conv_1 = tf.nn.conv2d(x_2, W_hidden1, strides=[1, 1, 1, 1], padding='SAME')
    #r_conv_1 = tf.nn.dropout(r_conv_1, dr_rate)  # dropout
    r_conv_1_1 = tf.nn.conv2d(x_2, W_hidden1_1, strides=[1, 1, 1, 1], padding='SAME')
    #r_conv_1_1 = tf.nn.dropout(r_conv_1_1, dr_rate)  # dropout
    r_conv_1_2 = tf.nn.conv2d(x_2, W_hidden1_2, strides=[1, 1, 1, 1], padding='SAME')
    #r_conv_1_2 = tf.nn.dropout(r_conv_1_2, dr_rate)  # dropout

    r_conv_2 = tf.nn.conv2d(r_conv_1, W_hidden2, strides=[1, 2, 2, 1], padding='SAME')
    #r_conv_2 = tf.nn.dropout(r_conv_2, dr_rate)  # dropout

    r_conv_3_0 = tf.nn.conv2d(r_conv_2, W_hidden3, strides=[1, 2, 2, 1], padding='SAME')
    #r_conv_3_0 = tf.nn.dropout(r_conv_3_0, dr_rate)  # dropout
    r_conv_3_1 = tf.nn.conv2d(r_conv_1_1, W_hidden3_1, strides=[1, 4, 4, 1], padding='SAME')
    #r_conv_3_1 = tf.nn.dropout(r_conv_3_1, dr_rate)  # dropout
    r_conv_3 = tf.concat([r_conv_3_0, r_conv_3_1], -1)

    r_conv_4_0 = tf.nn.conv2d(r_conv_3, W_hidden4_0, strides=[1, 2, 2, 1], padding='SAME')
    #r_conv_4_0 = tf.nn.dropout(r_conv_4_0, dr_rate)  # dropout
    r_conv_4_1 = tf.nn.conv2d(r_conv_3_1, W_hidden4_1, strides=[1, 2, 2, 1], padding='SAME')
    #r_conv_4_1 = tf.nn.dropout(r_conv_4_1, dr_rate)  # dropout
    r_conv_4_2 = tf.nn.conv2d(r_conv_1_2, W_hidden4_2, strides=[1, 8, 8, 1], padding='SAME')
    #r_conv_4_2 = tf.nn.dropout(r_conv_4_2, dr_rate)  # dropout
    r_conv_4 = tf.concat([r_conv_4_0, r_conv_4_1, r_conv_4_2], -1)

    r_conv_5 = tf.nn.conv2d(r_conv_4, W_hidden5, strides=[1, 1, 1, 1], padding='SAME')
    #r_conv_5 = tf.nn.dropout(r_conv_5, dr_rate)  # dropout

    sub_0 = tf.subtract(r_conv_5, conv_5)
    concat_0 = tf.concat([sub_0, conv_5], -1)

    conv_6 = tf.layers.conv2d(inputs=concat_0, filters=32, kernel_size=[3, 3], padding='SAME',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                              strides=(1, 1), use_bias=False, name='conv_6')
    phase = tf.placeholder(tf.bool, name='phase')
    after_bn = tf.contrib.layers.batch_norm(conv_6, center=True,
                                            is_training=phase)

    encoder_output = tf.nn.relu(after_bn)

    upsampled_0 = tf.image.resize_bilinear(encoder_output, size=[30, 40])

    # RB Weight
    RB_W_hidden1_1 = tf.get_variable("RB_W_hidden1_1", shape=[3, 3, 128, 64],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    RB_W_hidden1_2 = tf.get_variable("RB_W_hidden1_2", shape=[3, 3, 64, 2],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    RB_W_hidden2_1 = tf.get_variable("RB_W_hidden2_1", shape=[3, 3, 130, 64],  # conv3 채널이 64일 경우
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    RB_W_hidden2_2 = tf.get_variable("RB_W_hidden2_2", shape=[3, 3, 64, 2],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    RB_W_hidden3_1 = tf.get_variable("RB_W_hidden3_1", shape=[3, 3, 98, 64],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    RB_W_hidden3_2 = tf.get_variable("RB_W_hidden3_2", shape=[3, 3, 64, 2],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    RB_W_hidden4_1 = tf.get_variable("RB_W_hidden4_1", shape=[3, 3, 114, 64],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    RB_W_hidden4_2 = tf.get_variable("RB_W_hidden4_2", shape=[3, 3, 64, 2],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    # RB1 conv
    RB1_concat = tf.concat([upsampled_0, conv_4], -1)
    RB1_conv1 = tf.nn.conv2d(RB1_concat, RB_W_hidden1_1, strides=[1, 1, 1, 1], padding='SAME')
    #RB1_conv1 = tf.nn.dropout(RB1_conv1, dr_rate)  # dropout
    RB1_batch_norm = tf.contrib.layers.batch_norm(RB1_conv1, center=True, is_training=phase)
    RB1_conv2 = tf.nn.conv2d(RB1_batch_norm, RB_W_hidden1_2, strides=[1, 1, 1, 1], padding='SAME')
    #RB1_conv2 = tf.nn.dropout(RB1_conv2, dr_rate)  # dropout
    RB1_upsample1 = tf.image.resize_images(RB1_batch_norm, size=[60, 80])
    RB1_upsample2 = tf.image.resize_images(RB1_conv2, size=[60, 80])

    # RB2 conv
    RB2_concat = tf.concat([RB1_upsample1, RB1_upsample2, conv_3], -1)
    RB2_conv1 = tf.nn.conv2d(RB2_concat, RB_W_hidden2_1, strides=[1, 1, 1, 1], padding='SAME')
    #RB2_conv1 = tf.nn.dropout(RB2_conv1, dr_rate)  # dropout
    RB2_batch_norm = tf.contrib.layers.batch_norm(RB2_conv1, center=True, is_training=phase)
    RB2_conv2 = tf.nn.conv2d(RB2_batch_norm, RB_W_hidden2_2, strides=[1, 1, 1, 1], padding='SAME')
    #RB2_conv2 = tf.nn.dropout(RB2_conv2, dr_rate)  # dropout
    RB2_upsample1 = tf.image.resize_images(RB2_batch_norm, size=[120, 160])
    RB2_upsample2 = tf.image.resize_images(RB2_conv2, size=[120, 160])

    # RB3 conv
    RB3_concat = tf.concat([RB2_upsample1, RB2_upsample2, conv_2], -1)
    RB3_conv1 = tf.nn.conv2d(RB3_concat, RB_W_hidden3_1, strides=[1, 1, 1, 1], padding='SAME')
    #RB3_conv1 = tf.nn.dropout(RB3_conv1, dr_rate)  # dropout
    RB3_batch_norm = tf.contrib.layers.batch_norm(RB3_conv1, center=True, is_training=phase)
    RB3_conv2 = tf.nn.conv2d(RB3_batch_norm, RB_W_hidden3_2, strides=[1, 1, 1, 1], padding='SAME')
    #RB3_conv2 = tf.nn.dropout(RB3_conv2, dr_rate)  # dropout
    RB3_upsample1 = tf.image.resize_images(RB3_batch_norm, size=[240, 320])
    RB3_upsample2 = tf.image.resize_images(RB3_conv2, size=[240, 320])

    # RB4 conv
    RB4_concat = tf.concat([RB3_upsample1, RB3_upsample2, conv_1, conv_1_1, conv_1_2], -1)
    RB4_conv1 = tf.nn.conv2d(RB4_concat, RB_W_hidden4_1, strides=[1, 1, 1, 1], padding='SAME')
    #RB4_conv1 = tf.nn.dropout(RB4_conv1, dr_rate)  # dropout
    RB4_batch_norm = tf.contrib.layers.batch_norm(RB4_conv1, center=True, is_training=phase)
    y = tf.nn.conv2d(RB4_batch_norm, RB_W_hidden4_2, strides=[1, 1, 1, 1], padding='SAME')
    ya = tf.nn.softmax(y, -1)


    return x_1, x_2, y, global_step, learning_rate, RB1_upsample2, RB2_upsample2, RB3_upsample2, phase, ya #x_1, x_2, y_ 는 인풋이미지, y는 아웃풋 이미지

def lr(epoch):
    learning_rate = 0.0001
    reduce_factor = 0.8
    reduce_counter = 5
    if epoch < 100 :
        learning_rate = learning_rate * pow(reduce_factor, np.uint8(epoch/reduce_counter)) #10 or 5

    return learning_rate


'''
    #  RB1
    RB1_concat = tf.concat([upsampled_0, conv_4], -1)

    RB1_conv1 = tf.layers.conv2d(inputs=RB1_concat, filters=64, kernel_size=[3, 3], padding='SAME',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                              strides=(1, 1), use_bias=False, name='RB1_conv1')
    RB1_batch_norm = tf.contrib.layers.batch_norm(RB1_conv1, center=True, is_training=phase)
    RB1_conv2 = tf.layers.conv2d(inputs=RB1_batch_norm, filters=2, kernel_size=[3, 3], padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 strides=(1, 1), use_bias=False, name='RB1_conv2')
    RB1_upsample1 = tf.image.resize_bilinear(RB1_batch_norm, size=[60, 80])
    RB1_upsample2 = tf.image.resize_bilinear(RB1_conv2, size=[60, 80])

    #  RB2
    RB2_concat = tf.concat([RB1_upsample1, RB1_upsample2, conv_3], -1)
    RB2_conv1 = tf.layers.conv2d(inputs=RB2_concat, filters=64, kernel_size=[3, 3], padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 strides=(1, 1), use_bias=False, name='RB2_conv1')
    RB2_batch_norm = tf.contrib.layers.batch_norm(RB2_conv1, center=True, is_training=phase)
    RB2_conv2 = tf.layers.conv2d(inputs=RB2_batch_norm, filters=2, kernel_size=[3, 3], padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 strides=(1, 1), use_bias=False, name='RB2_conv2')
    RB2_upsample1 = tf.image.resize_bilinear(RB2_batch_norm, size=[120, 160])
    RB2_upsample2 =  tf.image.resize_bilinear(RB2_conv2, size=[120, 160])

    #  RB3
    RB3_concat = tf.concat([RB2_upsample1, RB2_upsample2, conv_2], -1)
    RB3_conv1 = tf.layers.conv2d(inputs=RB3_concat, filters=64, kernel_size=[3, 3], padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 strides=(1, 1), use_bias=False, name='RB3_conv1')
    RB3_batch_norm = tf.contrib.layers.batch_norm(RB3_conv1, center=True, is_training=phase)
    RB3_conv2 = tf.layers.conv2d(inputs=RB3_batch_norm, filters=2, kernel_size=[3, 3], padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 strides=(1, 1), use_bias=False, name='RB3_conv2')
    #RB3_upsample1 = tf.image.resize_images(RB3_batch_norm, size=[240, 320])
    #RB3_upsample2 = tf.image.resize_images(RB3_conv2, size=[240, 320])
    RB3_upsample1 = tf.image.resize_bilinear(RB3_batch_norm, size=[240, 320])
    RB3_upsample2 = tf.image.resize_bilinear(RB3_conv2, size=[240, 320])

    #  RB4
    RB4_concat = tf.concat([RB3_upsample1, RB3_upsample2, conv_1], -1)
    RB4_conv1 = tf.layers.conv2d(inputs=RB4_concat, filters=64, kernel_size=[3, 3], padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 strides=(1, 1), use_bias=False, name='RB4_conv1')
    RB4_batch_norm = tf.contrib.layers.batch_norm(RB4_conv1, center=True, is_training=phase)
    y= tf.layers.conv2d(inputs=RB4_batch_norm, filters=2, kernel_size=[3, 3], padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 strides=(1, 1), use_bias=False, name='y')
'''