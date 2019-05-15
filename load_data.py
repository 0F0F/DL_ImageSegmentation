import cv2
import os
import numpy as np
import tensorflow as tf

_BATCH_SIZE = 6


# this function try to read data folder and construct the list file containt path to image
# output list file have structure LIST = [[current_frame_path, reference_frame_path, groundtruth_frame_path],
#                                         [current_frame_path, reference_frame_path, groundtruth_frame_path],
#                                        ....
#                                        ]
with tf.device('/cpu:0'):
    def read_new_data(path,train = True,train_test_ration = 0.9):
        listFilesPath = []
        list_subFolder = os.listdir(path)
        for name_subfolder in list_subFolder:
            subfolder_path = path + '/' + name_subfolder
            gt_folder_path = subfolder_path + '/' + 'groundtruth'
            input_folder_path = subfolder_path + '/' + 'input'
            BG_folder_path = subfolder_path + '/bg'

            gt_files = [f for f in os.listdir(gt_folder_path) if os.path.isfile(os.path.join(gt_folder_path, f))]
            input_file = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]
            BG_file = [f for f in os.listdir(BG_folder_path) if os.path.isfile(os.path.join(BG_folder_path, f))]
            if train:
                for i in range(0,round(len(input_file)*train_test_ration)):
                    input_image_path = input_folder_path + '/'+input_file[i]
                    gt_image_path = gt_folder_path + '/'+ gt_files[i]
                    bg_image_path = BG_folder_path + '/'+ BG_file[i]
                    temp_path=[input_image_path, bg_image_path, gt_image_path]
                    listFilesPath.append(temp_path)
            else:
                for i in range(round(len(input_file)*train_test_ration),len(input_file)):
                    input_image_path = input_folder_path + '/'+input_file[i]
                    gt_image_path = gt_folder_path + '/'+ gt_files[i]
                    bg_image_path = BG_folder_path + '/'+ BG_file[i]
                    temp_path=[input_image_path, bg_image_path, gt_image_path]
                    listFilesPath.append(temp_path)
        return listFilesPath

    def read_old_data(path,train = False,train_test_ration = 0.9):
        listFilesPath = []
        list_subFolder = os.listdir(path)

        for name_subfolder in list_subFolder:
            subfolder_path = path + '/' + name_subfolder
            gt_folder_path = subfolder_path + '/' + 'groundtruth'
            input_folder_path = subfolder_path + '/' + 'input'
            BG_folder_path = subfolder_path + '/bg'

            #read infor_txt file
            temporalROI_path = subfolder_path + '/temporalROI.txt'
            content = open(temporalROI_path, 'r')
            content = content.readline( )
            data_content = [int(x) for x in content.split( )]

            start_frame = data_content[0]
            if len(data_content) == 3:
                stop_frame = data_content[2]
            else:
                stop_frame = data_content[1]

            number_frames = stop_frame - start_frame
            train_frame_end = start_frame+round(number_frames * train_test_ration)

            # gt_files = [f for f in os.listdir(gt_folder_path) if os.path.isfile(os.path.join(gt_folder_path, f))]
            # input_file = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]
            # BG_file = [f for f in os.listdir(BG_folder_path) if os.path.isfile(os.path.join(BG_folder_path, f))]
            if train:
                for i in range(start_frame, train_frame_end):
                    input_image_path = input_folder_path + '/in{0:06d}.jpg'.format(i + 1)
                    gt_image_path = gt_folder_path + '/gt{0:06d}.png'.format(i + 1)
                    bg_image_path = BG_folder_path + '/bg{0:06d}.jpg'.format(i + 1)
                    temp_path = [input_image_path, bg_image_path, gt_image_path]
                    listFilesPath.append(temp_path)
            else:
                for i in range(train_frame_end + 1, stop_frame):
                    input_image_path = input_folder_path + '/in{0:06d}.jpg'.format(i + 1)
                    gt_image_path = gt_folder_path + '/gt{0:06d}.png'.format(i + 1)
                    bg_image_path = BG_folder_path + '/bg{0:06d}.jpg'.format(i + 1)
                    temp_path = [input_image_path, bg_image_path, gt_image_path]
                    listFilesPath.append(temp_path)


        return listFilesPath

    def read_data(list_all, s):
      #  list_all = read_new_data(path,train)
        current_images = []
        reference_images = []
        groundtruth_images = []

        list = list_all[s*_BATCH_SIZE:min((s+1)*_BATCH_SIZE,len(list_all))]

        for _temp_ in list:

            _current_image_ = cv2.imread(_temp_[0]).astype('float') / 255
            _reference_image_ = cv2.imread(_temp_[1]).astype('float') / 255
            _groundtruth_image_ = cv2.imread(_temp_[2])

            _groundtruth_image_ = _groundtruth_image_[:,:,0]
            _groundtruth_image_[_groundtruth_image_ < 255] = 0
            _groundtruth_image_[_groundtruth_image_ == 255] = 1

            _current_image_ = cv2.resize(_current_image_, dsize=(320, 240))
            _reference_image_ = cv2.resize(_reference_image_, dsize=(320, 240))
            _groundtruth_image_ = cv2.resize(_groundtruth_image_, dsize=(320, 240))



            current_images.append(_current_image_)
            reference_images.append(_reference_image_)
            groundtruth_images.append(_groundtruth_image_)

        return [current_images,reference_images,groundtruth_images]
