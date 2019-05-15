import sys

RANDOM_SEED = 21
BATCH_SIZE = 6
EPOCH = 100
PLATFORM = sys.platform

if sys.platform == 'linux':
    SAVE_PATH = '/mnt/ssd1/grad/save'
    LOAD_PATH = '/mnt/ssd1/grad/data'
    DEVICE = '/device:GPU:0'
else:   # windows
    SAVE_PATH = 'C:/Users/user/Desktop/save'
    LOAD_PATH = 'C:/Users/user/Desktop/Data/new_data3'
    DEVICE = '/cpu:0'

DATA_SETS = ['0-22249','0-14606','0-1096']

TRAIN_RATION = 0.9
TEST_RATION = 0.9

THRESHOLD = 0.5