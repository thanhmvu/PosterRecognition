import random


TASK = 'randTrain' # or 'test'
NOTE = 'test'
CLASSES = 2
TOTAL_CLASSES = 100
POSTERS = sorted(random.sample(xrange(TOTAL_CLASSES), CLASSES))
# POSTERS = [2, 8, 11, 13, 16, 23, 24, 27, 28, 39, 45, 51, 64, 68, 78, 82, 88, 89, 90, 91]
NUM_VAR = 3 # number of variation for each ground image
NUM_OF_TEST_IMGS = 5


# CFGFILE_SRC = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/cfg-yolo2/yolo2_%dc.cfg" % (CFG.CLASSES)
# CFGFILE_DST = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/%dC_%dP_%s/yolo2_%dc.cfg" % (CLASSES,NUM_VAR,NOTE,CLASSES)

SRC_DIR = '/home/vut/PosterRecognition/DeepNet/database/realworld/set2/src/'
DST_DIR = '/home/vut/PosterRecognition/DeepNet/database/realworld/set2/%s/%dC_%dP_%s/' % (TASK,CLASSES,NUM_VAR,NOTE)
SRC_IMG_DIR = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/test/real_images/JPEGImages/"
DST_TEST_DIR = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTest/%dC_%dP_%s/" % (CLASSES,NUM_VAR,NOTE)

IMAGE_DIR = DST_DIR + 'JPEGImages/'
LABEL_DIR = DST_DIR + 'labels/'
BACKUP = DST_DIR + 'backup/'

STD_SIZE = 1000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

