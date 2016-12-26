import random


TASK = 'randTrain' # or 'test'
NOTE = 'trial3'
CLASSES = 30
POSTERS = sorted(random.sample(xrange(100), CLASSES))
NUM_VAR = 2000 # number of variation for each ground image


SRC_DIR = '../../../../database/realworld/set2/src/'
DST_DIR = '../../../../database/realworld/set2/%s/%dC_%dP_%s/' % (TASK,CLASSES,NUM_VAR,NOTE)

IMAGE_DIR = DST_DIR + 'JPEGImages/'
LABEL_DIR = DST_DIR + 'labels/'
BACKUP = DST_DIR + 'backup/'

STD_SIZE = 1000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

