import cv2
import os
import numpy as np
import glob

import config as CFG
import utils
import posterTrans as pTr

import datetime
from shutil import copyfile
""" ============================================================================================================
This script generates a set of images for training darknet from the ground database of 400 posters.
For each transformation methods, keep in mind that the deep net should use the title of the posters to recognize the whole poster.

The title of a poster is represent using the following parameters:
    <x> <y> <width> <height>
Where x, y, width, and height are relative to the image's width and height, with x and y are the location of the center of the object:
    x = x_center / img_w
    y = y_center / img_h
    with = obj_w / img w
    height = obj_h / img_h
============================================================================================================ """


def transform(img):
# 	# [TITLE ONLY]
#   # standardize the size of imput image. extract ONLY the title of the poster.
# 	h,w = img.shape[:2]
# 	imgT = img[:h/5,:] # crop out the title
# 	imgT = utils.resize(imgT, CFG.STD_SIZE)
# 	h,w = imgT.shape[:2]
# 	title = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)] # coords of the title is that of the img itself
	
  # standardize the size of imput image
	imgT = utils.resize(img, CFG.STD_SIZE)
	imgT = pTr.addOcclusions(imgT)
	h,w = imgT.shape[:2]
	r = CFG.TITLE_RATIO
	title = [(0,0), (w-1,0), (w-1,int(h*r)), (0,int(h*r))]
	
	imgT, title = pTr.lightBlob(imgT, title)
	imgT, title = pTr.blur(imgT, title)
	imgT, title = pTr.scaleAndTranslate(imgT,title)
	imgT, title = pTr.perspective(imgT, title)
	imgT, title = pTr.rotate(imgT,title)
	
	titleArea = utils.boundingArea(title)
	tBox = utils.formatLabel(titleArea, imgT.shape[:2])
	
	return (imgT, tBox)
	
	

def saveData(trainData, imgIdx, objIdx):
	""" Method to save output training data to files.
	trainData = (trainImg, titleBox)
	"""
	# name format: trainIdx_groundIdx_trans_transType.jpg
	img_out = CFG.IMAGE_DIR + `objIdx`.zfill(6) +"_" +`imgIdx`.zfill(6) +'.jpg'
	label_out = CFG.LABEL_DIR + `objIdx`.zfill(6) +"_" +`imgIdx`.zfill(6) +'.txt'
			
	if trainData != None:
		imgT = trainData[0]
		titleBox = trainData[1]

		## Save the training image
# 		imgT = utils.drawTitleBox(imgT,titleBox)
		cv2.imwrite(img_out,imgT)
		print 'Created ' + img_out

		## Save the label file
		f = open(label_out,'w')
		line = `objIdx` + ' ' + `titleBox`.strip('()').replace(',','')
		f.write(line)
		f.close()
		print 'Created ' + label_out
	else:
		print 'Fail to create' + img_out

		
""" ======================================== Begining of main code ======================================== """

start = datetime.datetime.now()

if not os.path.exists(CFG.DST_DIR): os.mkdir(CFG.DST_DIR)
if not os.path.exists(CFG.IMAGE_DIR): os.mkdir(CFG.IMAGE_DIR)
if not os.path.exists(CFG.LABEL_DIR): os.mkdir(CFG.LABEL_DIR)
if not os.path.exists(CFG.BACKUP): os.mkdir(CFG.BACKUP)
if not os.path.exists(CFG.BACKUP+'classify_weights/'): os.mkdir(CFG.BACKUP+'classify_weights/')
if not os.path.exists(CFG.BACKUP+'detect_weights/'): os.mkdir(CFG.BACKUP+'detect_weights/')
if not os.path.exists(CFG.BACKUP+'yolo2_weights/'): os.mkdir(CFG.BACKUP+'yolo2_weights/')


# Loop through all ground images
for objIdx, posterIdx in enumerate(CFG.POSTERS):
	# read an image from the source folder
	path_in = CFG.SRC_DIR + `posterIdx`.zfill(6) + '.jpg'
	img = cv2.imread(path_in)
	
	if img is None: 
		print 'ERROR: Cannot read' + path_in
	else:
		# Loop and create x training variations for each ground image
		for imgIdx in range (0, CFG.NUM_VAR):
			tfOut = transform(img)
			saveData(tfOut, imgIdx, objIdx)

f = open(CFG.DST_DIR + 'data.txt','w')
f.write("CLASSES: " + `CFG.CLASSES` + "\n")
f.write("NUM_VAR: " + `CFG.NUM_VAR` + "\n")
f.write("POSTERS: " + ", ".join(str(e) for e in CFG.POSTERS) + "\n")
f.close()

print "POSTERS: " + ", ".join(str(e) for e in CFG.POSTERS)

# create file for darknet's input
def getImgDir():
  f = open(CFG.DST_DIR + CFG.TASK +'.txt','w')
  for img in sorted(glob.glob(CFG.IMAGE_DIR+"*")):
    f.write(os.path.realpath(img) + '\n')
  f.close()

getImgDir()

print "Finished generating train data"


""" ======================================== Generate test images ======================================== """
NUM_OF_TEST_IMGS = 5
src_img_dir = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/test/real_images/JPEGImages/"
dst_test_dir = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTest/%dC_%s/" % (CFG.CLASSES, CFG.NOTE)

if not os.path.exists(dst_test_dir): os.makedirs(dst_test_dir)
dst_img_dir = dst_test_dir+"JPEGImages/"
if not os.path.exists(dst_img_dir): os.makedirs(dst_img_dir)

file = open(dst_test_dir+"test.txt","w")
for i,posterIdx in enumerate(CFG.POSTERS):
	for imgIdx in range(NUM_OF_TEST_IMGS):
		img_in = src_img_dir+ `posterIdx`.zfill(6)+"_"+ `imgIdx`.zfill(6)+".jpg"
		img_out = dst_img_dir+ `i`.zfill(6)+"_"+ `imgIdx`.zfill(6)+".jpg"
		copyfile(img_in, img_out)
		file.write(img_out+"\n")
		print "FROM: %s to %s" % (`posterIdx`.zfill(6)+"_"+ `imgIdx`.zfill(6),`i`.zfill(6)+"_"+ `imgIdx`.zfill(6))

print "Finished generating test data"


""" ======================================== Prepare to train ======================================== """
copyfile(CFG.CFGFILE_SRC,CFG.CFGFILE_DST)

# train_command = "./darknet -i X poster_detect train ../../database/realworld/set2/randTrain/%dC_%dP_%s/poster_detect_%dc.cfg ../../database/realworld/set2/randTrain/%dC_%dP_%s/randTrain.txt ../../database/realworld/set2/randTrain/%dC_%dP_%s/backup/detect_weights ../../database/extraction.conv.weights" % (CFG.CLASSES,CFG.NUM_VAR,CFG.NOTE,CFG.CLASSES,CFG.CLASSES,CFG.NUM_VAR,CFG.NOTE,CFG.CLASSES,CFG.NUM_VAR,CFG.NOTE)
train_command = "To train: \n=> Adjust params in train_detector method in detector.c;\n=> Run make;\n=> Run command: ./darknet detector train x x x -gpus 0 1 2 3"

# test_command = "./darknet -i X poster_detect valid ../../database/realworld/set2/randTrain/%dC_%dP_%s/poster_detect_%dc.cfg ../../database/realworld/set2/randTrain/%dC_%dP_%s/backup/detect_weights/poster_detect_%dc_%%d.weights ../../database/realworld/set2/randTest/%dC_%s/test.txt -saveImg 0" % (CFG.CLASSES,CFG.NUM_VAR,CFG.NOTE,CFG.CLASSES,CFG.CLASSES,CFG.NUM_VAR,CFG.NOTE,CFG.CLASSES,CFG.CLASSES,CFG.NOTE)
test_command = "To validate: \n=> Adjust params in multivalid method in detector.c;\n=> Run make;\n=> Run command: ./darknet detector multivalid x x x"

print "\n",train_command, "\n"
print test_command, "\n"



end = datetime.datetime.now()
runtime = end - start
print "Minutes, seconds: ", divmod(runtime.days * 86400 + runtime.seconds, 60)
print "Total seconds: ", (runtime).seconds