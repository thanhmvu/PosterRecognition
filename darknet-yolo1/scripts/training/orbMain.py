import cv2
import os
import numpy as np
import glob

import orbConfig as CFG
import utils
import posterTrans as pTr

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
  # standardize the size of imput image
	imgT = utils.resize(img, CFG.STD_SIZE)
# 	imgT = pTr.addOcclusions(imgT)
	h,w = imgT.shape[:2]
	r = CFG.TITLE_RATIO
	title = [(0,0), (w-1,0), (w-1,int(h*r)), (0,int(h*r))]
	
# 	imgT, title = pTr.scaleAndTranslate(imgT,title)
# 	imgT, title = pTr.perspective(imgT, title)
# 	imgT, title = pTr.rotate(imgT,title)
# 	imgT, title = pTr.blur(imgT, title)
	
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

if not os.path.exists(CFG.DST_DIR): os.mkdir(CFG.DST_DIR)
if not os.path.exists(CFG.IMAGE_DIR): os.mkdir(CFG.IMAGE_DIR)
if not os.path.exists(CFG.LABEL_DIR): os.mkdir(CFG.LABEL_DIR)
if not os.path.exists(CFG.BACKUP): os.mkdir(CFG.BACKUP)
if not os.path.exists(CFG.BACKUP+'classify_weights/'): os.mkdir(CFG.BACKUP+'classify_weights/')
if not os.path.exists(CFG.BACKUP+'detect_weights/'): os.mkdir(CFG.BACKUP+'detect_weights/')


# Loop through all ground images
for objIdx in range (0, CFG.CLASSES):
	# read an image from the source folder
	path_in = CFG.SRC_DIR + `CFG.LABELS[objIdx]`.zfill(6) + '.jpg'
	img = cv2.imread(path_in)
	
	if img is None: 
		print 'ERROR: Cannot read' + path_in
	else:
		# Loop and create x training variations for each ground image
		for imgIdx in range (0, CFG.NUM_VAR):
			tfOut = transform(img)
			saveData(tfOut, imgIdx, objIdx)

f = open(CFG.DST_DIR + 'data.txt','w')
f.write("LABEL_RANGE: " + `CFG.LIB_RANGE` + "\n")
f.write("CLASSES: " + `CFG.CLASSES` + "\n")
f.write("NUM_VAR: " + `CFG.NUM_VAR` + "\n")
f.close()

print CFG.LABELS

# create file for darknet's input
def getImgDir():
  f = open(CFG.DST_DIR + CFG.TASK +'.txt','w')
  for img in sorted(glob.glob(CFG.IMAGE_DIR+"*")):
    f.write(os.path.realpath(img) + '\n')
  f.close()

getImgDir()
print "Done!"
