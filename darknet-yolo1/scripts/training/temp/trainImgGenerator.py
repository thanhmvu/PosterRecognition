import cv2
import numpy as np
import os
import random
import imgRotation as rot

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

SRC_DIR = '../../../db-deepnet/srcPosters/'
DST_DIR = '../../../db-deepnet/training/'
if not os.path.exists(DST_DIR): os.mkdir(DST_DIR)
LABELS_DIR = '../../../db-deepnet/labels/'
if not os.path.exists(LABELS_DIR): os.mkdir(LABELS_DIR)

NUM_OF_IMG = 2 # the total number is 400 ground images
STD_SIZE = 500 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

PT_RANGE = range(3,31,27) # range(3,31,3)
RT_RANGE = range(4,41,36) # range(4,41,4)
SC_RANGE = range(1,22,6) # range(1,22,1)
# BL_RANGE = range(0,1,1)
# TL_RANGE = range(0,1,1)

""" Method to visualize the title box.
Draws the bounding box of the title and its center onto given image """
def drawTitleBox(img,tBox):
	# Extract the coordinates
	h,w = img.shape[:2]
	x = int(tBox[0] * w) # x-center of the title box
	y = int(tBox[1] * h) # y-center
	wB = int(tBox[2] * w) # the width of the box
	hB = int(tBox[3] * h) # the height
	
	# Calculate the position of top left corner
	x1 = x - wB/2; x2 = x1 + wB
	y1 = y - hB/2; y2 = y1 + hB
	
	# Draw the box and the center
	cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),3) 
	cv2.circle(img,(x,y),10,(0,0,255),3)
	return img


""" This method makes all image to have the same width as a way of standardizing their sizes """
def sizeStandardize(img,std_width): 
  height, width = img.shape[:2]
  ratio = float(height)/width  
  dim = (std_width, (int)(std_width*ratio)) # calculate new dimensions
  res = cv2.resize(img,dim)  
  return res


""" This method simulate the poster's audience as occlusions.
The number of occlusions generated is random but in the range of [0,10].
The width of all occlusions is the same, which is a third of the poster's standard width (STD_SIZE/3) """
def addOcclusions(img):
	h,w = img.shape[:2]
	
	numOfOcc = random.randint(0,10)
	occWidth = STD_SIZE/3
	# (x-start, x-end, x-step). same for ys 
	# The step is to make sure the occlusions are spaced out reasonably
	xs = (0, w-1, occWidth/4) 
	ys = (int(h*TITLE_RATIO), int(h*0.5), int(h*0.1)) 
	
	# Generate occlusions
	for it in range(0,numOfOcc):
		# Generate top point
		x1 = random.randrange(xs[0], xs[1], xs[2])		
		y1 = random.randrange(ys[0], ys[1], ys[2])
		pt1 = (x1,y1)
		# Calculate end point
		pt2 = (x1 + occWidth, h-1)
		# Add occlusion
		cv2.rectangle(img,pt1,pt2,(0,0,0),-1)
	
# 	return img 
	
	
""" Method to save output training data to files.
	trainData = (trainImg, titleBox)
	cnt = trainIdx
	i = groundIdx
	trans = transformation name (perspective, rotation, ...)
	transType = transformation info (ratio, angle, ...)
"""
def saveData(trainData, cnt, i, trans, transType):
	# name format: trainIdx_groundIdx_trans_transType.jpg
	img_out = DST_DIR + `cnt`.zfill(6) +'_'+ `i`.zfill(3) +'_'+ trans +'_'+ transType  +'.jpg'
	label_out = LABELS_DIR + `cnt`.zfill(6) +'.txt'
			
	if trainData != None:
		imgT = trainData[0]
		titleBox = trainData[1]

		## Save the training image
		cv2.imwrite(img_out,imgT)
		print 'Created ' + img_out

		## Save the label file
		f = open(label_out,'w')
		line = `i` + ' ' + `titleBox`.strip('()').replace(',','')
		f.write(line)
		f.close()
		print 'Created ' + label_out			
	else:
		print 'Fail to create' + img_out

	
""" This method generates training images from the ground images using perspective transformation.
The input ratios r1-r4 are used to generate the 4 corners of new image in the following way:
	top-l-point: (0, h * r1)     , top-r-point: (w, h * r2),
	bottom-l-pt: (0, h - h * r2) , bottom-r-pt: (w, h - h * r4) 
"""
def perspectiveTransform (img, (r1,r2,r3,r4)):
	if img is None: 
		print 'ERROR: Input image is None'
		return None
	else:
		h,w = img.shape[:2]

		# the original 4 points (corners) of the image to transform
		src_points = np.float32([ [0,0], [w,0],
		                          [0,h], [w,h] ])
		# the corresponding new 4 location to transform original points to
		y1 = 0 + int(h*r1) ; y2 = 0 + int(h*r2)
		y3 = h - int(h*r3) ; y4 = h - int(h*r4)
		dst_points = np.float32([ [0,y1], [w,y2],
		                          [0,y3], [w,y4] ])
		
		# compute the transform matrix and apply it
		M = cv2.getPerspectiveTransform(src_points,dst_points)
		ptImg = cv2.warpPerspective(img,M,(w,h))
		
		# compute the title box for training detection
		minY = min(y1,y2)
		maxY1 = y1 + int((y3-y1)*TITLE_RATIO)
		maxY2 = y2 + int((y4-y2)*TITLE_RATIO)
		maxY = max(maxY1,maxY2) # the lowest corner of the title area
		
		x = 0.5 # x_center / img_w (since the title's width = the poster's width)
		y = (minY + maxY)*0.5 / ptImg.shape[0] # y_center / img_h
		width = 1 # obj_w / img w (since the title's width = the poster's width)
		height = float(maxY - minY + 1) / ptImg.shape[0] # obj_h / img_h
		
		tBox = (x, y, width, height)
		
# 		ptImg = drawTitleBox(ptImg,tBox)
		return (ptImg,tBox)

""" Help method for perpspectiveTranasform.
Prints out the tuple of perpspective ratios as string """
def ratiosToStr(tupleR):
	out = ''
	for r in tupleR:
		if r == 0: out +="x"
		else: out += `int(r*100)`.zfill(2)
	return out


""" 
The corners of the title:
    TopLeft: 1, TopRight: 2
    BottomL: 3, BottomR:  4
Input c1 - c4 are the 4 corners of the poster image
imgW, imgH are the dimensions of that poster
"""
def getTitleBox(c1,c2,c3,c4,imgW,imgH):
	# Get the title corners
	tc1 = c1 ; tc2 = c2
	tc3 = tuple(c1[i] + int((c3[i] - c1[i]) * TITLE_RATIO) for i in [0,1])
	tc4 = tuple(c2[i] + int((c4[i] - c2[i]) * TITLE_RATIO) for i in [0,1])
	
	# Get bounding box
	minX = min(tc1[0], tc2[0], tc3[0], tc4[0])
	maxX = max(tc1[0], tc2[0], tc3[0], tc4[0])
	minY = min(tc1[1], tc2[1], tc3[1], tc4[1])
	maxY = max(tc1[1], tc2[1], tc3[1], tc4[1])
  
	# Calculate width, height, and center's coordinates
	wTitle = maxX - minX + 1
	hTille = maxY - minY + 1
	xCenter = (maxX + minX) /2
	yCenter = (maxY + minY) /2
	
	# Return the tile box's values relative to the containing image's dimensions
	x = float(xCenter) / imgW
	y = float(yCenter) / imgH
	w = float(wTitle) / imgW
	h = float(hTille) / imgH
	
	return (x, y, w, h)


""" This method generates training images from the ground images using rotation """
def rotate(img, angle):
	if img is None: 
		print 'ERROR: Input image is None'
		return None
	else:
		crop = False
		rotOut = rot.rotate(img,angle,crop)
		rtImg = rotOut[0]
		
		# compute the title box for training detection
		c = rotOut[1] # 4 corners of rotated poster
		h,w = rtImg.shape[:2]
		tBox = getTitleBox(c[0],c[1],c[2],c[3],w,h)
		
# 		rtImg = drawTitleBox(rtImg,tBox)
		return (rtImg,tBox)


""" This method generates training images from the ground images by rescaling them 
scale the poster, not the image. scale down

recalculate the tbox everytime
"""


def scale(img, mult):	
	if img is None: 
		print 'ERROR: Input image is None'
		return None
	else:
		h,w = img.shape[:2]
		dim = ((int)(mult*w), (int)(mult*h)) # calculate new dimensions
		scImg = cv2.resize(img,dim, interpolation = cv2.INTER_LINEAR)
		
		
		return (scImg,tBox)
	
	
""" ======================================== Begining of main code ======================================== """
cnt = 0
for i in range (0,NUM_OF_IMG):
	# read the image from the source folder
	path_in = SRC_DIR + `i`.zfill(3) + '.jpg'
	img = cv2.imread(path_in)
	
	if img is None: 
		print 'ERROR: Cannot read' + path_in
	else:
		img = sizeStandardize(img,STD_SIZE)
		addOcclusions(img)
		
# 		### Perspective Transformation
# 		ptRange = [x*0.01 for x in PT_RANGE]
# 		rightRatios = [(r,0,r,0) for r in ptRange]
# 		leftRatios  = [(0,r,0,r) for r in ptRange]
# 		ratios = rightRatios + leftRatios
		
# 		for rs  in ratios:
# 			# perform transformation
# 			ptOut = perspectiveTransform(img,rs)
# 			# save output data
# 			transType = ratiosToStr(rs)
# 			saveData(ptOut,cnt,i,'ptrans',transType)
# 			# increase image index
# 			cnt += 1

# 		### Rotation
# 		posAngles = [x for x in RT_RANGE]
# 		negAngles = [-x for x in posAngles]
# 		angles = posAngles + negAngles
		
# 		for angle  in angles:
# 			# perform transformation
# 			rtOut = rotate(img,angle)
# 			# save output data
# 			saveData(rtOut,cnt,i,'rotate',`angle`)
# 			# increase image index
# 			cnt += 1
		
		### Scaling
		mults = [round(x*0.1,2) for x in SC_RANGE if x != 10] 
			
		for mult in mults:
			# perform transformation
			scOut = scale(img,mult)
			# save output data
			saveData(scOut,cnt,i,'scale',`mult`)
			# increase image index
			cnt += 1
			
		
# 		### Blurring

# 		### Translation


