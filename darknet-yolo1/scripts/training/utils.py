import cv2
import numpy as np

# Resize the image to have the width of given size 
# preserve the ratios between 2 side 
def resize(img,size): 
  h,w = img.shape[:2]
  ratio = float(h)/w
  dim = (size, int(size*ratio)) # calculate new dimensions (w,h)
  return cv2.resize(img,dim)


# Method to visualize the title box by drawing the bounding box of the title and its center onto given image
# tBox = ((xCenter/imgW), (yCenter/imgH), (objW/imgW), (objH/imgH))
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


def boundingArea(list):
	""" Methods returns the boundingArea surrounding all given 2D points """
	minX = min([pt[0] for pt in list])
	maxX = max([pt[0] for pt in list])
	minY = min([pt[1] for pt in list])
	maxY = max([pt[1] for pt in list])
	return (minX,maxX,minY,maxY)

	
def formatLabel(titleArea, dim):
	""" Convert from the bounding area of the poster's title (minX,maxX,minY,maxY) 
	to print-out label format (<x>, <y>, <width>, <height>) relative to the poster's dimensions
	
	@param titleArea - (minX,maxX,minY,maxY) of the area containing the tile
	@param dim - the dimensions (H,W) of the whole poster
	@return - (xCenter/imgW), (yCenter/imgH), (objW/imgW), (objH/imgH)
	
	"""
	H,W = dim
	minX,maxX,minY,maxY = titleArea
	
	# Calculate the title's output 
	x = float(maxX + minX) /2 /W
	y = float(maxY + minY) /2 /H
	w = float(maxX - minX + 1) /W
	h = float(maxY - minY + 1) /H	
	
	return (x,y,w,h)
	

def transform2D(pt, mat):
	tmp = np.array([[pt[0]],[pt[1]],[1]])
	ptMat = np.dot(mat, tmp)
	x = int(ptMat[0][0]/ptMat[2][0])
	y = int(ptMat[1][0]/ptMat[2][0])
	return (x,y)