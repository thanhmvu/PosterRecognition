import cv2
import numpy as np
import random
import math

######################### TEST RANDVIEWPOINT ####################

def randViewpoint(C,w,h):
	""" Method that generates a point for the perpsective plane (the normal of this plane would be toward the center C of the poster)
	The point is generates randomly within a hardcoded range, relative to C,w,h. The range is illustrated below.
	
			      |_poster_|           .      a = 45 degree
			    / |        | \         |      zRange = (zC + R/2) +-  R/2
			  / a |        | a \       | R    xRange =     xC     -+ (w/2 + R.sin(a))
			/     |        |     \     |      yRange = (yC - h/2) +-  h/2
			\_____|________|_____/     |	
			       --------
			          w
	
	@param C - the center of the poster
	@param w - the width of the poster
	@param h - the height of the poster
	@return (x,y,z) - the coordinates of the generated point
	
	"""
	(xC,yC,zC) = C # poster's center
	unit = 1 # estimated to be roughly equivalent to 1/2 -> 1 inch in real world unit
	R = random.randrange(int(w*0.75), int(w*1.25), unit)
	
	# Generate x
	xStart = int(xC - w/2 - R/math.sqrt(2))
	xEnd = int(xC + w/2 + R/math.sqrt(2))
	x = random.randrange(xStart,xEnd,unit)
	
	# Generate z
	dx = abs(xC - x) - w/2
	dz = math.sqrt(R*R - dx*dx)
	z = int(zC + dz) if dx > 0 else int(zC + R)
	
	# Generate y
	y = random.randrange(yC, yC + h, unit)
	
	return (x,y,z)

img = np.zeros((600,1000,3), np.uint8)
img2 = np.zeros((600,1000,3), np.uint8)
C = (500,300,0)
h = 200
w = 300

cv2.circle(img,(C[0],C[2]),5,(0,0,255),-1)
cv2.line(img,(C[0]-w/2,C[2]),(C[0]+w/2,C[2]),(0,0,255),3)
cv2.circle(img2,(C[0],C[1]),5,(0,0,255),-1)
cv2.line(img2,(C[0]-w/2,C[1]),(C[0]+w/2,C[1]),(0,0,255),3)

for x in range (0,1000):
	pt = randViewpoint(C,w,h)
	img[pt[2],pt[0]] = (0,255,0)
	img2[pt[1],pt[0]] = (0,255,0)

cv2.imshow("temp",img) ; cv2.waitKey(0)
cv2.imshow("temp2",img2) ; cv2.waitKey(0)
cv2.destroyAllWindows()
