import os

"""
Reformat the name of images:
	0.jpg		-> 000.jpg
	1.jpg 	-> 001.jpg
	...
	399.jpg -> 399.jpg
	
"""

def rename():
	SRC = "../../../deepDB/srcPosters/"
	DST = "../../../deepDB/srcPosters/"
	# read file in this dir
	for i in range (400):
		try:
			file = SRC + `i` + '.jpg'
			newname = DST + `i`.zfill(3) + '.jpg'
			os.rename(file,newname)
			print (file)
		except IOError as e:
			print e

rename()