import os
import glob


def getImgDir():
  file = open("./test.txt",'w')
  for img in sorted(glob.glob("./JPEGImages/*")):
    file.write(os.path.realpath(img) + '\n')
  file.close()

print "Done!"
getImgDir()