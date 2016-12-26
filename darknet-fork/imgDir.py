import os
import glob


def getImgDir():
  file = open("./train.txt",'w')
  for img in sorted(glob.glob("/home/vut/PosterRecognition/DeepNet/darknet/darknet-fork/trainPosRec/images/*.jpg")):
    file.write(img + '\n')
  file.close()

getImgDir()