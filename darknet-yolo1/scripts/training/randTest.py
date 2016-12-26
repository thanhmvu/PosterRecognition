from shutil import copyfile
import os


RUN = "10c1"
CLASSES = [8, 14, 19, 24, 59, 66, 83, 85, 89, 92]


CURRENT_DIR = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/test/rand/"
src_img_dir = "../real_images/JPEGImages/"
NUM_OF_TEST_IMGS = 5
dir = RUN+"/"
if not os.path.exists(dir): os.makedirs(dir)
dst_img_dir = dir+"JPEGImages/"
if not os.path.exists(dst_img_dir): os.makedirs(dst_img_dir)

file = open(dir+"test_"+RUN+".txt","w")
for i,posterIdx in enumerate(CLASSES):
  for imgIdx in range(NUM_OF_TEST_IMGS):
    img_in = src_img_dir+ `posterIdx`.zfill(6)+"_"+ `imgIdx`.zfill(6)+".jpg"
    img_out = dst_img_dir+ `i`.zfill(6)+"_"+ `imgIdx`.zfill(6)+".jpg"
    copyfile(img_in, img_out)
    file.write(CURRENT_DIR+img_out+"\n")
    print "FROM: %s to %s" % (img_in,img_out)
    

print "Finished"