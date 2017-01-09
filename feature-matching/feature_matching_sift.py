import numpy as np

# order=[0,8,2,3,4,5,6,1,7,9,10,11]
# order=[11, 10, 9, 7, 1, 6, 5, 4, 3, 2, 8, 0]
# import sys
# sys.path=[sys.path[i] for i in order]
import cv2

from matplotlib import pyplot as plt
import glob
import os
from collections import Counter
import datetime
import random

now = datetime.datetime.now()
time = now.strftime("%Y-%m-%dT%H:%M:%S")

train_path  = '/home/vut/PosterRecognition/DeepNet/database/realworld/set2/src/'
query_path  = '/home/vut/PosterRecognition/DeepNet/database/realworld/set2/test/real_images/JPEGImages/'
output_path = '/home/vut/PosterRecognition/DeepNet/database/featureMatching/results/sift/sift_' + time + '/'

# train_var = 1
test_var = 5
title_ratio = 0.2
CLASSES = 10
TOTAL_CLASSES = 100
POSTERS = sorted(random.sample(xrange(TOTAL_CLASSES), CLASSES))

std_width = 500 # last max size was 300
number_of_kp = 500
number_of_out_imgs = 10 # ? is this used anywhere?
colorDescriptor = False
isFiltered = False
bsb_thres = 5 # threshold for Best Second Best filter

train_lib = [] # list of [file_name, keypoints, descriptors, image]
des_lib = None # a matrix of all descriptors in the training library
des_dict = [] # a look-up table for descriptors in des_lib: des_lib[des_index] = [img_index, img_des_index]

detector = cv2.xfeatures2d.SIFT_create()  # Initiate ORB detector
bf = cv2.BFMatcher() 
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # Create BFMatcher object

######################################## Methods #############################################
##############################################################################################

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    Implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = des_dict[mat.trainIdx][1]

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour green
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (0, 255, 0))   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (0, 255, 0))

        # Draw a line in between the two points
        # thickness = 1
        # colour green
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)


    # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
  
### Define the resizing method 
def sizeStandardize(img,std_width):  
  height, width = img.shape[:2]
  ratio = float(height)/width  
  dim = (std_width, (int)(std_width*ratio)) # calculate new dimensions
  res = cv2.resize(img,dim)  
  return res


def saveOutputImages(img1, kp1, img2, kp2, matches, outPath):
  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)
  
  # Draw first 10 matches.
  img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])
#   fig, ax = plt.subplots(figsize=(18, 2))
  #plt.figure(figsize=(15,15))
#   plt.imshow(img3), plt.show()

  # Save image
  cv2.imwrite(outPath,img3)
  
  # Return the saved image
  return img3
  
  
def detectAndCompute(file, withColor):
  ''' Detect the keypoints and compute the descriptors for a given image '''
  
  img = cv2.imread(file,0) # 0 = grayscale  
  if img is None: 
    print 'ERROR: Cannot read' + file
    return None
  
  elif not withColor: # grayscale descriptors
    imgS = sizeStandardize(img,std_width) # Resize input image
    
    # crop out the title of the poster
    h,w = imgS.shape[:2] 
    imgS = imgS[:int(h*title_ratio),:]
    
    kp, des = detector.detectAndCompute(imgS,None) 
    
  else: # color descriptors
    imgC = cv2.imread(file,1) # >1= bgr  
    imgS = sizeStandardize(imgC,std_width) # Resize input image
    
    # crop out the title of the poster
    h,w = imgS.shape[:2] 
    imgS = imgS[:int(h*title_ratio),:]
 
    kp, des = detector.detectAndCompute(imgS,None)
    
    imgB = np.array([[pixel[0] for pixel in line] for line in imgS])
    imgG = np.array([[pixel[1] for pixel in line] for line in imgS])
    imgR = np.array([[pixel[2] for pixel in line] for line in imgS])

    desB = detector.compute(imgB,kp)[1]
    desG = detector.compute(imgG,kp)[1]
    desR = detector.compute(imgR,kp)[1]

    rows, cols = desB.shape
    desBS = desB[0:rows, 0:(cols/2)]
    desGS = desG[0:rows, 0:(cols/2)]
    desRS = desR[0:rows, 0:(cols/2)]

    # descriptor's length = 16 * 3
    des = np.concatenate((desBS,desGS), axis = 1)
    des = np.concatenate((des,desRS), axis = 1)    
  
  return (kp, des, imgS)

def detectAndCompute_test(file, withColor):
  ''' Detect the keypoints and compute the descriptors for a given image '''
  
  img = cv2.imread(file,0) # 0 = grayscale  
  if img is None: 
    print 'ERROR: Cannot read' + file
    return None
  
  elif not withColor: # grayscale descriptors
    imgS = sizeStandardize(img,std_width) # Resize input image
    
    kp, des = detector.detectAndCompute(imgS,None) 
    
  else: # color descriptors
    imgC = cv2.imread(file,1) # >1= bgr  
    imgS = sizeStandardize(imgC,std_width) # Resize input image
    
    kp, des = detector.detectAndCompute(imgS,None)
    
    imgB = np.array([[pixel[0] for pixel in line] for line in imgS])
    imgG = np.array([[pixel[1] for pixel in line] for line in imgS])
    imgR = np.array([[pixel[2] for pixel in line] for line in imgS])

    desB = detector.compute(imgB,kp)[1]
    desG = detector.compute(imgG,kp)[1]
    desR = detector.compute(imgR,kp)[1]

    rows, cols = desB.shape
    desBS = desB[0:rows, 0:(cols/2)]
    desGS = desG[0:rows, 0:(cols/2)]
    desRS = desR[0:rows, 0:(cols/2)]

    # descriptor's length = 16 * 3
    des = np.concatenate((desBS,desGS), axis = 1)
    des = np.concatenate((des,desRS), axis = 1)    
  
  return (kp, des, imgS)

################################# Begining of main code ##################################
### Define the training method
def trainImage():
  # extract the descriptor for each training image
  for objIdx in POSTERS:
    file = train_path + `objIdx`.zfill(6) +'.jpg'
    dnc = detectAndCompute(file, colorDescriptor) # Detect keypoints, compute descriptors
    if dnc is None: 
      print 'ERROR: Cannot read' + file
    else:
      kp, des, imgT = dnc # get the keypoints, descriptors, and the training image
      train_lib.append([file,kp,des,imgT]) # store in an array 
      print "Generated descriptors for " + file
      
  # init des_lib and des_dict
  global des_lib
  train_des_0 = train_lib[0][2]
  des_lib = train_des_0
  for i in range(train_des_0.shape[0]): des_dict.append([0,i])
  
  for img_i in range(1, len(train_lib)):
    train_des = train_lib[img_i][2]
    # create des_lib
#     print "- des_lib size: "+`des_lib.shape[:2][0]` + " x "+`des_lib.shape[:2][1]`
    des_lib = np.concatenate((des_lib,train_des), axis = 0)
#     print "+ des_lib size: "+`des_lib.shape[:2][0]` + " x "+`des_lib.shape[:2][1]`
    # create des_dict
    for des_i in range(train_des.shape[0]): des_dict.append([img_i, des_i])
  print "Created training library"

### Define the query method
def retrieveImage(query_path,isFiltered):
  correct_retrievals = 0
  unclear_retrievals = 0
  wrong_imgs = []
  unclear_imgs = []
  
  # prepare the output directories
  correct_dir = output_path + "correct_imgs/"
  wrong_dir = output_path + "wrong_imgs/"
  unclear_dir = output_path + "unclear_imgs/"
  for folder in [correct_dir, wrong_dir, unclear_dir]:
    if not os.path.exists(folder): os.mkdir(folder) 
  
  # run experiment
  correct_imgs_cnt = 0
  wrong_imgs_cnt = 0
  unclear_imgs_cnt = 0
  num_imgs_tested = 0
  for objIdx in POSTERS:
    for imgIdx in range (0, test_var):
      file = query_path + `objIdx`.zfill(6) +"_" +`imgIdx`.zfill(6) +'.jpg'
      dnc = detectAndCompute_test(file, colorDescriptor) # Detect keypoints, compute descriptors
      if dnc is None: 
        print 'ERROR: Cannot read' + file
        break
      else:
        # Get the keypoints, descriptors, and the training image
        kp, des, imgQ = dnc   

        # Match descriptors
        matches = bf.match(des,des_lib) # The result is a list of DMatch objects

        # Determine the retrieval images
        out_imgs = [des_dict[match.trainIdx][0] for match in matches]      
        f = Counter(out_imgs).most_common()
        best_img_1 = f[0] # return (img_i, frequency)
        best_img_2 = f[1] if len(f) > 1 else (-1,0)

        # Apply Best-Second-Best filter
        img_path = ''
        img_index = -1
        if best_img_1[0] == objIdx:
          if isFiltered:
            diff = best_img_1[1] - best_img_2[1]
            isGoodMatch = (diff * diff > bsb_thres * best_img_1[1])
            if isGoodMatch: 
              correct_retrievals += 1
              # prepare data to save output images
              img_path = correct_dir
              img_index = best_img_1[0]
              correct_imgs_cnt += 1
            else: 
              unclear_retrievals += 1
              unclear_imgs.append((`objIdx`.zfill(6) +"_" +`imgIdx`.zfill(6),best_img_1[0],best_img_2[0]))
              # prepare data to save output images
              img_path = unclear_dir
              img_index = best_img_2[0]
              unclear_imgs_cnt += 1
          else:
            correct_retrievals += 1  
            # prepare data to save output images
            img_path = correct_dir
            img_index = best_img_1[0]
            correct_imgs_cnt += 1  
        else:
          wrong_imgs.append((`objIdx`.zfill(6) +"_" +`imgIdx`.zfill(6),best_img_1[0],best_img_2[0]))
          # prepare data to save output images
          img_path = wrong_dir
          img_index = best_img_1[0]
          wrong_imgs_cnt += 1

        # Save images of the result 
        if not colorDescriptor:
          if ((img_path == correct_dir) and (correct_imgs_cnt <= 10)) or ((img_path == wrong_dir) and (wrong_imgs_cnt <= 10)) or ((img_path == unclear_dir) and (unclear_imgs_cnt <= 10)):
            train_img_1 = train_lib[img_index]
            matches_1 = [dmatch for dmatch in matches if des_dict[dmatch.trainIdx][0] == img_index]
            outPath = img_path + `objIdx`.zfill(6) +"_" +`imgIdx`.zfill(6) +'_'+ `img_index` + '.jpg'      
            saveOutputImages(imgQ,kp,train_img_1[3],train_img_1[1],matches_1,outPath)
        else:
          print "[Can't save ouput images when using color descriptor]"

        num_imgs_tested += 1
        mrate = float(correct_retrievals)/ num_imgs_tested
        acc = float(correct_retrievals + unclear_retrievals)/ num_imgs_tested
        print file + '\tAccuracy: %.2f    \t,Match Rate: %.2f' % (acc*100,mrate*100)
      
  match_rate = float(correct_retrievals)/ (CLASSES* test_var)
  accuracy = float(correct_retrievals + unclear_retrievals)/ (CLASSES* test_var)
  print 'Final Accuracy: %.2f    \t,Match Rate: %.2f' % (acc*100,mrate*100)
  return (accuracy, match_rate, wrong_imgs, unclear_imgs)
      
      
################# Train images
print '\n'
trainImage()
print '\n'
      
################# Query images
if not os.path.exists(output_path): os.mkdir(output_path) 
  
text_out = output_path + 'outputData.txt'
f = open(text_out,'w')
f.write('Library size: %d \nNumber of keypoints per image: %d \nFiltered: %s \nColor descriptors: %s \n' % (CLASSES,number_of_kp,isFiltered,colorDescriptor)) 
f.write("Poster indexes: " + ", ".join(str(e) for e in POSTERS) + "\n\n") 

text_outS = output_path + 'outputData_summary.txt'
fS = open(text_outS,'w')
fS.write('Library size: %d \nNumber of keypoints per image: %d \nFiltered: %s \nColor descriptors: %s \n' % (CLASSES,number_of_kp,isFiltered,colorDescriptor)) 
fS.write("Poster indexes: " + ", ".join(str(e) for e in POSTERS) + "\n\n") 

# query all images
print query_path
out = retrieveImage(query_path,isFiltered)
# print query_path, out

acc = round(out[0] * 100, 2)
mrate = round(out[1] * 100, 2)
wrong_imgs = out[2]
unclear_imgs = out[3]
f.write(query_path + ': \n' 
        '>>> Accuracy: %s, Match Rate: %s \n' % (acc,mrate) +
        '    Incorrect (object, best_match, 2nd_best_match):' + ', '.join(map(str, wrong_imgs)) + '\n' +
        '    Unclear (object, best_match, 2nd_best_match):' + ', '.join(map(str, unclear_imgs)) + '\n' + '\n'
       )
fS.write(query_path + ':\tAccuracy: %s,\tMatch Rate: %s \n\n' % (acc,mrate))
f.close()
fS.close()

print '\n'

