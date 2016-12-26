[Generate Labels for VOC]

Now we need to generate the label files that Darknet uses. Darknet wants a .txt file for each image with a line for each ground truth object in the image that looks like:

<object-class> <x> <y> <width> <height>

Where x, y, width, and height are relative to the image's width and height, with x and y are the location of the center of the object:

    x = x_center / img_w
    y = y_center / img_h
    with = obj_w / img w
    height = obj_h / img_h

<objectClass> <xCenter/imgW> <yCenter/imgH> <objW/imgW> <objH/imgH>

[train.txt - darknet input file]
train.txt contains the names of all training images (one for each line)
the text file itself can be located anywhere.

[images/ - labels/]
the labels and the images folders should be in the same location
to look up the code where the 2 folders are used, go to darknet github and search for "labels" using the search engine