import os
import cv2

img_path = '/home/tianz/datsets/ucfimgs/'
dirs = os.listdir(img_path)

for allDir in dirs:
    child = os.path.join('%s%s' % (img_path, allDir))
    vids = os.listdir(child)
    for filename in vids:
        file_path = child + '/' + filename
        s1 = allDir + '/' + filename
        files = os.listdir(file_path)
        for file in files:
            img = cv2.imread(file_path+'/'+file)
            if img is None:
                print(file_path+'/'+file)
                os.remove(file_path+'/'+file)
