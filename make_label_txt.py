import os

img_path = '/home/tianz/datsets/ucfimgs/'
f1 = open('ucfTrainTestlist/train_file.txt','r')
f2 = open('ucfTrainTestlist/test_file.txt','r')

train_list = f1.readlines()
test_list = f2.readlines()

f3 = open('train_list.txt', 'w')
f4 = open('test_list.txt', 'w')

for line in train_list:
    name = line.split(' ')[0]
    image_path = img_path+name
    label = line.split(' ')[-1]
    images = os.listdir(image_path)
    nb = len(images) // 16
    for i in range(nb):
        f3.write(name+' '+ str(i*16+1)+' '+label)


for line in test_list:
    name = line.split(' ')[0]
    image_path = img_path+name
    label = line.split(' ')[-1]
    images = os.listdir(image_path)
    nb = len(images) // 16
    for i in range(nb):
        f4.write(name+' '+ str(i*16+1)+' '+label)

f1.close()
f2.close()
f3.close()
f4.close()
