import os
from os import listdir
import pytesseract
import cv2
from PIL import Image
#Path to base folder 
cur=os.path.abspath(os.curdir)
os.system('tar xvf SRGAN_pre-trained.tar') #Running the pretrained model
#The input images are loaded from /data/myimage 
#Path to folder with images
path=cur+'/data/myimage'

#Reading images from myimage and converting them to png
os.chdir(path)
l=0
files={i for i in listdir(path)}
print(files)
for filename in files:
	img=Image.open(filename)
	print(filename)
	l=l+1
	img.save(str(l)+'.png')
	os.remove(filename)

#Returning to the base folder
os.chdir(cur)

#After running the pretrained model on our images the output images are stored in
#/result/images  **The images folder is formed after the model runs
os.system('sh inference.sh')

#Extracting information from enhanced images
path_o=cur+'/result/images'
os.chdir(path_o)
custom_config=r'--oem 3 --psm 6 outputbase digits'
a=0
images={ i for i in listdir(path_o)}
for im in images:
	img=Image.open(im)
	print("Data for image"+ im+"\n")
	print(pytesseract.image_to_string(img,config=custom_config))


