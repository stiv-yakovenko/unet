import os
import shutil
import cv2

dir = "./data/roofBlocks/train/image/"
out="./data/roof_tf_unet/"
for filename in os.listdir(dir):
 i1=cv2.imread(dir+filename)
 cv2.imwrite(out+filename,i1)
 i2=cv2.imread(dir + "../label/"+filename,0)
 cv2.imwrite(out + (filename).replace(".png","")+"_mask.png",i2)

