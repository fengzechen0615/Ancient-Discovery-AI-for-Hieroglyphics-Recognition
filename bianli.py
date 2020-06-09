# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:48:43 2019

@author: wuyikun
"""

from skimage import transform,io,color
import os
import string

picdir="/Users/glh/Desktop/甲骨文收集/甲骨文-金文/"
pic = os.listdir(picdir)
for file in pic:
    dir2 = picdir+file+"/"
    address0 = os.listdir(dir2)
    for folder in address0:
        if folder.find(".")>=0:
            continue;
        address = dir2+folder+"/png/"
        picaddress = os.listdir(address)
        for filename in picaddress:
            img = io.imread(address+filename)
            img2size = transform.resize(img,(96,96))
            io.imsave(address+filename,img2size)
    
        
        for filename in picaddress:
        
            newname = filename
            newname = newname.split(".")
            if newname[-1] == "png":
                newname[-1] = "jpg"
                newname = str.join(".",newname) 

                filename = address+filename

                newname = address+newname

                os.rename(filename,newname)
    