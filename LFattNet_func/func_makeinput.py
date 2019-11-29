# -*- coding: utf-8 -*-

import imageio
import numpy as np
import os

def make_epiinput(image_path,seq1,image_h,image_w,view_n,RGB):
    traindata_tmp=np.zeros((1,image_h,image_w,len(view_n)),dtype=np.float32)
    i=0
    if(len(image_path)==1):
        image_path=image_path[0]
        
    for seq in seq1:
        tmp  = np.float32(imageio.imread(image_path+'/input_Cam0%.2d.png' % seq)) 
        traindata_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
    return traindata_tmp


def make_input(image_path, image_h, image_w, view_n):
    RGB = [0.299, 0.587, 0.114]  ## RGB to Gray // 0.299 0.587 0.114

    '''
    data from http://hci-lightfield.iwr.uni-heidelberg.de/
    Sample images ex: Cam000~ Cam080.png  
    '''

    output_list = []
    for i in range(81):
        if(image_path[:12]=='hci_dataset/'):
            A = make_epiinput(image_path, [i], image_h, image_w, [0], RGB)
        # print(A.shape)
        output_list.append(A)

    return output_list