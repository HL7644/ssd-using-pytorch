import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dsets
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import random

import json
import pandas as pd
import os.path
import xml.etree.ElementTree as ET
import PIL
from google.colab import drive

#import python files
from ftns_for_loss import *
from layers import *
from image_loader import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_data=get_pvoc_images(N=256) #N is limited due to GPU limit

class SSD(nn.Module):
  def __init__(self):
    super(SSD, self).__init__()
    #SSD Architecture
    self.base=Base_Layer()
    self.additional_conv=Additional_Convs()
    self.feat_extractor=Feature_extractor()

  def train(self, images, bboxes, cls_labels):
    batch=images.size(0)
    #Obtain predictions
    first_featmap, second_featmap=self.base(images)
    third_featmap, fourth_featmap, fifth_featmap, sixth_featmap=self.additional_conv(second_featmap)
    predictions=self.feat_extractor(first_featmap, second_featmap, third_featmap, fourth_featmap, fifth_featmap, sixth_featmap)
    offsets=predictions[:,:,:4].to(device) #(cx,cy,w,h)
    confidences=predictions[:,:,4:].to(device) #21 confidences

    featmap_sizes=[[38,38],[19,19],[10,10],[5,5],[3,3],[1,1]]
    default_box_dims=get_box_dims()
    #default box center coordinates for 6 shapes of feature maps
    center_coords=get_center_coords(featmap_sizes)
    #default boxes for each feature map (list of 8732)
    default_boxes=get_default_boxes(center_coords, default_box_dims) #default boxes are in same order as prediction
    total_loss=torch.FloatTensor([0]).to(device)
    #For each image:
    for n in range(batch):
      #gt_data
      image=images[n].to(device)
      bboxes_img=bboxes[n].to(device) #single object per image in P-VOC
      cls_labels_img=cls_labels[n]
      #prediction data
      offset=offsets[n].to(device)
      confidence=confidences[n].to(device)
      #calculate loss
      total_loss+=get_loss(default_boxes, bboxes_img, offset, cls_labels_img, confidence)
    return total_loss
  
  def test(self, images):
    batch,_,img_h,img_w=images.size()
    #Obtain predictions
    first_featmap, second_featmap=self.base(images)
    third_featmap, fourth_featmap, fifth_featmap, sixth_featmap=self.additional_conv(second_featmap)
    predictions=self.feat_extractor(first_featmap, second_featmap, third_featmap, fourth_featmap, fifth_featmap, sixth_featmap)
    offsets=predictions[:,:,:4].to(device) #(cx,cy,w,h)
    confidences=predictions[:,:,4:].to(device) #21 confidences

    #default box dimensions for 6 shapes of feature maps
    aspect_ratio=[1,2,3,1/2,1/3]
    featmap_sizes=[[38,38],[19,19],[10,10],[5,5],[3,3],[1,1]]
    default_box_dims=get_box_dims()
    #default box center coordinates for 6 shapes of feature maps
    center_coords=get_center_coords(featmap_sizes)
    #default boxes for each feature map (list of 8732)
    default_boxes=get_default_boxes(center_coords, default_box_dims) #default boxes are in same order as prediction
    N_defbox=default_boxes.size(0)
    pred_bboxes=[]
    pred_classes=[]
    for n in range(batch):
      pred_bbox_image=torch.Tensor([])
      pred_class_image=[]
      image=images[n]
      offset=offsets[n]
      confidence=confidences[n]
      for idx, conf_vector in enumerate(confidence):
        argmax_conf, argmax_class=torch.max(conf_vector, dim=0)
        if argmax_class!=0: #if not background
          if argmax_conf>0.05:
            pred_clslabel=int(argmax_class)
            pred_class=clslabel_to_cls[pred_clslabel]
            offset_box=offset[idx]
            pred_bbox=get_prediction(default_boxes[idx], offset_box)
            pred_bbox_image=torch.cat((pred_bbox_image, pred_bbox.unsqueeze(dim=0)), dim=0)
            pred_class_image.append(pred_class)
      pred_bboxes.append(pred_bbox_image)
      pred_classes.append(pred_class_image)
    return pred_bboxes, pred_classes

ssd=SSD().to(device)
optimizer=optim.SGD(ssd.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
dataloader=torch.utils.data.DataLoader(images_data, batch_size=64, shuffle=True)

#Train
iter=500
for epoch in range(1, iter+1):
    for batch_data in dataloader:
        images, bboxes, cls_labels=batch_data
        cost=ssd.train(images, bboxes, cls_labels)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if epoch%50==0:
            print("Epoch:{:d}, Cost:{:.3f}".format(epoch, cost.item()))

