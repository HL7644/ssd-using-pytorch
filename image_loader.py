import torch
import torchvision
import torchvision.transforms as transforms
import os
import PIL
import os.path
import xml.etree.ElementTree as ET
from google.colab import drive
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drive.mount('/content/gdrive') #using google colaboratory

#create class dictionary for PASCAL-VOC images (20 classes)
N_class=20
cls_to_clslabel=dict()
clslabel_to_cls=dict()
cls_to_clslabel['aeroplane']=1
cls_to_clslabel['bicycle']=2
cls_to_clslabel['bird']=3
cls_to_clslabel['boat']=4
cls_to_clslabel['bottle']=5
cls_to_clslabel['bus']=6
cls_to_clslabel['car']=7
cls_to_clslabel['cat']=8
cls_to_clslabel['chair']=9
cls_to_clslabel['cow']=10
cls_to_clslabel['diningtable']=11
cls_to_clslabel['dog']=12
cls_to_clslabel['horse']=13
cls_to_clslabel['motorbike']=14
cls_to_clslabel['person']=15
cls_to_clslabel['pottedplant']=16
cls_to_clslabel['sheep']=17
cls_to_clslabel['sofa']=18
cls_to_clslabel['train']=19
cls_to_clslabel['tvmonitor']=20
for val, key in enumerate(cls_to_clslabel):
  clslabel_to_cls[val+1]=key

#Custom Dataset of images
class Images_Data(torch.utils.data.Dataset):
  def __init__(self, images, bboxes, cls_labels):
    self.images=images
    self.bboxes=bboxes
    self.cls_labels=cls_labels
  
  def __len__(self):
    return self.images.size(0)
  
  def __getitem__(self, idx):
    image=self.images[idx]
    bbox=self.bboxes[idx]
    cls_label=self.cls_labels[idx]
    return image, bbox, cls_label

def reshape_img(img, size):
  w=size[1]
  h=size[0]
  scale=[0,0]
  _, img_h, img_w=img.size() #Image tensor is in format of CxHxW
  #step1: rescale into desired size
  image_rescale=torchvision.transforms.Resize(size).to(device)
  rescaled_tensor=image_rescale(img).to(device)
  scale[0]=h/img_h
  scale[1]=w/img_w

  return rescaled_tensor, scale

def format_bbox(bbox, scale): #formatting for single bbox
  #format bbox from (x1,y1,y1,y2) into (row1,col1,row2,col2), shift & rescale
  h_scale=scale[0]
  w_scale=scale[1]
  r1=int((bbox[0])*h_scale)
  c1=int((bbox[1])*w_scale)
  r2=int((bbox[2])*h_scale)
  c2=int((bbox[3])*w_scale)

  reshaped_bbox=torch.FloatTensor([r1,c1,r2,c2]).to(device)
  return reshaped_bbox  

#implement on pascal-voc 2007
#multi_bboxes for each image

def get_pvoc_images(N):
    img_dir_2007='/content/gdrive/My Drive/Colab Notebooks/VOC2007 Set/JPEGImages/'
    annot_dir_2007='/content/gdrive/My Drive/Colab Notebooks/VOC2007 Set/Annotations/'
    images=torch.Tensor([]).to(device)
    bboxes=[]
    cls_labels=[]
    goal_size=(300,300)
    img_to_tensor=transforms.ToTensor()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_normalize=transforms.Normalize(mean, std)

    N=64
    counter=0 
    for label in range(5, 9962):
        len_zero=5-int(np.log10(label))
    zero_string=''
    for _ in range(len_zero):
        zero_string+='0'
    img_name=img_dir_2007+zero_string+str(label)+'.jpg'
    annot_name=annot_dir_2007+zero_string+str(label)+'.xml'
    if os.path.isfile(annot_name):
        #get images
        img=PIL.Image.open(img_name)
        img=img_to_tensor(img).to(device)
        #normalize image
        img=img_normalize(img)
        reshaped_img, scale=reshape_img(img, goal_size)
        images=torch.cat((images, reshaped_img.unsqueeze(dim=0)), dim=0)
        #get annotations
        bbox_img=torch.Tensor([]).to(device)
        cls_label_img=torch.Tensor([]).to(device)
        tree=ET.parse(annot_name)
        for obj in tree.findall('./object'):
            bndbox=obj.find('bndbox')
            cls=obj.findtext('name')
            cls_label=cls_to_clslabel[cls]-1
            cls_label=torch.LongTensor([cls_label]).to(device)
            c1=int(bndbox.findtext('xmin'))
            r1=int(bndbox.findtext('ymin'))
            c2=int(bndbox.findtext('xmax'))
            r2=int(bndbox.findtext('ymax'))
            bbox=torch.FloatTensor([r1,c1,r2,c2]).to(device)
            reshaped_bbox=format_bbox(bbox, scale)
            bbox_img=torch.cat((bbox_img, reshaped_bbox.unsqueeze(dim=0)), dim=0)
            cls_label_img=torch.cat((cls_label_img, cls_label.unsqueeze(dim=0)), dim=0)
        bboxes.append(bbox_img)
        cls_labels.append(cls_label_img.long())
        counter+=1
        if counter==N:
            images_data=Images_Data(images, bboxes, cls_labels)
            return images_data


