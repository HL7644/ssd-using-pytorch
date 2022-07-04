import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_class=20
images_size=(300,300)

def get_box_dims():
  ar1=[1,2,1/2]
  ar2=[1,2,3,1/2,1/3]
  default_box_dims=[]
  img_h, img_w=images_size[2], images_size[3]
  scales=[] #scales for each feature map
  smin=0.2
  smax=0.9
  scales.append(0.1)
  for k in range(1,6):
    scales.append(smin+(smax-smin)/(5-1)*(k-1))
  scales.append(1.)
  len_s=len(scales)-1 #1 is optional scale
  for idx in range(len_s):
    if idx==1 or idx==2 or idx==3:
      scale=scales[idx]
      #use ar2
      box_dim=torch.zeros(6,2) #h,w format
      extra_scale=np.sqrt(scale*scales[idx+1])
      for ar_idx, ar in enumerate(ar2):
        h=scale*np.sqrt(ar)
        w=scale/np.sqrt(ar)
        box_dim[ar_idx,:]=torch.FloatTensor([h*img_h,w*img_w])
      box_dim[-1,:]=torch.FloatTensor([extra_scale*img_h, extra_scale*img_w])
    else:
      scale=scales[idx]
      #use ar1
      box_dim=torch.zeros(4,2).to(device)
      extra_scale=np.sqrt(scale*scales[idx+1])
      for ar_idx, ar in enumerate(ar1):
        h=scale*np.sqrt(ar)
        w=scale/np.sqrt(ar)
        box_dim[ar_idx,:]=torch.FloatTensor([h*img_h,w*img_w])
      box_dim[-1,:]=torch.FloatTensor([extra_scale*img_h, extra_scale*img_w])
    default_box_dims.append(box_dim)
  return default_box_dims

featmap_sizes=[[38,38],[19,19],[10,10],[5,5],[3,3],[1,1]]

def get_center_coords(featmap_sizes):
  img_h, img_w=images_size[2], images_size[3]
  center_coords=[]
  for size in featmap_sizes:
    feat_h=size[0]
    feat_w=size[1]
    center_coords_f=torch.zeros(feat_h, feat_w, 2).to(device) #(row, col) form
    for row in range(feat_h):
      for col in range(feat_w):
        center_coords_f[row,col,0]=(row+0.5)/feat_h*img_h
        center_coords_f[row,col,1]=(col+0.5)/feat_w*img_w
    center_coords.append(center_coords_f)
  return center_coords

def get_default_boxes(center_coords, default_box_dims):
  img_h, img_w=images_size[2], images_size[3]
  len_f=len(center_coords) #No. of feature maps
  default_boxes=torch.Tensor([]).to(device)
  for i in range(len_f):
    center_coord=center_coords[i]
    default_box_dim=default_box_dims[i]
    feat_h, feat_w,_=center_coord.size()
    default_box=torch.zeros(feat_h, feat_w, len(default_box_dim), 4).to(device) #r1,c1,r2,c2 format
    for row in range(feat_h):
      for col in range(feat_w):
        #center in h,w format
        r_center=center_coord[row,col,0]
        c_center=center_coord[row,col,1]
        #for each center location generate default boxes
        for idx, dim in enumerate(default_box_dim):
          h=dim[0]
          w=dim[1]
          r1=r_center-0.5*h
          if r1<0:
            r1=0
          c1=c_center-0.5*w
          if c1<0:
            c1=0
          r2=r_center+0.5*h
          if r2>img_h:
            r2=img_h
          c2=c_center+0.5*w
          if c2>img_w:
            c2=img_w
          default_box[row,col,idx,:]=torch.FloatTensor([r1,c1,r2,c2])
    default_box=default_box.reshape(-1,4)
    default_boxes=torch.cat((default_boxes, default_box),dim=0)
  return default_boxes

def get_iou(box1, box2):
  r1=box1[0] #coordinates in row-column perspective
  c1=box1[1]
  r2=box1[2]
  c2=box1[3]
  tr1=box2[0]
  tc1=box2[1]
  tr2=box2[2]
  tc2=box2[3]
  r1_max=max(r1,tr1)
  r2_min=min(r2,tr2)
  c1_max=max(c1,tc1)
  c2_min=min(c2,tc2)
  if r1_max>r2_min or c1_max>c2_min:
    iou=0
  else:
    intersection=(r2_min-r1_max)*(c2_min-c1_max)
    union=(r2-r1)*(c2-c1)+(tr2-tr1)*(tc2-tc1)-intersection
    iou=intersection/union
  return iou

def parametrize_bbox(default_box, bbox):
  d_xc=(default_box[1]+default_box[3])/2
  d_yc=(default_box[0]+default_box[2])/2
  d_w=default_box[3]-default_box[1]
  d_h=default_box[2]-default_box[0]
  bbox_xc=(bbox[1]+bbox[3])/2
  bbox_yc=(bbox[0]+bbox[2])/2
  bbox_w=bbox[3]-bbox[1]
  bbox_h=bbox[2]-bbox[0]
  tx=(bbox_xc-d_xc)/d_w
  ty=(bbox_yc-d_yc)/d_h
  tw=torch.log(bbox_w/d_w)
  th=torch.log(bbox_h/d_h)
  p_bbox=torch.FloatTensor([tx,ty,tw,th]).to(device)
  return p_bbox

def get_loss(default_boxes, bboxes_img, offset, cls_labels_img, confidence):
  #per image loss
  loc_loss=torch.FloatTensor([0]).to(device)
  conf_loss=torch.FloatTensor([0]).to(device)
  negative_losses=torch.FloatTensor([]).to(device)
  #No. of bboxes in particular image
  len_bbox=bboxes_img.size(0)
  #No. of default boxes
  len_defbox=default_boxes.size(0)
  N_pos=0
  iou_tensor=torch.zeros(len_defbox, len_bbox).to(device)
  #for each default box
  for def_idx, default_box in enumerate(default_boxes):
    offset_vector=offset[def_idx]
    conf_vector=confidence[def_idx]
    #get iou for each bbox
    iou_vector=torch.zeros(len_bbox).to(device)
    for bb_idx, bbox in enumerate(bboxes_img):
      iou=get_iou(bbox, default_box)
      iou_tensor[def_idx,bb_idx]=iou
    #select argmax
    argmax_iou, argmax_bb_idx=torch.max(iou_tensor[def_idx], dim=0)
    #corresponding cls label
    argmax_clslabel=cls_labels_img[argmax_bb_idx]
    if argmax_iou>=0.5: #if positive
      N_pos+=1
      #loc_loss
      gt_loc_vector=parametrize_bbox(default_box, bbox)
      loc_loss=loc_loss+F.smooth_l1_loss(offset_vector, gt_loc_vector)
      #conf_loss
      gt_conf_vector=torch.zeros(N_class+1).to(device)
      gt_conf_vector[argmax_clslabel]=1.
      sfmax_conf_vector=F.softmax(conf_vector, dim=0)
      conf_loss=conf_loss+F.cross_entropy(sfmax_conf_vector.unsqueeze(dim=0), gt_conf_vector.unsqueeze(dim=0))
    else: #if negative
      gt_conf_vector=torch.zeros(N_class+1).to(device)
      gt_conf_vector[0]=1.
      sfmax_conf_vector=F.softmax(conf_vector, dim=0)
      neg_loss_box=F.cross_entropy(sfmax_conf_vector.unsqueeze(dim=0), gt_conf_vector.unsqueeze(dim=0))
      negative_losses=torch.cat((negative_losses, neg_loss_box.unsqueeze(dim=0)), dim=0)
  #add maximum IoU defbox to bboxes
  for bb_idx in range(len_bbox):
    iou_vector=iou_tensor[:,bb_idx]
    argmax_iou, argmax_defidx=torch.max(iou_vector, dim=0)
    default_box=default_boxes[argmax_defidx]
    #if unselected in above iteration
    if argmax_iou<0.5:
      N_pos+=1
      #loc_loss
      gt_loc_vector=parametrize_bbox(default_box, bbox)
      loc_loss=loc_loss+F.smooth_l1_loss(offset_vector, gt_loc_vector)
      #conf_loss
      cls_label=cls_labels_img[bb_idx]
      gt_conf_vector=torch.zeros(N_class+1).to(device)
      gt_conf_vector[cls_label]=1.
      sfmax_conf_vector=F.softmax(conf_vector, dim=0)
      conf_loss=conf_loss+F.cross_entropy(sfmax_conf_vector.unsqueeze(dim=0), gt_conf_vector.unsqueeze(dim=0))

  #apply hard-negative mining
  sorted_negative_losses,_=torch.sort(negative_losses, dim=0, descending=True)
  sorted_negative_losses=sorted_negative_losses[:3*N_pos]
  conf_loss+=torch.sum(sorted_negative_losses).unsqueeze(dim=0)
  if N_pos==0:
    total_loss=0
  else:
    total_loss=(loc_loss+conf_loss)/N_pos
  return total_loss

def get_prediction(default_box, offset):
  img_h, img_w=images_size[2], images_size[3]
  #offset: cx,cy,w,h form
  #goal: (r1,c1,r2,c2) form
  d_cx=(default_box[1]+default_box[3])/2
  d_cy=(default_box[0]+default_box[2])/2
  d_w=default_box[3]-default_box[1]
  d_h=default_box[2]-default_box[0]
  g_cx=offset[0]*d_w+d_cx
  g_cy=offset[1]*d_h+d_cy
  g_w=torch.exp(offset[2])*d_w
  g_h=torch.exp(offset[3])*d_h
  r1=g_cy-0.5*g_h
  if r1<0:
    r1=0
  c1=g_cx-0.5*g_w
  if c1<0:
    c1=0
  r2=g_cy+0.5*g_h
  if r2>img_h:
    r2=img_h
  c2=g_cx+0.5*g_w
  if c2>img_w:
    c2=img_w
  prediction=torch.FloatTensor([r1,c1,r2,c2]).to(device)
  return prediction