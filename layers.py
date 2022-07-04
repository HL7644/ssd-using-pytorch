import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_class=20

def subsampling(tensor, target_size): #subsample target w/ target size (every N element)
  N_dim=len(tensor.size())
  sampled_tensor=torch.zeros(target_size).to(device)
  tensor_size=tensor.size()
  indices=[]
  for idx, size in enumerate(target_size):
    indices_dim=torch.zeros(size).long().to(device)
    t_size=tensor_size[idx]
    sample_rate=int(t_size/size)
    for s_idx in range(size):
      indices_dim[s_idx]=sample_rate*(s_idx+1)-1
    indices.append(indices_dim)
  sampled_tensor=torch.index_select(tensor, 0, indices[0])
  for dim in range(1,N_dim):
    sampled_tensor=torch.index_select(sampled_tensor, dim, indices[dim])
  return sampled_tensor

class Base_Layer(nn.Module):
  def __init__(self):
    super(Base_Layer, self).__init__()
    #Base Layer of SSD: First&Second feature map creation
    vgg16_feat=torchvision.models.vgg16(pretrained=True).features.to(device)
    vgg16_classif=torchvision.models.vgg16(pretrained=True).classifier.to(device)
    #dimensions of fully connecting layers
    fc6=vgg16_classif[0]
    w6=fc6.weight.reshape(4096,512,7,7)
    fc7=vgg16_classif[3]
    w7=fc7.weight.reshape(4096,4096,1,1)
    #1st feature map
    self.conv1_1=vgg16_feat[0].to(device)
    self.conv1_2=vgg16_feat[2].to(device)
    self.maxpool1=vgg16_feat[4].to(device)
    self.conv2_1=vgg16_feat[5].to(device)
    self.conv2_2=vgg16_feat[7].to(device)
    self.maxpool2=vgg16_feat[9].to(device)
    self.conv3_1=vgg16_feat[10].to(device)
    self.conv3_2=vgg16_feat[12].to(device)
    self.conv3_3=vgg16_feat[14].to(device)
    self.maxpool3=nn.MaxPool2d((2,2),stride=2, ceil_mode=True).to(device)
    self.conv4_1=vgg16_feat[17].to(device)
    self.conv4_2=vgg16_feat[19].to(device)
    self.conv4_3=vgg16_feat[21].to(device)
    #2nd feature map
    self.maxpool4=vgg16_feat[23].to(device)
    self.conv5_1=vgg16_feat[24].to(device)
    self.conv5_2=vgg16_feat[26].to(device)
    self.conv5_3=vgg16_feat[28].to(device)
    self.maxpool5=nn.MaxPool2d((3,3),stride=1,padding=1).to(device)
    #subsample weights from fc6,7 -> define conv6,7
    self.conv6=nn.Conv2d(512, 1024, (3,3),  padding=6, dilation=6).to(device) #Atrous Convolution
    self.conv6.weight=nn.Parameter(subsampling(w6, self.conv6.weight.size()))
    self.conv7=nn.Conv2d(1024, 1024, (1,1)).to(device)
    self.conv7.weight=nn.Parameter(subsampling(w7, self.conv7.weight.size()))

  def forward(self, image):
    relu=nn.ReLU(inplace=True).to(device)
    first_layer=nn.Sequential(self.conv1_1, relu, self.conv1_2, relu, self.maxpool1, self.conv2_1, relu, self.conv2_2, relu, self.maxpool2, 
                              self.conv3_1, relu, self.conv3_2, relu, self.conv3_3, relu, self.maxpool3, self.conv4_1, relu, 
                              self.conv4_2, relu, self.conv4_3, relu).to(device)
    first_featmap=first_layer(image).to(device)
    second_layer=nn.Sequential(self.maxpool4, self.conv5_1, relu, self.conv5_2, relu, self.conv5_3, relu, self.maxpool5, 
                               self.conv6, relu, self.conv7, relu).to(device)
    second_featmap=second_layer(first_featmap).to(device)

    return first_featmap, second_featmap

class Additional_Convs(nn.Module):
  def __init__(self):
    super(Additional_Convs, self).__init__()
    self.conv8_1=nn.Conv2d(1024,256,(1,1)).to(device)
    self.conv8_2=nn.Conv2d(256,512,(3,3), stride=2, padding=1).to(device)
    self.conv9_1=nn.Conv2d(512,128,(1,1)).to(device)
    self.conv9_2=nn.Conv2d(128,256,(3,3), stride=2, padding=1).to(device)
    self.conv10_1=nn.Conv2d(256,128,(1,1)).to(device)
    self.conv10_2=nn.Conv2d(128,256,(3,3)).to(device)
    self.conv11_1=nn.Conv2d(256,128,(1,1)).to(device)
    self.conv11_2=nn.Conv2d(128,256,(3,3)).to(device)
    self.conv_init()

  def conv_init(self): #initialize convolutions u/Xavier uniform
    for instance in self.children():
      if isinstance(instance, nn.Conv2d):
        nn.init.xavier_uniform_(instance.weight) #inplace operation
        nn.init.constant_(instance.bias, 0.)

  def forward(self, second_featmap):
    relu=nn.ReLU().to(device)
    third_layer=nn.Sequential(self.conv8_1, relu, self.conv8_2, relu).to(device)
    third_featmap=third_layer(second_featmap).to(device)
    fourth_layer=nn.Sequential(self.conv9_1, relu, self.conv9_2, relu).to(device)
    fourth_featmap=fourth_layer(third_featmap).to(device)
    fifth_layer=nn.Sequential(self.conv10_1, relu, self.conv10_2, relu).to(device)
    fifth_featmap=fifth_layer(fourth_featmap).to(device)
    sixth_layer=nn.Sequential(self.conv11_1, relu, self.conv11_2, relu).to(device)
    sixth_featmap=sixth_layer(fifth_featmap).to(device)

    return third_featmap, fourth_featmap, fifth_featmap, sixth_featmap  

class Feature_extractor(nn.Module):
  def __init__(self):
    super(Feature_extractor, self).__init__()
    self.first_extractor=nn.Conv2d(512, 4*(N_class+1+4),(3,3),stride=1, padding=1).to(device)
    self.second_extractor=nn.Conv2d(1024, 6*(N_class+1+4),(3,3),stride=1, padding=1).to(device)
    self.third_extractor=nn.Conv2d(512, 6*(N_class+1+4),(3,3),stride=1, padding=1).to(device)
    self.fourth_extractor=nn.Conv2d(256, 6*(N_class+1+4),(3,3),stride=1, padding=1).to(device)
    self.fifth_extractor=nn.Conv2d(256, 4*(N_class+1+4),(3,3),stride=1, padding=1).to(device)
    self.sixth_extractor=nn.Conv2d(256, 4*(N_class+1+4),(3,3),stride=1, padding=1).to(device)
    self.conv_init()
  
  def conv_init(self):
    for instance in self.children():
      if isinstance(instance, nn.Conv2d):
        nn.init.xavier_uniform_(instance.weight) #inplace operation
        nn.init.constant_(instance.bias, 0.)
  
  def forward(self, first_featmap, second_featmap, third_featmap, fourth_featmap, fifth_featmap, sixth_featmap):
    #normalize first features
    first_featmap=F.normalize(first_featmap, p=2, dim=1)
    first_featmap=first_featmap*nn.Parameter(torch.full((1,512,1,1), 20.).to(device))
    first_features=self.first_extractor(first_featmap).to(device)
    #permute&reshape into (batch,8732,4+21)
    first_features=first_features.permute(0,2,3,1)
    first_features=first_features.reshape(first_features.size(0),-1,(N_class+1+4))

    second_features=self.second_extractor(second_featmap).to(device)
    second_features=second_features.permute(0,2,3,1)
    second_features=second_features.reshape(second_features.size(0),-1,(N_class+1+4))

    third_features=self.third_extractor(third_featmap).to(device)
    third_features=third_features.permute(0,2,3,1)
    third_features=third_features.reshape(third_features.size(0),-1,(N_class+1+4))

    fourth_features=self.fourth_extractor(fourth_featmap).to(device)
    fourth_features=fourth_features.permute(0,2,3,1)
    fourth_features=fourth_features.reshape(fourth_features.size(0),-1,(N_class+1+4))

    fifth_features=self.fifth_extractor(fifth_featmap).to(device)
    fifth_features=fifth_features.permute(0,2,3,1)
    fifth_features=fifth_features.reshape(fifth_features.size(0),-1,(N_class+1+4))

    sixth_features=self.sixth_extractor(sixth_featmap).to(device)
    sixth_features=sixth_features.permute(0,2,3,1)
    sixth_features=sixth_features.reshape(sixth_features.size(0),-1,(N_class+1+4))
    predictions=torch.cat((first_features, second_features, third_features, fourth_features, fifth_features, sixth_features), dim=1).to(device)

    return predictions
