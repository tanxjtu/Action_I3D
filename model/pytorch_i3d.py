import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.insert(0, '..')
from lib.model.roi_align.modules.roi_align import RoIAlignAvg, RoIAlignMax
import numpy as np
import os
import sys
import json
from collections import OrderedDict
import pickle
import math


def load_bbox(bs_dir, vid, fm_list):
    file_name = bs_dir + vid + '.json'
    vd_bbox = json.load(open(file_name, 'r'))
    out_bbox = []
    for i in fm_list:
        out_bbox.append(vd_bbox['%06d' % i])
    return out_bbox


def Comp_ROI(ZIP_list):
    human_ROI = []
    Object_ROI = []
    for NO_Fm, ech_fm in enumerate(ZIP_list):
        for ec_clc in ech_fm:
            for ec_ins in ec_clc:
                if int(ec_ins[-1]) == 1:
                    human_ROI.append(np.insert(np.around(np.array(ec_ins[:4]), 2), 0, NO_Fm))
                else:
                    Object_ROI.append(np.insert(np.around(np.array(ec_ins[:4]), 2), 0, NO_Fm))
    return np.array(human_ROI), np.array(Object_ROI)


def H_O_BBOX(H_BOX, O_BOX):
     H_num = H_BOX.shape[0]
     H_O_ROI = []
     for i in range(H_num):
         H_i_fm, H_x1, H_y1, H_x2, H_y2 = H_BOX[i,:]
         for O_No in np.where(O_BOX[:,0]==H_i_fm)[0].tolist():
             _, O_x1, O_y1, O_x2, O_y2 = O_BOX[O_No, :]
             H_O_x1, H_O_y1, H_O_x2, H_O_y2  = min(H_x1,O_x1), min(H_y1, O_y1), max(H_x2, O_x2), max(H_y2, O_y2)
             H_O_ROI.append([H_i_fm, H_O_x1, H_O_y1, H_O_x2, H_O_y2])
     return np.array(H_O_ROI)


class Torch_ROI(nn.Module):
    def __init__(self, feature_scal=14):
        super(Torch_ROI, self).__init__()
        self.Adp_Avg_Pool = torch.nn.AdaptiveAvgPool2d((14, 14))
        self.Adp_Max_Pool = torch.nn.AdaptiveMaxPool2d((14, 14))
        self.scale = 1 / 16.0
        self.fea_scal = feature_scal

    def forward(self, tensor, ROI):
        ROI = (ROI.data.cpu().numpy()).tolist()
        out = []
        for Sg_ROI in ROI:
            fm_No = int(Sg_ROI[0])
            fm = torch.index_select(tensor, 0, Variable(torch.LongTensor([fm_No])).cuda())
            x1, y1, x2, y2 = self.get_Cord(np.array(Sg_ROI[1:]) * self.scale)
            # ROI_fea = fm[:, :, x1:x2, y1:y2].contiguous() # Bug
            ROI_fea = fm[:, :, y1:y2, x1:x2].contiguous()
            Pooled_feat = self.Adp_Avg_Pool(ROI_fea)
            out.append(Pooled_feat)
        final_out = torch.cat(out, 0)
        return final_out

    def get_Cord(self, float_ROI):
        x1, y1, x2, y2 = float_ROI
        x1 = max(math.floor(x1), 0)
        y1 = max(math.floor(y1), 0)
        x2 = min(math.ceil(x2), self.fea_scal)
        y2 = min(math.ceil(y2), self.fea_scal)  # I think using this function x1 < x2-1 and  y1 < y2-1
        return x1, y1, x2, y2


class RoI_layer(nn.Module):
    def __init__(self, out_size, phase, in_im_sz, fm_use):
        """Initializes RoI_layer module."""
        super(RoI_layer, self).__init__()

        self.phase = phase  # in order to get the RoI reigon
        self.out_size = out_size
        self.in_img_sz = in_im_sz
        self.tm_scale = 8
        self.fm_ROI = int(fm_use / 4)
        self.Dense_scale = int(self.tm_scale / 2)

        if phase == 'train':
            data_index_file = './data/Charades_train.pkl'
        elif phase == 'eval':
            data_index_file = './data/Charades_Val_Video.pkl'
        else:
            assert 0, 'The data can not find'
        self.bx_dir = '/VIDEO_DATA/BBOX/'
        self.data_index = pickle.load(open(data_index_file, 'rb'))  # in order to  get the bbox (RPN)

        # define rpn
        self.ROI_Align = RoIAlignAvg(out_size, out_size, 1 / 16.0)  # scale need to change

        self.ROI_Pool = _RoIPooling(out_size, out_size, 1 / 16.0)  # scale need to change

        self.Ptorch_ROI = Torch_ROI(feature_scal=(self.in_img_sz / 16))

        self.Scene_Roi = np.array([[i, 0, 0, self.in_img_sz - 32, self.in_img_sz - 32] for i in range(self.fm_ROI)])
        # 32 = scale * 2 = 16*2  for  ROI Align
        self.Scens_Full = np.array([[i, 0, 0, self.in_img_sz - 16, self.in_img_sz - 16] for i in range(self.fm_ROI)])
        self.Scens_Pytorch = np.array([[i, 0, 0, self.in_img_sz, self.in_img_sz] for i in range(self.fm_ROI)])
        self.Scens_Sparse = np.array([[i, 0, 0, self.in_img_sz, self.in_img_sz] for i in range(1, self.fm_ROI, 2)])

    def forward(self, input, BBox_info=None):
        batch_size = BBox_info.shape[0]
        # assert input.shape[0] ==batch_size, 'Bug'   # only used for test
        V_index = BBox_info.data.cpu().numpy().astype(np.int)
        batch_out = []
        for batch in range(batch_size):
            Each_bc = torch.index_select(input, 0, Variable(torch.LongTensor([batch])).cuda()).squeeze(0)
            VD_Batch = Each_bc.permute(1, 0, 2, 3).contiguous()

            # ----------init----------
            out_key = [1, 1, 1, 1]
            index_H = []
            index_O = []
            index_H_O = []
            # ----------init----------

            # -----------get vd info------------
            vid, *_ = self.data_index[V_index[batch, 0]]
            BBOX = load_bbox(self.bx_dir, vid, V_index[batch, 3:])  # 64 long list
            R_H_ROI, R_O_ROI = Comp_ROI(BBOX)  # original ratio: need to mul resize ratio
            IMG_H, IMG_W = V_index[batch, 1], V_index[batch, 2]  # sacle ratio
            rs_rt_H, rs_rt_W = np.round(self.in_img_sz / IMG_H, 3), np.round(self.in_img_sz / IMG_W, 3)
            V_info = [vid, V_index[batch, 3]]
            # -----------get vd info------------

            #  ---------------test -only test ---------------
            # -------ROI Align
            # V_S_ROI = Variable(torch.from_numpy(self.Scene_Roi).float().cuda())
            # S_node = self.ROI_Align(VD_Batch, V_S_ROI)
            # -------ROI Pooling
            # V_S_ROI = Variable(torch.from_numpy(self.Scens_Full).float().cuda())
            # S_node = self.ROI_Pool(VD_Batch, V_S_ROI)
            # -------pure mode
            # S_node = VD_Batch
            # -------Pytorch Version
            # V_S_ROI_ptch = Variable(torch.from_numpy(self.Scens_Pytorch).float().cuda())
            # S_node = self.Ptorch_ROI(VD_Batch, V_S_ROI_ptch)
            # H_Node = None
            # out_key[1] = 0
            # O_Node = None
            # out_key[2] = 0
            # H_O_Node = None
            # out_key[3] = 0
            #  ---------------test -only test ---------------

            #  ---------------Scene node--------------- select one to excute
            # ROI Align
            # V_S_ROI = Variable(torch.from_numpy(self.Scene_Roi).float().cuda())
            # S_node = self.ROI_Align(VD_Batch, V_S_ROI)
            # ROI Pooling
            # V_S_ROI = Variable(torch.from_numpy(self.Scens_Full).float().cuda())
            # S_node = self.ROI_Pool(VD_Batch, V_S_ROI)
            # Pytorch Version
            # V_S_ROI_ptch = Variable(torch.from_numpy(self.Scens_Pytorch).float().cuda())   # dense
            # V_S_ROI_ptch = Variable(torch.from_numpy(self.Scens_Sparse).float().cuda())     # sparse
            # S_node = self.Ptorch_ROI(VD_Batch, V_S_ROI_ptch)

            #  ------------- not using the Pytorch Pooling ------------
            index = np.array([i * 2 + 1 for i in range(int(self.fm_ROI / 2))])
            S_node = torch.index_select(VD_Batch, 0, Variable(torch.LongTensor(index)).cuda())
            #  ------------- not using the Pytorch Pooling ------------
            #  ---------------Scene node---------------

            # -----------Human node------------
            if len(R_H_ROI) > 0:
                H_ROI = np.round(R_H_ROI * [1, rs_rt_W, rs_rt_H, rs_rt_W, rs_rt_H], 3)
                H_zip_HOI = np.array(
                    [item for item in H_ROI.tolist() if (item[0] + 0.5 * self.tm_scale) % self.tm_scale == 0])
                index_H = [int(index[0] / self.Dense_scale) for index in H_zip_HOI]
                if len(index_H) > 0:
                    H_zip_HOI[:, 0] = index_H
                    V_H_ROI = Variable(torch.from_numpy(H_zip_HOI).float().cuda())  # Human ROI
                    # -----Faster R-CNN---
                    # H_Node = self.RCNN_roi_align(VD_Batch, V_H_ROI)
                    # -----Pytorch--------
                    H_Node = self.Ptorch_ROI(VD_Batch, V_H_ROI)
                else:
                    H_Node = None
                    out_key[1] = 0
            else:
                H_Node = None
                out_key[1] = 0
            # -----------Human node------------

            # -----------Object node-----------
            if len(R_O_ROI) > 0:
                O_ROI = np.round(R_O_ROI * [1, rs_rt_W, rs_rt_H, rs_rt_W, rs_rt_H], 3)
                O_zip_HOI = np.array(
                    [item for item in O_ROI.tolist() if (item[0] + 0.5 * self.tm_scale) % self.tm_scale == 0])
                index_O = [int(index[0] / self.Dense_scale) for index in O_zip_HOI]
                if len(index_O) > 0:
                    O_zip_HOI[:, 0] = index_O
                    V_O_ROI = Variable(torch.from_numpy(O_zip_HOI).float().cuda())  # Object ROI
                    # -----Faster R-CNN---
                    # O_Node = self.RCNN_roi_align(VD_Batch, V_O_ROI)
                    # -----Pytorch--------
                    O_Node = self.Ptorch_ROI(VD_Batch, V_O_ROI)
                else:
                    O_Node = None
                    out_key[2] = 0
            else:
                O_Node = None
                out_key[2] = 0
            # -----------Object node-----------

            # -----------Human_object node-----------
            if len(R_H_ROI) > 0 and len(R_O_ROI) > 0:
                H_O_ROI = H_O_BBOX(H_BOX=H_ROI, O_BOX=O_ROI)
                H_O_zip_HOI = np.array(
                    [item for item in H_O_ROI.tolist() if (item[0] + 0.5 * self.tm_scale) % self.tm_scale == 0])
                index_H_O = [int(index[0] / self.Dense_scale) for index in H_O_zip_HOI]
                if len(index_H_O) > 0:
                    H_O_zip_HOI[:, 0] = index_H_O
                    V_H_O_ROI = Variable(torch.from_numpy(H_O_zip_HOI).float().cuda())  # Union ROI
                    # -----Faster R-CNN---
                    # H_O_Node = self.RCNN_roi_align(VD_Batch, V_H_O_ROI)
                    # -----Pytorch--------
                    H_O_Node = self.Ptorch_ROI(VD_Batch, V_H_O_ROI)
                else:
                    H_O_Node = None
                    out_key[3] = 0
            else:
                H_O_Node = None
                out_key[3] = 0
            # -----------Human_object node-----------

            batch_out.append([S_node, H_Node, O_Node, H_O_Node, out_key, [index_H, index_O, index_H_O], V_info])

        return batch_out
        # return Variable(torch.from_numpy(np.array([0])).float().cuda())


class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name=' '):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5, phase='train',
                 in_size=320, fm_use=64):

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.Phase = phase
        self.IN_FM_Scale = in_size
        self.Fm_use = fm_use

        self.Conv3d_1a_7x7 = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                    stride=(2, 2, 2), padding=(3, 3, 3))

        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)

        self.Conv3d_2b_1x1 = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0)

        self.Conv3d_2c_3x3 = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1)

        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)

        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])

        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])

        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)

        self.Mixed_4b = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64])

        self.Mixed_4c = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64])

        self.Mixed_4d = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64])

        self.Mixed_4e = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64])

        self.Mixed_4f = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128])

        #----------I3D final predict -------------------------------
        # self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)  # original
        # self.Mixed_5b = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128])
        #
        # self.Mixed_5c = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128])
        #
        # self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
        #
        # self.dropout = nn.Dropout(dropout_keep_prob)
        #
        # self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes, kernel_shape=[1, 1, 1],
        #                      padding=0, activation_fn=None, use_batch_norm=False, use_bias=True)
        # ----------I3D final predict -------------------------------

        # ----------I3D resc4 dense out
        self.RoI_layer = RoI_layer(out_size=14, phase=self.Phase, in_im_sz=self.IN_FM_Scale, fm_use=self.Fm_use)

        self.Graph = Enhance_Graph(self.Fm_use)

        self.resC4 = ResNet_C4(Bottleneck, [3, 4, 23, 3])

        # ----------I3D resc4 dense out

        print('The I3D builded')

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes, kernel_shape=[1, 1, 1],
                             padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def I3D_predict(self, out_ROI):
        batch_size = len(out_ROI)
        batch_out = []
        for batch in out_ROI:
            scene_node = batch[0].permute(1, 0, 2, 3).contiguous().unsqueeze(0)
            vd_out = []
            for fm in range(int(self.Fm_use/4)):
                fm_tensor = torch.index_select(scene_node, 2, Variable(torch.LongTensor([fm])).cuda())
                fm_tensor = fm_tensor.repeat(1, 1, 8, 1, 1)
                sg_pred = self.MaxPool3d_5a_2x2(fm_tensor)
                sg_pred = self.Mixed_5b(sg_pred)
                sg_pred = self.Mixed_5c(sg_pred)
                sg_pred = self.avg_pool(sg_pred)
                sg_pred = self.dropout(sg_pred)
                sg_pred = self.logits(sg_pred)
                vd_out.append(sg_pred)
            vd_pred = torch.cat(vd_out, 2)
            batch_out.append(vd_pred)
        batch_pred = torch.cat(batch_out, 0 )
        return batch_pred
        
    def forward(self, x, BBox_info=None):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        out_roi = self.RoI_layer(x, BBox_info)

        # -------------resnet dense out predict ----

        x1 = self.Graph(out_roi)
        out = self.resC4(x1)
        logits = out.permute(0, 2, 1).contiguous()

        # -------------resnet dense out predict ----

        # -------------I3D suqeeze out predict ---

        # x = self.MaxPool3d_5a_2x2(x)
        # x = self.Mixed_5b(x)
        # x = self.Mixed_5c(x)
        # x = self.avg_pool(x)
        # x = self.dropout(x)
        # x = self.logits(x)
        # logits = x.squeeze(3).squeeze(3)

        # -------------I3D suqeeze out predict ---

        # -------------I3D dense out predict ---------
        # logits = self.I3D_predict(out_roi)
        # logits = logits.squeeze(3).squeeze(3)
        # -------------I3D dense out predict ---------

        return torch.nn.Sigmoid()(logits), logits
