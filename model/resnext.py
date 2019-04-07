import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
import pickle
import json
import sys
sys.path.insert(0, '..')
from lib.model.roi_align.modules.roi_align import RoIAlignAvg, RoIAlignMax
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
__all__ = ['ResNeXt', 'resnet50', 'resnet101']


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


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Torch_ROI(nn.Module):
    def __init__(self, feature_scal=14):
        super(Torch_ROI, self).__init__()
        self.Adp_Avg_Pool = torch.nn.AdaptiveAvgPool2d((14, 14))
        self.Adp_Max_Pool = torch.nn.AdaptiveMaxPool2d((14, 14))
        self.scale = 1/16.0
        self.fea_scal = feature_scal

    def forward(self, tensor, ROI):
        ROI = (ROI.data.cpu().numpy()).tolist()
        out = []
        for Sg_ROI in ROI:
            fm_No = int(Sg_ROI[0])
            fm = torch.index_select(tensor, 0, Variable(torch.LongTensor([fm_No])).cuda())
            x1, y1, x2, y2 = self.get_Cord(np.array(Sg_ROI[1:])*self.scale)
            # ROI_fea = fm[:, :, x1:x2, y1:y2].contiguous() # Bug
            ROI_fea = fm[:, :, y1:y2, x1:x2].contiguous()
            Pooled_feat = self.Adp_Avg_Pool(ROI_fea)
            out.append(Pooled_feat)
        final_out = torch.cat(out, 0)
        return final_out

    def get_Cord(self,float_ROI):
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
        self.Dense_scale = int(self.tm_scale/2)

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
        self.Scens_Sparse = np.array([[i, 0, 0, self.in_img_sz, self.in_img_sz] for i in range(1,self.fm_ROI,2)])

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
            index = np.array([i*2 + 1 for i in range(int(self.fm_ROI/2))])
            S_node = torch.index_select(VD_Batch, 0, Variable(torch.LongTensor(index)).cuda())
            #  ------------- not using the Pytorch Pooling ------------
            #  ---------------Scene node---------------

            # -----------Human node------------
            if len(R_H_ROI) > 0:
                H_ROI = np.round(R_H_ROI * [1, rs_rt_W, rs_rt_H, rs_rt_W, rs_rt_H], 3)
                H_zip_HOI = np.array([item for item in H_ROI.tolist() if (item[0] + 0.5*self.tm_scale) % self.tm_scale == 0])
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
                O_zip_HOI = np.array([item for item in O_ROI.tolist() if (item[0] + 0.5*self.tm_scale) % self.tm_scale == 0])
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
                H_O_zip_HOI = np.array([item for item in H_O_ROI.tolist() if (item[0] + 0.5*self.tm_scale) % self.tm_scale == 0])
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


class Down_Size(nn.Module):
    def __init__(self):
        """Initializes RoI_layer module."""
        super(Down_Size, self).__init__()
        self.Down_Conv = torch.nn.Conv2d(1024, 512, (3, 3), 2, 1, bias=False)
        self.node_Bn = torch.nn.BatchNorm2d(512)
        self.Relu = torch.nn.ReLU()

    def forward(self, nodes):
        batch_size = len(nodes)
        out = []
        for batch in nodes:
            scene_node, hum_node, object_node, H_O_node, key, fm_ex = batch[0], batch[1], batch[2], batch[3], batch[4], \
                                                                      batch[5]
            # '[S_node, H_Node, O_Node, H_O_Node]'
            # S_nodes = self.Relu(self.node_Bn(self.Down_Conv(scene_node)))
            S_nodes = self.Down_Conv(scene_node)
            S_nodes = self.node_Bn(S_nodes)
            S_nodes = self.Relu(S_nodes)

            # -------rest only ---------
            H_nodes = None
            O_nodes = None
            H_O_edges = None
            # -------rest only ---------

            '''
            if key[1]:
                H_nodes = self.Relu(self.node_Bn(self.Down_Conv(hum_node)))
            else:
                H_nodes = None
            if key[2]:
                O_nodes = self.Relu(self.node_Bn(self.Down_Conv(object_node)))
            else:
                O_nodes = None
            if key[3]:
                H_O_edges = self.Relu(self.node_Bn(self.Down_Conv(H_O_node)))
            else:
                H_O_edges = None

            '''

            out.append([S_nodes, H_nodes, O_nodes, H_O_edges, key, fm_ex])
        return out


class Enhance_Graph(nn.Module):
    def __init__(self, fm_use):
        """Initializes RoI_layer module."""
        super(Enhance_Graph, self).__init__()
        self.Fm_use = fm_use
        # self.Down_size = Down_Size()
        self.inter_step = 3

    def forward(self, nodes):
        # nodes_edges = self.Down_size(nodes)  # contain many batches :2
        nodes_edges = nodes
        test_Scene_batch = []
        for batch in nodes_edges:  # vd : contain many frames woneed to class each fm

            S_nodes, H_nodes, O_nodes, H_O_edges, key, fm_index,_ = batch
            test_out_S_node = []
            '''
            for fm_No in range(int(self.Fm_use/4)):
                fm_S_N = torch.index_select(S_nodes, 0, Variable(torch.LongTensor([fm_No])).cuda())
                fm_H_N = []
                fm_O_N = []
                fm_H_O_E = []


                # Human node in frame  may mulity prople
                if len(np.where(np.array(fm_index[0])==fm_No)[0]):  # exist people
                    for num in np.where(np.array(fm_index[0])==fm_No)[0]:
                        fm_H_N.append(torch.index_select(H_nodes, 0, Variable(torch.LongTensor([int(num)]).cuda())))

                # object node in frame may mulity object
                if len(np.where(np.array(fm_index[1]) == fm_No)[0]):  # exist object
                    for num in np.where(np.array(fm_index[1]) == fm_No)[0]:
                        fm_O_N.append(torch.index_select(O_nodes, 0, Variable(torch.LongTensor([int(num)]).cuda())))

                # object human edge in frame may mulity object human edges
                if len(np.where(np.array(fm_index[2]) == fm_No)[0]):  # exist object human edge
                    for num in np.where(np.array(fm_index[2]) == fm_No)[0]:
                        fm_H_O_E.append(torch.index_select(H_O_edges, 0, Variable(torch.LongTensor([int(num)])).cuda()))

                # print(1)

                test_out_S_node.append(fm_S_N)
            test_vd_S = torch.cat(test_out_S_node, 0).unsqueeze(0)
            # test_Scene_batch.append(test_vd_S)
            '''
            test_vd_S = S_nodes.unsqueeze(0)  # changed
            test_Scene_batch.append(test_vd_S)

        test_out_batch = torch.cat(test_Scene_batch, 0)
        return test_out_batch


class Bottleneck(nn.Module):
    expansion = 2
    groups = 32
    # expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        # self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, groups=self.groups,
                               padding=1, bias=False )
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        groups=self.groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_C4(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet_C4, self).__init__()
        # self.inplanes = 512
        self.inplanes = 1024
        self.layerC5 = self._make_layer(block, 1024, layers[3], stride=2)  # change resnet expansion
        # self.layerC5 = self._make_layer(block, 512, layers[3], stride=2)
        self.drop = torch.nn.Dropout()
        self.cls = torch.nn.Linear(2048, 157, True)
        self.Avg_Pool = nn.AvgPool2d((7, 7),stride=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # batch_size = x.shape[0]
        # out = []
        # for i in range(batch_size):
        #     vd = torch.index_select(x, 0, Variable(torch.LongTensor([i])).cuda()).squeeze(0)
        #     vd = vd.permute(1, 0, 2, 3).contiguous()
        #     vd_out = self.layerC5(vd).mean(2).mean(2)
        #     vd_out = self.drop(vd_out)
        #     vd_out = self.cls(vd_out)
        #     vid_out = vd_out.unsqueeze(0)
        #     out.append(vid_out)
        # final_out = torch.cat(out, 0)
        # return final_out
        # ----------V1 --------
        # out = self.layerC5(x)
        # out = out.mean(2).mean(2)
        # ----------V1 --------
        # ----------V2 --------
        out = self.layerC5(x)
        out = self.Avg_Pool(out)
        out = self.drop(out)
        out = out.squeeze(2).squeeze(2)
        # ----------V2 --------

        return out


class Dropout(nn.Module):
    def __init__(self, fm=4):
        super(Dropout, self).__init__()
        self.drop = nn.Dropout()
        self.fm = fm

    def forward(self, x):
        batch_size = x.shape[0]
        out = []
        for batch in range(batch_size):
            batch_out = []
            Each_bc = torch.index_select(x, 0, Variable(torch.LongTensor([batch])).cuda())
            for fm in range(self.fm):
                fm_tensor = torch.index_select(Each_bc, 2, Variable(torch.LongTensor([fm])).cuda())
                batch_out.append(self.drop(fm_tensor))
            out.append(torch.cat(batch_out, 2))
        final = torch.cat(out, 0)
        return final


class Message(nn.Module):
    def __init__(self, feat_size=2048):
        super(Message, self).__init__()
        self.feat_size = feat_size
        self.Liner_node = nn.Linear(self.feat_size, int(self.feat_size*0.5), bias=True)
        self.Liner_edge = nn.Linear(self.feat_size, int(self.feat_size*0.5), bias=True)

    def forward(self, node, edge):
        Message_node = self.Liner_node(node)
        Message_edge = self.Liner_edge(edge)
        Message = torch.cat([Message_edge, Message_node], 1)
        return Message


class Pred_Fn(nn.Module):
    def __init__(self, ):
        super(Pred_Fn, self).__init__()
        self.class_num = 157
        self.In_channal = 2048
        self.Logist = nn.Linear(self.In_channal, self.class_num, bias=True)

    def forward(self, input):
        x = self.Logist(x)
        return x


class O_H_LinK(nn.Module):
    def __init__(self, ):
        super(O_H_LinK, self).__init__()
        self.channal = 2048
        self.L1 = nn.Linear(self.channal, int(0.5*self.channal), bias=True)
        self.L2 = nn.Linear(int(0.5*self.channal), 1, bias=True)
        self.ReLu = nn.ReLU()
        self.Softmax = nn.Softmax(0)

    def forward(self, edge):
        input = edge
        x = self.L1(input)
        x = self.ReLu(x)
        x = self.L2(x)
        x = self.Softmax(x)
        return x


class Update_H(nn.Module):
    def __init__(self, Msg_Size=2048, Node_Size=2048):
        super(Update_H, self).__init__()
        self.message_size = Msg_Size
        self.node_size = Node_Size
        self.num_layer = 1
        self.Drop = False
        self.GRU = nn.GRU(self.message_size, self.node_size, num_layers=self.num_layer,
                          bias=True, dropout=self.Drop)

    def forward(self, message, H_O):
        Msg = message.unsqueeze(0).contiguous()
        H = H_O.unsqueeze(0).contiguous()
        out, H_t = self.GRU(Msg, H)
        return H_t.squeeze(0)


class Update_S(nn.Module):
    def __init__(self, Mag_Size=2048, Node_size=2048):
        super(Update_S, self).__init__()
        self.message_size = Mag_Size
        self.node_size = Node_size
        self.num_layer = 1
        self.Drop = False
        self.GRU = nn.GRU(self.message_size, self.node_size, num_layers=self.num_layer,
                          bias=True, dropout=self.Drop)

    def forward(self, s, S):
        s_ = s.unsqueeze(0)
        S_ = S.unsqueeze(0)
        out, S_t = self.GRU(s_, S_)
        return S_t


class Graph_Enhance_model(nn.Module):
    def __init__(self, fm_duration):
        super(Graph_Enhance_model, self).__init__()
        self.O_to_link = O_H_LinK()
        self.Message_Fn = Message(feat_size=2048)
        self.Update_H_Fn = Update_H(Msg_Size=2048, Node_Size=2048)
        self.Update_S_Fn = Update_S(Mag_Size=2048, Node_size=2048)
        self.Predict_Fn = Pred_Fn()
        self.propagation_step = 3
        self.fm_num = fm_duration  # used for 32  # BUG  this here
        self.Stride = 2

    def forward(self, Aulix_node, final_S_node):
        # def_message_key = [1, 1, 1, 1]  # four type of key to test which get best performance
        # process each batch one by one
        batch_out = []
        for ba_No, batch in enumerate(Aulix_node):
            # each batch is a video

            # ready node info
            S_node_C4, H_nodes, O_nodes, H_O_edges, out_key, nodes_num_info, V_info = batch
            S_base_nodes = torch.index_select(final_S_node, 0, Variable(torch.LongTensor([ba_No])).cuda()).squeeze()
            # ready node info

            vd_cls = []
            # fm_num = len(nodes_num_info[0])
            for fm_No in range(self.fm_num):
                s_C4_node = torch.index_select(S_node_C4, 0, Variable(torch.LongTensor([fm_No])).cuda()).squeeze()
                S_f_node = torch.index_select(S_base_nodes, 1, Variable(torch.LongTensor([fm_No])).cuda()).squeeze()

                idx = fm_No * self.Stride + 1   # need to change
                fm_key = [1, 0, 0, 0]  # whether exist this type node
                n_info = [1, 0, 0, 0]  # count each node num

                # search H_nodes
                if out_key[1] and idx in nodes_num_info[0]: # this fm exist human node
                    H_num, H_st = nodes_num_info[0].count(idx), nodes_num_info[0].index(idx)
                    slect_idx = [H_st + i for i in range(H_num)]
                    H_node = torch.index_select(H_nodes, 0, Variable(torch.LongTensor(slect_idx)).cuda())
                    fm_key[1] = 1
                    n_info[1] = H_num

                # search O_nodes
                if out_key[2] and idx in nodes_num_info[1]:  # exist object
                    O_num, O_st = nodes_num_info[1].count(idx), nodes_num_info[1].index(idx)
                    slect_idx = [O_st + i for i in range(O_num)]
                    O_node = torch.index_select(O_nodes, 0, Variable(torch.LongTensor(slect_idx)).cuda())
                    fm_key[2] = 1
                    n_info[2] = O_num

                # search Edge between human and Object
                if out_key[3] and idx in nodes_num_info[2]:  # exist edges
                    E_num, E_st = nodes_num_info[2].count(idx), nodes_num_info[2].index(idx)
                    slect_idx = [E_st + i for i in range(E_num)]
                    Edeg = torch.index_select(H_O_edges, 0, Variable(torch.LongTensor(slect_idx)).cuda())
                    fm_key[3] = 1
                    n_info[3] = E_num

                # to process graph
                # --------hidden state ready ----------
                hidden_S_C4 = [s_C4_node.clone() for _ in range(self.propagation_step)]
                hidden_S = [S_f_node.clone() for _ in range(self.propagation_step)]
                if fm_key[1] == 1:  # exist human
                    hidden_H = [H_node.clone() for _ in range(self.propagation_step)]
                if fm_key[2] == 1:  # exist object
                    O_node = O_node   # O node not need to update
                if fm_key[3] == 1:  # exist edge
                    hidden_Edeg = [Edeg.clone() for _ in range(self.propagation_step)]
                # if fm_key[2] == 1:  # exist object
                #     hidden_O = [O_node.clone() for _ in range(self.propagation_step)]
                # --------hidden state ready ----------

                '''
                # for step in range(self.propagation_step):
                if fm_key == [1, 1, 1, 1]:  # exist human object edge
                    for step in range(self.propagation_step-1):
                    # O-> P -> s -> S
                        M_Coll = []
                        H_Coll = []
                        for human_No in range(n_info[1]):  # this  fm human num
                            O_num = n_info[2]
                            edge_index = [human_No*O_num+i for i in range(O_num)] # the edge to this human
                            edge = torch.index_select(hidden_Edeg[step], 0, Variable(torch.LongTensor(edge_index)).cuda())
                            human = torch.index_select(hidden_H[step], 0, Variable(torch.LongTensor([human_No])).cuda())
                            Msg_weig = self.O_to_link(edge)
                            # Massage = self.Message_Fn(hidden_O[step], edge)  # i think object node not need t oupdate
                            Message = self.Message_Fn(O_node, edge)
                            Updated_Message = Msg_weig * Message
                            M_Sum = Updated_Message.mean(0).unsqueeze(0)
                            Human_updated= self.Update_H_Fn(M_Sum, human)
                            M_Coll.append(Updated_Message)
                            H_Coll.append(Human_updated)
                        New_Edge = torch.cat(M_Coll, 0)
                        All_H_update= torch.cat(H_Coll, 0)
                        hidden_Edeg[step + 1] = New_Edge    # GPNN  Chun-song Zhu
                        hidden_H[step] = All_H_update  # GPNN
                        # hidden_H[step + 1] = All_H_update   # My
                        # hidden_Edeg[step] = New_Edge        # My

                    All_human = hidden_H[-2].mean(0).unsqueeze(0)
                    hidden_S_C4_out =self.Update_S_Fn(All_human, hidden_S_C4[0].unsqueeze(0))
                    S_node_cls = self.Update_S_Fn(hidden_S_C4_out.squeeze(0), hidden_S[0].unsqueeze(0))
                    S_node_cls = S_node_cls.squeeze().unsqueeze(0)
                elif fm_key == [1, 1, 0, 0]:  # exist human only
                    # P -> s -> S
                    All_human = hidden_H[-1].mean(0).unsqueeze(0)
                    hidden_S_C4_out = self.Update_S_Fn(All_human, hidden_S_C4[0].unsqueeze(0))
                    S_node_cls = self.Update_S_Fn(hidden_S_C4_out.squeeze(0), hidden_S[0].unsqueeze(0))
                    S_node_cls = S_node_cls.squeeze().unsqueeze(0)
                elif out_key == [1, 0, 1, 0]:  # only exist Object nodes
                    # O -> s -> S
                    S_node_cls = self.Update_S_Fn(hidden_S_C4[0].unsqueeze(0), hidden_S[0].unsqueeze(0))
                    S_node_cls = S_node_cls.squeeze().unsqueeze(0)
                    # I think no nothing  only object have no info
                else:
                    S_node_cls = hidden_S[0]
                    S_node_cls = S_node_cls.squeeze().unsqueeze(0)
                '''

                # test V1 if s node can Up S performance ? GRU model  -----
                S_node_cls = self.Update_S_Fn(hidden_S_C4[0].unsqueeze(0), hidden_S[0].unsqueeze(0))
                S_node_cls = S_node_cls.squeeze().unsqueeze(0)
                # test one if s node can Up S performance ? GRU model  -----
                vd_cls.append(S_node_cls)
            vd_out = torch.cat(vd_cls, 0).unsqueeze(0)
            batch_out.append(vd_out)
        batch_final = torch.cat(batch_out, 0)
        return batch_final


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration,
                 shortcut_type='B', cardinality=32, num_classes=400, dropout_keep_prob=0.5,
                 phase='val', Dense_out=False):
        self.inplanes = 64
        super(ResNeXt, self).__init__()

        self.Phase = phase
        self.IN_FM_Scale = sample_size
        self.Fm_use = sample_duration
        self.Dense_out = Dense_out

        last_duration = int(math.ceil(sample_duration / 8))
        last_size = int(math.ceil(sample_size / 32))
        self.last_duration = last_duration
        self.last_size = last_size

        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)

        if self.Dense_out:
            self.Change_Conv3D_Pad_stride()

        self.avgpool = nn.AvgPool3d((1, last_size, last_size), stride=1)

        self.logits = torch.nn.Conv3d(2048, 400, (1, 1, 1), 1, 0, bias=True)

        # ------------------my test--------------------------------
        # self.Maxpool = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        # self.Avgpool_after_Max = nn.AvgPool3d((1, 3, 3), stride=1)
        # self.dropout = nn.Dropout(dropout_keep_prob)
        # ------------------my test--------------------------------
        # self.drop = Dropout(fm=last_duration)
        # self.Linear = nn.Linear(2048, 157, bias=True)

        # ----------R3D resc4 dense out
        # self.RoI_layer = RoI_layer(out_size=14, phase=self.Phase, in_im_sz=self.IN_FM_Scale, fm_use=self.Fm_use)
        # self.Graph = Enhance_Graph(self.Fm_use)
        # self.resC4 = ResNet_C4(Bottleneck, [3, 4, 23, 3])  # original !
        # self.resC4 = ResNet_C4(Bottleneck, [3, 4, 23, 4])
        # ----------R3D resc4 dense out

        # ---------graph model------------
        # self.graph = Graph_Enhance_model(fm_duration=last_duration)
        # ---------graph model------------

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def replace_logits(self, num_classes):
        self.logits = torch.nn.Conv3d(2048, num_classes, (1, 1, 1), 1, 0, bias=True)

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                if planes==1024 and blocks == 3 and self.Dense_out:  # Added by Haoliang Tan Only used for layer 4
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1,
                                  stride=(2, stride, stride), bias=False),
                        nn.BatchNorm3d(planes * block.expansion))
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1,
                                  stride=stride, bias=False),
                        nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def Change_Conv3D_Pad_stride(self):
        self.layer4[0].conv2 = torch.nn.Conv3d(1024, 1024, kernel_size=(3, 3, 3),
                                               stride=(1, 2, 2), padding=(0, 1, 1), groups=32, bias=False)

    def forward(self, x, BBox_info=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # -----------original resnext---------------
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.logits(x)
        logits = x.squeeze(3).squeeze(3)
        # -----------original resnext---------------
        return torch.nn.Sigmoid()(logits), logits


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
