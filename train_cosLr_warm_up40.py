import os
import sys
import argparse
import shutil
import torch
import torch.nn as nn
import torch.utils.data
from utils import videotransform
from data import Charades, Charades_VD
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
from model import InceptionI3d, resnext
from utils import map
import math

parser = argparse.ArgumentParser(description='Action Recognition')
#   data info
parser.add_argument('--dataname', default='Charades', type=str, help='Data name [Charades] ')

parser.add_argument('--version', default=22, type=int, help='version 1,2,3,4...')

parser.add_argument('--batch_size', default=2, type=int, help='5 for every GPU (NVIDIA 1080Ti)')

parser.add_argument('--fm_use', default=32, type=int, help='5 for every GPU (NVIDIA 1080Ti)')

parser.add_argument('--in_fm_sz', default=224, type=int, help='frame size in to net in order to cal scale for ROI')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# optim info

parser.add_argument('--weight-decay', '--wd', default=0.0000001, type=float, help='weight decay (default: 1e-7)')

parser.add_argument('--init_lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')

parser.add_argument('--epoch', default=10, type=int, help='epoch all ')

# net info
parser.add_argument('--arch', default='R3D', type=str, help='model architecture | (I3D Res3D)')

parser.add_argument('--dropout', default=0.50, type=float, help='dropout Prob (default: 0.3)')

parser.add_argument('--bk_loss_threshold', default=32, type=int, help='How much to new parameters')

parser.add_argument('--mAP', default=0, type=float, metavar='DIR to save pth')

args = parser.parse_args()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Make_Dir(args):
    version = args.version   # 1
    data_name = args.dataname  # str 'Charades'
    arch = args.arch  # I3D or R3D
    dir_name = os.path.join('./sv_model', str(version) + '_' + data_name + '_' + arch)

    train_log_dir = os.path.join(dir_name, 'trainlog')
    val_log_dir = os.path.join(dir_name, 'vallog')
    model_dir = os.path.join(dir_name, 'model')
    code_dir = os.path.join(dir_name, 'code')
    train_py = sys.argv[0]
    val_py = './val.py'
    data_py = './data/Charades.py'
    transforms_py = './utils/videotransform.py'
    I3D_net_model = './model/pytorch_i3d.py'
    Resnet_3D_model = './model/resnext.py'
    if os.path.exists(dir_name):
        print('The Dir exist Continue? ', dir_name)
        a = str.lower(input("Continue? YES or NO"))
        assert a =='yes', "Check the path Correctly"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        os.mkdir(train_log_dir)
        os.mkdir(val_log_dir)
        os.mkdir(model_dir)
        os.mkdir(code_dir)
    shutil.copyfile(train_py, os.path.join(code_dir, train_py.split('/')[-1]))
    shutil.copyfile(val_py, os.path.join(code_dir, 'val.py'))
    shutil.copyfile(data_py, os.path.join(code_dir, 'Charades.py'))
    shutil.copyfile(transforms_py, os.path.join(code_dir, 'videotransform.py'))
    shutil.copyfile(I3D_net_model, os.path.join(code_dir, 'pytorch_i3d.py'))
    shutil.copyfile(Resnet_3D_model, os.path.join(code_dir, 'resnext.py'))

    return dir_name, train_log_dir, model_dir


def save_log(args, lr, loss, epoch, base_dir):
    # txt_log_Dir = os.path.join(args.txtlog, args.dataname, args.arch + str(epoch)+'.txt')
    txt_log_Dir = os.path.join(base_dir, args.arch + str(epoch)+'.txt')
    file = open(txt_log_Dir, mode='w')
    for i in args.__dict__.keys():
        file.write(i +'\t' + str(args.__dict__[i])+'\n')
    file.write('lr   :' + str(lr)+'\n')
    file.write('Avg_loss   :' + str(loss) + '\n')
    file.close()
    print('The train log saved to :', txt_log_Dir)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warm_up(start_lr, end_lr, step, step_max=40):
    if step < step_max:
        lr = (step/step_max)*(end_lr-start_lr) + start_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('warm up ')


def Fix_block(model, keys):
    for name,value in model.named_parameters():
        if name.split('.')[0] in keys:
            value.requires_grad = False


def init_weight_bias(model, keys):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d) and name.split('.')[0] in keys:
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d) and name.split('.')[0] in keys :
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def lock_Bn(model, keys):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm3d) and name.split('.')[0] in keys:
            for p in m.parameters(): p.requires_grad = False
        if isinstance(m, nn.BatchNorm2d) and name.split('.')[0] in keys:
            for p in m.parameters(): p.requires_grad = False


def winsmooth(mat, ker_nl_clip=1, ker_clips=1):  # only sommth every 64 and finally to smooth 20th finaly

    # print('applying smoothing with kernelsize {}'.format(kernelsize))
    out_vids = []
    gts = []
    name_vds = []
    mat.keys()
    for No, V_id in enumerate(mat.keys()):
        # process each video

        # add this vd label and name
        label = mat[V_id]['label']
        gts.append(label)
        name_vds.append(V_id)
        # add this vd label and name

        # loop to process each video logist
        out_vid = []
        outs_vd = mat[V_id]['out']
        time_order = [st for _, st in outs_vd]
        time_sort = np.argsort(np.array(time_order))
        # process each clip
        for i in time_sort:
            clp_out = outs_vd[i][0].T    # every clip
            temp = outs_vd[i][0].T.copy()
            n = clp_out.shape[0]
            for frm in range(n):
                st = max(0, frm-ker_nl_clip)
                ed = min(n-1, frm + ker_nl_clip)
                clp_out[frm, :] = temp[st:ed+1,:].mean(0)
            # out_vid.append(clp_out.max(0))  # original
            out_vid.append(clp_out.max(0))  # V1
        out_vids.append(np.stack(out_vid).max(0))
    return out_vids, gts, name_vds


if __name__ == '__main__':

    # ===========some info not need to add into args =====================
    Charades_train_CSV = './data/Charades/Charades_v1_train.csv'  # this is needed
    Charades_val_CSV = './data/Charades/Charades_v1_test.csv'
    data_dir = '/VIDEO_DATA/Charades_v1_rgb'                      # dir store the rgb frames
    Bbox_dir = '/VIDEO_DATA/BBOX/'
    data_workers = 16
    momentum = 0.9
    print_freq = 10
    save_freq = 1
    # ===========some info not need to add into args =====================
    for i in args.__dict__.keys():
        print(i, ':\t', args.__dict__[i])

    if args.arch == 'I3D':
        I3D = InceptionI3d(400, in_channels=3, dropout_keep_prob=args.dropout, phase='train', in_size=args.in_fm_sz,
                           fm_use=args.fm_use)
        # I3D.replace_logits(157)  # Charades: 157   only used for load trained model
        # -----------load pre_train model -------- ------------
        Pre_tn_model = './model/pretred_mod/rgb_imagenet.pt'
        pretrained_dict = torch.load(Pre_tn_model)
        model_dict = I3D.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        I3D.load_state_dict(model_dict)
        I3D.replace_logits(157)  # used for load Knitices model
        Model = I3D
        train_trans = transforms.Compose([videotransform.IMG_resize(args.in_fm_sz, args.in_fm_sz),
                                          videotransform.ToTensor()])  # contain normalize
        # -----------load pre_train model --------------------
    elif args.arch == "R3D":
        R3D = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32, sample_size=args.in_fm_sz,
                                sample_duration=args.fm_use, phase='train', Dense_out=False)
        R3D.replace_logits(157)
        # -----------load pre_train model --------------------
        # Pre_tn_model = './model/pretred_mod/resnext-101-kinetics.pth'
        Pre_tn_model = './model/pretred_mod/resnext-101-64f-kinetics.pth'
        # Pre_tn_model = './model/pretred_mod/R3D025_Charades_model.pth'
        pretrained_dict = torch.load(Pre_tn_model)
        model_dict = R3D.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        R3D.load_state_dict(model_dict)
        Model = R3D
        train_trans = transforms.Compose([videotransform.IMG_resize(args.in_fm_sz, args.in_fm_sz),
                                          videotransform.Normalize_R3D([114.7748, 107.7354, 99.4750], [1, 1, 1])])
        # -----------load pre_train model --------------------

    # dir_name, train_log_dir, model_dir = Make_Dir(args)
    dir_name = './sv_model/14_Charades_R3D'
    train_log_dir = './sv_model/14_Charades_R3D/trainlog'
    model_dir = './sv_model/14_Charades_R3D/model'

    train_dataset = Charades(data_info_scv=Charades_train_CSV, data_root=data_dir, fm_use=args.fm_use,
                             mode='rgb', transforms=train_trans, bbox_dir=Bbox_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=data_workers, pin_memory=True)

    print('Find {} train samples '.format(len(train_dataset)))

    # ---------------------------------------fine tune ----------------------------------------------
    Model_block = list(Model._modules.keys())
    # ----------fix bn--------------
    fixed_bn = 0
    if fixed_bn:
        bn_key = Model_block[:8]                     # make sure the keys Check please !
        lock_Bn(Model, bn_key)
    # ----------fix bn--------------
    # -------fix parameters --------
    fixed_base_layer = 1
    if fixed_base_layer:
        fix_key = Model_block[:10]
        Fix_block(Model, fix_key)                       # make sure the keys Check please !
    # -------fix parameters --------
    # -----init some parameters-----
    init_later_layer = 0                                     # make sure the keys Check please !
    if init_later_layer:                                     # not to init the trained and fixed parameters
        init_key = Model_block[13:]
        init_weight_bias(Model, init_key)
    # -----init some parameters-----

    if fixed_bn ==1 or fixed_base_layer == 1:
        param = filter(lambda p: p.requires_grad, Model.parameters())
        optimizer = torch.optim.SGD(param, lr=args.init_lr, momentum=momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(Model.parameters(), lr=args.init_lr, momentum=momentum,
                                    weight_decay=args.weight_decay)
    # ---------------------------------------fine tune ----------------------------------------------

    Model = nn.DataParallel(Model).cuda()
    Model.train()
    bk_loss_threshold = args.bk_loss_threshold
    Epoch = len(train_loader)
    num_steps_per_update = int(bk_loss_threshold/args.batch_size)  # accum gradient
    epoch = 1
    warm = 0

    for steps in range(args.epoch):  # one epoch = 40   (8000/200)
        losses = AverageMeter()
        loss_loc = AverageMeter()
        loss_cls = AverageMeter()
        num_iter = 0
        lr = args.init_lr * 0.5 * (math.cos((steps/args.epoch)*math.pi)+1)
        adjust_learning_rate(optimizer, lr)
        print('This batch lr:\t', optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()

        out_dict = {}  # to test Map
        # for i, (input, target, data_info) in enumerate(train_loader):
        for i, (input, target, data_info, V_id, V_Gt) in enumerate(train_loader):
        # for i in range(len(train_loader)):  # test lr only

            num_iter += 1
            epoch += 1

            data_info = torch.autograd.Variable(data_info)
            input_var = torch.autograd.Variable(input.float().cuda(async=True))
            target_var = torch.autograd.Variable(target.float().cuda(async=True))
            t = input.size(2)

            # output, out_logits = I3D(input_var)
            output, out_logits = Model(input_var, data_info)
            C_logits = F.upsample(out_logits, t, mode='linear')

            loc_loss = torch.nn.BCEWithLogitsLoss()(C_logits, target_var)
            cls_loss = torch.nn.BCEWithLogitsLoss()(torch.max(C_logits, dim=2)[0], torch.max(target_var, dim=2)[0])

            loss = (0.52 * loc_loss + 0.48 * cls_loss)/num_steps_per_update
            loss.backward()

            losses.update(loss.data[0]*num_steps_per_update, input.size(0))
            loss_loc.update(loc_loss.data[0], input.size(0))
            loss_cls.update(cls_loss.data[0], input.size(0))

            # add to mAP test
            C_logits_d = C_logits.data.cpu().numpy()
            batch_sz = input.shape[0]
            for j in range(batch_sz):
                if V_id[j] in out_dict.keys():
                    out_dict[V_id[j]]['out'].append((C_logits_d[j, :, :], int(data_info.data.cpu().numpy()[j,3])))
                else:
                    out_dict[V_id[j]] = {'out': [], 'label': None}
                    out_dict[V_id[j]]['label'] = V_Gt[j, :, 0].cpu().numpy().astype(np.uint8)
                    out_dict[V_id[j]]['out'].append((C_logits_d[j, :, :],int(data_info.data.cpu().numpy()[j,3])))
            # add to mAP test

            if num_iter == num_steps_per_update:
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()
                # warm = warm + num_steps_per_update
                warm = warm + 1
                warm_up(0.0001, args.init_lr, warm, step_max=int((Epoch/num_steps_per_update)))  # just pass_
            if epoch % int(0.05*Epoch) == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                      'Loss_loc {loss_loc.val:.4f} ({loss_loc.avg:.4f})\t'
                      .format(round((epoch/Epoch),3), i, len(train_loader), loss=losses, loss_cls=loss_cls, loss_loc=loss_loc))
            if epoch % int(0.2 * Epoch) == 0:
                print('This batch lr:\t', optimizer.param_groups[0]['lr'])

        Map = winsmooth(out_dict)
        mAP, _, ap = map.charades_map(np.vstack(Map[0]), np.vstack(Map[1]))
        print('The final mAp is:', mAP)

        print('Lr: {:.5f}\t'.format(optimizer.param_groups[0]['lr']))
        checkpoint_name = "%03d_%s" % (int((epoch+1)/Epoch), args.dataname+"_model.pth")
        cur_path = os.path.join(model_dir, args.arch + checkpoint_name)
        torch.save(Model.module.state_dict(), cur_path)
        args.mAP = mAP
        save_log(args, optimizer.param_groups[0]['lr'], losses.avg, int((epoch+1)/Epoch), train_log_dir)


