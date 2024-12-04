# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml



###############################################################################################
###############################################################################################

# '''
#     anchor based model
# '''

# device = select_device('1', batch_size=1)
# # model trained by cfg/training/*.yaml
# ckpt = torch.load('cfg/ckpt/VGPD_ab.pt', map_location=device)
# # reparameterized model in cfg/deploy/*.yaml
# model = Model('cfg/deploy/VGPD_ab.yaml', ch=3, nc=1).to(device)

# with open('cfg/deploy/VGPD_ab.yaml') as f:
#     yml = yaml.load(f, Loader=yaml.SafeLoader)
# anchors = len(yml['anchors'][0]) // 2

# # copy intersect weights
# state_dict = ckpt['model'].float().state_dict()
# exclude = []
# intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
# model.load_state_dict(intersect_state_dict, strict=False)
# model.names = ckpt['model'].names
# model.nc = ckpt['model'].nc

# idx = 149
# idx2 = 152

# # copy weights of lead head
# model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
# model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
# model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
# model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
# model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
# model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
# model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
# model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
# model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
# model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
# model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
# model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data


# # reparametrized YOLOR
# for i in range((model.nc+5)*anchors):
#     model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
#     model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
#     model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
# model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
# model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
# model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
# model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
# model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
# model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()


# # model to be saved
# ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
#         'optimizer': None,
#         'training_results': None,
#         'epoch': -1}

# # save reparameterized model
# torch.save(ckpt, 'cfg/Thesis_deploy/ckpt/SC_k3_L_ASFF_PAN_aux_Caltech_Finetune.pt')

###############################################################################################
###############################################################################################

# '''
#     anchor free model
# '''

# device = select_device('0', batch_size=1)
# # model trained by cfg/training/*.yaml
# ckpt = torch.load('cfg/ckpt/VGPD_af.pt', map_location=device)
# # reparameterized model in cfg/deploy/*.yaml
# model = Model('cfg/deploy/VGPD_af.yaml', ch=3, nc=1).to(device)


# # copy intersect weights
# state_dict = ckpt['model'].float().state_dict()
# exclude = []
# intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
# model.load_state_dict(intersect_state_dict, strict=False)
# model.names = ckpt['model'].names
# model.nc = ckpt['model'].nc

# idx = 149
# idx2 = 152

# # copy weights of lead head
# for k, v in state_dict.items():
#     layer = int(k.split('.')[1])
#     if layer == idx2 and k.split('.')[2] in ['cv2', 'cv3', 'dfl']:
#         print(f'{k.replace(str(idx2), str(idx))} -> {k}')
#         model.state_dict()[k.replace(str(idx2), str(idx))].data -=  model.state_dict()[k.replace(str(idx2), str(idx))].data
#         model.state_dict()[k.replace(str(idx2), str(idx))].data +=  state_dict[k].data


# # model to be saved
# ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
#         'optimizer': None,
#         'training_results': None,
#         'epoch': -1}

# # save reparameterized model
# torch.save(ckpt, 'cfg/deploy/VGPD_af.pt')

########################################################################
# '''
#     to visualization visible box
# '''
# device = select_device('0', batch_size=1)
# # model trained by cfg/training/*.yaml
# ckpt = torch.load('cfg/ckpt/VGPD_ab_Vis.pt', map_location=device)
# # reparameterized model in cfg/deploy/*.yaml
# model = Model('cfg/deploy/VGPD_ab_Vis.yaml', ch=3, nc=1).to(device)


# # copy intersect weights
# state_dict = ckpt['model'].float().state_dict()
# exclude = []
# intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
# model.load_state_dict(intersect_state_dict, strict=True)
# model.names = ckpt['model'].names
# model.nc = ckpt['model'].nc

# # model to be saved
# ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
#         'optimizer': None,
#         'training_results': None,
#         'epoch': -1}

# # save reparameterized model
# torch.save(ckpt, 'cfg/deploy/VGPD_Vis.pt')