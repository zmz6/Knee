import torch
from network.head_scn_net import Net

model = Net(filter=128, num_class=28)

model_dict = model.state_dict()
# for k, v in model_dict.items():
#     print(k)

pretrained_dict = torch.load('/home/liushenyao/code/box/model/hip_box/Fully_Supervised_iter_copy_10_labeled/scn_size[128, 128]_b1_epoch300_lossL1/scn_best_model.pth')

print(pretrained_dict)