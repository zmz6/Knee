#  将两位医生的数据与徐声明的金标准进行对比


import argparse
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataloader.single_dataset import RandomGenerator, TestDataSets
from network.net_factory import net_factory
from val_unet import test_single_volume, test_single_angel_volume, \
    test_single_down_angel_volume, test_single_depth_volume, get_max_preds, judge

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_classes = 5
    path_size = [256, 256]
    root_path = 'D:/code/Knee/data/knee'

    state = "test"

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../data/knee")
    parser.add_argument("--save_result_path", type=str, default=state)
    args = parser.parse_args()

    best_mode_path = "D:\code\Knee\model\\all_data_swap\Fully_Supervised_iter_cv3\\nl_unet_size[256, 256]_b4_iter25000_lossMSE_dropout0.5_sigma3\\nl_unet_best_model.pth"
    # best_mode_path = os.path.join(snapshot_path, 'unet_best_model.pth')
    print(best_mode_path)
    test_transform = RandomGenerator(output_size=path_size,
                                     downsample=1,
                                     sigma=3,
                                     state='val'
                                     )

    db_test = TestDataSets(base_dir=root_path, split=state, num=None, transform=test_transform, cv=3,
                           num_class=num_classes)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                            num_workers=5)


    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type='nl_unet', in_chns=1, class_num=num_classes, dropout=0.5)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model()

    model.load_state_dict(torch.load(best_mode_path))
    model.eval()

    all_index = list()
    wrong_img = set()
    pred_wrong_img = set()

    dict_doctor_gt_3cm = {}
    dict_doctor_gt_normal = {}

    dict_rookie_3cm = {}
    dict_rookie_normal = {}

    dict_rookie1_3cm = {}
    dict_rookie1_normal = {}

    dict_rookie2_3cm = {}
    dict_rookie2_normal = {}

    for i_batch, sampled_batch in enumerate(testloader):
        image = sampled_batch['image'].cuda()
        rookie = sampled_batch['ori_kp'][0].cpu().numpy()
        rookie1 = sampled_batch['ori_1'][0].cpu().numpy()
        rookie2 = sampled_batch['ori_2'][0].cpu().numpy()
        doctor = sampled_batch['gt_kp'][0].cpu().numpy()
        ori_h = sampled_batch['ori_h'].cpu().numpy()
        ori_w = sampled_batch['ori_w'].cpu().numpy()
        img_index = sampled_batch['img_index'][0]
        target_heatmap = sampled_batch["target_hm"].cuda()
        space = sampled_batch['space'].cpu().numpy()

        if "3cm" in img_index:
            if img_index[:-3] not in all_index:
                all_index.append(img_index[:-3])

            rookie_3cm = judge(rookie, space, type="3cm")
            doctor_3cm = judge(doctor, space, type="3cm")
            rookie1_3cm = judge(rookie1, space, type="3cm")
            rookie2_3cm = judge(rookie2, space, type="3cm")

            dict_doctor_gt_3cm[img_index] = doctor_3cm
            dict_rookie_3cm[img_index] = rookie_3cm
            dict_rookie1_3cm[img_index] = rookie1_3cm
            dict_rookie2_3cm[img_index] = rookie2_3cm

        else:
            if img_index not in all_index:
                all_index.append(img_index)
            rookie_normal = judge(rookie, space, type="ori")
            doctor_normal = judge(doctor, space, type="ori")
            rookie1_normal = judge(rookie1, space, type="ori")
            rookie2_normal = judge(rookie2, space, type="ori")

            dict_doctor_gt_normal[img_index] = doctor_normal
            dict_rookie_normal[img_index] = rookie_normal
            dict_rookie1_normal[img_index] = rookie1_normal
            dict_rookie2_normal[img_index] = rookie2_normal

    fold = "result"
    rookie_name = "wang"
    doctor_name = "xu"
    all_index.sort()

    with open(os.path.join(fold, rookie_name + "_3cm.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{} 3cm:{}\n".format(v, dict_rookie_3cm[v + "3cm"]))

    with open(os.path.join(fold, rookie_name + ".txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{}:{}\n".format(v, dict_rookie_normal[v]))

    ######################################################################

    with open(os.path.join(fold, doctor_name + "_3cm.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{} 3cm:{}\n".format(v, dict_doctor_gt_3cm[v + "3cm"]))

    with open(os.path.join(fold, doctor_name + ".txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{}:{}\n".format(v, dict_doctor_gt_normal[v]))

    #######################################################################
    with open(os.path.join(fold, rookie_name + "1_3cm.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{} 3cm:{}\n".format(v, dict_rookie1_3cm[v + "3cm"]))

    with open(os.path.join(fold, rookie_name + "1.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{}:{}\n".format(v, dict_rookie1_normal[v]))

    ########################################################################

    with open(os.path.join(fold, rookie_name + "2_3cm.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{} 3cm:{}\n".format(v, dict_rookie2_3cm[v + "3cm"]))

    with open(os.path.join(fold, rookie_name + "2.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{}:{}\n".format(v, dict_rookie2_normal[v]))
