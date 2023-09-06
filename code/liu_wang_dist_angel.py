import argparse
import os
import torch
import numpy as np
import pandas as pd
import math
import xlwt
import xlrd

from torch.utils.data import DataLoader
from dataloader.single_dataset import RandomGenerator, TestDataSets
from network.net_factory import net_factory
from val_unet import test_single_volume, test_single_angel_volume, \
    test_single_down_angel_volume, test_single_depth_volume, get_max_preds, judge

def get_foot_point(a, b, c):
    da = a[1] - b[1]
    db = b[0] - a[0]
    dc = -da * a[0] - db * a[1]
    return (
        (db * db * c[0] - da * db * c[1] - da * dc) / (da * da + db * db),
        (da * da * c[1] - da * db * c[0] - db * dc) / (da * da + db * db)
    )


def get_dist_angle(landmarks, space, type="3cm"):
    if type == "3cm":
        foot1 = get_foot_point(landmarks[3], landmarks[4], landmarks[0])
        foot2 = get_foot_point(landmarks[3], landmarks[4], landmarks[1])
        foot3 = get_foot_point(landmarks[3], landmarks[4], landmarks[2])

        a = np.sqrt((foot1[0] - landmarks[0][0]) ** 2 +
                    (foot1[1] - landmarks[0][1]) ** 2) * space
        c = np.sqrt((foot2[0] - landmarks[1][0]) ** 2 +
                    (foot2[1] - landmarks[1][1]) ** 2) * space
        b = np.sqrt((foot3[0] - landmarks[2][0]) ** 2 +
                    (foot3[1] - landmarks[2][1]) ** 2) * space
        dist = (a + b) / 2 - c

        # print(dist)

        d = np.sqrt((landmarks[0][0] - landmarks[1][0]) ** 2 +
                    (landmarks[0][1] - landmarks[1][1]) ** 2) * space

        e = np.sqrt((landmarks[1][0] - landmarks[2][0]) ** 2 +
                    (landmarks[1][1] - landmarks[2][1]) ** 2) * space

        angle = e / d

        return dist.item(), angle.item()

    elif type == "ori":
        foot1 = get_foot_point(landmarks[3], landmarks[4], landmarks[0])
        foot2 = get_foot_point(landmarks[0], foot1, landmarks[1])

        vec_2_1 = landmarks[0] - landmarks[1]
        vec_2_foot2 = foot2 - landmarks[1]

        vec_2_1 = torch.from_numpy(vec_2_1)
        vec_2_foot2 = torch.from_numpy(vec_2_foot2)

        angel = torch.acos(torch.nn.functional.cosine_similarity(vec_2_1, vec_2_foot2, dim=0)) * (180 / math.pi)

        # print(angel)

        return angel.item()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_classes = 5
    path_size = [256, 256]
    root_path = 'D:/code/Knee/data/knee'
    savepath = 'D:/code/Knee/code/result/王距离角度.xls'

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

    all_datalist_3cm = []
    all_datalist_ori = []

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
            datalist_3cm = []
            datalist_3cm.append(img_index)
            # rookie_3cm = judge(rookie, space, type="3cm")
            # doctor_3cm_dist, doctor_3cm_angel = get_dist_angle(doctor, space, type="3cm")
            rookie1_3cm_dist, rookie1_3cm_angel = get_dist_angle(rookie1, space, type="3cm")
            rookie2_3cm_dist, rookie2_3cm_angel = get_dist_angle(rookie2, space, type="3cm")
            datalist_3cm.append(rookie1_3cm_dist)
            datalist_3cm.append(rookie2_3cm_dist)
            datalist_3cm.append(rookie1_3cm_angel)
            datalist_3cm.append(rookie2_3cm_angel)

            all_datalist_3cm.append(datalist_3cm)
            # dict_doctor_gt_3cm[img_index] = doctor_3cm
            # dict_rookie_3cm[img_index] = rookie_3cm
            # dict_rookie1_3cm[img_index] = rookie1_3cm
            # dict_rookie2_3cm[img_index] = rookie2_3cm

        else:
            datalist_ori = []
            datalist_ori.append(img_index)
            # rookie_normal = judge(rookie, space, type="ori")
            # doctor_normal = judge(doctor, space, type="ori")
            rookie1_normal_angel = get_dist_angle(rookie1, space, type="ori")
            rookie2_normal_angel = get_dist_angle(rookie2, space, type="ori")
            datalist_ori.append(rookie1_normal_angel)
            datalist_ori.append(rookie2_normal_angel)
            all_datalist_ori.append(datalist_ori)
            # dict_doctor_gt_normal[img_index] = doctor_normal
            # dict_rookie_normal[img_index] = rookie_normal
            # dict_rookie1_normal[img_index] = rookie1_normal
            # dict_rookie2_normal[img_index] = rookie2_normal

    with open((r'D:/code/Knee/code/result/王距离角度.xls'),'w') as f:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet1 = book.add_sheet('王3cm距离角度', cell_overwrite_ok=True)
        sheet2 = book.add_sheet('王ori距离角度', cell_overwrite_ok=True)
        col1 = ('患者ID', '第一次(a+b)/2-c(mm)', '第二次(a+b)/2-c', '第一次e/d', '第二次e/d')
        col2 = ('患者ID', '第一次ori角度', '第二次ori角度')
        # , 'ori角度实际值', 'ori角度预测值', 'ori角度误差'
        for i in range(0, len(col1)):
            sheet1.write(0, i, col1[i])
        for i in range(0, len(col2)):
            sheet2.write(0, i, col2[i])

        for i in range(0, 94):
            data = all_datalist_3cm[i]
            print(data)
            for j in range(0, len(col1)):
                sheet1.write(i + 1, j, str(data[j]))

        for i in range(0, 94):
            data = all_datalist_ori[i]
            print(data)
            for j in range(0, len(col2)):
                sheet2.write(i + 1, j, str(data[j]))

        book.save(savepath)