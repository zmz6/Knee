import argparse
import os
import torch
import numpy as np
import time

from torch.utils.data import DataLoader

from dataloader.box_dataset import RandomGenerator, BaseDataSets
from network.net_factory import net_factory
from val_unet import test_single_volume, test_single_angel_volume, \
    test_single_down_angel_volume, test_single_depth_volume, get_max_preds, judge

if __name__ == '__main__':
    time_start = time.time()
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

    db_test = BaseDataSets(base_dir=root_path, split=state, num=None, transform=test_transform, cv=3)

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

    all_index = set()
    wrong_img = set()
    pred_wrong_img = set()

    for i_batch, sampled_batch in enumerate(testloader):
        image = sampled_batch['image'].cuda()
        label = sampled_batch['ori_kp']
        ori_h = sampled_batch['ori_h'].cpu().numpy()
        ori_w = sampled_batch['ori_w'].cpu().numpy()
        img_index = sampled_batch['img_index'][0]
        target_heatmap = sampled_batch["target_hm"].cuda()
        space = sampled_batch['space'].cpu().numpy()

        model.eval()
        with torch.no_grad():
            out = model(image)
            if isinstance(out, list):
                pre_hm = out[-1]
            else:
                pre_hm = out

        batch_size, num_landmarks, out_h, out_w = pre_hm.shape
        heatmap = pre_hm.cpu().numpy()

        label = label[0].cpu().numpy()

        pre_landmarks = get_max_preds(heatmap)
        pre_landmarks = pre_landmarks[0]
        pre_landmarks[:, 0] = pre_landmarks[:, 0] * (ori_w / out_w)
        pre_landmarks[:, 1] = pre_landmarks[:, 1] * (ori_h / out_h)

        if "3cm" in img_index:
            all_index.add(img_index[:-3])
            gt_3cm = judge(pre_landmarks, space, type="3cm")
            if gt_3cm:
                pred_wrong_img.add(img_index[:-3])
        else:
            all_index.add(img_index)
            gt = judge(pre_landmarks, space, type="ori")
            if gt:
                pred_wrong_img.add(img_index)

        if "3cm" in img_index:
            all_index.add(img_index[:-3])
            gt_3cm = judge(label, space, type="3cm")
            if gt_3cm:
                wrong_img.add(img_index[:-3])
        else:
            all_index.add(img_index)
            gt = judge(label, space, type="ori")
            if gt:
                wrong_img.add(img_index)

    normal_img = all_index - wrong_img
    pre_normal_img = all_index - pred_wrong_img

    print(normal_img)
    print(len(normal_img))
    print(pre_normal_img)
    print(len(pre_normal_img))
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum, "s")

    # print(normal_img.copy().add(wrong_img))

    good_index = normal_img
    bad_index = all_index - good_index


    # if not os.path.exists("good.txt"):
    #     os.makedirs("good.txt")
    # if not os.path.exists("bad.txt"):
    #     os.makedirs("bad.txt")
    #
    # with open("good.txt", "w") as f:
    #     for index in good_index:
    #         f.writelines(index + "\n")
    #
    # with open("bad.txt", "w") as f:
    #     for index in bad_index:
    #         f.writelines(index + "\n")
    '''
    print("不正常预测为正常:{}".format(pre_normal_img - normal_img))
    print("正常预测为不正常:{}".format(pred_wrong_img - wrong_img))
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for pre_wrong in pred_wrong_img:
        if pre_wrong in wrong_img:
            # 是滑车预测为滑车
            TP += 1
        else:
            # 是正常预测为滑车
            FP += 1

    for pre_normal in pre_normal_img:
        if pre_normal in normal_img:
            # 是正常预测为正常
            TN += 1
        else:
            # 是滑车预测为正常
            FN += 1
    # print(TP)
    #     # print(FP)
    #     # print(FN)
    #     # print(TN)
    acc = (TP + TN) / (len(db_test) / 2)
    pec = TP / (TP + FP)
    rec = TP / (TP + FN)
    spe = TN / (TN + FP)
    print("acc:{}".format(acc))
    print("pec:{}".format(pec))
    print("rec:{}".format(rec))
    print("spe:{}".format(spe))
    print("f1:{}".format((2 * pec * rec) / (pec + rec)) )
    '''