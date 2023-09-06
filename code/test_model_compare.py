import argparse
import os
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataloader.box_dataset import RandomGenerator, BaseDataSets
from network.net_factory import net_factory
from val_unet import test_single_volume, test_single_angel_volume, \
    test_single_down_angel_volume, test_single_depth_volume, get_max_preds, judge

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_classes = 5
    path_size = [256, 256]
    root_path = 'D:/Code/Python/Knee/data/knee'

    state = "test"

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../data/knee")
    parser.add_argument("--save_result_path", type=str, default=state)
    args = parser.parse_args()

    best_mode_path = "D:\Code\Python\Knee\model\\all_data_swap\Fully_Supervised_iter_cv3\\nl_unet_size[256, 256]_b4_iter25000_lossMSE_dropout0.5_sigma3\\nl_unet_best_model.pth"
    # best_mode_path = os.path.join(snapshot_path, 'unet_best_model.pth')
    print(best_mode_path)
    test_transform = RandomGenerator(output_size=path_size,
                                     downsample=1,
                                     sigma=3,
                                     state='val'
                                     )

    db_test = BaseDataSets(base_dir=root_path, split=state, num=None, transform=test_transform, cv=3)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                            num_workers=10)


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

    dict_pred_3cm = {}
    dict_pred_normal = {}


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
            if img_index[:-3] not in all_index:
                all_index.append(img_index[:-3])
            gt_3cm = judge(pre_landmarks, space, type="3cm")
            dict_pred_3cm[img_index] = gt_3cm
        else:
            if img_index not in all_index:
                all_index.append(img_index)
            gt = judge(pre_landmarks, space, type="ori")
            dict_pred_normal[img_index] = gt

    fold = "result"
    all_index.sort()

    with open(os.path.join(fold,  "model_3cm.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{} 3cm:{}\n".format(v, dict_pred_3cm[v + "3cm"]))

    with open(os.path.join(fold, "model.txt"), "w") as f:
        for k, v in enumerate(all_index):
            f.writelines("{}:{}\n".format(v, dict_pred_normal[v]))