import argparse
import os
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataloader.box_dataset import RandomGenerator, BaseDataSets
from network.net_factory import net_factory
from val_unet import test_single_volume, test_single_angel_volume, test_single_volume_judge, judge

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    num_classes = 5
    path_size = [256, 256]
    root_path = '../data/knee'

    state = "val"

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../data/knee")
    parser.add_argument("--save_result_path", type=str, default=state)
    args = parser.parse_args()

    print(parser)

    best_mode_path = "/home/liushenyao/code/Knee/model/debug_knee_5/Fully_Supervised_iter_10_labeled/nl_unet_size[256, 256]_b2_iter15000_lossMSE_dropout0.5_sigma3/nl_unet_best_model.pth"
    # best_mode_path = os.path.join(snapshot_path, 'unet_best_model.pth')

    img_save_path = "/home/liushenyao/code/Knee/model/debug_knee_5/Fully_Supervised_iter_10_labeled/nl_unet_size[256, 256]_b2_iter15000_lossMSE_dropout0.5_sigma3/nl_unet_best_model.pth"
    print(best_mode_path)
    test_transform = RandomGenerator(output_size=path_size,
                                     downsample=1,
                                     sigma=3,
                                     state='val'
                                     )

    db_test = BaseDataSets(base_dir=root_path, split=state, num=None, num_class=num_classes, transform=test_transform)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                            num_workers=10)

    result = np.array([0, 0, 0, 0])
    test_metric_list = 0.0

    all_index = set()
    wrong_img = set()

    for i_batch, sampled_batch in enumerate(testloader):
        label = sampled_batch['ori_kp'][0].numpy()
        img_index = sampled_batch['img_index'][0]
        space = sampled_batch['space'].cpu().numpy()


        if "3cm" in img_index:
            all_index.add(img_index[:-3])
            gt_3cm = judge(label, space, type="3cm")
            if gt_3cm:
                wrong_img.add(img_index[:-3])

        else:
            all_index.add(img_index)
            gt = judge(label, space, type="ori")
            if gt:
                # print(img_index)
                wrong_img.add(img_index)

    normal_img = all_index - wrong_img

    print(normal_img)
    print(len(normal_img))
    print(wrong_img)
    print(len(wrong_img))

    # with open("../data/knee/old/good.txt", 'w') as f:
    #     for index in normal_img:
    #         f.write(str(index) + '\n')
    #
    # with open("../data/knee/old/bad.txt", 'w') as f:
    #     for index in wrong_img:
    #         f.write(str(index) + '\n')
