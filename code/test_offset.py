import argparse
import os
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataloader.box_dataset import RandomGenerator, BaseDataSets
from network.net_factory import net_factory, RegNet_factory
from network.regession.net import RegessionNet
from val_unet import test_single_volume, test_single_volume_position

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    num_classes = 18
    path_size = [256, 256]
    base_filter = 128
    root_path = '../data/hip_all'

    state = "val"

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../data/hip_all")
    parser.add_argument('--patch_size', type=list, default=[256, 256],
                        help='patch size of network input')
    parser.add_argument("--save_result_path", type=str, default=state)
    args = parser.parse_args()

    print(parser)

    init_model_path = "/home/liushenyao/code/box/model/hip_18_split/Fully_Supervised_iter_10_labeled/four_unet_size[256, 256]_b1_epoch500_lossMSE_dropout0_sigma3/four_unet_best_model.pth"
    # best_mode_path = os.path.join(snapshot_path, 'unet_best_model.pth')

    offset_model_path = "/home/liushenyao/code/box/model/hip_18_split/offsetNet_10_labeled/four_unet_fc_size[256, 256]_b1_epoch100_lossMSE_dropout0.5/four_unet_best_model.pth"
    img_save_path = "/home/liushenyao/code/box/model/hip_18/att_offsetNet_10_labeled/four_unet_size[256, 256]_b1_epoch200_lossMSE_dropout0.5/"

    test_transform = RandomGenerator(output_size=path_size,
                                     downsample=1,
                                     sigma=3,
                                     state='val'
                                     )

    db_test = BaseDataSets(base_dir=root_path, split="val", num=None, transform=test_transform)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                            num_workers=10)

    def create_model(ema=False, base_filter=32, bilinear=True):
        # Network definition
        model = net_factory(net_type='four_unet', in_chns=1, class_num=num_classes,
                            base_filter=base_filter)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    init_model = create_model(base_filter=base_filter)
    init_model.load_state_dict(torch.load(init_model_path))
    init_model.eval()

    offset_model =  RegNet_factory(net_type="fc", num_classes=num_classes)
    offset_model.load_state_dict(torch.load(offset_model_path))
    offset_model.eval()

    acc = np.array([0, 0, 0, 0])
    test_metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume_position(sampled_batch, init_model, offset_model, img_save_path=None, args=args, att=True)
        acc[0] += np.sum(metric_i < int(3 / 0.143))
        acc[1] += np.sum(metric_i < int(3.5 / 0.143))
        acc[2] += np.sum(metric_i < int(4 / 0.143))
        acc[3] += np.sum(metric_i < int(5 / 0.143))
        test_metric_list += np.array(metric_i)
    mean_test_metric_list = test_metric_list / len(db_test)

    for class_i in range(num_classes):
        print("The {} landmark mse loss is {}".format(class_i + 1, mean_test_metric_list[class_i]))

    test_performance = np.sum(test_metric_list) / (len(db_test) * num_classes)
    print(test_performance)

    print(acc / (len(db_test) * num_classes))
