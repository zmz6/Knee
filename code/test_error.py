import argparse
import os
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataloader.box_dataset import RandomGenerator, BaseDataSets
from network.net_factory import net_factory
from val_unet import test_single_volume, test_single_angel_volume, \
    test_single_down_angel_volume, test_single_depth_volume, get_max_preds, judge

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/knee', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='all_data_swap/Fully_Supervised_iter', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='nl_unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=5,
                    help='max number epoch to train')
parser.add_argument('--base_filter', type=int, default=128,
                    help='base_filter')
parser.add_argument('--num_epoch', type=int, default=200,
                    help='max number epoch to train')
parser.add_argument('--max_iterations', type=int, default=25000,
                    help='max number iterations to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
# h, w
# parser.add_argument('--patch_size', nargs='+', type=int,
#                     help='patch size of network input')

parser.add_argument('--sigma', type=int, default=3,
                    help='sigma')
parser.add_argument('--downsample', type=int, default=1,
                    help='downsample')

parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
parser.add_argument('--loss', type=str, default="MSE",
                    help='sigma')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout')
parser.add_argument("--save_result_path", type=str, default="test")
parser.add_argument("--cv", type=int, default=1)
parser.add_argument('--cuda', type=int, default=1,
                    help='cuda')
args = parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_classes = 5
    path_size = [256, 256]
    root_path = 'D:/code/Knee/data/knee'

    state = "test"

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root_path", type=str, default="../data/knee")
    # parser.add_argument("--save_result_path", type=str, default=state)
    # args = parser.parse_args()

    best_mode_path = "D:\code\Knee\model\\all_data_swap\Fully_Supervised_iter_cv3\\nl_unet_size[256, 256]_b4_iter25000_lossMSE_dropout0.5_sigma3\\nl_unet_best_model.pth"
    # best_mode_path = os.path.join(snapshot_path, 'unet_best_model.pth')
    print(best_mode_path)
    test_transform = RandomGenerator(output_size=args.patch_size,
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

    snapshot_path = "../model/{}_cv{}/{}_size{}_b{}_iter{}_loss{}_dropout{}_sigma{}".format(
        args.exp, args.cv, args.model, args.patch_size,
        args.batch_size, args.max_iterations, args.loss, args.dropout, args.sigma)
    best_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    best_model.load_state_dict(torch.load(best_mode_path))

    best_model.eval()

    acc = np.array([0, 0, 0, 0])
    test_metric_list = 0.0
    result = []

    for i_batch, sampled_batch in enumerate(testloader):
        image = sampled_batch['image'].cuda()
        label = sampled_batch['ori_kp']
        ori_h = sampled_batch['ori_h'].cpu().numpy()
        ori_w = sampled_batch['ori_w'].cpu().numpy()
        img_index = sampled_batch['img_index'][0]
        target_heatmap = sampled_batch["target_hm"].cuda()
        space = sampled_batch['space'].cpu().numpy()

        metric_i, _ = test_single_volume(sampled_batch, best_model, snapshot_path, args)
        result.append(metric_i)
        acc[0] += np.sum(metric_i < int(3 / 0.143))
        acc[1] += np.sum(metric_i < int(3.5 / 0.143))
        acc[2] += np.sum(metric_i < int(4 / 0.143))
        acc[3] += np.sum(metric_i < int(5 / 0.143))
        test_metric_list += np.array(metric_i)

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

    mean_test_metric_list = test_metric_list / len(db_test)
    normal_img = all_index - wrong_img
    pre_normal_img = all_index - pred_wrong_img


    for class_i in range(num_classes):
        print("The {} landmark mse loss is {}".format(class_i + 1, mean_test_metric_list[class_i]))

    mtx = np.array(result).reshape(num_classes, len(db_test))

    print("The landmarks' average loss is {}".format(np.mean(mtx, axis=1)))

    print("ave: {}%".format(np.mean(mtx)))
    print("std: {}".format(np.std(mtx[:, :])))