import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader.box_dataset import BaseDataSets, RandomGenerator
from network.net_factory import net_factory, RegNet_factory
from network.regession.net import RegessionNet
from utils.intergral import softmax_integral_heatmap
from utils.loss import JointsMSELoss, Loss_weighted, AWing
from val_unet import test_single_volume, test_single_volume_position

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/hip_all', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='hip_18_split/offsetNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='four_unet', help='model_name')
parser.add_argument('--regmodel', type=str,
                    default='fc', help='reg model_name')
parser.add_argument('--num_classes', type=int, default=18,
                    help='max number epoch to train')
parser.add_argument('--base_filter', type=int, default=128,
                    help='base_filter')
parser.add_argument('--num_epoch', type=int, default=100,
                    help='max number epoch to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
# h, w
# parser.add_argument('--patch_size', nargs='+', type=int,
#                     help='patch size of network input')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
parser.add_argument('--loss', type=str, default="MSE",
                    help='sigma')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout')
args = parser.parse_args()

H, W = args.patch_size
batch_size = args.batch_size


def generate_joint_location_label(patch_width, patch_height, joints):
    joints[:, :, 0] = joints[:, :, 0] / patch_width - 0.5
    joints[:, :, 1] = joints[:, :, 1] / patch_height - 0.5

    joints = joints.reshape((joints.shape[0], -1))
    return joints


def patients_to_slices(dataset, patiens_num):
    ref_dict = {"1": 35, "2": 70,
                "3": 105, "4": 140, "5": 175, "6": 210, "7": 245, "8": 280, "9": 315, "10": 165}
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    init_model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes, base_filter=args.base_filter,
                             dropout=args.dropout)

    ####################################
    pretrain_path = "/home/liushenyao/code/box/model/hip_18_split/Fully_Supervised_iter_10_labeled/four_unet_size[256, 256]_b1_epoch500_lossMSE_dropout0_sigma3/four_unet_best_model.pth"
    init_model.load_state_dict(torch.load(pretrain_path))

    ####################################

    offset_model = RegNet_factory(net_type="fc", num_classes=num_classes)

    train_transform = RandomGenerator(output_size=args.patch_size,
                                      downsample=1,
                                      sigma=3,
                                      state='train'
                                      )
    val_transform = RandomGenerator(output_size=args.patch_size,
                                    downsample=1,
                                    sigma=3,
                                    state='val'
                                    )

    # NOTE: 由于最初训练时两个验证集交换了，所以这里也进行交换
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, transform=train_transform)

    db_val = BaseDataSets(base_dir=args.root_path, split="test", num=None, transform=val_transform)

    db_test = BaseDataSets(base_dir=args.root_path, split="val", num=None, transform=val_transform)

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=5, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=5)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                            num_workers=5)
    init_model.eval()
    offset_model.train()

    optimizer = optim.Adam([{'params': offset_model.parameters(), 'initial_lr': base_lr}], lr=base_lr)

    # mse_loss = nn.MSELoss().cuda()
    sml1 = nn.SmoothL1Loss().cuda()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0

    # max_iterations = args.max_iterations
    # max_epoch = int(max_iterations / len(trainloader))
    max_epoch = args.num_epoch
    max_iterations = max_epoch * len(trainloader)
    logging.info("The total iteration is {}".format(max_iterations))

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.97, max_epoch)
    # 每5轮训练验证一次数据集
    val_batch_iteration = len(trainloader) * 1

    best_performance = 999999

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['target_kp']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            with torch.no_grad():
                outputs = init_model(volume_batch)
                init_pos = softmax_integral_heatmap(outputs, normal=False).view(batch_size, -1)

            label = generate_joint_location_label(args.patch_size[0], args.patch_size[1], label_batch)

            offset_pos = offset_model(volume_batch)

            loss = sml1(init_pos + offset_pos, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
                print('lr: ' + str(lr_))

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))

            if iter_num > 0 and iter_num % val_batch_iteration == 0:
                offset_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_position(sampled_batch, init_model, offset_model, None, args)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes):
                    writer.add_scalar('info/val_{}_dist'.format(class_i + 1),
                                      metric_list[class_i], iter_num)

                performance = np.mean(metric_list, axis=0)

                writer.add_scalar('info/val_mean_dist', performance, iter_num)

                if performance < best_performance:
                    best_performance = performance
                    if iter_num > val_batch_iteration:
                        os.remove(save_mode_path)
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(offset_model.state_dict(), save_mode_path)
                    torch.save(offset_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_distance : %f' % (iter_num, performance))

            if iter_num >= max_iterations:
                break
        scheduler.step()
        if iter_num >= max_iterations:
            break

    # save the last epoch model
    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(iter_num) + '.pth')
    torch.save(offset_model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))

    # Test
    best_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    best_model = RegNet_factory(net_type=args.regmodel, num_classes=num_classes)
    best_model.load_state_dict(torch.load(best_mode_path))

    best_model.eval()

    acc = np.array([0, 0, 0, 0])
    test_metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume_position(sampled_batch, init_model, best_model, snapshot_path, args)
        acc[0] += np.sum(metric_i < int(3 / 0.143))
        acc[1] += np.sum(metric_i < int(3.5 / 0.143))
        acc[2] += np.sum(metric_i < int(4 / 0.143))
        acc[3] += np.sum(metric_i < int(5 / 0.143))
        test_metric_list += np.array(metric_i)
    mean_test_metric_list = test_metric_list / len(db_test)

    for class_i in range(num_classes):
        logging.info("The {} landmark mse loss is {}".format(class_i + 1, mean_test_metric_list[class_i]))

    test_performance = np.sum(test_metric_list) / (len(db_test) * num_classes)

    logging.info(acc / (len(db_test) * num_classes))

    logging.info("The test performance is %f" % test_performance)
    iterator.close()
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}_{}_size{}_b{}_epoch{}_loss{}_dropout{}".format(
        args.exp, args.labeled_num, args.model, args.regmodel, args.patch_size,
        args.batch_size, args.num_epoch, args.loss, args.dropout)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
