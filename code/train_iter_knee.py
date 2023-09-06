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
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.box_dataset import BaseDataSets, RandomGenerator
from network.net_factory import net_factory
from val_unet import test_single_volume

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

H, W = args.patch_size
batch_size = args.batch_size


def patients_to_slices(dataset, patiens_num):
    ref_dict = {"1": 35, "2": 70,
                "3": 105, "4": 140, "5": 175, "6": 210, "7": 245, "8": 280, "9": 315, "10": 231}
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes,
                        dropout=args.dropout)

    train_transform = RandomGenerator(output_size=args.patch_size,
                                      downsample=args.downsample,
                                      sigma=args.sigma,
                                      state='train'
                                      )
    val_transform = RandomGenerator(output_size=args.patch_size,
                                    downsample=args.downsample,
                                    sigma=args.sigma,
                                    state='val'
                                    )

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, num_class=args.num_classes,
                            transform=train_transform, cv=args.cv)

    db_val = BaseDataSets(base_dir=args.root_path, split="val", num=None, num_class=args.num_classes,
                          transform=val_transform, cv=args.cv)

    db_test = BaseDataSets(base_dir=args.root_path, split="test", num=None, num_class=args.num_classes,
                           transform=val_transform, cv=args.cv)

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=5, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=5)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                            num_workers=5)
    model.train()

    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': base_lr}], lr=base_lr)

    criterion = nn.MSELoss().cuda()
    # sml1 = nn.SmoothL1Loss().cuda()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0

    max_iterations = args.max_iterations
    max_epoch = int(max_iterations / len(trainloader))
    # max_epoch = args.num_epoch
    # max_iterations = max_epoch * len(trainloader)
    logging.info("The total iteration is {}".format(max_iterations))

    # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.98, max_epoch)
    # 每1轮训练验证一次数据集
    val_batch_iteration = len(trainloader) * 1

    best_performance = 999999

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['target_hm']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], label_batch)
                for output in outputs[1:]:
                    loss += criterion(output, label_batch)
            else:
                output = outputs
                loss = criterion(output, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                print('lr: ' + str(lr_))

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))

            if iter_num > 0 and iter_num % val_batch_iteration == 0:
                model.eval()
                metric_list = 0.0
                all_val_loss = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i, val_loss = test_single_volume(sampled_batch, model, None, args)
                    all_val_loss += val_loss
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes):
                    writer.add_scalar('info/val_{}_dist'.format(class_i + 1),
                                      metric_list[class_i], iter_num)

                mean_val_loss = all_val_loss / len(db_val)
                performance = np.mean(metric_list, axis=0)

                writer.add_scalar('info/val_mean_dist', performance, iter_num)

                if performance < best_performance:
                    best_performance = performance
                    if iter_num > val_batch_iteration:
                        os.remove(save_mode_path)
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_distance : %f' % (iter_num, performance))
                logging.info(
                    'iteration %d : mean_val_loss : %f' % (iter_num, mean_val_loss))

            if iter_num >= max_iterations:
                break
        # scheduler.step()
        if iter_num >= max_iterations:
            break

    # save the last epoch model
    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(iter_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))

    # Test
    best_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    best_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    best_model.load_state_dict(torch.load(best_mode_path))

    best_model.eval()

    acc = np.array([0, 0, 0, 0])
    test_metric_list = 0.0
    result = []
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i, _ = test_single_volume(sampled_batch, best_model, snapshot_path, args)
        result.append(metric_i)
        acc[0] += np.sum(metric_i < int(3 / 0.143))
        acc[1] += np.sum(metric_i < int(3.5 / 0.143))
        acc[2] += np.sum(metric_i < int(4 / 0.143))
        acc[3] += np.sum(metric_i < int(5 / 0.143))
        test_metric_list += np.array(metric_i)
    mean_test_metric_list = test_metric_list / len(db_test)

    for class_i in range(num_classes):
        logging.info("The {} landmark mse loss is {}".format(class_i + 1, mean_test_metric_list[class_i]))

    mtx = np.array(result).reshape(num_classes, len(db_test))

    logging.info("ave: {}%".format(np.mean(mtx)))
    logging.info("std: {}".format(np.std(mtx[:, :])))

    test_performance = np.sum(test_metric_list) / (len(db_test) * num_classes)

    logging.info(acc / (len(db_test) * num_classes))

    logging.info("The test performance is %f" % test_performance)
    iterator.close()
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
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

    snapshot_path = "../model/{}_cv{}/{}_size{}_b{}_iter{}_loss{}_dropout{}_sigma{}".format(
        args.exp, args.cv, args.model, args.patch_size,
        args.batch_size, args.max_iterations, args.loss, args.dropout, args.sigma)
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
