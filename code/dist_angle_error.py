import argparse
import os
import torch
import numpy as np
import math
import xlwt

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
    savepath = 'D:/code/Knee/data/knee/距离角度误差.xls'
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

    all_datalist_3cm = []

    all_datalist_ori = []

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
            datalist_3cm = []
            datalist_3cm.append(img_index)
            pred_dist_3cm, pred_angel_3cm = get_dist_angle(pre_landmarks, space, type="3cm")

        else:
            datalist_ori = []
            datalist_ori.append(img_index)
            pred_angel = get_dist_angle(pre_landmarks, space, type="ori")

        if "3cm" in img_index:
            dist_3cm, angel_3cm = get_dist_angle(label, space, type="3cm")
            datalist_3cm.append(dist_3cm)
            datalist_3cm.append(pred_dist_3cm)
            datalist_3cm.append(abs(dist_3cm-pred_dist_3cm))
            datalist_3cm.append(angel_3cm)
            datalist_3cm.append(pred_angel_3cm)
            datalist_3cm.append(abs(angel_3cm-pred_angel_3cm))
            all_datalist_3cm.append(datalist_3cm)
        else:
            angel = get_dist_angle(label, space, type="ori")
            datalist_ori.append(angel)
            datalist_ori.append(pred_angel)
            datalist_ori.append(abs(angel-pred_angel))
            all_datalist_ori.append(datalist_ori)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open((r'D:/code/Knee/data/knee/距离角度误差.xls'),'w') as f:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet1 = book.add_sheet('3cm距离角度误差', cell_overwrite_ok=True)
        sheet2 = book.add_sheet('ori距离角度误差', cell_overwrite_ok=True)
        col1 = ('患者ID', '(a+b)/2-c实际值(mm)', '(a+b)/2-c预测值(mm)', '(a+b)/2-c误差', 'e/d实际值', 'e/d预测值', 'e/d误差')
        col2 = ('患者ID', 'ori角度实际值', 'ori角度预测值', 'ori角度误差')
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



