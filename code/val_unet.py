import os
import pickle

import cv2
import numpy as np
from utils.intergral import *
import math


def test_single_volume(sampled_batch, net, img_save_path=None, args=None):
    image = sampled_batch['image'].cuda()
    label = sampled_batch['ori_kp']
    ori_h = sampled_batch['ori_h'].cpu().numpy()
    ori_w = sampled_batch['ori_w'].cpu().numpy()
    img_index = sampled_batch['img_index']
    target_heatmap = sampled_batch["target_hm"].cuda()
    space = sampled_batch['space'].cpu().numpy()

    net.eval()
    with torch.no_grad():
        out = net(image)
        if isinstance(out, list):
            pre_hm = out[-1]
        else:
            pre_hm = out
        temp = target_heatmap.resize_(1, 5, 256, 256)
        loss = F.mse_loss(temp, pre_hm)

    batch_size, num_landmarks, out_h, out_w = pre_hm.shape
    heatmap = pre_hm.cpu().numpy()

    label = label[0].cpu().numpy()
    pre_landmarks = get_max_preds(heatmap)
    pre_landmarks = pre_landmarks[0]
    pre_landmarks[:, 0] = pre_landmarks[:, 0] * (ori_w / out_w)
    pre_landmarks[:, 1] = pre_landmarks[:, 1] * (ori_h / out_h)

    dist_landmark = np.sqrt((pre_landmarks[:, 0] - label[0:5, 0]) ** 2 +
                            (pre_landmarks[:, 1] - label[0:5, 1]) ** 2) * space

    if img_save_path:
        root_path = args.root_path
        save_result_path = args.save_result_path
        ori_img = cv2.imread(os.path.join(root_path, str(img_index[0]) + ".jpg"))
        for idx in range(num_landmarks):
            #  ±êÇ©ÎªBLUE
            cv2.circle(ori_img, (int(label[idx, 0]), int(label[idx, 1])), 2, (255, 0, 0), -1, 1)
            # Ô¤²âÎªRED
            cv2.circle(ori_img, (int(pre_landmarks[idx, 0]), int(pre_landmarks[idx, 1])), 2, (0, 0, 255), -1, 1)
            cv2.putText(ori_img, str(idx + 1), (int(label[idx, 0]), int(label[idx, 1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)

        if not os.path.exists(os.path.join(img_save_path, save_result_path)):
            os.makedirs(os.path.join(img_save_path, save_result_path))

        cv2.imwrite(os.path.join(img_save_path, save_result_path, '{}.jpg'.format(img_index[0])), ori_img)

    return dist_landmark, loss


def test_single_volume_judge(sampled_batch, net):
    image = sampled_batch['image'].cuda()
    label = sampled_batch['ori_kp']
    ori_h = sampled_batch['ori_h'].cpu().numpy()
    ori_w = sampled_batch['ori_w'].cpu().numpy()
    img_index = sampled_batch['img_index'][0]
    target_heatmap = sampled_batch["target_hm"].cuda()
    space = sampled_batch['space'].cpu().numpy()

    net.eval()
    with torch.no_grad():
        out = net(image)
        if isinstance(out, list):
            pre_hm = out[-1]
        else:
            pre_hm = out
        loss = F.mse_loss(target_heatmap, pre_hm)

    batch_size, num_landmarks, out_h, out_w = pre_hm.shape
    heatmap = pre_hm.cpu().numpy()

    label = label[0].cpu().numpy()

    pre_landmarks = get_max_preds(heatmap)
    pre_landmarks = pre_landmarks[0]
    pre_landmarks[:, 0] = pre_landmarks[:, 0] * (ori_w / out_w)
    pre_landmarks[:, 1] = pre_landmarks[:, 1] * (ori_h / out_h)

    dist_landmark = np.sqrt((pre_landmarks[:, 0] - label[:, 0]) ** 2 +
                            (pre_landmarks[:, 1] - label[:, 1]) ** 2) * space
    if "3cm" in img_index:
        pred = judge(pre_landmarks, space, type="3cm")
        gt = judge(label, space, type="3cm")
    else:
        pred = judge(pre_landmarks, space, type="ori")
        gt = judge(label, space, type="ori")

    # print("The {} gt is {}".format(img_index, str(gt)))
    # print("The {} pred is {}".format(img_index, str(pred)))
    # print("The {} label is {}".format(img_index, str(knee_type.item())))
    # print()

    np.array([0, 0, 0, 0])
    if pred == True and gt == True:
        res = np.array([1, 0, 0, 0])
        # TP, FP, FN, TN
    elif pred == False and gt == True:
        res = np.array([0, 1, 0, 0])
    elif pred == False and gt == False:
        res = np.array([0, 0, 0, 1])
    elif pred == True and gt == False:
        res = np.array([0, 0, 1, 0])
    else:
        raise NotImplemented
    return dist_landmark, loss, res


def judge(landmarks, space, type="3cm"):
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

        if dist <= 3:
            return True

        d = np.sqrt((landmarks[0][0] - landmarks[1][0]) ** 2 +
                    (landmarks[0][1] - landmarks[1][1]) ** 2) * space

        e = np.sqrt((landmarks[1][0] - landmarks[2][0]) ** 2 +
                    (landmarks[1][1] - landmarks[2][1]) ** 2) * space

        # print(e / d)

        if (e / d) <= 0.4:
            return True

        return False
    elif type == "ori":
        foot1 = get_foot_point(landmarks[3], landmarks[4], landmarks[0])
        foot2 = get_foot_point(landmarks[0], foot1, landmarks[1])

        vec_2_1 = landmarks[0] - landmarks[1]
        vec_2_foot2 = foot2 - landmarks[1]

        vec_2_1 = torch.from_numpy(vec_2_1)
        vec_2_foot2 = torch.from_numpy(vec_2_foot2)

        angel = torch.acos(torch.nn.functional.cosine_similarity(vec_2_1, vec_2_foot2, dim=0)) * (180 / math.pi)

        # print(angel)

        if angel.item() <= 11:
            return True
        else:
            return False
    else:
        raise NotImplemented


def test_single_volume_position(sampled_batch, init_model, offset_model, img_save_path=None, args=None, att=False):
    image = sampled_batch['image'].cuda()
    label = sampled_batch['ori_kp']
    ori_h = sampled_batch['ori_h'].cpu().numpy()
    ori_w = sampled_batch['ori_w'].cpu().numpy()
    img_index = sampled_batch['img_index']

    init_model.eval()
    offset_model.eval()
    with torch.no_grad():
        outputs = init_model(image)
        init_pos = softmax_integral_heatmap(outputs, normal=False).view(outputs.shape[0], -1)
        if att:
            heatmap_att = F.sigmoid(torch.sum(outputs, dim=1, keepdim=True))
            offset_pos = offset_model(image * heatmap_att)
        else:
            offset_pos = offset_model(image)
        final_pos = init_pos + offset_pos

    num_landmarks = int(final_pos.shape[1] / 2)

    label = label[0].cpu().numpy()

    assert args is not None, 'args is None'

    pre_landmarks = get_result_from_normal_coords(final_pos, args.patch_size[0], args.patch_size[1])
    pre_landmarks = pre_landmarks[0]
    pre_landmarks[:, 0] = pre_landmarks[:, 0] * (ori_w / args.patch_size[0])
    pre_landmarks[:, 1] = pre_landmarks[:, 1] * (ori_h / args.patch_size[1])

    dist_landmark = np.sqrt((pre_landmarks[:, 0] - label[:, 0]) ** 2 +
                            (pre_landmarks[:, 1] - label[:, 1]) ** 2)

    if img_save_path:
        if "cep" in args.root_path:
            ori_img = cv2.imread(os.path.join(args.root_path, str(img_index[0]) + ".bmp"))
        elif "sp" in args.root_path:
            ori_img = cv2.imread(os.path.join(args.root_path, str(img_index[0]) + ".jpg"))
        elif "hip" in args.root_path:
            ori_img = cv2.imread(os.path.join(args.root_path, str(img_index[0]) + ".jpg"))
        else:
            raise NotImplemented
        for idx in range(num_landmarks):
            cv2.circle(ori_img, (int(label[idx, 0]), int(label[idx, 1])), 7, (255, 0, 0), -1, 1)
            cv2.circle(ori_img, (int(pre_landmarks[idx, 0]), int(pre_landmarks[idx, 1])), 7, (0, 0, 255), -1, 1)

        if not os.path.exists(os.path.join(img_save_path, "val")):
            os.mkdir(os.path.join(img_save_path, "val"))

        cv2.imwrite(os.path.join(img_save_path, "val", '{}.jpg'.format(img_index[0])), ori_img)

    return dist_landmark


def get_foot_point(a, b, c):
    da = a[1] - b[1]
    db = b[0] - a[0]
    dc = -da * a[0] - db * a[1]
    return (
        (db * db * c[0] - da * db * c[1] - da * dc) / (da * da + db * db),
        (da * da * c[1] - da * db * c[0] - db * dc) / (da * da + db * db)
    )


def test_single_depth_volume(sampled_batch, net, img_save_path=None, args=None):
    image = sampled_batch['image'].cuda()
    label = sampled_batch['ori_kp']
    ori_h = sampled_batch['ori_h'].cpu().numpy()
    ori_w = sampled_batch['ori_w'].cpu().numpy()
    img_index = sampled_batch['img_index']

    net.eval()
    with torch.no_grad():
        out = net(image)
        if len(out) > 1:
            pre_hm = out[0]
        else:
            pre_hm = out

    batch_size, num_landmarks, out_h, out_w = pre_hm.shape
    heatmap = pre_hm.cpu().numpy()

    label = label[0].cpu().numpy()

    pre_landmarks = get_max_preds(heatmap)
    pre_landmarks = pre_landmarks[0]
    pre_landmarks[:, 0] = pre_landmarks[:, 0] * (ori_w / out_w)
    pre_landmarks[:, 1] = pre_landmarks[:, 1] * (ori_h / out_h)

    # deep kuojiu right
    # index_1 = 1
    # index_2 = 22
    # index_3 = 2

    # deep kuojiu left
    # index_1 = 12
    # index_2 = 22
    # index_3 = 13

    # deep leg right
    # index_1 = 5
    # index_2 = 6
    # index_3 = 3

    # deep leg left
    index_1 = 17
    index_2 = 16
    index_3 = 14

    pred_foot_point = get_foot_point(pre_landmarks[index_1], pre_landmarks[index_2], pre_landmarks[index_3])
    gt_foot_point = get_foot_point(label[index_1], label[index_2], label[index_3])

    pred_dist_landmark = np.sqrt((pred_foot_point[0] - pre_landmarks[index_3][0]) ** 2 +
                                 (pred_foot_point[1] - pre_landmarks[index_3][1]) ** 2)

    gt_dist_landmark = np.sqrt((gt_foot_point[0] - label[index_3][0]) ** 2 +
                               (gt_foot_point[1] - label[index_3][1]) ** 2)

    return abs(pred_dist_landmark - gt_dist_landmark)


def test_single_down_angel_volume(sampled_batch, net, img_save_path=None, args=None):
    image = sampled_batch['image'].cuda()
    label = sampled_batch['ori_kp']
    ori_h = sampled_batch['ori_h'].cpu().numpy()
    ori_w = sampled_batch['ori_w'].cpu().numpy()
    img_index = sampled_batch['img_index']

    net.eval()
    with torch.no_grad():
        out = net(image)
        if len(out) > 1:
            pre_hm = out[0]
        else:
            pre_hm = out

    batch_size, num_landmarks, out_h, out_w = pre_hm.shape
    heatmap = pre_hm.cpu().numpy()

    label = label[0].cpu().numpy()

    pre_landmarks = get_max_preds(heatmap)
    pre_landmarks = pre_landmarks[0]
    pre_landmarks[:, 0] = pre_landmarks[:, 0] * (ori_w / out_w)
    pre_landmarks[:, 1] = pre_landmarks[:, 1] * (ori_h / out_h)

    # right jinggujiao
    # vect_1 = 3
    # vect_2 = 4
    # vect_3 = 6
    # vect_4 = 5

    # left jinggujiao
    vect_1 = 14
    vect_2 = 15
    vect_3 = 16
    vect_4 = 17

    vect_5_4 = label[vect_1, :] - label[vect_2, :]
    vect_6_7 = label[vect_3, :] - label[vect_4, :]

    vect_5_4 = torch.from_numpy(vect_5_4)
    vect_6_7 = torch.from_numpy(vect_6_7)

    gt_angel = torch.acos(torch.nn.functional.cosine_similarity(vect_6_7, vect_5_4, dim=0)) * (180 / math.pi)

    pred_vect_5_4 = pre_landmarks[vect_1, :] - pre_landmarks[vect_2, :]
    pred_vect_6_7 = pre_landmarks[vect_3, :] - pre_landmarks[vect_4, :]

    pred_vect_5_4 = torch.from_numpy(pred_vect_5_4)
    pred_vect_6_7 = torch.from_numpy(pred_vect_6_7)

    pred_angel = torch.acos(torch.nn.functional.cosine_similarity(pred_vect_6_7, pred_vect_5_4, dim=0)) * (
            180 / math.pi)

    print("gt" + str(gt_angel))
    print("pred" + str(pred_angel))

    return abs(gt_angel - pred_angel)


def test_single_angel_volume(sampled_batch, net, img_save_path=None, args=None):
    image = sampled_batch['image'].cuda()
    label = sampled_batch['ori_kp']
    ori_h = sampled_batch['ori_h'].cpu().numpy()
    ori_w = sampled_batch['ori_w'].cpu().numpy()
    img_index = sampled_batch['img_index']

    net.eval()
    with torch.no_grad():
        out = net(image)
        if len(out) > 1:
            pre_hm = out[0]
        else:
            pre_hm = out

    batch_size, num_landmarks, out_h, out_w = pre_hm.shape
    heatmap = pre_hm.cpu().numpy()

    label = label[0].cpu().numpy()

    pre_landmarks = get_max_preds(heatmap)
    pre_landmarks = pre_landmarks[0]
    pre_landmarks[:, 0] = pre_landmarks[:, 0] * (ori_w / out_w)
    pre_landmarks[:, 1] = pre_landmarks[:, 1] * (ori_h / out_h)

    # right_sharp
    # vect_1 = 1
    # vect_2 = 8
    # vect_3 = 19

    # left_sharp
    # vect_1 = 12
    # vect_2 = 19
    # vect_3 = 8

    # right_ce
    # vect_1 = 1
    # vect_2 = 3
    # vect_3 = 14

    # left_ce
    vect_1 = 12
    vect_2 = 14
    vect_3 = 3

    vect_5_4 = label[vect_1, :] - label[vect_2, :]
    vect_6_7 = label[vect_3, :] - label[vect_2, :]

    vect_5_4 = torch.from_numpy(vect_5_4)
    vect_6_7 = torch.from_numpy(vect_6_7)

    gt_angel = torch.acos(torch.nn.functional.cosine_similarity(vect_6_7, vect_5_4, dim=0)) * (180 / math.pi)

    pred_vect_5_4 = pre_landmarks[vect_1, :] - pre_landmarks[vect_2, :]
    pred_vect_6_7 = pre_landmarks[vect_3, :] - pre_landmarks[vect_2, :]

    pred_vect_5_4 = torch.from_numpy(pred_vect_5_4)
    pred_vect_6_7 = torch.from_numpy(pred_vect_6_7)

    pred_angel = torch.acos(torch.nn.functional.cosine_similarity(pred_vect_6_7, pred_vect_5_4, dim=0)) * (
            180 / math.pi)

    return abs(gt_angel - pred_angel)


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds
