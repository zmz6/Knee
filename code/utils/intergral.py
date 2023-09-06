import torch
import torch.nn.functional as F


def get_label_func():
    return generate_joint_location_label


def generate_joint_location_label(patch_width, patch_height, joints):
    joints[:, :, 0] = joints[:, :, 0] / patch_width - 0.5
    joints[:, :, 1] = joints[:, :, 1] / patch_height - 0.5

    joints = joints.reshape((joints.shape[0], -1))
    return joints


def generate_joint_location_labelv2(patch_width, patch_height, joints):
    joints[:, :, 0] = joints[:, :, 0] / patch_width - 0.5
    joints[:, :, 1] = joints[:, :, 1] / patch_height - 0.5

    return joints


def get_joint_location_result(patch_width, patch_height, preds):
    # TODO: This cause imbalanced GPU useage, implement cpu version
    # 感觉它的输出方式是 BXC
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]
    num_joints = preds.shape[1]

    pred_jts = softmax_integral_tensor(preds, num_joints, hm_width, hm_height)
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 2), 2))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height

    return coords


def get_result_from_normal_coordsV2(pred_jts, patch_width, patch_height):
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height

    return coords


def get_result_from_normal_coords(pred_jts, patch_width, patch_height):
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 2), 2))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height

    return coords


def get_result_func():
    return get_joint_location_result


def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)

    accu_x = accu_x * torch.arange(x_dim).float().cuda()
    accu_y = accu_y * torch.arange(y_dim).float().cuda()

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    return accu_x, accu_y


def softmax_integral_tensor(preds, num_joints, hm_width, hm_height):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    alpha = 4000
    preds = F.softmax(preds * alpha, 2)

    # integrate heatmap into joint location
    x, y = generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    preds = torch.cat((x, y), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 2))
    return preds


def softmax_integral_out(preds):
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]
    num_joints = preds.shape[1]

    preds = preds.reshape((preds.shape[0], num_joints, -1))
    alpha = 1
    preds = F.softmax(preds * alpha, 2)

    # integrate heatmap into joint location
    x, y = generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    preds = torch.cat((x, y), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints, 2))

    return preds


def softmax_integral_heatmap(preds, normal=True):
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]
    num_joints = preds.shape[1]

    preds = preds.reshape((preds.shape[0], num_joints, -1))
    alpha = 5000
    preds = F.softmax(preds * alpha, 2)

    # integrate heatmap into joint location
    x, y = generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    preds = torch.cat((x, y), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints, 2))

    if normal:
        preds[:, :, 0] = (preds[:, :, 0] + 0.5) * hm_width
        preds[:, :, 1] = (preds[:, :, 1] + 0.5) * hm_height

    return preds
