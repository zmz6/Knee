import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import cv2
from torch.autograd import Variable

from utils.intergral import softmax_integral_tensor, softmax_integral_heatmap


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx].reshape(-1)),
                    heatmap_gt.mul(target_weight[:, idx].reshape(-1))
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class EdgeOHKMMSELoss(nn.Module):
    def __init__(self, type="MSE", topk=15):
        super().__init__()
        self.edge = get_edge()
        if type == "MSE":
            self.loss = nn.MSELoss(reduction='none').cuda()
        elif type == "SML1":
            self.loss = nn.SmoothL1Loss(reduction='none').cuda()
        elif type == "L1":
            self.loss = nn.L1Loss(reduction='none').cuda()
        else:
            raise NotImplemented
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)

        pred = softmax_integral_heatmap(output, normal=False)
        target = softmax_integral_heatmap(target, normal=False)
        # pred_landmark = pred.reshape((batch_size, num_joints, -1)).split(1, 1)
        # target_landmark = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for i in range(num_joints):
            loss_one = 0
            for j in range(num_joints):
                if self.edge[i][j] == 1:
                    pred_vec = pred[:, i] - pred[:, j]
                    gt_vec = target[:, i] - target[:, j]
                    loss_one += self.loss(pred_vec, gt_vec)

            loss.append(loss_one)

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


class AWing(nn.Module):

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y))) * (self.alpha - y) * (
                (self.theta / self.epsilon) ** (self.alpha - y - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - y))
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1 + torch.abs((y[case1_ind] - y_pred[case1_ind]) / self.epsilon) ** (self.alpha - y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind] * torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]
        return lossMat.mean()


def get_edge(file_name="18_adj.txt"):
    with open("utils/" + file_name, "r") as f:
        adjs = f.readlines()
    relation_list = [adj.strip().split() for adj in adjs]
    num_class = 18
    relation_matrix = [[0 for i in range(num_class)] for i in range(num_class)]

    for (x, y) in relation_list:
        x_index = int(x) - 1
        y_index = int(y) - 1
        relation_matrix[x_index][y_index] = 1
        relation_matrix[y_index][x_index] = 1
    return relation_matrix


class EdgeLoss(nn.Module):
    def __init__(self, type="MSE", line_type=1):
        super().__init__()
        if line_type == 1:
            self.edge = get_edge("18_adj.txt")
        elif line_type == 2:
            self.edge = get_edge("18_adj_2.txt")
        else:
            raise NotImplemented
        if type == "MSE":
            self.loss = nn.MSELoss().cuda()
        elif type == "SML1":
            self.loss = nn.SmoothL1Loss().cuda()
        elif type == "L1":
            self.loss = nn.L1Loss().cuda()
        else:
            raise NotImplemented

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)

        pred = softmax_integral_heatmap(output, normal=False)
        target = softmax_integral_heatmap(target, normal=False)
        # pred_landmark = pred.reshape((batch_size, num_joints, -1)).split(1, 1)
        # target_landmark = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        for i in range(num_joints):
            loss_one = 0
            for j in range(num_joints):
                if self.edge[i][j] == 1:
                    pred_vec = pred[:, i] - pred[:, j]
                    gt_vec = target[:, i] - target[:, j]
                    loss_one += self.loss(pred_vec, gt_vec)
            loss += loss_one

        return loss / num_joints


class Loss_weighted(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWing(alpha, omega, epsilon, theta)

    def forward(self, y_pred, y, M):
        M = M.float()
        Loss = self.Awing(y_pred, y)
        weighted = Loss * (self.W * M + 1.)
        return weighted.mean()


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             # N,C,H,W => N,C,H*W
#             input = input.view(input.size(0), input.size(1), -1)
#             input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input, dim=1)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        # print(score)
        return score


def show_heatmap(kernel, filename=""):
    # kernel的类型是 tensor,shape: (batch_size,channel,w,h)
    kernel = kernel.squeeze(dim=0).cpu().numpy()
    # 先计算第一个通道的heatmap
    heatmap = kernel[0] * 255
    heatmap = heatmap.astype(np.uint8)
    final_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    for i in range(1, len(kernel)):
        heatmap = kernel[i] * 255
        if (type(heatmap) == 'Tensor'):
            heatmap = heatmap.numpty()
        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)  # 生成伪热力图
        final_img = final_img + heatmap
    cv2.imwrite(filename, final_img)
    cv2.waitKey(0)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        batch_size = predict.size(0)
        num_joints = predict.size(1)
        heatmaps_pred = predict.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += self.criterion(heatmap_pred, heatmap_gt.float())

        return loss / num_joints


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob)
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()
#
#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss
#
#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict & target shape do not match'
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                             keepdim=True)
    return ent_map


def cam_activation(batch_feature, channel_weight):
    # batch_feature = batch_feature.permute(0,2,3,1)#48 7 7 1024
    # activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))#48*49*1024

    # attention = activations.permute(1,0,2)#.mul(channel_weight)#49*48*1024
    # attention = attention.permute(1,2,0)#48*1024*49
    # attention = F.softmax(attention, -1)#48*1024*49

    # activations2 = activations.permute(0, 2, 1) #48 1024 49
    # activations2 = activations2 * attention
    # activations2 = torch.sum(activations2, -1)#48*1024
    batch_feature = batch_feature.permute(0, 2, 3, 1)
    # 48*49*1024
    activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))

    # 49*48*1024
    attention = activations.permute(1, 0, 2).mul(channel_weight)
    # 48*49*1024
    attention = attention.permute(1, 0, 2)
    # 48*49
    attention = torch.sum(attention, -1)
    attention = F.softmax(attention, -1)

    activations2 = activations.permute(2, 0, 1)  # 1024*48*49
    activations2 = activations2 * attention
    activations2 = torch.sum(activations2, -1)  # 1024*48
    # 48 1024
    activations2 = activations2.permute(1, 0)

    return activations2


def relation_mse_loss_cam(activations, ema_activations, model, label):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = model.module.densenet121.classifier[0].weight
    # 48*1024
    channel_weight = label.mm(weight)

    activations = cam_activation(activations.clone(), channel_weight)
    ema_activations = cam_activation(ema_activations.clone(), channel_weight)

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2
    return similarity_mse_loss


def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    batch_size, c, h, w = activations.shape

    activations = torch.reshape(activations, (batch_size * c, h * w))
    ema_activations = torch.reshape(ema_activations, (batch_size * c, h * w))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2

    return similarity_mse_loss


def draw_map(activations, ema_activations):
    batch_size, c, h, w = activations.shape

    activations = torch.reshape(activations, (batch_size * c, h * w))
    ema_activations = torch.reshape(ema_activations, (batch_size * c, h * w))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    pass


def relation_nl_mse_loss(q_out, v_out, ema_q_out, ema_v_out):
    assert q_out.size() == ema_q_out.size()
    assert v_out.size() == ema_v_out.size()

    batch_size, c, h, w = q_out.shape

    q_out = torch.reshape(q_out, (batch_size * c, h * w))
    v_out = torch.reshape(v_out, (h * w, batch_size * c))

    similarity = q_out.mm(v_out)
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_q_out = torch.reshape(ema_q_out, (batch_size * c, h * w))
    ema_v_out = torch.reshape(ema_v_out, (h * w, batch_size * c))

    ema_similarity = ema_q_out.mm(ema_v_out)
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2

    return similarity_mse_loss


# class JointsMSELoss(nn.Module):
#     def __init__(self, use_target_weight):
#         super(JointsMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(reduction='mean')
#         self.use_target_weight = use_target_weight
#
#     def forward(self, output, target, target_weight):
#         batch_size = output.size(0)
#         num_joints = output.size(1)
#         heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
#         heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
#         loss = 0
#
#         for idx in range(num_joints):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             if self.use_target_weight:
#
#                 loss += 0.5 * self.criterion(
#                     heatmap_pred.mul(target_weight[:, idx].view(batch_size, 1)),
#                     heatmap_gt.mul(target_weight[:, idx].view(batch_size, 1))
#                 )
#             else:
#                 loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
#
#         return loss / num_joints


def relation_landmark_loss(outputs, final_feature, ema_outputs, ema_final_feature):
    assert outputs.size() == ema_outputs.size()
    assert final_feature.size() == ema_final_feature.size()

    batch_size, f_c, f_h, f_w = final_feature.shape
    _, c, h, w = outputs.shape

    # Students网络的输出值
    stu_relation = []
    ema_relation = []
    for index in range(batch_size):
        # 1.
        one_final_feature = final_feature[index]
        one_output = outputs[index]
        feature_soft = one_final_feature.view(1 * f_c, -1)
        output_soft = one_output.view(1 * c, -1)
        fuse_feature = feature_soft.mm(output_soft.t())
        norm = torch.reshape(torch.norm(fuse_feature, 2, 1), (-1, 1))
        fuse_feature = fuse_feature / norm

        relation_heatmap = fuse_feature.t().mm(fuse_feature)
        norm = torch.reshape(torch.norm(relation_heatmap, 2, 1), (-1, 1))
        relation_heatmap = relation_heatmap / norm

        stu_relation.append(relation_heatmap.unsqueeze(0))
        # 2.
        one_ema_final_feature = ema_final_feature[index]
        one_ema_input = ema_outputs[index]
        feature_soft = one_ema_final_feature.view(1 * f_c, -1)
        output_soft = one_ema_input.view(1 * c, -1)
        fuse_feature = feature_soft.mm(output_soft.t())
        norm = torch.reshape(torch.norm(fuse_feature, 2, 1), (-1, 1))
        fuse_feature = fuse_feature / norm

        relation_heatmap = fuse_feature.t().mm(fuse_feature)
        norm = torch.reshape(torch.norm(relation_heatmap, 2, 1), (-1, 1))
        relation_heatmap = relation_heatmap / norm

        ema_relation.append(relation_heatmap.unsqueeze(0))

    stu_landmark_heatmap = torch.cat(stu_relation, dim=0)
    ema_landmark_heatmap = torch.cat(ema_relation, dim=0)
    similarity_mse_loss = (stu_landmark_heatmap - ema_landmark_heatmap) ** 2

    return similarity_mse_loss


def relation_feature_loss(outputs, final_feature, ema_outputs, ema_final_feature):
    batch_size, f_c, f_h, f_w = final_feature.shape
    _, c, h, w = outputs.shape

    # Students网络的输出值
    stu_relation = []
    ema_relation = []
    for index in range(batch_size):
        # 1.
        one_final_feature = final_feature[index]
        one_output = outputs[index]
        feature_soft = one_final_feature.view(1 * f_c, -1)
        output_soft = one_output.view(1 * c, -1)
        fuse_feature = feature_soft.mm(output_soft.t())
        norm = torch.reshape(torch.norm(fuse_feature, 2, 1), (-1, 1))
        fuse_feature = fuse_feature / norm

        relation_heatmap = fuse_feature.mm(fuse_feature.t())
        norm = torch.reshape(torch.norm(relation_heatmap, 2, 1), (-1, 1))
        relation_heatmap = relation_heatmap / norm

        stu_relation.append(relation_heatmap.unsqueeze(0))
        # 2.
        one_ema_final_feature = ema_final_feature[index]
        one_ema_input = ema_outputs[index]
        feature_soft = one_ema_final_feature.view(1 * f_c, -1)
        output_soft = one_ema_input.view(1 * c, -1)
        fuse_feature = feature_soft.mm(output_soft.t())
        norm = torch.reshape(torch.norm(fuse_feature, 2, 1), (-1, 1))
        fuse_feature = fuse_feature / norm

        relation_heatmap = fuse_feature.mm(fuse_feature.t())
        norm = torch.reshape(torch.norm(relation_heatmap, 2, 1), (-1, 1))
        relation_heatmap = relation_heatmap / norm

        ema_relation.append(relation_heatmap.unsqueeze(0))

    stu_landmark_heatmap = torch.cat(stu_relation, dim=0)
    ema_landmark_heatmap = torch.cat(ema_relation, dim=0)
    similarity_mse_loss = (stu_landmark_heatmap - ema_landmark_heatmap) ** 2

    return similarity_mse_loss


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class L1JointLocationLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(L1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, *args):
        gt_joints = args[0]

        num_joints = preds.shape[1]
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]

        pred_jts = softmax_integral_tensor(preds, num_joints, hm_width, hm_height)

        _assert_no_grad(gt_joints)
        return weighted_l1_loss(pred_jts, gt_joints, self.size_average)


class L2JointLocationLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, *args):
        gt_joints = args[0]
        num_joints = preds.shape[1]
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]

        pred_jts = softmax_integral_tensor(preds, num_joints, hm_width, hm_height)

        _assert_no_grad(gt_joints)
        return weighted_mse_loss(pred_jts, gt_joints, self.size_average)


def weighted_mse_loss(input, target, size_average):
    out = (input - target) ** 2
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()


def weighted_l1_loss(input, target, size_average):
    out = torch.abs(input - target)
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()


def L2(f_):
    return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2) / ((f_T.shape[-1] * f_T.shape[-2]) ** 2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, feat_S, feat_T):
        # feat_S = preds_S[self.feat_ind]
        # feat_T = preds_T[self.feat_ind]

        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                               ceil_mode=True)  # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss


if __name__ == '__main__':
    edgeloss = EdgeOHKMMSELoss("SML1")

    target_hm = torch.randn((1, 24, 128, 128)).cuda()
    pred_hm = torch.randn((1, 24, 128, 128)).cuda()
    loss = edgeloss(target_hm, pred_hm)

    pass
