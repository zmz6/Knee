import argparse
import os
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataloader.test_dataset import RandomGenerator, TestDataSets
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

    best_mode_path = "D:\Code\Python\Knee\model\\all_data_swap\Fully_Supervised_iter_cv3\\nl_unet_size[256, 256]_b4_iter25000_lossMSE_dropout0.5_sigma3\\iter_24940.pth"

    img_save_path = "D:\Code\Python\Knee\model\\all_data_swap\Fully_Supervised_iter_cv3\\nl_unet_size[256, 256]_b4_iter25000_lossMSE_dropout0.5_sigma3\\final_test"
    # best_mode_path = os.path.join(snapshot_path, 'unet_best_model.pth')
    print(best_mode_path)
    test_transform = RandomGenerator(output_size=path_size,
                                     downsample=1,
                                     sigma=3,
                                     state='val'
                                     )

    db_test = TestDataSets(base_dir=root_path, split=state, num=None, transform=test_transform, cv=3, num_class=num_classes)

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

    model = create_model()

    model.load_state_dict(torch.load(best_mode_path))

    model.eval()
    acc = np.array([0, 0, 0, 0])
    test_metric_list = 0.0
    result = []
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i, _ = test_single_volume(sampled_batch, model, img_save_path=img_save_path, args=args)
        result.append(metric_i)
        acc[0] += np.sum(metric_i < 1.5)
        acc[1] += np.sum(metric_i < 2)
        acc[2] += np.sum(metric_i < 2.5)
        acc[3] += np.sum(metric_i < 4)
        test_metric_list += np.array(metric_i)
    mean_test_metric_list = test_metric_list / len(db_test)

    for class_i in range(num_classes):
        print("The {} landmark mse loss is {}".format(class_i + 1, mean_test_metric_list[class_i]))

    test_performance = np.sum(test_metric_list) / (len(db_test) * num_classes)
    print(test_performance)

    print(acc / (len(db_test) * num_classes))

    mtx = np.array(result).reshape(num_classes, len(db_test))

    print("ave: {}%".format(np.mean(mtx)))
    print("std: {}".format(np.std(mtx[:, :])))

    test_performance = np.sum(test_metric_list) / (len(db_test) * num_classes)

    print(acc / (len(db_test) * num_classes))

    print("The test performance is %f" % test_performance)

    for i in range(num_classes):
        print("The std of {} landmark is {}".format(i + 1, np.std(mtx[i, :])))
