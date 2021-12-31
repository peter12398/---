from MLAdaptiveFilter.modules import Precoder
from datasets import VimeoDataset
from config import TorchHelper
from tools.io import tensor_yuv_to_rgb, tensor_yuv_to_rgb_grad, tensor_rgb_to_yuv
from tools.visualization import plot_tensor
from torch.utils.data import DataLoader
import torch
from MLAdaptiveFilter.modules.syntax import scatter_frames, gather_frames, AFSizeSeg
from MLAdaptiveFilter.modules.transform import DCTModule, InvDCTModule
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(image):
    h, w = image.shape[-2:]
    dct = DCTModule()
    seg = AFSizeSeg(resolution=(h, w), block_sz=32)
    image_scattered = scatter_frames(image, seg)
    image_freq = dct(image_scattered)
    image_freq = gather_frames(image_freq, seg)

    # hist = torch.histc(image_freq, bins=256, min=0, max=1)
    # hist_np = hist.detach().cpu().numpy()
    # x_axe = np.arange(0, 256, 1)
    # plt.bar(x_axe, hist_np)

    image_freq_np = image_freq.detach().reshape(-1).cpu().numpy().reshape(-1)
    plt.hist(image_freq_np, bins=256)
    plt.show()


def plot_freq(image, *args, **kwargs):
    h, w = image.shape[-2:]
    dct = DCTModule()
    seg = AFSizeSeg(resolution=(h, w), block_sz=32)
    image_scattered = scatter_frames(image, seg)
    image_freq = dct(image_scattered)
    image_freq = gather_frames(image_freq, seg)
    plot_tensor(image_freq, *args, **kwargs)


if __name__ == '__main__':
    from config import TorchHelper
    PATH_TO_CHECKPOINT = './model_output/2021-05-08T23:43:54+0800_epoch=64_loss=-18.339759_lr=0.000500.pth'

    PATH_TO_DATASET = '/mnt/hd1/yutong/Vimeo/vimeo_septuplet/'
    USE_CUDA = False
    DEVICE = torch.device('cuda', 1) if USE_CUDA else torch.device('cpu')

    running_model = Precoder()
    running_model.load_state_dict(torch.load(PATH_TO_CHECKPOINT, map_location=DEVICE))
    # TorchHelper.load_from_DataParallel(running_model, PATH_TO_CHECKPOINT, map_location=DEVICE)

    if USE_CUDA:
        running_model = running_model.to(DEVICE)
    running_model.eval()
    eval_dataloader = DataLoader(dataset=VimeoDataset(path_to_dataset=PATH_TO_DATASET, mode='test', grayscale=False), shuffle=True, batch_size=1)

    for image, mask, _ in eval_dataloader:
        with torch.no_grad():
            if USE_CUDA:
                image = image.to(DEVICE)
                mask = mask.to(DEVICE)
            image_yuv = tensor_rgb_to_yuv(image)

            pred_y = running_model(image_yuv[:, :, 0:1, :, :])  # batch size，gops的数量，channels
            # pred_y = image_yuv[:, :, 0:1, :, :]

            image_yuv[:, :, 0, :, :] = pred_y[:, :, 0, :, :]
            pred = tensor_yuv_to_rgb(image_yuv)

            margin = mask - pred
            plot_tensor(pred[0], 1, 0, data_range=1)
            plot_tensor(image[0], 1, 0, data_range=1)
            diff = (torch.floor(255 * pred[0][1]) - torch.floor(255 * image[0][1])).unsqueeze(0)
            plot_tensor(torch.abs(diff), 0, 0, data_range=2)
            diff_sum = torch.sum(torch.abs(diff))
            print(diff_sum)
            print('finish')