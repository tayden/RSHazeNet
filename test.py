import sys
import time
import cv2
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_msssim import ssim
from FasterRSDNet import FasterRSDNet
from skimage import img_as_ubyte
from datasets import *
from options import Options


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask


if __name__ == '__main__':

    opt = Options()

    myNet = FasterRSDNet()
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    datasetTest = MyTestDataSet(opt.Input_Path_Test, opt.Target_Path_Test)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True)

    print('--------------------------------------------------------------')
    if opt.CUDA_USE:
        myNet.load_state_dict(torch.load(opt.MODEL_PRE_PATH))
    else:
        myNet.load_state_dict(torch.load(opt.MODEL_PRE_PATH, map_location=torch.device('cpu')))
    myNet.eval()

    PSNR = 0
    SSIM = 0
    MSE = 0
    L = 0

    with torch.no_grad():
        timeStart = time.time()
        for index, (x, y, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_test = x.cuda() if opt.CUDA_USE else x
            target = y.cuda() if opt.CUDA_USE else y

            _, _, h, w = input_test.shape
            input_test, mask = expand2square(input_test, factor=128)
            restored_ = myNet(input_test).clamp_(-1, 1)

            restored = restored_ * 0.5 + 0.5
            target = target * 0.5 + 0.5

            restored = torch.masked_select(restored, mask.bool()).reshape(1, 3, h, w)

            mse_val = F.mse_loss(restored, target)
            psnr_val = 10 * torch.log10(1 / mse_val).item()

            _, _, H, W = restored.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(restored, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

            MSE += mse_val.item()
            PSNR += psnr_val
            SSIM += ssim_val

            L = index + 1

            # print(psnr_val, ssim_val, mse_val.item())  # current metrical scores

            restored = restored_.cpu().numpy().squeeze().transpose((1, 2, 0))
            cv2.imwrite(opt.Result_Path_Test + name[0], cv2.cvtColor(img_as_ubyte(restored), cv2.COLOR_RGB2BGR))

        timeEnd = time.time()
        print('---------------------------------------------------------')
        print(PSNR / L, SSIM / L, MSE / L)
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))
