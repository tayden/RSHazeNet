import sys
import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import torchPSNR
from model import RSHazeNet
from torch.cuda.amp import autocast, GradScaler
from datasets import *
from options import Options
import torch.nn.functional as F


if __name__ == '__main__':

    opt = Options()
    cudnn.benchmark = True

    best_psnr = 0
    best_epoch = 0

    myNet = RSHazeNet()
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    optimizer = optim.Adam(myNet.parameters(), lr=opt.Learning_Rate)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Epoch, eta_min=1e-8)

    datasetTrain = MyTrainDataSet(opt.Input_Path_Train, opt.Target_Path_Train, patch_size=opt.Patch_Size_Train)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=opt.Batch_Size_Train, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    datasetValue = MyValueDataSet(opt.Input_Path_Val, opt.Target_Path_Val, patch_size=opt.Patch_Size_Val)  # 实例化评估数据集类
    valueLoader = DataLoader(dataset=datasetValue, batch_size=opt.Batch_Size_Val, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists(opt.MODEL_PRE_PATH):
        if opt.CUDA_USE:
            myNet.load_state_dict(torch.load(opt.MODEL_PRE_PATH))
        else:
            myNet.load_state_dict(torch.load(opt.MODEL_PRE_PATH, map_location=torch.device('cpu')))

    scaler = GradScaler()
    for epoch in range(opt.Epoch):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            if opt.CUDA_USE:
                input_train, target = Variable(x).cuda(), Variable(y).cuda()
            else:
                input_train, target = Variable(x), Variable(y)

            with autocast(True):
                restored = myNet(input_train)
                loss = F.mse_loss(restored, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epochLoss += loss.item()
            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, opt.Epoch, loss.item()))

        if epoch % 3 == 0:
            myNet.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_, target_value = (x.cuda(), y.cuda()) if opt.CUDA_USE else (x, y)
                with torch.no_grad():
                    output_value = myNet(input_).clamp_(-1, 1)
                for output_value, target_value in zip(output_value, target_value):
                    psnr_val_rgb.append(torchPSNR(output_value, target_value))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(myNet.state_dict(), './model_best.pth')

        if epoch % 20 == 0:
            torch.save(myNet.state_dict(), f'./model_{epoch}.pth')
        scheduler.step()
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}, current psnr:  {:.3f}, best psnr:  {:.3f}.".format(
                epoch + 1, timeEnd - timeStart, epochLoss, psnr_val_rgb, best_psnr))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))





