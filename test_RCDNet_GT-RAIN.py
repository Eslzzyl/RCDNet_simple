import os
import cv2
import argparse
from glob import glob
import numpy as np
import torch
from utils import *
from rcdnet import RCDNet
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from natsort import natsorted
from PIL import Image

parser = argparse.ArgumentParser(description="RCDNet_Test")
parser.add_argument("--model_dir", type=str,
                    default="./models/", help='path to model files')
parser.add_argument("--data_path", type=str,
                    default="/root/autodl-tmp/GT-RAIN/GT-RAIN_train/", help='path to testing data')
parser.add_argument('--num_M', type=int, default=32,
                    help='the number of rain maps')
parser.add_argument('--num_Z', type=int, default=32,
                    help='the number of dual channels')
parser.add_argument('--T', type=int, default=4,
                    help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=17,
                    help='the number of iterative stages in RCDNet')
parser.add_argument("--use_GPU", type=bool,
                    default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str,
                    default="/root/autodl-tmp/GT-RAIN/model-results/RCDNet/", help='path to derained results')
opt = parser.parse_args()
try:
    os.makedirs(opt.save_path, exist_ok=True)
except OSError:
    pass

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    # Build model
    print('Loading model ...\n')
    model = RCDNet(opt)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(
        opt.model_dir, 'DerainNet_state_100.pt')))
    model.eval()
    time_test = 0
    outer_count = 0
    # 列出 opt.data_path 下的所有目录
    scene_names = os.listdir(opt.data_path)
    print('Testing model...')
    psnr_in_total = 0
    ssim_in_total = 0
    psnr_out_total = 0
    ssim_out_total = 0
    for scene_name in scene_names:
        scene_path = os.path.join(opt.data_path, scene_name)
        out_path = os.path.join(opt.save_path, scene_name)
        os.makedirs(out_path, exist_ok=True)
        psnr_in = 0
        ssim_in = 0
        psnr_out = 0
        ssim_out = 0
        inner_count = 0
        clean_img_path = glob(scene_path + '/*C-000.png')[0]
        clean_img = np.array(Image.open(clean_img_path))
        for img_path in natsorted(glob(scene_path + '/*R-*.png')):
            img_name = img_path.split('/')[-1]
            if is_image(img_name):
                # input image
                O = cv2.imread(img_path)
                O = cv2.cvtColor(O, cv2.COLOR_BGR2RGB)

                input_img = np.array(O)
                temp = psnr(clean_img, input_img)
                psnr_in += temp
                psnr_in_total += temp
                # 以下的计算SSIM指标的方法是去雨任务中约定俗成的方式。
                # 将图像从RGB转换到YCbCr
                clean_img_ycbcr = cv2.cvtColor(clean_img, cv2.COLOR_RGB2YCrCb)
                input_img_ycbcr = cv2.cvtColor(input_img, cv2.COLOR_RGB2YCrCb)
                # 分解YCbCr图像为三个通道
                y_clean, _, _ = cv2.split(clean_img_ycbcr)
                y_input, _, _ = cv2.split(input_img_ycbcr)
                # 在亮度通道（Y通道）上计算SSIM
                temp = ssim(y_clean, y_input, data_range=255)
                ssim_in += temp
                ssim_in_total += temp

                O = np.expand_dims(O.transpose(2, 0, 1), 0)
                O = torch.Tensor(O)
                if opt.use_GPU:
                    O = O.cuda()
                with torch.no_grad():
                    if opt.use_GPU:
                        torch.cuda.synchronize()
                    start_time = time.time()
                    _, ListB, _ = model(O)
                    out = ListB[-1]
                    out = torch.clamp(out, 0., 255.)
                    end_time = time.time()
                    dur_time = end_time - start_time
                    time_test += dur_time
                    # print(img_name, ': ', dur_time)
                if opt.use_GPU:
                    save_out = np.uint8(
                        out.data.cpu().numpy().squeeze())   # back to cpu
                else:
                    save_out = np.uint8(out.data.numpy().squeeze())
                save_out = save_out.transpose(1, 2, 0)

                temp = psnr(clean_img, save_out)
                psnr_out += temp
                psnr_out_total += temp
                # 将图像从RGB转换到YCbCr
                clean_img_ycbcr = cv2.cvtColor(clean_img, cv2.COLOR_RGB2YCrCb)
                out_img_ycbcr = cv2.cvtColor(save_out, cv2.COLOR_RGB2YCrCb)
                # 分解YCbCr图像为三个通道
                y_clean, _, _ = cv2.split(clean_img_ycbcr)
                y_out, _, _ = cv2.split(out_img_ycbcr)
                # 在亮度通道（Y通道）上计算SSIM
                temp = ssim(y_clean, y_out, data_range=255)
                ssim_out += temp
                ssim_out_total += temp

                save_out = cv2.cvtColor(save_out, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(out_path, img_name), save_out)

                inner_count += 1
                outer_count += 1
        psnr_in /= inner_count
        ssim_in /= inner_count
        psnr_out /= inner_count
        ssim_out /= inner_count
        print(f"Scene: {scene_name}, PSNR in: {psnr_in:.2f}, SSIM in: {ssim_in:.4f} PSNR out: {psnr_out:.2f}, SSIM out: {ssim_out:.4f}")
    psnr_in_total /= outer_count
    ssim_in_total /= outer_count
    psnr_out_total /= outer_count
    ssim_out_total /= outer_count
    print(f"Overall PSNR in: {psnr_in_total:.2f}, SSIM in: {ssim_in_total:.4f} PSNR out: {psnr_out_total:.2f}, SSIM out: {ssim_out_total:.4f}")
    print('Avg. time:', time_test / outer_count)


if __name__ == "__main__":
    main()
