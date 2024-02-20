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
    for scene_name in scene_names:
        print("Scene: " + scene_name)
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
                psnr_in += psnr(clean_img, input_img)
                ssim_in += ssim(clean_img, input_img, multichannel=True, channel_axis=-1, data_range=255)

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

                psnr_out += psnr(clean_img, save_out)
                ssim_out += ssim(clean_img, save_out, multichannel=True, channel_axis=-1, data_range=255)

                save_out = cv2.cvtColor(save_out, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(out_path, img_name), save_out)

                inner_count += 1
                outer_count += 1
        psnr_in /= inner_count
        ssim_in /= inner_count
        psnr_out /= inner_count
        ssim_out /= inner_count
        print(f"Scene: {scene_name}, PSNR in: {psnr_in}, SSIM in: {ssim_in} PSNR out: {psnr_out}, SSIM out: {ssim_out}")
    print('Avg. time:', time_test/outer_count)


if __name__ == "__main__":
    main()
