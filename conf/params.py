import argparse
import os
from datetime import datetime

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# log_path = os.path.join(basedir, f'log/asr_{datetime.now().strftime("%Y-%m-%d %H:%M:%S_%f")}.log')


def parse_args():
    parser = argparse.ArgumentParser(description='arc service')
    parser.add_argument('--log_path', default=basedir + '/log/arc.log', type=str, required=False, help='日志路径')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3'))
    parser.add_argument('--model_path', type=str, default=basedir+'/data/pretrained_models', help='model path')
    parser.add_argument('-i', '--input', type=str, default='', help='Input folder')
    parser.add_argument('-o', '--output', type=str, default=basedir+'/data/results', help='Output folder')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--GFPGAN_path', type=str, default=basedir+'/data/pretrained_models/GFPGANv1.3.pth', help='model path')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    # parser.add_argument(
    #     '--bf', action='store_true', help='Use bfloat16 precision during inference.')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    args = parser.parse_args()
    return args
