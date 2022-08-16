import argparse
import base64
from cmath import log
import cv2
import glob
import os

import numpy as np
from commons.utils import image_to_base64
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from model.realesrgan import RealESRGANer
from model.realesrgan.archs.srvgg_arch import SRVGGNetCompact


def initial_model(args):
    """Inference demo for Real-ESRGAN.
    """
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in ['realesr-animevideov3']:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    model_path = os.path.join(args.model_path, args.model_name + '.pth')
    # if not os.path.isfile(model_path):
    #     model_path = os.path.join('realesrgan/weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    return upsampler


def initial_model_demo(args):
    """Inference demo for Real-ESRGAN.
    """
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in ['realesr-animevideov3']:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    model_path = os.path.join(args.model_path, args.model_name + '.pth')
    # if not os.path.isfile(model_path):
    #     model_path = os.path.join('realesrgan/weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    return upsampler


def infer(args, model, logger):
    if args.type == 'path':
        if os.path.isfile(args.input):
            paths = args.input
        os.makedirs(args.output, exist_ok=True)
        imgname, extension = os.path.splitext(os.path.basename(paths))
        logger.info('Predicting  ' + imgname)

        img = cv2.imread(paths, cv2.IMREAD_UNCHANGED)

    elif args.type.startswith('base64'):
        extension = args.type[6:]
        imgname = None
        # 转换为np数组
        img_array = np.fromstring(args.input, np.uint8)
        # 转换成opencv可用格式
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None
    # 动态缩放
    max_s = max(img.shape)
    scale = 1
    if max_s <= 100:
        scale = 4
    elif max_s <= 200:
        scale = 3
    elif max_s <= 300:
        scale = 2
    logger.info(f'当前尺寸缩放比例为{scale}')

    try:
        if args.face_enhance:
            _, _, output = model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = model.enhance(img, outscale=scale)
    except RuntimeError as error:
        logger.info('Error', error)
        logger.info('If you encounter CUDA out of memory, try to set --tile with a smaller number.')

    if args.ext == 'auto':
        extension = extension[1:]
    else:
        extension = args.ext
    if img_mode == 'RGBA':  # w RGBA images should be saved in png format
        extension = 'png'
    if imgname:
        if args.suffix == '':
            save_path = os.path.join(args.output, f'{imgname}.{extension}')
        else:
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        # cv2.imwrite(save_path, output)
        # logger.info(f"图片恢复完成,输出在{save_path}")
    else:
        save_path = None
    logger.info(f"图片格式为 {extension}")
    image_code = image_to_base64(output, extension)
    return image_code, save_path


if __name__ == '__main__':
    pass
