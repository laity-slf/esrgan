import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
print(__file__,__name__)
from gfpgan import GFPGANer
from model.realesrgan import RealESRGANer
from model.realesrgan.archs.srvgg_arch import SRVGGNetCompact
from commons.get_logger import get_logger


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
    model_path = os.path.join('../../data/pretrained_models', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('weights', args.model_name + '.pth')
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

    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=args.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)

    return upsampler, face_enhancer


def infer(args, model, logger):
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    os.makedirs(args.output, exist_ok=True)
    imgname, extension = os.path.splitext(os.path.basename(paths))
    print('Testing', imgname)

    img = cv2.imread(paths, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    try:
        if args.face_enhance:
            _, _, output = model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = model.enhance(img, outscale=args.outscale)
    except RuntimeError as error:
        logger.info('Error', error)
        logger.info('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        if args.ext == 'auto':
            extension = extension[1:]
        else:
            extension = args.ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        if args.suffix == '':
            save_path = os.path.join(args.output, f'{imgname}.{extension}')
        else:
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        cv2.imwrite(save_path, output)
    return "success"


if __name__ == '__main__':
    pass
