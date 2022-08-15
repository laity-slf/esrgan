import sys
import os
import logging
import gradio as gr
import cv2

sys.path.insert(0, sys.path[0] + '/../')
sys.path.insert(0, sys.path[0] + '/commons/')
from conf.params import parse_args
from commons.get_logger import get_logger
from model.inference_realesrgan import initial_model_demo

args = parse_args()
logger = get_logger(args.log_path, name='arc')
handler = logging.FileHandler(args.log_path)
model_= initial_model_demo(args)
logger.info('模型加载成功')


def infer_demo(img, model=model_):
    try:
        if args.face_enhance:
            _, _, output = model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = model.enhance(img, outscale=4)
    except RuntimeError as error:
        logger.info('Error', error)
        logger.info('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        logger.info("success enhance")
        return output


if __name__ == "__main__":
    interface = gr.Interface(fn=infer_demo, inputs="image", outputs="image")
    interface.launch(share=True)
