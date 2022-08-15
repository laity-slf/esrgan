import sys
import os
import logging
from flask import Flask, request, jsonify
sys.path.insert(0, sys.path[0] + '/../')
sys.path.insert(0, sys.path[0] + '/commons/')
from conf.params import parse_args
from commons.get_logger import get_logger
from model.inference_realesrgan import initial_model, infer

app = Flask('esrgan')
app.config['JSON_AS_ASCII'] = False

@app.route('/nlp/arc', methods=['POST'])
def arc():
    try:
        data = request.get_json()
        img_file = data.get('input')
        base_name = os.path.dirname(img_file)
        app.logger.info('成功接收图片路径: ' + img_file)
        out_path = data.get('output') if data.get('output') else os.path.join(base_name, 'results')
        do_face_enhance = True if data.get("face_enhance") == 1 else False
        # 是否存在图片文件
        img_file = os.path.abspath(img_file)
        if not os.path.isfile(img_file):
            app.logger.info('Invalid audio_file')
            return jsonify(status=112002, msg='Invalid img_file')
        # 参数写入arg
        args.input = img_file
        args.output = out_path
        args.face_enhance = do_face_enhance
        # arc
        res = infer(args, model, logger)
        return jsonify(status=200,output_path=res, msg="success")

    except Exception as e:
        app.logger.info(e)
        return jsonify(status=112001, msg='fail')


# 执行
if __name__ == "__main__":
    args = parse_args()
    logger = get_logger(args.log_path, name='arc')
    handler = logging.FileHandler(args.log_path)
    model = initial_model(args)
    logger.info('模型加载成功')
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.run(
        host='0.0.0.0',
        port=5073
    )
