import base64
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
        type = data.get('type')
        if type == 'path':
            img_file = data.get('input')
            base_name = os.path.dirname(img_file)
            app.logger.info('成功接收图片路径: ' + img_file)
            out_path = data.get('output') if data.get('output') else os.path.join(base_name, 'results')
            # 是否存在图片文件
            img_file = os.path.abspath(img_file)
            if not os.path.isfile(img_file):
                app.logger.info('Invalid image_file')
                return jsonify(status=112002, msg='Invalid img_file')
            # 参数写入arg
            args.input = img_file
            args.output = out_path
        elif type.startswith('base64'):
            try:
                img_data = base64.b64decode(data.get('input'))
            except Exception as e:
                raise Exception("不是 base64 编码")
            # 参数写入arg
            args.input = img_data
            args.output = None
        else:
            return jsonify(status=112002, msg='None type')

        # type 参数
        args.type = type
        # arc
        r1, r2 = infer(args, model, logger)
        if not r1:
            return jsonify(status=112002, data=None, msg=r2)
        data = {
            "base64": r1
        }
        return jsonify(status=200, data=data, msg="success")

    except Exception as e:
        app.logger.info(e)
        return jsonify(status=112001, data=None, msg='fail', )


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
