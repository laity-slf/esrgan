#项目介绍
> 图片超分辨率恢复

# 启动说明

        1、安装python>=3.7
        2、创建虚拟环境
        3、在项目目录下执行 
            bash install.sh
        4、运行 
            make run_esr

# 接口返回说明

ESR接口

        请求类型：post 
        功能：输入一张图片的地址，输出进行相应分辨率恢复的图片地址
        请求输入输出参数如下：
        请求参数说明：
        input：输入图片地址
        output：输出图片地址(默认输出在输入地址同级目录的results文件夹中)
        type:输入类型，本地路径或者base64.文件格式

输入

        {
            "input":"/home/shanhoo4/slf/esrgan/0014.jpg"
            "type":"path"
        }
        {
            "input":"/home/shanhoo4/slf/esrgan/0014.jpg"
        }

输出

        {
            "msg": "success",
            "output_path": "/home/shanhoo4/slf/esrgan/results/0014_out.jpg",
            "status": 200
        }

# 调用方式

	    测试环境式例（图片分辨率恢复）：curl -H "Accept: application/json" -H "Content-type: application/json" -X POST  -d ' {"input":"/home/shanhoo4/slf/esrgan/0014.jpg"}'  http://localhost:5073/nlp/arc
