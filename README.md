cal4Mask:
1. 用于处理灰度分割图像（未进行数字标号）
2. 需要依赖：os, numpy, cv2, pandas
3. 直接运行cellcal.py
4. line 349~360 为Excel中输出的集合参数，根据情况做修改
5. line 525 修改为输入jpg/png图像存放的文件夹全路径
6. line 526 修改为输出图像及Excel文件的相对路径

cal4No:
1. 用于识别并分割mask图像（已进行数字标号或部分进行数字标号）
2. models文件为模型训练参数（若无.pk文件时使用，但是标号无法对应，前后帧图像同数字标号则一一对应）
3. 需要依赖：torch, cv2, sys, numpy, os, pickle, transforms
4. infer.py中用命令行运行：
   python infer.py --root ./data/ --model ./models/9800.pth --output ./res/ --save_vis False
  1) --root 数据的路径，路径下包括图片文件和pk文件（其中pk文件可存在可不存在），如果pk文件存在则基于pk文件内容处理；反之，则利用图片文件和训练好的模型进行处理；
  2) --model 指定深度模型的参数路径；
  3) --output 指定xslx文件的储存位置，如果不指定则会输出至--root指定的数据路径下
  4) --save_vis 若为True，则将图像可视化储存在--output中，包括最小外接框和序号；默认为Fasle；
