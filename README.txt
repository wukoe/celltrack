各个阶段的使用方法

# cell mask 优化和 track 部分
在得到微调后的SAM等模型输出的mask（无论针对荧光还是明场图像）之后，通过使用 celltrack.ipynb 的若干cell实现处理。

## 针对明场多帧图像，在有人工点标注数据条件下的操作流程
1. 执行 cell "load and process data" 
—— 先在其中修改得到的mask文件路径：
  video_dir：原始多帧图像文件目录
  annotation_dir：标注文件路径，在现有的目录组织结构下一般和video_dir在同一目录
  cell_mask_dir：SAM模型预测得到的各帧png图像所在目录。
  output_dir：输出的优化结果保存目录

2. 执行 cell "load video frames and point coordinates."
—— 修改dscode和fcode
  fcode：对应不同药物浓度的 0,1,5几个目录
  dscode：对应最下面具样本所在目录的名称。

3. 执行 cell "get segmentation mask."

4. 执行 cell "optimize mask and sequence"
这一步将人工标注的coord序号和预测得到mask关联起来，为mask赋予正确的ID号。
同时实现 mask的过滤、优化等操作，以及track之间的补帧。

4. 执行 cell "保存细胞轨迹结果"
保存优化后的ID值正确的新mask。

