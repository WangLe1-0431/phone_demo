# 点云分割demo By WangLe1_
此项目主要用于：通过**Orbbec Gemini 335l**相机，基于YOLOv8实现物品的分割和点云的生成。
## 快速开始
### 环境配置
此项目测试环境基于Windows11 24H2，`Python == 3.9.2` 。
```
pip install -r requirements.txt
```
需要说明的是：**`requirements.txt`文件中的版本为本人电脑上的版本**，若报错则需要自行修改。

### 一键启动
```
python sync_align_yolo_phone.py --model 选择合适的模型
```
