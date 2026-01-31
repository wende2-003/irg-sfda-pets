# PETS + IRG 训练运行指南（FoggyCityscapes）

开始训练的流程如下（按顺序执行）：
1. 激活环境
   source /root/miniconda3/etc/profile.d/conda.sh
   conda activate irg_sfda

2. 设置数据集根目录
   export DETECTRON2_DATASETS=/root/autodl-tmp

3. 进入项目目录
   cd /root/irg-sfda

4. （可选）清理旧训练输出
   rm -rf /root/irg-sfda/checkpoint/foggy

5. 启动训练（正式训练）
   CUDA_VISIBLE_DEVICES=0 python tools/train_st_sfda_net.py \
     --config-file detectron2/model_zoo/configs/sfda/sfda_foggy.yaml \
     --model-dir /root/autodl-tmp/model_final.pth

6. 观察日志
   tail -f /root/irg-sfda/checkpoint/foggy/log.txt

说明
- 训练仅使用 FoggyCityscapes（目标域），Cityscapes 权重只用于初始化。
- PETS 关键超参在 config 中：CONF_THRESH、IOU_THRESH、BETA、EMA_KEEP_RATE、WARMUP_EPOCHS。
- 训练结果输出到：/root/irg-sfda/checkpoint/foggy
- 训练过程曲线输出到：/root/irg-sfda/checkpoint/foggy/*.png

常用改动位置
- detectron2/model_zoo/configs/sfda/sfda_foggy.yaml
  - SOLVER.MAX_ITER
  - SOLVER.BASE_LR
  - SOLVER.IMS_PER_BATCH
  - TEST.EVAL_PERIOD
  - SOLVER.CHECKPOINT_PERIOD
  - SOURCE_FREE.PETS.*

可选运行（评估动态教师）
   python tools/plain_test_net.py \
     --config-file detectron2/model_zoo/configs/sfda/sfda_foggy.yaml \
     --num-gpus 1 \
     MODEL.WEIGHTS /root/irg-sfda/checkpoint/foggy/model_teacher_10.pth
