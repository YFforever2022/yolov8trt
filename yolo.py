"""
A code for training, predicting, verifying, and exporting models using yaml
"""
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils.checks import check_yaml

import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="你应该附带这些参数")
    parse.add_argument('--mode', default="train", help="默认为train，模式总共有train(训练)、predict(预测)、val(验证)、export(模型转换)")
    parse.add_argument('--cfg', default="cfg.yaml", help="默认为运行目录下的cfg.yaml文件")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    m_cfg = args.cfg
    m_mode = args.mode

    m_overrides = yaml_load(check_yaml(m_cfg), append_filename=True)
    m_model_file = m_overrides['model']
    model = YOLO(m_model_file)

    if m_mode == "train":
        results = model.train(cfg = m_cfg)
    elif m_mode == "predict":
        results = model.predict(cfg = m_cfg)
    elif m_mode == "val":
        results = model.val(cfg=m_cfg)
    elif m_mode == "export":
        results = model.export(cfg=m_cfg)
