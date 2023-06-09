# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Benchmark a YOLO model formats for speed and accuracy

Usage:
    from ultralytics.yolo.utils.benchmarks import run_benchmarks
    run_benchmarks(model='yolov8n.pt', imgsz=160)

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlmodel
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/
"""

import platform
import time
from pathlib import Path

import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.yolo.engine.exporter import export_formats
from ultralytics.yolo.utils import LOGGER, SETTINGS
from ultralytics.yolo.utils.checks import check_yolo
from ultralytics.yolo.utils.files import file_size


def run_benchmarks(model=Path(SETTINGS['weights_dir']) / 'yolov8n.pt',
                   imgsz=640,
                   half=False,
                   device='cpu',
                   hard_fail=False):
    device = torch.device(int(device) if device.isnumeric() else device)
    model = YOLO(model)

    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu) in export_formats().iterrows():  # index, (name, format, suffix, CPU, GPU)
        try:
            assert i not in (9, 10), 'inference not supported'  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == 'Darwin', 'inference only supported on macOS>=10.13'  # CoreML

            if 'cpu' in device.type:
                assert cpu, 'inference not supported on CPU'
            if 'cuda' in device.type:
                assert gpu, 'inference not supported on GPU'

            # Export
            if format == '-':
                filename = model.ckpt_path
                export = model  # PyTorch format
            else:
                filename = model.export(imgsz=imgsz, format=format, half=half, device=device)  # all others
                export = YOLO(filename)
            assert suffix in str(filename), 'export failed'

            # Validate
            if model.task == 'detect':
                data, key = 'coco128.yaml', 'metrics/mAP50-95(B)'
            elif model.task == 'segment':
                data, key = 'coco128-seg.yaml', 'metrics/mAP50-95(M)'
            elif model.task == 'classify':
                data, key = 'imagenet100', 'metrics/accuracy_top5'

            results = export.val(data=data, batch=1, imgsz=imgsz, plots=False, device=device, half=half, verbose=False)
            metric, speed = results.results_dict[key], results.speed['inference']
            y.append([name, '✅', round(file_size(filename), 1), round(metric, 4), round(speed, 2)])
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f'Benchmark --hard-fail for {name}: {e}'
            LOGGER.warning(f'ERROR ❌️ Benchmark failure for {name}: {e}')
            y.append([name, '❌', None, None, None])  # mAP, t_inference

    # Print results
    LOGGER.info('\n')
    check_yolo(device=device)  # print system info
    c = ['Format', 'Status❔', 'Size (MB)', key, 'Inference time (ms/im)'] if map else ['Format', 'Export', '', '']
    df = pd.DataFrame(y, columns=c)
    LOGGER.info(f'\nBenchmarks complete for {Path(model.ckpt_path).name} on {data} at imgsz={imgsz} '
                f'({time.time() - t0:.2f}s)')
    LOGGER.info(str(df if map else df.iloc[:, :2]))

    if hard_fail and isinstance(hard_fail, str):
        metrics = df[key].array  # values to compare to floor
        floor = eval(hard_fail)  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f'HARD FAIL: metric < floor {floor}'


if __name__ == '__main__':
    run_benchmarks()
