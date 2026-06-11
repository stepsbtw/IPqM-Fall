import os
import torch
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from pathlib import Path
from modelos import CNN1Conv

CHECKPOINT_DIR = Path("checkpoints")
TFLITE_DIR = Path("tflite_models")
TFLITE_DIR.mkdir(exist_ok=True)

COMBINACOES = {
    "ALL": 24,
    "CHEST_LEFT": 16,
    "CHEST_RIGHT": 16,
    "LEFT_RIGHT": 16,
    "CHEST": 8,
    "LEFT": 8,
    "RIGHT": 8
}

def representative_dataset_gen(num_features):
    def _gen():
        for _ in range(100):
            yield [np.random.normal(0, 1, (1, num_features, 180)).astype(np.float32)]
    return _gen

for nome_modelo, num_features in COMBINACOES.items():
    pth_path = CHECKPOINT_DIR / nome_modelo / f"{nome_modelo}_FINAL.pth"
    onnx_path = TFLITE_DIR / f"{nome_modelo}.onnx"
    tf_saved_model_dir = TFLITE_DIR / f"{nome_modelo}_tf"
    tflite_path = TFLITE_DIR / f"{nome_modelo}_quantizado.tflite"

    if not pth_path.exists():
        continue

    try:
        modelo_pt = CNN1Conv(num_features)
        modelo_pt.load_state_dict(torch.load(pth_path, map_location="cpu"))
        modelo_pt.eval()

        dummy_input = torch.randn(1, num_features, 180, dtype=torch.float32)

        torch.onnx.export(
            modelo_pt, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen(num_features)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

    except Exception:
        pass
