#!/usr/bin/env python3
"""用 onnxruntime 1.31.1 验证策略模型可正常加载推理（对齐官方评测环境）。

用法：
    python scripts/check_onnx.py policy.onnx
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort

path = sys.argv[1] if len(sys.argv) > 1 else 'policy.onnx'
model = onnx.load(path)
onnx.checker.check_model(model)
print('onnx 检查通过，ir_version=%d, opset=%s' % (model.ir_version,
      [(o.domain, o.version) for o in model.opset_import]))

session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
for item in session.get_inputs():
    print('input : %s %s' % (item.name, item.shape))
for item in session.get_outputs():
    print('output: %s %s' % (item.name, item.shape))

obs = np.random.randn(*session.get_inputs()[0].shape).astype(np.float32)
action = session.run(None, {session.get_inputs()[0].name: obs})[0]
print('推理通过：obs %s -> action %s' % (obs.shape, action.shape))
print('动作范围检查: min=%.3f max=%.3f' % (action.min(), action.max()))
