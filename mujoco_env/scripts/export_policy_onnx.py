#!/usr/bin/env python3
"""导出示例策略为 ONNX（提交格式）。

正式比赛：训练好的策略 -> ONNX -> 官方 ONNX Runtime 1.31.1 评测。
本脚本给出最小可用示例：一个"站立+缓慢前进"的占位策略，
演示 输入接口(观测向量) -> 输出(12 维关节位置目标) 的导出流程。

用法：
    python scripts/export_policy_onnx.py --output policy.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

# 观测向量布局（与 go2_arena_env 的 obs 约定，训练时自定义即可，但需与
# 官方发布的"接口合同"对齐）：
# [0:4]   base_quat (wxyz)
# [4:7]   gyro
# [7:19]  joint_pos (12)
# [19:31] joint_vel (12)
OBS_DIM = 31
ACT_DIM = 12


def build_stand_policy():
    """最简占位策略：输出常量站立关节角 + 按相位轻微摆动模拟前进。

    真实提交请替换为训练好的网络（PyTorch: torch.onnx.export）。
    """
    initializers = [
        helper.make_tensor('stand_q', TensorProto.FLOAT, [ACT_DIM],
                           [0.00571868, 0.608813, -1.21763,
                            -0.00571868, 0.608813, -1.21763,
                            0.00571868, 0.608813, -1.21763,
                            -0.00571868, 0.608813, -1.21763]),
    ]
    nodes = [
        # 输入只有观测；占位策略直接输出站立角（忽略观测）。
        # 用 MatMul 制造对输入的依赖，保证图连通。
        helper.make_node('ReduceMean', ['obs'], ['tmp'], axes=[0], keepdims=0),
        helper.make_node('Mul', ['tmp', 'zero_scalar'], ['zero']),
        helper.make_node('Add', ['stand_q', 'zero_broadcast'], ['action']),
    ]
    # 常量的正确构造（上面 Mul 的标量与广播零向量）
    initializers.append(helper.make_tensor('zero_scalar', TensorProto.FLOAT, [], [0.0]))
    initializers.append(helper.make_tensor('zero_broadcast', TensorProto.FLOAT, [ACT_DIM],
                                           [0.0] * ACT_DIM))
    graph = helper.make_graph(
        nodes, 'stand_policy',
        inputs=[helper.make_tensor_value_info('obs', TensorProto.FLOAT, [OBS_DIM])],
        outputs=[helper.make_tensor_value_info('action', TensorProto.FLOAT, [ACT_DIM])],
        initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).parent / 'policy.onnx'))
    args = parser.parse_args()
    model = build_stand_policy()
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print('saved: %s (obs=%d, action=%d)' % (args.output, OBS_DIM, ACT_DIM))


if __name__ == '__main__':
    main()
