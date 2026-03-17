"""
MQTT 全色锐化服务端 - 部署在远程算例
接收本地发送的 MS 和 PAN，运行模型推理，返回融合图和黑白误差图
"""

import argparse
import json
import base64
import time
import numpy as np
import torch
import torch.nn.functional as F
import paho.mqtt.client as mqtt
import sys
import os

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.PreMixHuge import GreHuge


def get_args_parser():
    parser = argparse.ArgumentParser('MQTT Pansharpening Server')
    parser.add_argument('--broker', default='localhost', help='MQTT broker 地址')
    parser.add_argument('--port', default=1883, type=int, help='MQTT 端口')
    parser.add_argument('--topic_request', default='pansharpening/request', help='请求主题')
    parser.add_argument('--topic_response', default='pansharpening/response', help='响应主题')
    parser.add_argument('--ckpt', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--ms_chans', default=8, type=int)
    parser.add_argument('--embed_dim', default=32, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--pf_kernel', default=3, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--EWFM', action='store_true')
    parser.add_argument('--activation', default=None, type=str)
    parser.add_argument('--beta', default=None, type=float)
    parser.add_argument('--sensor', default='wv3', type=str)
    parser.add_argument('--rgb_c', default='4,2,1', type=str)
    parser.add_argument('--max_val', default=2047.0, type=float, help='WV3 最大像素值')
    return parser


def load_model(args):
    """加载 PreMixHuge 模型"""
    model = GreHuge.load_from_checkpoint(
        args.ckpt,
        map_location='cpu',
        lr=1e-3,
        epochs=300,
        bands=args.ms_chans,
        rgb_c=args.rgb_c,
        sensor=args.sensor,
        embed_dim=args.embed_dim,
        kernel_size=args.kernel_size,
        pf_kernel=args.pf_kernel,
        enable_EWFM=args.EWFM,
        num_layers=args.num_layers,
        beta=args.beta,
        act=args.activation,
    )
    model.eval()
    return model


def pansharpen(model, ms_np, pan_np, max_val, device):
    """
    ms_np: (H_ms, W_ms, C) 如 (16, 16, 8)
    pan_np: (H_pan, W_pan) 或 (1, H_pan, W_pan)
    """
    with torch.no_grad():
        # 归一化
        ms = torch.from_numpy((ms_np.astype(np.float32) / max_val)).permute(2, 0, 1).unsqueeze(0).to(device)
        if pan_np.ndim == 2:
            pan = torch.from_numpy((pan_np.astype(np.float32) / max_val)).unsqueeze(0).unsqueeze(0).to(device)
        else:
            pan = torch.from_numpy((pan_np.astype(np.float32) / max_val)).reshape(1, 1, *pan_np.shape[-2:]).to(device)

        # 上采样 MS 到 PAN 尺寸得到 LMS
        up_ms = F.interpolate(ms, size=pan.shape[-2:], mode='bicubic', align_corners=False)

        # 推理
        out, _, _ = model(up_ms, ms, pan)
        pred = out['pred']  # (1, C, H, W)

        # 融合图: (C, H, W) 转 (H, W, C)
        fused = pred[0].permute(1, 2, 0).cpu().numpy()

        # 黑白误差图: |pred - up_ms| 各通道平均，归一化到 [0,1]
        up_ms_single = up_ms[0]  # (C, H, W)
        err = torch.abs(pred[0] - up_ms_single).mean(dim=0).cpu().numpy()  # (H, W)
        err_max = err.max()
        if err_max > 0:
            err = err / err_max
        error_map = np.clip(err, 0, 1).astype(np.float32)

    return fused, error_map


class PansharpeningServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {args.ckpt} on {self.device}...")
        self.model = load_model(args).to(self.device)
        self.client = mqtt.Client(client_id="pansharpening_server")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to broker. Subscribing to {self.args.topic_request}")
            client.subscribe(self.args.topic_request)
        else:
            print(f"Connection failed, code: {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            request_id = payload.get('request_id', 'unknown')
            client_id = payload.get('client_id', 'unknown')

            ms_b64 = payload['ms']
            pan_b64 = payload['pan']
            max_val = payload.get('max_val', self.args.max_val)

            # 解码
            ms_bytes = base64.b64decode(ms_b64)
            pan_bytes = base64.b64decode(pan_b64)

            ms_np = np.frombuffer(ms_bytes, dtype=np.float32).reshape(payload['ms_shape'])
            pan_np = np.frombuffer(pan_bytes, dtype=np.float32).reshape(payload['pan_shape'])

            # 推理（记录耗时）
            t0 = time.perf_counter()
            fused, error_map = pansharpen(
                self.model, ms_np, pan_np, max_val, self.device
            )

            # 编码响应
            fused_b64 = base64.b64encode(fused.astype(np.float32).tobytes()).decode('utf-8')
            err_b64 = base64.b64encode(error_map.tobytes()).decode('utf-8')

            processing_time_ms = int((time.perf_counter() - t0) * 1000)
            response = {
                'request_id': request_id,
                'client_id': client_id,
                'fused': fused_b64,
                'fused_shape': list(fused.shape),
                'error_map': err_b64,
                'error_map_shape': list(error_map.shape),
                'processing_time_ms': processing_time_ms,
            }

            self.client.publish(
                self.args.topic_response,
                json.dumps(response),
                qos=1
            )
            print(f"Processed request {request_id} for client {client_id}")

        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            try:
                err_response = {
                    'request_id': payload.get('request_id', 'unknown'),
                    'client_id': payload.get('client_id', 'unknown'),
                    'error': str(e),
                }
                self.client.publish(self.args.topic_response, json.dumps(err_response), qos=1)
            except Exception:
                pass

    def run(self):
        print(f"Connecting to {self.args.broker}:{self.args.port}...")
        self.client.connect(self.args.broker, self.args.port, 60)
        self.client.loop_forever()


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    server = PansharpeningServer(args)
    server.run()


if __name__ == '__main__':
    main()
