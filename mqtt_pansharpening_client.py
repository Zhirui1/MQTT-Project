"""
MQTT 全色锐化客户端 - 部署在本地电脑
发送 MS 和 PAN 到远程服务器，接收融合图和黑白误差图并保存
"""

import argparse
import json
import base64
import numpy as np
import paho.mqtt.client as mqtt
import uuid
import time
import os
from scipy import io
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('MQTT Pansharpening Client (Local)')
    parser.add_argument('--broker', default='localhost', help='MQTT broker 地址（远程服务器 IP 或域名）')
    parser.add_argument('--port', default=1883, type=int, help='MQTT 端口')
    parser.add_argument('--topic_request', default='pansharpening/request', help='请求主题')
    parser.add_argument('--topic_response', default='pansharpening/response', help='响应主题')
    parser.add_argument('--ms_path', type=str, help='MS 图像路径 (.mat 或 .npy)')
    parser.add_argument('--pan_path', type=str, help='PAN 图像路径 (.mat 或 .npy)')
    parser.add_argument('--mat_path', type=str, help='含 I_MS 和 I_PAN 的 .mat 路径（可替代 ms_path+pan_path）')
    parser.add_argument('--output_dir', default='./pansharpening_output', help='输出目录')
    parser.add_argument('--client_id', default='local_client', help='客户端 ID')
    parser.add_argument('--timeout', default=60, type=int, help='等待响应超时秒数')
    parser.add_argument('--max_val', default=2047.0, type=float, help='最大像素值 (WV3=2047)')
    return parser


def load_ms(path):
    """加载 MS，返回 (H, W, C) float32"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        mat = io.loadmat(path)
        if 'I_MS' in mat:
            ms = mat['I_MS']  # (H, W, C)
        elif 'ms' in mat:
            ms = mat['ms']
        else:
            key = [k for k in mat.keys() if not k.startswith('_')][0]
            ms = mat[key]
        if ms.ndim == 3 and ms.shape[2] < ms.shape[0]:
            pass  # (H,W,C)
        elif ms.ndim == 3 and ms.shape[0] < ms.shape[2]:
            ms = ms.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        return ms.astype(np.float32)
    elif ext == '.npy':
        ms = np.load(path).astype(np.float32)
        if ms.ndim == 3 and ms.shape[0] < ms.shape[2]:
            ms = ms.transpose(1, 2, 0)
        return ms
    else:
        raise ValueError(f"Unsupported format: {ext}")


def load_pan(path):
    """加载 PAN，返回 (H, W) 或 (1, H, W) float32"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        mat = io.loadmat(path)
        if 'I_PAN' in mat:
            pan = mat['I_PAN']
        elif 'pan' in mat:
            pan = mat['pan']
        else:
            key = [k for k in mat.keys() if not k.startswith('_')][0]
            pan = mat[key]
        if pan.ndim == 3:
            pan = pan.squeeze()
        return pan.astype(np.float32)
    elif ext == '.npy':
        pan = np.load(path).astype(np.float32)
        if pan.ndim == 3:
            pan = pan.squeeze()
        return pan
    else:
        raise ValueError(f"Unsupported format: {ext}")


def save_results(fused, error_map, output_dir, prefix='result'):
    """保存融合图和误差图"""
    os.makedirs(output_dir, exist_ok=True)

    # 融合图：取 RGB 通道 (4,2,1 for WV3) 保存为彩色图
    if fused.shape[2] >= 4:
        rgb = fused[:, :, [3, 1, 0]]  # band 4,2,1 -> R,G,B
    else:
        rgb = fused[:, :, :3]
    rgb = np.clip(rgb, 0, 1)
    fused_path = os.path.join(output_dir, f'{prefix}_fused.png')
    plt.imsave(fused_path, rgb)
    print(f"Saved fused: {fused_path}")

    # 误差图：黑白
    err_path = os.path.join(output_dir, f'{prefix}_error.png')
    plt.imsave(err_path, error_map, cmap='gray')
    print(f"Saved error map: {err_path}")

    # 同时保存原始数组供后续使用
    np.save(os.path.join(output_dir, f'{prefix}_fused.npy'), fused)
    np.save(os.path.join(output_dir, f'{prefix}_error.npy'), error_map)


def run_client(args):
    result_holder = {'response': None}
    request_id = str(uuid.uuid4())

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(args.topic_response)

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode('utf-8'))
        if payload.get('request_id') != request_id:
            return  # 忽略其他请求的响应
        if 'error' in payload:
            result_holder['response'] = {'error': payload['error']}
        else:
            result_holder['response'] = payload

    # 加载数据
    if args.mat_path:
        mat = io.loadmat(args.mat_path)
        ms = mat['I_MS'].astype(np.float32) if 'I_MS' in mat else mat['ms'].astype(np.float32)
        pan = mat['I_PAN'].squeeze().astype(np.float32) if 'I_PAN' in mat else mat['pan'].squeeze().astype(np.float32)
        if ms.ndim == 3 and ms.shape[0] < ms.shape[2]:
            ms = ms.transpose(1, 2, 0)
        print(f"Loaded from {args.mat_path}: MS {ms.shape}, PAN {pan.shape}")
    else:
        if not args.ms_path or not args.pan_path:
            raise ValueError("请指定 --mat_path 或同时指定 --ms_path 和 --pan_path")
        print(f"Loading MS from {args.ms_path}...")
        ms = load_ms(args.ms_path)
        print(f"Loading PAN from {args.pan_path}...")
        pan = load_pan(args.pan_path)
    print(f"MS shape: {ms.shape}, PAN shape: {pan.shape}")

    # 编码
    ms_b64 = base64.b64encode(ms.astype(np.float32).tobytes()).decode('utf-8')
    pan_b64 = base64.b64encode(pan.astype(np.float32).tobytes()).decode('utf-8')

    payload = {
        'request_id': request_id,
        'client_id': args.client_id,
        'ms': ms_b64,
        'pan': pan_b64,
        'ms_shape': list(ms.shape),
        'pan_shape': list(pan.shape),
        'max_val': args.max_val,
    }

    client = mqtt.Client(client_id=f"{args.client_id}_{uuid.uuid4().hex[:8]}")
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {args.broker}:{args.port}...")
    client.connect(args.broker, args.port, 60)
    client.loop_start()

    # 等待连接
    time.sleep(1)
    print("Publishing request...")
    client.publish(args.topic_request, json.dumps(payload), qos=1)

    # 等待响应
    start = time.time()
    while result_holder['response'] is None and (time.time() - start) < args.timeout:
        time.sleep(0.1)

    client.loop_stop()
    client.disconnect()

    resp = result_holder['response']
    if resp is None:
        print("Timeout: No response received.")
        return

    if 'error' in resp:
        print(f"Server error: {resp['error']}")
        return

    # 解码
    fused_bytes = base64.b64decode(resp['fused'])
    fused = np.frombuffer(fused_bytes, dtype=np.float32).reshape(resp['fused_shape'])
    err_bytes = base64.b64decode(resp['error_map'])
    error_map = np.frombuffer(err_bytes, dtype=np.float32).reshape(resp['error_map_shape'])

    print("Received results. Saving...")
    save_results(fused, error_map, args.output_dir, prefix=request_id[:8])
    print("Done.")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    run_client(args)


if __name__ == '__main__':
    main()
