"""
MQTT 全色锐化客户端 - 本地部署
复制此文件到本地电脑，pip install -r requirements.txt 后运行
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


def main():
    parser = argparse.ArgumentParser('MQTT Pansharpening Client (Local)')
    parser.add_argument('--broker', default='localhost', help='远程服务器 IP 或域名')
    parser.add_argument('--port', default=1883, type=int)
    parser.add_argument('--ms_path', type=str, help='MS 图像路径')
    parser.add_argument('--pan_path', type=str, help='PAN 图像路径')
    parser.add_argument('--mat_path', type=str, help='含 I_MS 和 I_PAN 的 .mat（可替代上面两个）')
    parser.add_argument('--output_dir', default='./pansharpening_output')
    parser.add_argument('--timeout', default=60, type=int)
    parser.add_argument('--max_val', default=2047.0, type=float)
    args = parser.parse_args()

    result_holder = {'response': None}
    request_id = str(uuid.uuid4())

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe('pansharpening/response')

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode('utf-8'))
        if payload.get('request_id') != request_id:
            return
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
    else:
        if not args.ms_path or not args.pan_path:
            print("请指定 --mat_path 或 --ms_path 与 --pan_path")
            return
        if args.ms_path.endswith('.mat'):
            m = io.loadmat(args.ms_path)
            ms = (m['I_MS'] if 'I_MS' in m else m['ms']).astype(np.float32)
        else:
            ms = np.load(args.ms_path).astype(np.float32)
        if ms.ndim == 3 and ms.shape[0] < ms.shape[2]:
            ms = ms.transpose(1, 2, 0)
        if args.pan_path.endswith('.mat'):
            p = io.loadmat(args.pan_path)
            pan = (p['I_PAN'] if 'I_PAN' in p else p['pan']).squeeze().astype(np.float32)
        else:
            pan = np.load(args.pan_path).squeeze().astype(np.float32)

    print(f"MS shape: {ms.shape}, PAN shape: {pan.shape}")

    payload = {
        'request_id': request_id,
        'client_id': 'local',
        'ms': base64.b64encode(ms.astype(np.float32).tobytes()).decode('utf-8'),
        'pan': base64.b64encode(pan.astype(np.float32).tobytes()).decode('utf-8'),
        'ms_shape': list(ms.shape),
        'pan_shape': list(pan.shape),
        'max_val': args.max_val,
    }

    client = mqtt.Client(client_id=f"client_{uuid.uuid4().hex[:8]}")
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"Connecting to {args.broker}:{args.port}...")
    client.connect(args.broker, args.port, 60)
    client.loop_start()
    time.sleep(1)
    print("Sending request...")
    client.publish('pansharpening/request', json.dumps(payload), qos=1)

    start = time.time()
    while result_holder['response'] is None and (time.time() - start) < args.timeout:
        time.sleep(0.1)
    client.loop_stop()
    client.disconnect()

    resp = result_holder['response']
    if resp is None:
        print("Timeout.")
        return
    if 'error' in resp:
        print(f"Error: {resp['error']}")
        return

    fused = np.frombuffer(base64.b64decode(resp['fused']), dtype=np.float32).reshape(resp['fused_shape'])
    error_map = np.frombuffer(base64.b64decode(resp['error_map']), dtype=np.float32).reshape(resp['error_map_shape'])

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = request_id[:8]
    rgb = np.clip(fused[:, :, [3, 1, 0]] if fused.shape[2] >= 4 else fused[:, :, :3], 0, 1)
    plt.imsave(os.path.join(args.output_dir, f'{prefix}_fused.png'), rgb)
    plt.imsave(os.path.join(args.output_dir, f'{prefix}_error.png'), error_map, cmap='gray')
    print(f"Saved to {args.output_dir}/{prefix}_fused.png and {prefix}_error.png")


if __name__ == '__main__':
    main()
