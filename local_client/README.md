# 本地客户端 - 复制到本地电脑使用

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 使用 .mat 文件（含 I_MS 和 I_PAN）
python mqtt_client.py --broker 192.168.1.100 --mat_path /path/to/data.mat

# 或分别指定 MS 和 PAN
python mqtt_client.py --broker 192.168.1.100 --ms_path ms.mat --pan_path pan.mat

# 指定输出目录
python mqtt_client.py --broker 192.168.1.100 --mat_path data.mat --output_dir ./results
```

将 `192.168.1.100` 替换为远程算例的 IP 地址。
