
import os
import h5py
import scipy.io as sio
import numpy as np

input_dir = "/map-vepfs/liniuniu/mengming/xzr/Test/training_wv3"
output_root = "/map-vepfs/liniuniu/mengming/xzr/Test/dataset"

for h5_name, split in [("train_wv3.h5", "train"), ("valid_wv3.h5", "val")]:
    split_dir = os.path.join(output_root, split)
    os.makedirs(split_dir, exist_ok=True)

    h5_path = os.path.join(input_dir, h5_name)
    with h5py.File(h5_path, 'r') as f:
        ms_all = f['ms'][()]       # (N, C, 16, 16)
        pan_all = f['pan'][()]     # (N, 1, 64, 64)
        gt_all = f['gt'][()]       # (N, C, 64, 64)
        lms_all = f['lms'][()]     # (N, C, 64, 64)

        N = ms_all.shape[0]
        print(f"{h5_name}: {N} samples, MS={ms_all.shape}, PAN={pan_all.shape}, GT={gt_all.shape}, LMS={lms_all.shape}")

        for n in range(N):
            ms = np.transpose(ms_all[n], (1, 2, 0))       # (16, 16, C)
            pan = pan_all[n, 0]                             # (64, 64)
            gt = np.transpose(gt_all[n], (1, 2, 0))        # (64, 64, C)
            lms = np.transpose(lms_all[n], (1, 2, 0))      # (64, 64, C)

            sio.savemat(
                os.path.join(split_dir, f"{n:05d}.mat"),
                {"I_MS": ms, "I_PAN": pan, "I_GT": gt, "I_LMS": lms}
            )

        print(f"  -> Saved to {split_dir}/")

print("Done.")
