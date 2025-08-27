import os
import argparse
import numpy as np
import time

import torch
import torch.nn as nn

from model_update import create_model


# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Modified DeepGRU Test')
parser.add_argument('--ckpt', type=str, default="checkpoint")
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)


# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
use_cuda = torch.cuda.is_available() and args.use_cuda

ckpt_path = f"./checkpoints/{args.ckpt}.tar"
seq_len = 12

idx_to_class = {0: 'CClock_index', 1: 'CClock_thumb',
                2: 'Clock_index', 3: 'Clock_thumb',
                4: 'Down_index', 5: 'Down_thumb',
                6: 'Left_index', 7: 'Left_thumb',
                8: 'Natural_index', 9: 'Natural_thumb',
                10: 'Right_index', 11: 'Right_thumb',
                12: 'Tap_index', 13: 'Tap_thumb',
                14: 'Up_index', 15: 'Up_thumb'}

# ----------------------------------------------------------------------------------------------------------------------
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the dataset



    # Instantiate the model
    model = create_model(num_features=78, num_classes=16)
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['model_state_dict']

    # "module." prefix가 있는지 확인
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    # prefix 제거
    if has_module_prefix:
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # Create sample data
    sample_pth = "./sample/1756185457.npy"
    sample_raw = np.load(sample_pth)

    data_action = []
    num_seq = len(sample_raw) // 2
    num_seq -= seq_len // 2
    for seq in range(num_seq):
        data_action.append(sample_raw[seq * 2:seq * 2 + seq_len])
    data_action.append(sample_raw[-seq_len:])


    with torch.no_grad():
        for x in data_action:
            input = torch.from_numpy(x).to(device).unsqueeze(0).float()

            outputs = model(input)

            pred = outputs.argmax(1).cpu().numpy()
            gesture = idx_to_class[pred[0]]

            ## visualize
            print("gesture : ", gesture)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
