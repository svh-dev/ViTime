import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.model import ViTime


def main(modelpath, savepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(modelpath, map_location=device)
    args = checkpoint['args']
    args.device = device
    args.flag = 'test'

    # Set upscaling parameters
    args.upscal = True  # True: max input length = 512, max prediction length = 720
                        # False: max input length = 1024, max prediction length = 1440
    model = ViTime(args=args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # Example data
    xData = np.sin(np.arange(512) / 10) + np.sin(np.arange(512) / 5 + 50) + np.cos(np.arange(512) + 50)
    args.realInputLength = len(xData)
    yp = model.inference(xData)

    # Plot results
    plt.plot(np.concatenate([xData, yp.flatten()], axis=0), label='Prediction')
    plt.plot(xData, label='Input Sequence')
    plt.legend()
    plt.savefig(savepath)  # 保存图形到指定路径
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViTime model inference')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--savepath', type=str, default='plot.png', help='Path to save the plot image (default: plot.png)')
    args = parser.parse_args()
    main(args.modelpath, args.savepath)
