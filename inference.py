import os
import argparse
import torch
import numpy as np
import PIL.Image as Image
import torchvision.models as models

from torch import nn
from tqdm import tqdm
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import csv
from src.helper_functions.augmentations import get_eval_trnsform
from src.data_loading.data_loader import TestImageLoader

def main(args):
    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = checkGPU()
    model = loadModel(args, device)
    trans = get_eval_trnsform()
    test_images_stg1 = TestImageLoader(args.data_path + 'test_stg1', transform=trans)
    test_images_stg2 = TestImageLoader(args.data_path + 'test_stg2', transform=trans)
    submission = []
    test_loader_stg1 = data.DataLoader(
        test_images_stg1,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )
    test_loader_stg2 = data.DataLoader(
        test_images_stg2,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )
    smax = nn.Softmax(dim=1)
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"])
        print("==========stg1===========")
        for img, name in tqdm(test_loader_stg1):
            output = model(img.to(device))
            output = smax(output).detach().cpu().numpy()
            for one_output, one_name in zip(output, name):
                row = [one_name]
                for f in one_output:
                    row.append(f)
                writer.writerow(row)
        print("==========stg2===========")
        for img, name in tqdm(test_loader_stg2):
            output = model(img.to(device))
            output = smax(output).detach().cpu().numpy()
            for one_output, one_name in zip(output, name):
                row = ["test_stg2/"+one_name]
                for f in one_output:
                    row.append(f)
                writer.writerow(row)

def checkGPU():
    print("torch version:" + torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Available GPUs: ", end="")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i), end=" ")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")
    return device


def loadModel(args, device):
    with torch.no_grad():
        model = torch.load(args.model_path)
        # model.load_state_dict(torch.load(args.model_path_dict)["state_dict"])
        model.eval().to(device)
    return model


def predict(root, path, trans, device, model, class_to_idx):
    img = Image.open(os.path.join(root, path)).convert("RGB")
    data = trans(img).unsqueeze(0).to(device)
    output = model(data)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" inference")
    parser.add_argument(
        "--data_path", type=str, default="../../DATA/fish/test"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    parser.add_argument("--output", type=str, default="answer.csv")
    args = parser.parse_args()

    main(args)
