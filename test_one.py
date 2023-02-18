import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from rotate_captcha_crack.config import CONFIG, device
from rotate_captcha_crack.helper import DEFAULT_NORM
from rotate_captcha_crack.model import RotationNet
from rotate_captcha_crack.utils import find_out_model_path

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", "-ts", type=int, default=0, help="Use which timestamp")
parser.add_argument("--epoch", type=int, default=0, help="Use which epoch")
opts = parser.parse_args()

if __name__ == "__main__":
    img_size = CONFIG.dataset.img_size
    trans = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            DEFAULT_NORM,
        ]
    )

    with torch.no_grad():
        model_dir = Path("models")
        model = RotationNet(train=False)
        model_path = find_out_model_path(opts.timestamp, opts.epoch)
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model = model.to(device)
        model.eval()
        img = Image.open("datasets/tieba/1615096451.jpg")

        img_tensor: Tensor = trans(img).unsqueeze_(0).to(device)
        predict: Tensor = model(img_tensor)
        degree: float = predict.cpu().item() * 360
        print(f"Predict degree: {degree:.4f}")

    img = img.rotate(
        -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
    )  # use neg degree to recover the img
    plt.figure("debug")
    plt.imshow(img)
    plt.show()