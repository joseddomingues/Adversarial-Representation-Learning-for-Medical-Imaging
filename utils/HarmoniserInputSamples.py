import math
import sys

sys.path.append("../")

import MedSinGAN.functions as functions
from MedSinGAN.config import get_arguments

if __name__ == "__main__":

    parser = get_arguments()
    parser.add_argument('--input_name', type=str, default="")
    parser.add_argument('--nc_im', type=int, default=3)
    parser.add_argument('--not_cuda', type=bool, default=False)

    parser.add_argument('--im_max_size', type=int, default=614)
    parser.add_argument('--train_stages', type=int, default=24)
    parser.add_argument('--im_min_size', type=int, default=120)

    parser.add_argument('--train_mode', type=str, default="harmonization")

    opt = parser.parse_args()

    # Reads the image
    # Needs: input_name, nc_im, not_cuda
    real = functions.read_image(opt)

    # Adjusts the scales of the image
    # Needs: im_max_size, scale1, stop_scale, train_stages, scale_factor, im_min_size,
    real = functions.adjust_scales2image(real, opt)
    opt.scale1 = min(opt.im_max_size / max([real.shape[2], real.shape[3]]), 1)
    opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = math.pow(opt.im_min_size / (min(real.shape[2], real.shape[3])), 1 / opt.stop_scale)

    # Create the scales reals pyramids
    # Needs: train_mode, stop_scale, scale_factor,
    reals = functions.create_reals_pyramid(real, opt)

    img_to_augment = functions.convert_image_np(reals[0]) * 255.0

    data = {"image": img_to_augment}
    aug = functions.Augment()
    augmented = aug.transform(**data)
    image = augmented["image"]
    functions.save_image(f"sample_image.png", image)
