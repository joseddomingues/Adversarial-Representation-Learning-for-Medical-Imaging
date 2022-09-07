import sys

sys.path.append("../")

import MedSinGAN.functions as functions
from MedSinGAN.config import get_arguments

if __name__ == "__main__":
    # TODO: Adjust these values to run on a colab notebook
    parser = get_arguments()
    parser.add_argument('--input_name', type=str, default="")
    parser.add_argument('--nc_im', type=int, default=0)
    parser.add_argument('--not_cuda', type=bool, default=True)

    parser.add_argument('--im_max_size', type=int, default=0)
    parser.add_argument('--scale1', type=int, default=0)
    parser.add_argument('--stop_scale', type=int, default=0)
    parser.add_argument('--train_stages', type=int, default=0)
    parser.add_argument('--scale_factor', type=str, default="")
    parser.add_argument('--im_min_size', type=int, default=0)

    parser.add_argument('--train_mode', type=str, default="")

    opt = parser.parse_args()

    # Reads the image
    # Needs: input_name, nc_im, not_cuda
    real = functions.read_image(opt)

    # Adjusts the scales of the image
    # Needs: im_max_size, scale1, stop_scale, train_stages, scale_factor, im_min_size,
    real = functions.adjust_scales2image(real, opt)

    # Create the scales reals pyramids
    # Needs: train_mode, stop_scale, scale_factor,
    reals = functions.create_reals_pyramid(real, opt)

    img_to_augment = functions.convert_image_np(reals[0]) * 255.0

    data = {"image": img_to_augment}
    aug = functions.Augment()
    augmented = aug.transform(**data)
    image = augmented["image"]
    functions.save_image(f"sample_image.png", image)
