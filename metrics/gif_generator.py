import imageio
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', help='Folder where the images for the gif are', required=True)
    parser.add_argument('--gif_name', help='Name of the future generated gif', required=True)
    opt = parser.parse_args()

    images_gif = []
    for imag in os.listdir(opt.images_folder):
        images_gif.append(imageio.imread(opt.images_folder + '/' + imag))
    imageio.mimsave(opt.images_folder + '/' + opt.gif_name + '.gif', images_gif)