import argparse
import os

import PIL.Image as Image
import numpy as np
import torch
import torch.autograd as autograd
import torchvision.transforms as tvt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from mammos_dataset import MammographyDataset
from models import Generator, Discriminator


def process_pipeline_images(transform, im_path):
    """

    @param transform:
    @param im_path:
    @return:
    """
    target_image = Image.open(im_path)

    if transform:
        target_image = transform(target_image)
    else:
        converter = tvt.ToTensor()
        target_image = converter(target_image)

    return target_image.numpy()


def compute_gradient_penalty(D, real_samples, fake_samples, dev, ds):
    """
    Calculates the gradient penalty loss for WGAN GP
    @param D:
    @param real_samples:
    @param fake_samples:
    @param dev:
    @return:
    """
    # Random weight term for interpolation between real and fake samples
    # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=dev)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.to(dev)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    with autocast():
        d_interpolates = D(interpolates)

    # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = torch.ones((real_samples.shape[0], 1), device=dev)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=ds.scale(d_interpolates),
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)  # [0]

    inv_scale = 1. / ds.get_scale()
    gradients = [p * inv_scale for p in gradients]
    gradients = gradients[0]

    gradients = gradients.view(gradients.size(0), -1)
    with autocast():
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    del alpha, fake
    return gradient_penalty


def perform_train(opt, img_shape, dev, lambda_gp):
    """

    @param opt:
    @param img_shape:
    @param dev:
    @param lambda_gp:
    @return:
    """

    d_scaler = GradScaler()
    g_scaler = GradScaler()

    # Initialize generator and discriminator
    generator = Generator(opt=opt, img_shape=img_shape)
    discriminator = Discriminator(img_shape=img_shape)

    generator.to(dev)
    discriminator.to(dev)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    transformations = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize([0.5], [0.5])
    ])

    mam_dataset = MammographyDataset(opt.train_folder, transformations=transformations)
    dataloader = DataLoader(mam_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

    ############################
    # TRAIN
    ############################

    writer = SummaryWriter("tensorboard_wgan_gp_logs")
    batches_done = 0
    _iter = tqdm(range(opt.n_epochs))
    for epoch in _iter:
        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        # )
        _iter.set_description('Epoch [{}/{}]:'.format(epoch + 1, opt.n_epochs))
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            # real_imgs = imgs.to(dev)

            real_imgs = []
            for elem in imgs:
                real_imgs.append(process_pipeline_images(transform=transformations, im_path=elem))
            real_imgs = torch.tensor(np.array(real_imgs), device=dev)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn((2, opt.latent_dim), device=dev)

            # Generate a batch of images
            with autocast():
                fake_imgs = generator(z)
                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity = discriminator(fake_imgs.detach())

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, dev, d_scaler)

            with autocast():
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_scaler.scale(d_loss).backward()
            d_scaler.step(optimizer_D)
            d_scaler.update()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                with autocast():
                    fake_imgs = generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                g_scaler.scale(g_loss).backward()
                g_scaler.step(optimizer_G)
                g_scaler.update()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                writer.add_scalar('Loss/train/D/{}'.format(i), d_loss.item(), epoch)
                writer.add_scalar('Loss/train/G/{}'.format(i), g_loss.item(), epoch)

                if batches_done % opt.sample_interval == 0:
                    save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += opt.n_critic


if __name__ == "__main__":
    # Create folder to save the generated images
    os.makedirs("images", exist_ok=True)

    # Create argparser for the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, required=True, help="Folder of the base dataset for the GAN")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()

    # Get images shapes and cuda device
    img_shape = (3, 614, 499)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weight for gradient penalty
    lambda_gp = 10

    perform_train(opt, img_shape, dev, lambda_gp)
