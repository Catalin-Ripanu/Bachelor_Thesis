from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy


from utils import *
from fid_score import *
from inception_score import *
from qrkt_gan import *
from utils.quantum_layer import *

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

import os

import numpy as np

lr_gen = 0.0001  # Learning rate for generator
lr_dis = 0.0001  # Learning rate for discriminator
latent_dim = 1024  # Latent dimension
gener_batch_size = 32  # Batch size for generator
dis_batch_size = 32  # Batch size for discriminator
epoch = 1  # Number of epoch
weight_decay = 1e-3  # Weight decay
drop_rate = 0.5  # dropout
n_critic = 5  #
max_iter = 500000
img_name = "img_name"
lr_decay = True

# architecture details by authors
image_size = 32  # H,W size of image for discriminator
initial_size = 8  # Initial size for generator
patch_size = 4  # Patch size for generated image
num_classes = 1  # Number of classes for discriminator
output_dir = "checkpoint"  # saved model path
dim = 384  # Embedding dimension
optimizer = "Adam"  # Optimizer
loss = "wgangp_eps"  # Loss function
phi = 1  #
beta1 = 0  #
beta2 = 0.99  #
diff_aug = "translation,cutout,color"  # data augmentation

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


def noise(n_samples, z_dim, device):
    return torch.randn(n_samples, z_dim).to(device)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


def inits_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight.data, 1.0)


def noise(imgs, latent_dim):
    return torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))


def gener_noise(gener_batch_size, latent_dim):
    return torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))


def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pth"):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, "checkpoint_best.pth"))

def validate(generator, writer_dict, fid_stat):
    

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']

        generator = generator.eval()
        fid_score = get_fid(fid_stat, epoch, generator, num_img=5000, val_batch_size=60*2, latent_dim=1024, writer_dict=None, cls_idx=None)


        print(f"FID score: {fid_score}")

        writer.add_scalar('FID_score', fid_score, global_steps)

        writer_dict['valid_global_steps'] = global_steps + 1
        return fid_score

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
        real_samples.get_device()
    )
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(
        real_samples.get_device()
    )
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train(
    noise,
    generator,
    discriminator,
    optim_gen,
    optim_dis,
    epoch,
    writer,
    schedulers,
    img_size=32,
    latent_dim=latent_dim,
    n_critic=n_critic,
    gener_batch_size=gener_batch_size,
    device="cuda:0",
):

    writer = writer_dict["writer"]
    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()

    transform = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=30, shuffle=True
    )

    for index, (img, _) in enumerate(train_loader):

        global_steps = writer_dict["train_global_steps"]

        real_imgs = img.type(torch.cuda.FloatTensor)

        noise = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (img.shape[0], latent_dim))
        )  # noise(img, latent_dim)#= args.latent_dim)

        optim_dis.zero_grad()
        real_valid = discriminator(real_imgs)
        fake_imgs = generator(noise).detach()

        # assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_valid = discriminator(fake_imgs)

        if loss == "hinge":
            loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(
                device
            ) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
        elif loss == "wgangp_eps":
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs, fake_imgs.detach(), phi
            )
            loss_dis = (
                -torch.mean(real_valid)
                + torch.mean(fake_valid)
                + gradient_penalty * 10 / (phi**2)
            )

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

        if global_steps % n_critic == 0:

            optim_gen.zero_grad()
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar("LR/g_lr", g_lr, global_steps)
                writer.add_scalar("LR/d_lr", d_lr, global_steps)

            gener_noise = torch.cuda.FloatTensor(
                np.random.normal(0, 1, (gener_batch_size, latent_dim))
            )

            generated_imgs = generator(gener_noise)
            fake_valid = discriminator(generated_imgs)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
            writer.add_scalar("gener_loss", gener_loss.item(), global_steps)

            gen_step += 1

            # writer_dict['train_global_steps'] = global_steps + 1

        if gen_step and index % 100 == 0:
            sample_imgs = generated_imgs[:25]
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(
                sample_imgs,
                f"generated_images/generated_img_{epoch}_{index % len(train_loader)}.jpg",
                nrow=5,
                normalize=True,
                scale_each=True,
            )
            tqdm.write(
                "[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch + 1,
                    index % len(train_loader),
                    len(train_loader),
                    loss_dis.item(),
                    gener_loss.item(),
                )
            )


device = torch.device(dev)

generator = Generator(
    quantum_circuit=get_circuit(),
    depth1=5,
    depth2=4,
    depth3=1,
    initial_size=8,
    dim=384,
    heads=4,
    mlp_ratio=4,
    drop_rate=0.5,
)  # ,device = device)
generator.to(device)

discriminator = Discriminator(
    diff_aug=diff_aug,
    quantum_circuit=get_circuit(),
    image_size=32,
    patch_size=4,
    input_channel=3,
    num_classes=1,
    dim=384,
    depth=7,
    heads=4,
    mlp_ratio=4,
    drop_rate=0.5,
)
discriminator.to(device)


print(generator.apply(inits_weight))
print(discriminator.apply(inits_weight))

if optimizer == "Adam":
    optim_gen = optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=lr_gen,
        betas=(beta1, beta2),
    )

    optim_dis = optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=lr_dis,
        betas=(beta1, beta2),
    )
elif optimizer == "SGD":
    optim_gen = optim.SGD(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=lr_gen,
        momentum=0.9,
    )

    optim_dis = optim.SGD(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=lr_dis,
        momentum=0.9,
    )

elif optimizer == "RMSprop":
    optim_gen = optim.RMSprop(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=lr_dis,
        eps=1e-08,
        weight_decay=weight_decay,
        momentum=0,
        centered=False,
    )

    optim_dis = optim.RMSprop(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=lr_dis,
        eps=1e-08,
        weight_decay=weight_decay,
        momentum=0,
        centered=False,
    )

gen_scheduler = LinearLrDecay(optim_gen, lr_gen, 0.0, 0, max_iter * n_critic)
dis_scheduler = LinearLrDecay(optim_dis, lr_dis, 0.0, 0, max_iter * n_critic)

# RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

print("optimizer:", optimizer)

fid_stat = './fid_stat/fid_stats_cifar10_train.npz'

writer = SummaryWriter()
writer_dict = {"writer": writer}
writer_dict["train_global_steps"] = 0
writer_dict["valid_global_steps"] = 0

best = 1e4

for epoch in range(epoch):

    lr_schedulers = (gen_scheduler, dis_scheduler) if lr_decay else None

    train(noise, generator, discriminator, optim_gen, optim_dis,
    epoch, writer, lr_schedulers,img_size=32, latent_dim = latent_dim,
    n_critic = n_critic,
    gener_batch_size=gener_batch_size)

    checkpoint = {'epoch':epoch, 'best_fid':best}
    checkpoint['generator_state_dict'] = generator.state_dict()
    checkpoint['discriminator_state_dict'] = discriminator.state_dict()

    score = validate(generator, writer_dict, fid_stat)

    print(f'FID score: {score} - best ID score: {best} || @ epoch {epoch+1}.')
    if epoch == 0 or epoch > 30:
        if score < best:
            save_checkpoint(checkpoint, is_best=(score<best), output_dir=output_dir)
            print("Saved Latest Model!")
            best = score


checkpoint = {'epoch':epoch, 'best_fid':best}
checkpoint['generator_state_dict'] = generator.state_dict()
checkpoint['discriminator_state_dict'] = discriminator.state_dict()
score = validate(generator, writer_dict, fid_stat) ####CHECK AGAIN
save_checkpoint(checkpoint,is_best=(score<best), output_dir=output_dir)
