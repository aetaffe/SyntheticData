import time

import torch.nn as nn
import random
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
from endoscopic_dataset import EndoscopicSurgicalDataset
import numpy as np
import cv2
from pathlib import Path

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Root directory for dataset
dataroot = "data/segmentation_data"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 4

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 4

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr_dis = 0.00001
lr_gen = 0.00009
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    gen_device = None
    disc_device = None
    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        print(f'Using {ngpu} GPUs')
        # Decide which device we want to run on
        gen_device = torch.device("cuda:0" if ngpu > 0 else "cpu")
        disc_device = torch.device("cuda:1" if ngpu > 1 else "cuda:0")
    else:
        gen_device = torch.device("cpu")
        disc_device = torch.device("cpu")
        print('Using CPU')

    # Create the generator
    netG = Generator(ngpu, num_latent_ch=nz, num_hidden_ch=ngf, num_img_ch=nc).to(gen_device)

    # Handle multi-GPU if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)
    # print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu, num_img_ch=nc, num_hidden_ch=ndf).to(disc_device)

    # Handle multi-GPU if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    # print(netD)

    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=gen_device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr_dis, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_gen, betas=(beta1, 0.999))

    # Create the dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(tuple(0.5 for _ in range(nc)), tuple(0.5 for _ in range(nc)))
    ])
    dataset = EndoscopicSurgicalDataset(dataroot, transform=transform)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[:,:3,:,:].to(gen_device)[:4], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('training_images.png')

    # Training Loop

    # Lists to keep track of progress
    # img_list = []
    fake_images = []
    G_losses = []
    D_losses = []
    iters = 0

    start = time.time()
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(disc_device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=disc_device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=gen_device)
            # Generate fake image batch with G
            fake = netG(noise)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            label.to(gen_device)
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            label.to(disc_device)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 200 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 200 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                # fake_images = np.transpose(vutils.make_grid(fake[:,:3,:,:], padding=2, normalize=True), (1, 2, 0))
                # Nomalize the image to be between 0 and 1 for saving
                fake_images = [np.transpose((fake[i, :3, :, :] * 0.5) + 0.5, (1, 2, 0)) for i in range(fake.shape[0])]
            iters += 1

    end = time.time()
    time_taken = end - start
    hours = time_taken // 60 // 60
    minutes = (time_taken - hours * 60) // 60
    seconds = time_taken - (hours * 60) - (minutes * 60)
    print('Time taken: {:.0f} hours, {:.0f} minutes, {:.2f} seconds'
          .format(hours, minutes, seconds))
    print('Finished Training')
    torch.save(netG.state_dict(), 'netG.pth')
    torch.save(netD.state_dict(), 'netD.pth')

    # plt.clf()
    # Plot the training losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('losses.png')

    # Save the fake images
    # cv2.imwrite('fake_images_' + str(num_epochs) + '.jpg', cv2.cvtColor(fake_images.numpy() * 255, cv2.COLOR_RGB2BGR))
    Path(f'fake_images/{num_epochs}_epochs').mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(fake_images):
        cv2.imwrite(f'fake_images/{num_epochs}_epochs/fake_image_{i}.jpg', cv2.cvtColor(img.numpy() * 255, cv2.COLOR_RGB2BGR))


