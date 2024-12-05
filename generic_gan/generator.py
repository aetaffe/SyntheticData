import torch.nn as nn

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, num_latent_ch=100, num_hidden_ch=64, num_img_ch=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_latent_ch, num_hidden_ch * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 16),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(num_hidden_ch * 16, num_hidden_ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 8),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(num_hidden_ch * 8, num_hidden_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 4),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(num_hidden_ch * 4, num_hidden_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 2),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(num_hidden_ch * 2, num_hidden_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch),
            nn.ReLU(True),
            # state size. ``(nc) x 64 x 64``
            nn.ConvTranspose2d(num_hidden_ch, num_img_ch, 4, 2, 1, bias=False),
            nn.Tanh()
<<<<<<< Updated upstream
            # state size. ``(nc) x 256 x 156``
=======
            # state size. ``(nc) x 128 x 128``
>>>>>>> Stashed changes
        )

    def forward(self, input):
        return self.main(input)