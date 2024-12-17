import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ngpu, num_img_ch, num_hidden_ch):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 256 x 256``
            nn.Conv2d(num_img_ch, num_hidden_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 128 x 128``
            nn.Conv2d(num_hidden_ch, num_hidden_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 64 x 64``
            nn.Conv2d(num_hidden_ch * 2, num_hidden_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 32 x 32``
            nn.Conv2d(num_hidden_ch * 4, num_hidden_ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 16 x 16``
            nn.Conv2d(num_hidden_ch * 8, num_hidden_ch * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden_ch * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*16) x 8 x 8``
            nn.Conv2d(num_hidden_ch * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)