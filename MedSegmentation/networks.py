import torch
import torch.nn as nn
import torch.utils.data


class U_Net(nn.Module):

    def __init__(self):
        super(U_Net, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.activ_1 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.activ_11 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        # Conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.activ_2 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.activ_22 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        # Conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.activ_3 = nn.ReLU()
        self.conv33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.activ_33 = nn.ReLU()

        # DeConv1
        self.deconv1 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)
        self.activ_4 = nn.ReLU()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # DeConv 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=48, out_channels=16, kernel_size=3, padding=1)
        self.activ_5 = nn.ReLU()
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)

        # DeConv 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=19, out_channels=1, kernel_size=5, padding=2)
        self.activ_6 = nn.Sigmoid()

    def forward(self, x):

        # First Extraction
        out_1 = x
        out = self.conv1(x)
        out = self.activ_1(out)
        out = self.conv11(out)
        out = self.activ_11(out)
        size1 = out.size()
        out, indices1 = self.pool1(out)

        # Second extraction
        out_2 = out
        out = self.conv2(out)
        out = self.activ_2(out)
        out = self.conv22(out)
        out = self.activ_22(out)
        size2 = out.size()
        out, indices2 = self.pool2(out)

        # Third extraction
        out_3 = out
        out = self.conv3(out)
        out = self.activ_3(out)
        out = self.conv33(out)
        out = self.activ_33(out)

        # Reconstruct 1
        out = torch.cat((out, out_3), dim=1)
        out = self.deconv1(out)
        out = self.activ_4(out)
        out = self.unpool1(out, indices2, size2)

        # Reconstruct 2
        out = torch.cat((out, out_2), dim=1)
        out = self.deconv2(out)
        out = self.activ_5(out)
        out = self.unpool2(out, indices1, size1)

        # Reconstruct 3
        out = torch.cat((out, out_1), dim=1)
        out = self.deconv3(out)
        out = self.activ_6(out)

        return out
