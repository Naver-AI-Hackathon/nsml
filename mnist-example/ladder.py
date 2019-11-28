import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchvision.utils import make_grid
from ops import Noise


class LadderModule(nn.Module):
    def __init__(self):
        super(LadderModule, self).__init__()

        self.layer_widths = [784, 1000, 500, 250, 250, 250, 10]
        self.layer_weights = [1000, 1, 0.01, 0.01, 0.01, 0.01, 0.01]

        assert len(self.layer_widths) == len(self.layer_weights), "# layer dimensions and weights are not the same."

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.v = nn.ModuleList()

        for index in range(len(self.layer_widths)):
            if index > 0:
                self.encoders.append(LadderEncoder(self.layer_widths[index - 1], self.layer_widths[index],
                                                   index == len(self.layer_widths) - 1))
                self.v.append(nn.Linear(self.layer_widths[index], self.layer_widths[index - 1], bias=False))
            self.decoders.append(LadderDecoder(self.layer_widths[index]))

        self.noise = Noise()

        self.supervised_loss = nn.NLLLoss(reduction='mean')

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        x_n = self.noise(x)

        # Encoder passes.
        hi, zi, h_ni, z_ni, mi, stdi = [x], [x], [x_n], [x_n], [], []

        for encoder in self.encoders:
            h, z, h_n, z_n, m, std = encoder(hi[-1], h_ni[-1])

            hi.append(h)
            zi.append(z)
            h_ni.append(h_n)
            z_ni.append(z_n)
            mi.append(m)
            stdi.append(std)

        # Decoder passes.

        loss, z_bn_hati, z_hati = 0, [], []

        for index, decoder in enumerate(reversed(self.decoders)):
            if index == 0:
                z_bn_hat, z_hat = decoder(h_ni[-1], z_ni[-1], mi[-1], stdi[-1])
            else:
                if index < len(self.decoders) - 1:
                    mean, std = mi[-index - 1], stdi[-index - 1]
                else:
                    mean, std = None, None
                z_bn_hat, z_hat = decoder(self.v[-index](z_hati[-1]), z_ni[-index - 1], mean, std)

            z_bn_hati.append(z_bn_hat)
            z_hati.append(z_hat)

            weight = (self.layer_weights[-index - 1] / self.layer_widths[-index - 1])
            loss += weight * torch.mean(torch.sum((zi[-index - 1] - z_bn_hat) ** 2, dim=-1))

        if labels is not None:
            loss += self.supervised_loss(h_ni[-1], labels)

        return loss, hi[-1], h_ni[-1], z_hati[-1]


class Ladder(nn.Module):
    def __init__(self):
        super(Ladder, self).__init__()

        self.e1 = LadderEncoder(784, 1000, False)
        self.e2 = LadderEncoder(1000, 500, False)
        self.e3 = LadderEncoder(500, 250, False)
        self.e4 = LadderEncoder(250, 250, False)
        self.e5 = LadderEncoder(250, 250, False)
        self.e6 = LadderEncoder(250, 10, True)

        self.d6 = LadderDecoder(10)
        self.d5 = LadderDecoder(250)
        self.d4 = LadderDecoder(250)
        self.d3 = LadderDecoder(250)
        self.d2 = LadderDecoder(500)
        self.d1 = LadderDecoder(1000)
        self.d0 = LadderDecoder(784)

        self.v6 = nn.Linear(10, 250, bias=False)
        self.v5 = nn.Linear(250, 250, bias=False)
        self.v4 = nn.Linear(250, 250, bias=False)
        self.v3 = nn.Linear(250, 500, bias=False)
        self.v2 = nn.Linear(500, 1000, bias=False)
        self.v1 = nn.Linear(1000, 784, bias=False)

        self.noise = Noise()

        self.supervised_loss = nn.NLLLoss(reduction='mean')

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)

        # Encoder passes.

        h0 = z0 = x
        h_n0 = z_n0 = self.noise(x)

        h1, z1, h_n1, z_n1, m1, std1 = self.e1(h0, h_n0)
        h2, z2, h_n2, z_n2, m2, std2 = self.e2(h1, h_n1)
        h3, z3, h_n3, z_n3, m3, std3 = self.e3(h2, h_n2)
        h4, z4, h_n4, z_n4, m4, std4 = self.e4(h3, h_n3)
        h5, z5, h_n5, z_n5, m5, std5 = self.e5(h4, h_n4)
        h6, z6, h_n6, z_n6, m6, std6 = self.e6(h5, h_n5)

        # Decoder passes.

        z_bn_hat6, z_hat6 = self.d6(h_n6, z_n6, m6, std6)
        z_bn_hat5, z_hat5 = self.d5(self.v6(z_hat6), z_n5, m5, std5)
        z_bn_hat4, z_hat4 = self.d4(self.v5(z_hat5), z_n4, m4, std4)
        z_bn_hat3, z_hat3 = self.d3(self.v4(z_hat4), z_n3, m3, std3)
        z_bn_hat2, z_hat2 = self.d2(self.v3(z_hat3), z_n2, m2, std2)
        z_bn_hat1, z_hat1 = self.d1(self.v2(z_hat2), z_n1, m1, std1)
        z_bn_hat0, z_hat0 = self.d0(self.v1(z_hat1), z_n0, None, None)

        loss = 0
        loss += (0.01 / 10) * torch.mean(torch.sum((z6 - z_bn_hat6) ** 2, dim=-1))
        loss += (0.01 / 250) * torch.mean(torch.sum((z5 - z_bn_hat5) ** 2, dim=-1))
        loss += (0.01 / 250) * torch.mean(torch.sum((z4 - z_bn_hat4) ** 2, dim=-1))
        loss += (0.01 / 250) * torch.mean(torch.sum((z3 - z_bn_hat3) ** 2, dim=-1))
        loss += (0.01 / 500) * torch.mean(torch.sum((z2 - z_bn_hat2) ** 2, dim=-1))
        loss += (1 / 1000) * torch.mean(torch.sum((z1 - z_bn_hat1) ** 2, dim=-1))
        loss += (1000 / 784) * torch.mean(torch.sum((z0 - z_bn_hat0) ** 2, dim=-1))

        if labels is not None:
            loss += self.supervised_loss(h_n6, labels)

        return loss, h6, h_n6, z_hat0


class LadderDecoder(nn.Module):
    def __init__(self, input_size):
        super(LadderDecoder, self).__init__()

        self.variance_weights = nn.ParameterList()
        self.mean_weights = nn.ParameterList()

        for index in range(5):
            self.variance_weights.append(nn.Parameter(torch.zeros(1, input_size)))
            self.mean_weights.append(nn.Parameter(torch.zeros(1, input_size)))

            if index == 1:
                self.variance_weights[-1].data.fill_(1)
                self.mean_weights[-1].data.fill_(1)

        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, u, z_noise, means=None, stds=None):
        u = self.norm(u)

        a1, a2, a3, a4, a5 = [weight.expand_as(u) for weight in self.variance_weights]
        a6, a7, a8, a9, a10 = [weight.expand_as(u) for weight in self.mean_weights]

        v = a1 * torch.sigmoid(a2 * u + a3) + a4 * u + a5
        m = a6 * torch.sigmoid(a7 * u + a8) + a9 * u + a10

        z_hat = (z_noise - m) * v + m

        if means is None and stds is None:
            z_hat_bn = z_hat
        else:
            means = means.reshape([means.numel(), 1])
            stds = stds.reshape([stds.numel(), 1])

            z_hat_bn = (z_hat - means.expand_as(z_hat)) / stds.expand_as(z_hat)

        return z_hat_bn, z_hat


class LadderEncoder(nn.Module):
    def __init__(self, input_size, output_size, is_last_layer):
        super(LadderEncoder, self).__init__()
        self.input_size = input_size

        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.norm = nn.BatchNorm1d(output_size)

        self.is_last_layer = is_last_layer

        self.beta = nn.Parameter(torch.zeros(1, 1))
        self.gamma = nn.Parameter(torch.ones(1, 1))

        self.noise = Noise()

    def forward(self, x, x_n):
        z_pre = self.linear(x)
        z = self.norm(z_pre)

        z_pre_noise = self.linear(x_n)
        z_noise = self.noise(self.norm(z_pre_noise))

        gamma = self.gamma.expand_as(z)
        beta = self.beta.expand_as(z)

        if self.is_last_layer:
            h = F.log_softmax(gamma * (z + beta), dim=1)
            h_n = F.log_softmax(gamma * (z_noise + beta), dim=1)
        else:
            h = F.relu(z + beta, inplace=True)
            h_n = F.relu(z_noise + beta, inplace=True)

        means = torch.mean(z_pre, dim=-1)
        stds = torch.std(z_pre, dim=-1)
        return h, z, h_n, z_noise, means, stds


if __name__ == "__main__":
    from tqdm import tqdm

    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = MNIST(root='datasets', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='datasets', train=False, transform=transform, download=True)
    training_data = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    testing_data = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

    input_size = 28 * 28

    model = LadderModule()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(100):
        running_loss, running_acc = 0, 0
        num_runs = 0

        model.train()

        for images, labels in tqdm(training_data):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            loss, clean_state, noise_state, _ = model(images, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = F.log_softmax(clean_state, 1).max(1)[1]
            acc = 100. * torch.sum(predictions.long() == labels).float() / batch_size

            running_loss += loss.data[0]
            running_acc += acc.data[0]
            num_runs += 1

        print("[Epoch %d] Loss: %f - Accuracy: %f" % (epoch, running_loss / num_runs, running_acc / num_runs))

        model.eval()

        running_loss, running_acc = 0, 0
        num_runs = 0
        for images, labels in tqdm(testing_data):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            loss, clean_state, noise_state, _ = model(images, labels)

            predictions = F.log_softmax(clean_state, 1).max(1)[1]
            acc = 100. * torch.sum(predictions.long() == labels).float() / batch_size

            running_loss += loss.data[0]
            running_acc += acc.data[0]
            num_runs += 1

        print("[Validation] Loss: %f - Accuracy: %f" % (running_loss / num_runs, running_acc / num_runs))
