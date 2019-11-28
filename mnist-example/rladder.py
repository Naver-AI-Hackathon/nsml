import torch
import torch.nn.functional as F
from torch import nn


class G1(nn.Module):
    def __init__(self, input_size, output_size):
        super(G1, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.w1 = nn.ParameterList()
        self.w2 = nn.ParameterList()

        for index in range(5):
            self.w1.append(nn.Parameter(torch.zeros(1, output_size)))
            self.w2.append(nn.Parameter(torch.zeros(1, output_size)))

            if index == 1:
                self.w1[-1].data.fill_(1)
                self.w2[-1].data.fill_(1)

    def forward(self, x_top, x_lat):
        u = self.linear(x_top)

        w0, w1, w2, w3, w4 = [w.expand_as(u) for w in self.w1]
        mu = w0 * torch.sigmoid(w1 * u + w2) + w3 * u + w4

        w0, w1, w2, w3, w4 = [w.expand_as(u) for w in self.w2]
        std = w0 * torch.sigmoid(w1 * u + w2) + w3 * u + w4

        y = x_lat * std + mu * (1 - std)

        return y, u, (mu, std)


class ConvG1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvG1, self).__init__()

        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.w1 = nn.ParameterList()
        self.w2 = nn.ParameterList()

        for index in range(5):
            self.w1.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1)))
            self.w2.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

            if index == 1:
                self.w1[-1].data.fill_(1)
                self.w2[-1].data.fill_(1)

    def forward(self, x_top, x_lat):
        u = self.linear(x_top)

        w0, w1, w2, w3, w4 = [w.expand_as(u) for w in self.w1]
        mu = w0 * torch.sigmoid(w1 * u + w2) + w3 * u + w4

        w0, w1, w2, w3, w4 = [w.expand_as(u) for w in self.w2]
        std = w0 * torch.sigmoid(w1 * u + w2) + w3 * u + w4

        y = x_lat * std + mu * (1 - std)

        return y, u, (mu, std)


class ConvG2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvG2, self).__init__()

        self.a1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.a2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.astd = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

        self.b1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.b2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bstd = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.w1 = nn.ParameterList()
        self.w2 = nn.ParameterList()
        self.wstd = nn.ParameterList()

        for index in range(5):
            self.w1.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1)))
            self.w2.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1)))
            self.wstd.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

            if index == 1:
                self.w1[-1].data.fill_(1)
                self.w2[-1].data.fill_(1)
                self.wstd[-1].data.fill_(1)

    def forward(self, x_top, x_lat):
        u1 = self.a1(x_top) + self.b1(x_lat)
        u2 = self.a2(x_top) + self.b2(x_lat)
        ustd = self.astd(x_top) + self.bstd(x_lat)

        w0, w1, w2, w3, w4 = [w.expand_as(u1) for w in self.w1]
        mu1 = w0 * torch.sigmoid(w1 * u1 + w2) + w3 * u1 + w4

        w0, w1, w2, w3, w4 = [w.expand_as(u2) for w in self.w2]
        mu2 = w0 * torch.sigmoid(w1 * u2 + w2) + w3 * u2 + w4

        w0, w1, w2, w3, w4 = [w.expand_as(ustd) for w in self.wstd]
        std = w0 * torch.sigmoid(w1 * ustd + w2) + w3 * ustd + w4

        y = mu1 * std + mu2 * (1 - std)

        return y, (u1, u2, ustd), (mu1, mu2, std)


class ConvG3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvG3, self).__init__()

        self.a1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.b1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.W1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.W2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.Wstd = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

        self.w1 = nn.ParameterList()

        for index in range(5):
            self.w1.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

            if index == 1:
                self.w1[-1].data.fill_(1)

    def forward(self, x_top, x_lat):
        u_pre = self.a1(x_top) + self.b1(x_lat)
        w0, w1, w2, w3, w4 = [w.expand_as(u_pre) for w in self.w1]

        u = F.relu(w0 * torch.sigmoid(w1 * u_pre + w2) + w3 * u_pre + w4, inplace=True)
        mu1 = self.W1(u)
        mu2 = self.W2(u)
        std = torch.sigmoid(self.Wstd(u))

        y = mu1 * std + mu2 * (1 - std)

        return y, u, (mu1, mu2, std)


if __name__ == "__main__":
    pass
# from tqdm import tqdm
#
# batch_size = 100
#
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# train_dataset = MNIST(root='datasets', train=True, transform=transform, download=True)
# test_dataset = MNIST(root='datasets', train=False, transform=transform, download=True)
# training_data = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
# testing_data = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
#
# input_size = 28 * 28
#
# model = RecurrentLadderModule()
# model.cuda()
#
# optimizer = optim.Adam(model.parameters(), lr=0.002)
#
# for epoch in range(100):
#     running_loss, running_acc = 0, 0
#     num_runs = 0
#
#     model.train()
#
#     for images, labels in tqdm(training_data):
#         images = Variable(images.cuda())
#         labels = Variable(labels.cuda())
#
#         loss, clean_state, noise_state, _ = model(images, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         predictions = F.log_softmax(clean_state).max(1)[1]
#         acc = 100. * torch.sum(predictions.long() == labels).float() / batch_size
#
#         running_loss += loss.data[0]
#         running_acc += acc.data[0]
#         num_runs += 1
#
#     print("[Epoch %d] Loss: %f - Accuracy: %f" % (epoch, running_loss / num_runs, running_acc / num_runs))
#
#     model.eval()
#
#     running_loss, running_acc = 0, 0
#     num_runs = 0
#     for images, labels in tqdm(testing_data):
#         images = Variable(images.cuda())
#         labels = Variable(labels.cuda())
#
#         loss, clean_state, noise_state, _ = model(images, labels)
#
#         predictions = F.log_softmax(clean_state).max(1)[1]
#         acc = 100. * torch.sum(predictions.long() == labels).float() / batch_size
#
#         running_loss += loss.data[0]
#         running_acc += acc.data[0]
#         num_runs += 1
#
#     print("[Validation] Loss: %f - Accuracy: %f" % (running_loss / num_runs, running_acc / num_runs))
#
#     # Visualize outputs.
#
#     vis_images, vis_labels = next(iter(testing_data))
#     vis_images = vis_images.cuda()
#     vis_labels = vis_labels.cuda()
#
#     loss, clean_state, noise_state, recovered_state = model(Variable(vis_images), Variable(vis_labels))
#
#     denoised_images = recovered_state.data.unsqueeze(1).unsqueeze(3).view(batch_size, 1, 28, 28)
#
#     images = torch.cat([vis_images, denoised_images], dim=0)
#
#     plt.ion()
#     plt.imshow(np.transpose(make_grid(images, nrow=int(batch_size ** 0.5)).cpu().numpy(), (1, 2, 0)),
#                cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
#     plt.show()
#
#     plt.pause(0.001)
