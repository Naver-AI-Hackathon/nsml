import argparse
import pickle
import os
import torch
import torch.nn.functional as F
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import read_label_file, read_image_file
from ladder import LadderModule
import nsml
from nsml import DATASET_PATH


class ClassToSave:
    def __init__(self):
        self.elem = 0
        self.elem2 = 1

    def method(self):
        print(self.elem, self.elem2)

    def update(self, val):
        self.elem2 = val


def bind_model(model, class_to_save, optimizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        with open(os.path.join(filename, 'class.pkl'), 'rb') as fp:
            temp_class = pickle.load(fp)
        nsml.copy(temp_class, class_to_save)
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))
        with open(os.path.join(filename, 'class.pkl'), 'wb') as fp:
            pickle.dump(class_to_save, fp)

    def infer(input, top_k=100):
        # if app_type is not defined
        try:
            image = Image.open(BytesIO(input[0])).convert('L')
            image = image.resize((28,28))
            input = [image]
        # if app_type is canvas
        except:
            pass

        model.eval()
        # from list to tensor
        image = torch.stack(preprocess(None, input))
        image = Variable(image.cuda())
        _, clean_state, _, _ = model(image, None)
        _, all_cls = clean_state.size()
        prediction = F.softmax(clean_state, dim=1).topk(min(top_k, all_cls))
        # output format
        # [[(prob, key), (prob, key)... ], ...]
        return list(zip(list(prediction[0].data.cpu().squeeze().tolist()),
                        list(prediction[1].data.cpu().squeeze().tolist())))

    nsml.bind(save=save, load=load, infer=infer)


def _normalize_image(image_tensor, label, transform):
    new_tensor = []
    for idx, image in enumerate(image_tensor):
        if torch.is_tensor(image):  # else, image is instance of PIL.Image
            image = Image.fromarray(image.numpy(), mode='L')
        if label is not None:
            new_tensor.append([transform(image), label[idx]])
        else:
            new_tensor.append(transform(image))
    return new_tensor


def preprocess(output_path, data):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    # if-statement for submit/infer
    if output_path:
        data_set = {
            'train': _normalize_image(data['train']['data'], data['train']['label'], transform),
            'test': _normalize_image(data['test']['data'], data['test']['label'], transform)
        }
        with open(output_path[0], 'wb') as file:
            torch.save(data_set, file)
    else:
        data_set = _normalize_image(data, None, transform)
        return data_set


# supported by dataset owner
def data_loader(root_path):
    root_path = os.path.join(root_path, 'train')
    data_dict = {
        'train': {
            'data': read_image_file(os.path.join(root_path, 'train', 'train-images-idx3-ubyte')),
            'label': read_label_file(os.path.join(root_path, 'train', 'train-labels-idx1-ubyte'))
        },
        'test': {
            'data': read_image_file(os.path.join(root_path, 'test', 't10k-images-idx3-ubyte')),
            'label': read_label_file(os.path.join(root_path, 'test', 't10k-labels-idx1-ubyte'))
        }
    }
    return data_dict


if __name__ == "__main__":
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--batch", type=int, default=100)
    args.add_argument("--lr", type=float, default=0.002)
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--top", type=int, default=100)
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)
    args.add_argument("--gpu", type=int, default=1)
    config = args.parse_args()

    # initialization
    batch_size = config.batch
    input_size = 28 * 28
    mode = config.mode.lower()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    model = LadderModule()
    if config.gpu:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    class_to_save = ClassToSave()
    bind_model(model, class_to_save, optimizer=optimizer)
    # test mode
    if config.pause:
        nsml.paused(scope=locals())

    # training mode
    if mode == 'train':
        preprocessed_file = ['./processed.pt']
        nsml.cache(preprocess, output_path=preprocessed_file, data=data_loader(DATASET_PATH))
        dataset = torch.load(preprocessed_file[0])
        training_data = DataLoader(dataset['train'], batch_size, shuffle=True, num_workers=4)
        testing_data = DataLoader(dataset['test'], batch_size, shuffle=False, num_workers=4)

        for epoch in range(config.epochs):
            running_loss, running_acc = 0, 0
            num_runs = 0
            model.train()
            total_length = len(training_data)
            for iter_idx, (images, labels) in enumerate(tqdm(training_data)):
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

                loss, clean_state, noise_state, _ = model(images, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predictions = F.log_softmax(clean_state, dim=1).max(1)[1]
                acc = 100. * torch.sum(predictions.long() == labels).float() / batch_size

                running_loss += loss.data.item()
                running_acc += acc.data.item()
                num_runs += 1
                # get current lr
                opt_params = optimizer.state_dict()['param_groups'][0]
                step = epoch*total_length+iter_idx

                nsml.report(
                    epoch=epoch+int(config.iteration),
                    epoch_total=config.epochs,
                    iter=iter_idx,
                    iter_total=total_length,
                    batch_size=batch_size,
                    train__loss=running_loss / num_runs,
                    train__accuracy=running_acc / num_runs,
                    step=epoch*total_length+iter_idx,
                    lr = opt_params['lr'],
                    scope=locals()
                )
            # test model
            model.eval()
            running_loss, running_acc = 0, 0
            num_runs = 0
            for images, labels in tqdm(testing_data):
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

                loss, clean_state, noise_state, _ = model(images, labels)
                predictions = F.log_softmax(clean_state, dim=1).max(1)[1]
                acc = 100. * torch.sum(predictions.long() == labels).float() / batch_size

                running_loss += loss.data.item()
                running_acc += acc.data.item()
                num_runs += 1
            cur_acc = running_acc / num_runs

            print("[Epoch %d] Loss: %f - Accuracy: %f" % (epoch, running_loss / num_runs, running_acc / num_runs))

            print("[Validation] Loss: %f - Accuracy: %f" % (running_loss / num_runs, cur_acc))
            nsml.report(
                summary=True,
                epoch=epoch,
                epoch_total=config.epochs,
                test__loss=running_loss / num_runs,
                test__accuracy=running_acc / num_runs,
                step=(epoch+1) * total_length,
                lr=opt_params['lr']
            )
            nsml.save(epoch)
        print("Done")
