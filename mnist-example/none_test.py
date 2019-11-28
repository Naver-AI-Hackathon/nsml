import argparse
import pickle
import os
import torch
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.datasets.mnist import read_label_file, read_image_file
from ladder import LadderModule
import nsml


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
        if isinstance(input, list):
            print('input[0] :', input[0], 'type :', type(input[0]))
            return input[0]
        print('input :', input, 'type : ', type(input))
        return input

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
    args.add_argument("--gpu", type=int, default=0)
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
        print('saving ping model')
        nsml.save('0')
        print("Done")
