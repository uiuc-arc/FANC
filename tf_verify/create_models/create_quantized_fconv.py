
import torch
import torch.nn as nn
import os
from foolbox.models import PyTorchModel

# Quantization documentation
# https://pytorch.org/docs/stable/quantization.html

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        # self.quant = torch.quantization.QuantStub()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),  # type: ignore
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10),
        )
        # DeQuantStub converts tensors from quantized to floating point
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        # x = self.quant(x)
        x = self.layers(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        # x = self.dequant(x)
        return x


class CNNQ(torch.nn.Module):
    def __init__(self):
        super(CNNQ, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),  # type: ignore
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10),
        )
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        # print(type(x))

        x = self.quant(x)
        x = self.layers(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def get_model():
    model = CNN()

    path = os.path.join(os.path.dirname(os.path.realpath('content')), "mnist_cnn.pth")
    state_dict = torch.load(path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for key, val in state_dict.items():
        new_state_dict['layers.' + key] = val

    model.load_state_dict(new_state_dict)  # type: ignore
    model.eval()
    return model

def get_quantized_model():
    model = CNNQ()

    path = os.path.join(os.path.dirname(os.path.realpath('content')), "mnist_cnn.pth")
    state_dict = torch.load(path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for key, val in state_dict.items():
        new_state_dict['layers.' + key] = val

    model.load_state_dict(new_state_dict)  # type: ignore
    model.eval()
    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # model_fp32_fused = torch.quantization.fuse_modules(model, [['conv', 'relu']])

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare(model)

    # How to do this?
    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    input_fp32 = torch.randn(4, 1, 28, 28)
    model_fp32_prepared(input_fp32)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    return model_int8

def create() -> PyTorchModel:
    model = get_model()
    model.eval()
    preprocessing = dict(mean=0.1307, std=0.3081)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return model

if __name__ == "__main__":

    import time
    start_time = time.time()

    # test the model
    model = create()

    import torchvision
    import torchvision.transforms as transforms
    batch_size = 100

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    inputs, classes = next(iter(test_loader))

    # This seems to worked somehow without the preprocessing?
    # It says some 60k neurons, that will be too slow.
    # So get back to this later

    # model.eval()
    torch.onnx.export(model,  # model being run
                      inputs[0].reshape(1, 1, 28 , 28),  # model input (or a tuple for multiple inputs)
                      'fconv.onnx',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )
