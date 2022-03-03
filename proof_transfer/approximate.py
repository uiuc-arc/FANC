''' 
Generate the approximated networks provided the original network. The input is in saved pytorch format. 
The generated outputs are in ONNX format. 
'''
from operator import mod
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os

from torch.nn import functional as F 
from model_defs import model_cnn_3layer, model_mlp_any, model_cnn_4layer, model_cnn_2layer, model_cnn_2layer_op, model_cnn_5layer, model_cnn_4layer_conv13, model_cnn_4layer_conv11, model_cnn_4layer_orig

DEVICE = 'cpu'

def generate_approximate_model(approx_type, model_name, dataset, tune_pt=False):
    per = [10*x for x in range(10)]
    post_template = False

    testloader = prepare_data(dataset, False)
    inputs, _ = next(iter(testloader))

    if model_name == 'fcn' and dataset == 'mnist':
        model = model_mlp_any(784, [200, 200, 200, 200, 200, 200, 200])
        checkpoint_name = 'nets/mlp_1400_best.pth'
        ex_input = inputs[0].reshape(-1, 784)
        model_type = 'fcnn7'
        # Layers are skipped in post_template approximation mode.
        skip_layer = 2
    elif model_name == 'conv' and dataset == 'mnist':
        model = model_cnn_2layer(1, 28, 1, 256)
        checkpoint_name = 'nets/cnn_2layer_width_1_best.pth'
        ex_input = inputs[0].reshape(1, 1, 28, 28)
        # Don't prune the conv layers
        skip_layer = 2
        model_type = 'fconv4'
    elif model_name == 'fcn' and dataset == 'cifar':
        model = model_mlp_any(3072, [200, 200, 200, 200, 200, 200, 200])
        # rand20 works better than guide20
        checkpoint_name = 'nets/fcnn7_cifar10_best.pth'
        ex_input = inputs[0].reshape(1, 3, 32, 32)
        skip_layer = 2
        model_type = 'fcnn7_cifar'
    elif model_name == 'conv' and dataset == 'cifar':  
        # First layer takes most time
        checkpoint_name = 'nets/cnn_4layer_linear_256_width_1_best.pth'
        model = model_cnn_4layer_conv13(3, 32, 1, 256)
        ex_input = inputs[0].reshape(1, 3, 32, 32)
        skip_layer = 5
        model_type = 'fconv4_cifar'
    

    checkpoint = torch.load(checkpoint_name, map_location=torch.device(DEVICE))

    model.load_state_dict(checkpoint['state_dict'])
    
    trainloader = prepare_data(dataset, False)
    
    if tune_pt:
        finetune(model, trainloader, testloader, dataset)

    check_accuracy(model, testloader)

    torch.onnx.export(model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      ex_input,
                      'nets/' + model_type + '.onnx',
                      # where to save the model (can be a file or file-like
                      # object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )

    # Only for post tempalte approximation layers are skipped
    if not post_template:
        skip_layer = 0

    if approx_type == 'prune':
        prune_model(model, dataset, testloader, model_type, ex_input, skip_layer, per)
    
    if approx_type == 'quant8':
        # skip_layer = 0
        dummy_quant(model, dataset, testloader, 8, model_type, ex_input, skip_layer)

    if approx_type == 'quant16':
        dummy_quant(model, dataset, testloader, 16, model_type, ex_input, skip_layer)
    
    if approx_type == 'float16':
        dummy_quant_float(model, dataset, testloader, model_type, ex_input)

def prune_model(model, dataset, testloader, model_type, ex_input, skip_layer, per=[5, 10, 15, 20, 25], post_finetune=False):
    density(model)
    prev_accuracy = check_accuracy(model, testloader)

    for i in range(len(per)):
        prune(model, per[i], skip_layer)

        check_accuracy(model, testloader)

        density(model)

        trainloader = prepare_data(dataset, True)
        print("Fine tune the network to get accuracy:", prev_accuracy)
        
        if(post_finetune):
            finetune(model, trainloader, testloader, dataset, req_accuracy=prev_accuracy)

        density(model)

        inputs, _ = next(iter(testloader))

        if not os.path.exists('nets/' + model_type + '_prune/'):
            os.makedirs('nets/' + model_type + '_prune/')
        
        generated_filename = 'nets/' + model_type + '_prune/' + model_type + '_prune_tune' + str(i) + '.onnx'
        torch.onnx.export(model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      ex_input,
                      generated_filename,
                      # where to save the model (can be a file or file-like
                      # object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )

    inputs, _ = next(iter(testloader))

    generated_filename = 'nets/' + model_type + '_prune.onnx'
    torch.onnx.export(model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      ex_input,
                      generated_filename,
                      # where to save the model (can be a file or file-like
                      # object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )
    print('Generated file: ', generated_filename)

def prepare_data(dataset, train):
    print('==> Preparing data..')

    if dataset == 'cifar':
        transform_test = transforms.Compose([
            transforms.ToTensor()
            ,
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
            ]
            )

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform_test)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        inputs, _ = next(iter(testloader))
    else:
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=train, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                                (0,), (1,))
                                       ])),
            batch_size=100, shuffle=True)

    return testloader


def prune(model, per, skip_layer):
    state_dict = model.state_dict()

    for name, param in state_dict.items():

        if "weight" not in name:
            continue

        per_it = per
        if skip_layer > 0:
            skip_layer -= 1
            per_it = 0

        print('Pruning layer: ', name, ' | Percentage: ', per_it)
        cutoff = np.percentile(np.abs(param), per_it)

        print(param.shape)

        if(len(param.shape) == 2):
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    if abs(param[i][j]) < cutoff:
                        param[i][j] = 0

        elif(len(param.shape) == 1):
            for i in range(param.shape[0]):
                if abs(param[i]) < cutoff:
                    param[i] = 0
    print('Done!')

def density(model):
    state_dict = model.state_dict()

    count = 0
    count_nz = 0

    for name, param in state_dict.items():
        # Don't update if this is not a weight.
        # print(name)

        if "weight" not in name:
            continue

        # Transform the parameter as required.
        # print(param.shape)

        if(len(param.shape) == 2):
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    count += 1
                    if param[i][j] != 0:
                        count_nz += 1

        elif(len(param.shape) == 1):
            for i in range(param.shape[0]):
                count += 1
                if param[i][j] != 0:
                    count_nz += 1

    print('Density :', count_nz * 1.0 / count)

'''
Get bounds for a layer using interval propogation. 

TODO: 
1. Use multiple layers
2. Refactor the propagation code to a separate analyzer module
'''
def get_bounds(images, params):
    eps = 0.05
    is_conv = True
    if not is_conv:

        lb = (images - eps).reshape(images.shape[0], -1)
        ub = (images + eps).reshape(images.shape[0], -1)
        
        pos_wt  = F.relu(params[0])
        neg_wt = -F.relu(-params[0])
        
        # print(params[0].shape)
        # print(ub.shape)

        oub = F.relu(ub @ pos_wt.T + lb @ neg_wt.T)
        olb = F.relu(lb @ pos_wt.T + ub @ neg_wt.T)
    else:
        lb = (images - eps).reshape(images.shape[0], -1)
        ub = (images + eps).reshape(images.shape[0], -1)
        
        weight = params[0]
        bias = params[1]

        num_kernel = weight.shape[0]

        k_h, k_w = 4, 4
        s_h, s_w = 2, 2
        p_h, p_w = 1, 1
    
        input_h, input_w = 28, 28

        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        linear_cof = []

        size = 784
        shape = (1,28,28)

        cof = torch.eye(size).reshape(size, *shape)
        pad2d = (p_w, p_w, p_h, p_h)
        cof = F.pad(cof, pad2d)

        for i in range(output_h):
            w_cof = []
            for j in range(output_w):
                h_start = i * s_h
                h_end = h_start + k_h
                w_start = j * s_w
                w_end = w_start + k_w

                w_cof.append(cof[:, :, h_start: h_end, w_start: w_end])
            
            linear_cof.append(torch.stack(w_cof, dim=1))
                
        linear_cof = torch.stack(linear_cof, dim=1).reshape(size, output_h, output_w, -1)
    
        new_weight = weight.reshape(num_kernel, -1).T
        new_cof = linear_cof @ new_weight
        new_cof = new_cof.permute(0,3,1,2).reshape(size,-1) 
   
        pos_wt  = F.relu(new_cof)
        neg_wt = -F.relu(-new_cof)

        bias = bias.view(-1,1,1).expand(num_kernel,output_h,output_w).reshape(1,-1) 

        oub = F.relu(ub @ pos_wt.T + lb @ neg_wt.T)
        olb = F.relu(lb @ pos_wt.T + ub @ neg_wt.T)

    return olb, oub

'''
Fine-tune the model to be more amenable to the proof transfer. 
'''
def finetune(model, trainloader, testloader, dataset, epochs=5, req_accuracy=1):
    
    loss_type = 'sum'

    if dataset == 'mnist':
        if loss_type == 'volume':
            learning_rate = 1e-7
        elif loss_type == 'sum':
            learning_rate = 1e-7
    elif dataset == 'cifar10':
        learning_rate = 1e-7

    num_epochs = epochs
    
    # Extract all the weights of the model
    params = []
    for param in model.parameters():
        params.append(param)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [params[0]],
        lr=learning_rate)

    # Train the model
    total_step = len(trainloader)
    first_time = True

    all_ims = []
    

    for epoch in range(num_epochs):
        cur_accuracy = check_accuracy(model, testloader)

        if cur_accuracy >= req_accuracy:
            break

        for i, (images, labels) in enumerate(trainloader):
            all_ims.append(images)
            # Move tensors to the configured device
            # images = images.reshape(-1, 28 * 28).to(DEVICE)
            if dataset == 'mnist':
                images = images.reshape(-1, 1, 28, 28).to(DEVICE)
            else:
                images = images.reshape(-1, 3, 32, 32).to(DEVICE)

            labels = labels.to(DEVICE)
            olb, oub = get_bounds(images, params)

            if loss_type == 'volume':
                loss2 = torch.sum(torch.log(oub-olb + 1))
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()
            
            elif loss_type == 'sum':
                loss2 = torch.sum(oub-olb)
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()
            
            elif loss_type == 'mix':
                # Forward pass
                outputs = model(images)
                print(olb.shape)
                loss2 = torch.sum(oub-olb)
                print(loss, olb[0], oub[0])
                loss1 = criterion(outputs, labels)

                if first_time:
                    gamma = (loss1/loss2).detach()
                    first_time = False

                loss = loss1 + gamma*loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()         

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(
                    epoch + 1, num_epochs, i + 1, total_step, loss2.item()))
                # print(loss2.item(), loss2.item())

        olb, oub = get_bounds(all_ims[0], params)
        loss2 = torch.sum(oub-olb)
        print(loss2.item())
        print((oub-olb)[:20])

def check_accuracy(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    inputs, labels = next(iter(testloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # print(inputs.shape)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
    return correct / total

def dummy_quant(model, dataset, testloader, quant_bit, model_type, ex_input, skip_layer):

    # Dummy quantized
    # Calculate max to do the quantization symmetric, per-tensor

    def quant(x, scale): return int(x * scale)
    def unquant(x, scale): return x / scale

    state_dict = model.state_dict()

    post_layer = skip_layer

    for name, param in state_dict.items():
        # Don't update if this is not a weight.
        # print(name, param.shape)

        if "weight" not in name:
            continue

        # Transform the parameter as required.
        if(skip_layer > 0):
            skip_layer -= 1
            continue

        if(len(param.shape) == 2):
            abs_max = 0

            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    abs_max = max(abs(param[i][j]), abs_max)

            scale = (2**(quant_bit - 1)) / abs_max

            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    # print((param[i][j].item()- unquant(quant(param[i][j], scale), scale).item())/param[i][j].item(), param[i][j].item(), quant(param[i][j], scale), unquant(quant(param[i][j], scale), scale).item(), param[i][j].item()- unquant(quant(param[i][j], scale), scale).item())
                    param[i][j] = unquant(quant(param[i][j], scale), scale)

        elif len(param.shape) == 1:
            abs_max = 0

            for i in range(param.shape[0]):
                abs_max = max(abs(param[i]), abs_max)

            scale = (2**(quant_bit - 1)) / abs_max

            for i in range(param.shape[0]):
                param[i] = unquant(quant(param[i], scale), scale)
        else:
            print('Param shape length is: ', len(param.shape))

        # Update the parameter.
        state_dict[name].copy_(param)

    inputs, _ = next(iter(testloader))

    if post_layer > 0:
        generated_filename = 'nets/' + model_type + '_quant' + str(quant_bit) + '_skip' + str(post_layer) + '.onnx'
    else:
        generated_filename = 'nets/' + model_type + '_quant' + str(quant_bit) + '.onnx'

    torch.onnx.export(model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      ex_input,
                      generated_filename,
                      # where to save the model (can be a file or file-like
                      # object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )
    print('Generated file: ', generated_filename)
    check_accuracy(model, testloader)

def dummy_quant_float(model, dataset, testloader, model_type, ex_input):

    # Dummy quantized
    # Calculate max to do the quantization symmetric, per-tensor

    def quant(x): return torch.Tensor([float(np.float16(x.item()))])
    # def unquant(x, scale): return x / scale

    state_dict = model.state_dict()

    for name, param in state_dict.items():
        # Don't update if this is not a weight.
        print(name)

        if "weight" not in name:
            continue

        # Transform the parameter as required.
        print(param.shape)

        if(len(param.shape) == 2):
            abs_max = 0

            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    abs_max = max(abs(param[i][j]), abs_max)

            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param[i][j] = quant(param[i][j])

        elif len(param.shape) == 1:
            abs_max = 0

            for i in range(param.shape[0]):
                abs_max = max(abs(param[i]), abs_max)

            for i in range(param.shape[0]):
                param[i] = quant(param[i])
        else:
            print('Param shape length is: ', len(param.shape))

        # Update the parameter.
        state_dict[name].copy_(param)

    inputs, _ = next(iter(testloader))

    generated_filename = 'nets/' + model_type + '_float16' + '.onnx'
    torch.onnx.export(model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      ex_input,
                      generated_filename,
                      # where to save the model (can be a file or file-like
                      # object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )
    print('Generated file: ', generated_filename)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--approx_type", help="specify a config file to run", default='quant16')
    parser.add_argument("--model", help="specify a config file to run", default='fcn')
    parser.add_argument("--dataset", help="specify a config file to run", default='mnist')
    parser.add_argument("--tune", help="specify a config file to run", default=False)

    args = parser.parse_args()

    print("Generating network for (", args.approx_type, args.model, args.dataset, args.tune, ")")
    generate_approximate_model(args.approx_type, args.model, args.dataset, args.tune)



