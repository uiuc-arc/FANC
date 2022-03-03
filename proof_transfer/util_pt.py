
import numpy as np
import csv
import argparse
import sys
import os
import onnxruntime as rt
from proof_zono import init_nn, init_nn_at_layer, get_optimizer
import random
from optimizer import *
from analyzer import *

'''
Generating one particular patch
'''
def get_spec_patch(image, xs, ys, xe, ye, dataset):
    specLB = np.clip(image, 0, 1)
    specUB = np.clip(image, 0, 1)

    if dataset == 'mnist':
        for i in range(xs, xe):
            for j in range(ys, ye):
                # print(specLB.shape)
                specLB[i * 28 + j] = 0
                specUB[i * 28 + j] = 1
    elif dataset == 'cifar10':
        for i in range(xs, xe):
            for j in range(ys, ye):
                specLB[i * 32 * 3 + j * 3] = 0
                specUB[i * 32 * 3 + j * 3] = 1
                specLB[i * 32 * 3 + j * 3 + 1] = 0
                specUB[i * 32 * 3 + j * 3 + 1] = 1
                specLB[i * 32 * 3 + j * 3 + 2] = 0
                specUB[i * 32 * 3 + j * 3 + 2] = 1

    normalize(specLB, dataset)
    normalize(specUB, dataset)
    return specLB, specUB

'''
Generating one particular patch
'''


def get_spec_patch_eps(image, xs, ys, xe, ye, dataset, epsilon):
    specLB = np.clip(image, 0, 1)
    specUB = np.clip(image, 0, 1)

    if dataset == 'mnist':
        for i in range(xs, xe):
            for j in range(ys, ye):
                specLB[i * 28 + j] = max(specLB[i * 28 + j] - epsilon, 0)
                specUB[i * 28 + j] = min(specUB[i * 28 + j] + epsilon, 1)
    elif dataset == 'cifar10':
        for i in range(xs, xe):
            for j in range(ys, ye):
                # specLB[i * 32 * 3 + j * 3] -= epsilon
                # specUB[i * 32 * 3 + j * 3] += epsilon
                # specLB[i * 32 * 3 + j * 3 + 1] -= epsilon
                # specUB[i * 32 * 3 + j * 3 + 1] += epsilon
                # specLB[i * 32 * 3 + j * 3 + 2] -= epsilon
                # specUB[i * 32 * 3 + j * 3 + 2] += epsilon

                specLB[i * 32 * 3 + j * 3] = max(specLB[i * 32 * 3 + j * 3] - epsilon, 0)
                specUB[i * 32 * 3 + j * 3] = min(specLB[i * 32 * 3 + j * 3] + epsilon, 1)
                specLB[i * 32 * 3 + j * 3 + 1] = max(specLB[i * 32 * 3 + j * 3 + 1] - epsilon, 0)
                specUB[i * 32 * 3 + j * 3 + 1] = min(specLB[i * 32 * 3 + j * 3 + 1] + epsilon, 1)
                specLB[i * 32 * 3 + j * 3 + 2] = max(specLB[i * 32 * 3 + j * 3 + 2] - epsilon, 0)
                specUB[i * 32 * 3 + j * 3 + 2] = min(specUB[i * 32 * 3 + j * 3 + 2] + epsilon, 1)

    normalize(specLB, dataset)
    normalize(specUB, dataset)

    # print(specLB[0], specUB[0])
    return specLB, specUB

def get_grad_sign(netname, image, i, j, label, dataset, epsilon):
    image_use = np.clip(image, 0, 1)
    _, out1 = get_onnx_model_output(netname, image_use, dataset)
    if dataset == 'mnist':
        image_use[i * 28 + j] += epsilon
    _, out2 = get_onnx_model_output(netname, image_use, dataset)

    if(out2[0][0][label]<out1[0][0][label]):
        return 1
    else: 
        return -1

def get_grad_patch_eps_perturb(netname, image, xs, ys, xe, ye, dataset, label, epsilon):
    image_out = np.clip(image, 0, 1)

    if dataset == 'mnist':
        for i in range(xs, xe):
            for j in range(ys, ye):
                sign = get_grad_sign(netname, image, i, j, label, dataset, epsilon)
                image_out[i * 28 + j] = image_out[i * 28 + j] + sign*epsilon

    elif dataset == 'cifar10':
        for i in range(xs, xe):
            for j in range(ys, ye):
                sign = get_grad_sign(netname, image, i, j, label, dataset, epsilon)
                image_out[i * 32 * 3 + j * 3] += sign*epsilon
                image_out[i * 32 * 3 + j * 3 + 1] += sign*epsilon
                image_out[i * 32 * 3 + j * 3 + 2] += sign*epsilon

    # print(specLB[0], specUB[0])
    return image_out

def get_random_patch_eps_perturb(image, xs, ys, xe, ye, dataset, epsilon):
    image_out = np.clip(image, 0, 1)

    if dataset == 'mnist':
        for i in range(xs, xe):
            for j in range(ys, ye):
                sign = [-1,1][random.randrange(2)]
                image_out[i * 28 + j] = image_out[i * 28 + j] + sign*epsilon

    elif dataset == 'cifar10':
        for i in range(xs, xe):
            for j in range(ys, ye):
                sign = [-1,1][random.randrange(2)]
                image_out[i * 32 * 3 + j * 3] += sign*epsilon
                image_out[i * 32 * 3 + j * 3 + 1] += sign*epsilon
                image_out[i * 32 * 3 + j * 3 + 2] += sign*epsilon

    # print(specLB[0], specUB[0])
    return image_out

'''
To verify the label
'''


def verify(lbs, ubs, label):
    for i in range(len(lbs)):
        if(i != label and ubs[i] > lbs[label]):
            return False
    return True

def get_model_output(model, image_start, dataset):
    specLB = preprocess_spec(image_start, dataset)
    specUB = preprocess_spec(image_start, dataset)

    nn, analyzer = init_nn(model, specLB, specUB)
    element, nlb2, nub2 = analyzer.get_abstract0()

    # assert nlb2[-1] == nub2[-1]
    return nlb2[-1]

def get_model_output_at_layer(model, image_start, dataset, k):
    specLB = preprocess_spec(image_start, dataset)
    specUB = preprocess_spec(image_start, dataset)

    # print(specLB)

    nn, analyzer = init_nn(model, specLB, specUB)
    element, _, nlb2, nub2, _ = analyzer.get_abstract0_at_layer(k)

    # assert nlb2[-1] == nub2[-1]
    return nlb2[-1]    

def get_onnx_model_output(netname, image_start, dataset):
    if dataset == 'cifar10':
        image = preprocess_spec(image_start, dataset)
        image = image.reshape((1,32,32,3)).transpose(0,3,1,2).reshape((3072))
  
        sess = rt.InferenceSession(netname)
        input_name = sess.get_inputs()[0].name
        pred_onnx = sess.run(None, {input_name: image.reshape(1,3,32,32).astype(np.float32)})
        # print(pred_onnx)
        return np.argmax(pred_onnx), pred_onnx
    elif dataset == 'mnist':
        image = preprocess_spec(image_start, dataset)
  
        sess = rt.InferenceSession(netname)
        input_name = sess.get_inputs()[0].name

        if 'conv' in netname: 
            image = image.reshape(1,1,28,28)
        else: 
            image = image.reshape(1, 784)
            
        pred_onnx = sess.run(None, {input_name: image.astype(np.float32)})
        # print(pred_onnx)
        return np.argmax(pred_onnx), pred_onnx

def l2_norm(y1, y2):
    val = 0
    for i in range(len(y1)):
        val += (y1[i]-y2[i])*(y1[i]-y2[i])    
    return val 

def linf_norm(y1, y2):
    val = 0
    val2 = 0
    for i in range(len(y1)):
        val = max(abs(y1[i]-y2[i]), val)
        val2 = max(abs(y1[i]), val2)
    return val/val2 

def invert_table(table):
    ii = len(table)
    jj = len(table[0])

    itable = []

    for j in range(jj):
        itable.append([])
        for i in range(ii):
            itable[-1].append(table[i][j])

    return itable

def layer_count(model, specLB):
    label = 1
    nn = layers()
    optimizer = get_optimizer(model)
    execute_list, _ = optimizer.get_deepzono(nn, specLB, specLB)
    analyzer = Analyzer(
        execute_list,
        nn,
        'deepzono',
        False,
        None,
        None,
        False,
        label,
        None,
        False)

    # for i in execute_list:
    #     print(i)
    return len(execute_list)

def preprocess_spec(spec, dataset):
    spec = np.clip(spec, 0, 1)
    normalize(spec, dataset)
    return spec


def normalize(image, dataset):
    # return
    if dataset == 'mnist':
        means = [0]
        stds = [1]
    else:
        # For the model that is loaded from cert def this normalization was
        # used
        stds = [0.2023, 0.1994, 0.2010]
        means = [0.4914, 0.4822, 0.4465]
        # means = [0.5, 0.5, 0.5]
        # stds = [1, 1, 1]

    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds is not None:
                image[i] /= stds[i]
    elif dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0]) / stds[0]
    elif(dataset == 'cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0]) / stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1]) / stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2]) / stds[2]
            count = count + 1

        # is_gpupoly = (domain == 'gpupoly' or domain == 'refinegpupoly')
        is_gpupoly = False
        is_conv = False

        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count + 1
                image[i + 1024] = tmp[count]
                count = count + 1
                image[i + 2048] = tmp[count]
                count = count + 1

        nimage = image.reshape((1,3,32,32)).transpose(0,2,3,1).reshape((3072))
        for i in range(3072):
            image[i] = nimage[i]

def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf', '.onnx', '.pb']:
        raise argparse.ArgumentTypeError(
            'only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname


def get_tests(dataset, geometric):
    if geometric:
        csvfile = open(
            '../../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        # if config.subset is None:
        csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')

    return tests
