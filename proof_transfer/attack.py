'''
Module to generate input region specifications for various attacks
'''

import util_pt
import numpy as np
import random
import copy
import logging

def get_specs(attack_type, imn, image_start, model, dataset):
    if attack_type == 'patch':
        specLBs, specUBs = get_patch_spec_set(image_start, 2, 2, dataset)
    elif attack_type == 'l0':
        specLBs, specUBs = get_l0_spec_set(image_start, 3, dataset, 20, model)
    elif attack_type == 'l0_center':
        specLBs, specUBs = get_l0_center_spec_set(image_start, 3, dataset, 16, model)
    elif attack_type == 'l0_random':
        specLBs, specUBs = get_l0_random_spec_set(image_start, 3, dataset, 1000)
    elif attack_type == 'bright':
        specLBs, specUBs = get_bright_spec_set(image_start, 0.02, dataset, 7)
    elif attack_type == 'rotation':
        specLBs, specUBs = get_rotation_spec_set(imn-1, dataset)
        # print(label)
        # label, _, _, _ = analyze_box(spec_lb, spec_ub, 'deeppoly', args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
    return specLBs, specUBs

'''
Generating all 729 patches calling the get_spec_patch function for MNIST and 961 for CIFAR10
'''


def get_patch_spec_set(image, patch_len, patch_wid, dataset):
    specLBs = []
    specUBs = []

    if dataset == 'mnist':
        for i in range(28 - patch_len + 1):
            for j in range(28 - patch_wid + 1):
                specLB, specUB = util_pt.get_spec_patch(
                    image, i, j, i + patch_len, j + patch_wid, dataset)
                specLBs.append(specLB)
                specUBs.append(specUB)

    elif dataset == 'cifar10':
        for i in range(32 - patch_len + 1):
            for j in range(32 - patch_wid + 1):
                specLB, specUB = util_pt.get_spec_patch(
                    image, i, j, i + patch_len, j + patch_wid, dataset)
                specLBs.append(specLB)
                specUBs.append(specUB)

    return specLBs, specUBs

'''
Generating L_inf specs
'''

def get_linf_spec_set(image, eps, dataset):
    specLBs = []
    specUBs = []

    if dataset == 'mnist':
        specLB = np.clip(image - eps, 0, 1)
        specUB = np.clip(image + eps, 0, 1)

        specLBs.append(specLB)
        specUBs.append(specUB)

    elif dataset == 'cifar10':
        specLB = np.clip(image - eps, 0, 1)
        specUB = np.clip(image + eps, 0, 1)

        specLBs.append(specLB)
        specUBs.append(specUB)

    return specLBs, specUBs

'''
Generating N specs with L0 attack
'''


def get_l0_spec_set(image, l0_norm, dataset, count, model):
    specLBs = []
    specUBs = []

    if dataset == 'mnist':
        # Find top 20 pixels with max gradient
        delta = 0.01
        grad_pix_list = [] 

        y1 = util_pt.get_model_output(model, image, dataset)

        for i in range(28):
            for j in range(28):           
                image[i * 28 + j] += delta 
                y2 = util_pt.get_model_output(model, image, dataset)
                image[i * 28 + j] -= delta
                nrm = util_pt.l2_norm(y1, y2)

                grad_pix_list.append((nrm, (i, j)))

        grad_pix_list = sorted(grad_pix_list, key=lambda x: x[0], reverse=True)[:count]

        for i in range(count):
            for j in range(i+1, count):
                for k in range(j+1, count):

                    pixels = [grad_pix_list[i][1], grad_pix_list[j][1], grad_pix_list[k][1]]
                    
                    specLB = np.clip(image, 0, 1)
                    specUB = np.clip(image, 0, 1) 

                    for pp in range(l0_norm):
                        x = pixels[pp][0]
                        y = pixels[pp][1]

                        specLB[x * 28 + y] = 0
                        specUB[x * 28 + y] = 1

                    util_pt.normalize(specLB, dataset)
                    util_pt.normalize(specUB, dataset)

                    specLBs.append(specLB)
                    specUBs.append(specUB)
    elif dataset == 'cifar10':
        # Find top 20 pixels with max gradient
        delta = 0.01
        grad_pix_list = [] 

        y1 = util_pt.get_model_output(model, image, dataset)

        for i in range(32):
            for j in range(32):           
                image[i * 32 * 3 + j * 3] += delta 
                image[i * 32 * 3 + j * 3 + 1] += delta 
                image[i * 32 * 3 + j * 3 + 2] += delta 
                y2 = util_pt.get_model_output(model, image, dataset)
                image[i * 32 * 3 + j * 3] -= delta
                image[i * 32 * 3 + j * 3 + 1] -= delta
                image[i * 32 * 3 + j * 3 + 2] -= delta
                nrm = util_pt.l2_norm(y1, y2)

                grad_pix_list.append((nrm, (i, j)))

        grad_pix_list = sorted(grad_pix_list, key=lambda x: x[0], reverse=True)[:count]

        for i in range(count):
            for j in range(i+1, count):
                for k in range(j+1, count):

                    pixels = [grad_pix_list[i][1], grad_pix_list[j][1], grad_pix_list[k][1]]
                
                    specLB = np.clip(image, 0, 1)
                    specUB = np.clip(image, 0, 1)

                    for pp in range(l0_norm):
                        x = pixels[pp][0]
                        y = pixels[pp][1]

                        specLB[x * 32 * 3 + y * 3] = 0
                        specUB[x * 32 * 3 + y * 3] = 1
                        specLB[x * 32 * 3 + y * 3 + 1] = 0
                        specUB[x * 32 * 3 + y * 3 + 1] = 1
                        specLB[x * 32 * 3 + y * 3 + 2] = 0
                        specUB[x * 32 * 3 + y * 3 + 2] = 1
                    
                    util_pt.normalize(specLB, dataset)
                    util_pt.normalize(specUB, dataset)

                    specLBs.append(specLB)
                    specUBs.append(specUB)

    return specLBs, specUBs

'''
Generating N specs with L0 attack on randomly chosen pixels
'''


def get_l0_random_spec_set(image, l0_norm, dataset, count):
    specLBs = []
    specUBs = []

    if dataset == 'mnist':
        for i in range(count):
            length = 28
            width = 28

            specLB = np.clip(image, 0, 1)
            specUB = np.clip(image, 0, 1) 

            for _ in range(l0_norm):
                x = random.randint(0, length-1)
                y = random.randint(0, width-1)

                specLB[x * 28 + y] = 0
                specUB[x * 28 + y] = 1

            util_pt.normalize(specLB, dataset)
            util_pt.normalize(specUB, dataset)

            specLBs.append(specLB)
            specUBs.append(specUB)

    elif dataset == 'cifar10':
        for i in range(count):
            length = 32
            width = 32

            specLB = np.clip(image, 0, 1)
            specUB = np.clip(image, 0, 1)

            for _ in range(l0_norm):
                x = random.randint(0, length-1)
                y = random.randint(0, width-1)

                specLB[x * 32 * 3 + y * 3] = 0
                specUB[x * 32 * 3 + y * 3] = 1
                specLB[x * 32 * 3 + y * 3 + 1] = 0
                specUB[x * 32 * 3 + y * 3 + 1] = 1
                specLB[x * 32 * 3 + y * 3 + 2] = 0
                specUB[x * 32 * 3 + y * 3 + 2] = 1

            util_pt.normalize(specLB, dataset)
            util_pt.normalize(specUB, dataset)

            specLBs.append(specLB)
            specUBs.append(specUB)

    return specLBs, specUBs

'''
Generating N specs with L0 attack pixels from the center
'''


def get_l0_center_spec_set(image, l0_norm, dataset, count, model):
    specLBs = []
    specUBs = []
    count = 20

    if dataset == 'mnist':
        cent_x = 12
        cent_y = 12

        funx = lambda c : cent_x + (c//4)
        funy = lambda c : cent_y + (c%5)

        for i in range(count):
            for j in range(i+1, count):
                for k in range(j+1, count):

                    pixels = [(funx(i), funy(i)), (funx(j), funy(j)), (funx(k),funy(k))]
                    # logging.info(pixels)

                    specLB = np.clip(image, 0, 1)
                    specUB = np.clip(image, 0, 1) 

                    for pp in range(l0_norm):
                        x = pixels[pp][0]
                        y = pixels[pp][1]

                        specLB[x * 28 + y] = 0
                        specUB[x * 28 + y] = 1

                    util_pt.normalize(specLB, dataset)
                    util_pt.normalize(specUB, dataset)

                    specLBs.append(specLB)
                    specUBs.append(specUB)
        logging.info(len(specLBs))
    elif dataset == 'cifar10':
        cent_x = 15
        cent_y = 15

        funx = lambda c : cent_x + (c//4)
        funy = lambda c : cent_y + (c%5)

        for i in range(count):
            for j in range(i+1, count):
                for k in range(j+1, count):

                    pixels = [(funx(i), funy(i)), (funx(j), funy(j)), (funx(k),funy(k))]
                
                    specLB = np.clip(image, 0, 1)
                    specUB = np.clip(image, 0, 1)

                    for pp in range(l0_norm):
                        x = pixels[pp][0]
                        y = pixels[pp][1]

                        specLB[x * 32 * 3 + y * 3] = 0
                        specUB[x * 32 * 3 + y * 3] = 1
                        specLB[x * 32 * 3 + y * 3 + 1] = 0
                        specUB[x * 32 * 3 + y * 3 + 1] = 1
                        specLB[x * 32 * 3 + y * 3 + 2] = 0
                        specUB[x * 32 * 3 + y * 3 + 2] = 1
                    
                    util_pt.normalize(specLB, dataset)
                    util_pt.normalize(specUB, dataset)

                    specLBs.append(specLB)
                    specUBs.append(specUB)

    return specLBs, specUBs

'''
Generating specs for Brightness attack with 2^log_splits splits
'''


def get_bright_spec_set(image, delta, dataset, log_splits):
    specLBs = []
    specUBs = []
    dif_ind_pairs = []

    if dataset == 'mnist':
        length = 28
        width = 28
    
        specLB = np.clip(image, 0, 1)
        specUB = np.clip(image, 0, 1) 

        for i in range(length):
            for j in range(width):
                if(image[i * length + j] > (1-delta)):
                    specLB[i * length + j] = image[i * length + j]
                    specUB[i * length + j] = 1
                    dif_ind_pairs.append((1-image[i * length + j], i * length + j))

        logging.info('Pixels brightned: %s', len(dif_ind_pairs))
        dif_ind_pairs = sorted(dif_ind_pairs, key=lambda x: x[0], reverse=True)[:log_splits]

        i = 0
        while i < (1<<log_splits):
            specLBC = copy.deepcopy(specLB)
            specUBC = copy.deepcopy(specUB)

            j = i
            k = 0

            while (j!=0):
                index = dif_ind_pairs[k][1]

                if j%2 == 0:
                    specLBC[index] = image[index] + (1 - image[index])/2
                else:
                    specUBC[index] = image[index] + (1 - image[index])/2

                j = j//2
                k += 1

            util_pt.normalize(specLB, dataset)
            util_pt.normalize(specUB, dataset)

            specLBs.append(specLBC)
            specUBs.append(specUBC)
            i += 1

    elif dataset == 'cifar10':
        length = 32
        width = 32  

        logging.info(type(image))
        logging.info(np.max(image))
        logging.info(np.min(image))

        ma = 0
        # calculate max av
        for i in range(length):
            for j in range(width):
                av = (image[i * 32 * 3 + j * 3] + image[i * 32 * 3 + j * 3 + 1] + image[i * 32 * 3 + j * 3 + 2])/3
                ma = max(ma, av)
        # ma = np.max(image)

        specLB = np.clip(image, 0, 1)
        specUB = np.clip(image, 0, 1)

        for i in range(length):
            for j in range(width):
                av = (image[i * 32 * 3 + j * 3] + image[i * 32 * 3 + j * 3 + 1] + image[i * 32 * 3 + j * 3 + 2])/3
                # for k in range(3):
                    # logging.info(image[i * 32 * 3 + j * 3 + k] )
                if(av > (1-delta)*ma):
                    specLB[i * 32 * 3 + j * 3] = image[i * 32 * 3 + j * 3]
                    specUB[i * 32 * 3 + j * 3] = 1
                    specLB[i * 32 * 3 + j * 3 + 1] = image[i * 32 * 3 + j * 3 + 1]
                    specUB[i * 32 * 3 + j * 3 + 1] = 1
                    specLB[i * 32 * 3 + j * 3 + 2] = image[i * 32 * 3 + j * 3 + 2]
                    specUB[i * 32 * 3 + j * 3 + 2] = 1
                    dif_ind_pairs.append((1-av, i * 32 * 3 + j))
         

        logging.info('Pixels brightned:', len(dif_ind_pairs))
        dif_ind_pairs = sorted(dif_ind_pairs, key=lambda x: x[0], reverse=True)[:log_splits]
        logging.info(dif_ind_pairs)

        i = 0
        while i < (1<<log_splits):
            specLBC = copy.deepcopy(specLB)
            specUBC = copy.deepcopy(specUB)

            j = i
            k = 0

            while (j!=0):
                index = dif_ind_pairs[k][1]

                if j%2 == 0:
                    specLBC[index] = image[index] + (1 - image[index])/2
                else:
                    specUBC[index] = image[index] + (1 - image[index])/2

                j = j//2
                k += 1
            
            util_pt.normalize(specLB, dataset)
            util_pt.normalize(specUB, dataset)

            specLBs.append(specLBC)
            specUBs.append(specUBC)
            i += 1

    return specLBs, specUBs

import os

# Not working with CIFAR10
def get_rotation_spec_set(imn, dataset):

    if dataset == 'mnist':
        specs_file = os.path.join('../data/rotation', '{}.csv'.format(imn))
        dim = 784
    else:
        assert False, "Does not work!"
        dim = 3072

    num_params = 1

    k = num_params + 1 + 1 + dim
    specLBs = []
    specUBs = []

    with open(specs_file, 'r') as fin:
        lines = fin.readlines()
        print('Number of lines: ', len(lines))
        assert len(lines) % k == 0
    
        for i, line in enumerate(lines):
            specLB = np.zeros(dim)
            specUB = np.zeros(dim)

            if i % k < num_params:
                # read specs for the parameters
                values = np.array(list(map(float, line[:-1].split(' '))))
                assert values.shape[0] == 2
                param_idx = i % k
                # spec_lb[dim + param_idx] = values[0]
                # spec_ub[dim + param_idx] = values[1]

            elif i % k == num_params:
                # read interval bounds for image pixels
                # print('line:', i)

                values = np.array(list(map(float, line[:-1].split(','))))

                specLB = values[::2]
                specUB = values[1::2]
                # if args.debug:
                #     show_ascii_spec(spec_lb, spec_ub)
                # util_pt.normalize(specLB, dataset)
                # util_pt.normalize(specUB, dataset)

                specLBs.append(specLB)
                specUBs.append(specUB)
                # print('tot:', (specUBs[-1] - specLBs[-1] > 0.0001).sum())  
                 
            
    return specLBs, specUBs 