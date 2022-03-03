'''
The module is responsible for constructing the templates. 
'''
import util_pt as util
from proof_zono import init_nn, init_nn_at_layer, init_nn_at_layer_with_optimizer, init_nn_with_optimizer, get_optimizer
from elina_abstract0 import elina_abstract0_free
import numpy as np
import sys
import logging
from attack import get_patch_spec_set
import torch
import util_pt

sys.path.insert(0, '../../ELINA/python_interface/')
from onnx import numpy_helper
from torch.nn import functional as F
from zonoml import *

class Template:
    def __init__(self, nlb, nub, element, man, k):
        self.nlb = nlb
        self.nub = nub
        self.element = element
        self.man = man
        self.k = k

    def free_memory(self):
        elina_abstract0_free(self.man, self.element)

def free_template_memory(templates):
    for template in templates:
        template.free_memory()

def get_patch_eps_templates(model, netname, image_start, label, k, plen, pwid, dataset, least_count, template_search):
    templates = []
    optimizer = get_optimizer(model)

    if dataset == 'mnist':
       width = 28
    else:
        width = 32     
    import time
    i = 0
    while i <= width - plen:
        j = 0
        while j <= width - pwid:
            
            # Find the maximum verifiable epsilon for 
            eps_min = 0
            eps_max = 1

            if template_search == 'bs':
                start_bs = time.time()
                # Standard binary search:
                while(eps_max - eps_min > least_count):
                    alpha = 0.5
                    eps_med = alpha*eps_min + (1-alpha)*eps_max

                    specLB, specUB = util.get_spec_patch_eps(
                    image_start, i, j, i + plen, j + pwid, dataset, eps_med)

                    is_verified = propagate_zono_as_box_at_layer_k(optimizer, specLB, specUB, label, k)
                    # is_verified = analyze_with_box(nn, specLB, specUB, [], [], label)

                    if is_verified:
                        eps_min = eps_med
                    else:
                        eps_max = eps_med

                
                if eps_max == 1:
                    epsilon = eps_max
                else:
                    epsilon = eps_min

                logging.info("Binary search time: %s Epsilon: %s", time.time()-start_bs, epsilon)

            else:
                
                if dataset=='mnist':
                    eps_cur = 0.25
                else:
                    eps_cur = 0.03125

                specLB, specUB = util.get_spec_patch_eps(
                    image_start, i, j, i + plen, j + pwid, dataset, eps_cur)
                
                is_verified = propagate_zono_as_box_at_layer_k(optimizer, specLB, specUB, label, k)

                if(is_verified):
                    while(eps_cur < 1):
                        eps_cur *= 2
                        specLB, specUB = util.get_spec_patch_eps(
                            image_start, i, j, i + plen, j + pwid, dataset, eps_cur)

                        is_verified = propagate_zono_as_box_at_layer_k(optimizer, specLB, specUB, label, k)

                        if(not is_verified):
                            epsilon = eps_cur/2
                            break
                    
                    if(eps_cur >= 1):
                        epsilon = 1
                else:
                    while(eps_cur > least_count):
                        eps_cur /= 2
                        specLB, specUB = util.get_spec_patch_eps(
                            image_start, i, j, i + plen, j + pwid, dataset, eps_cur)

                        is_verified = propagate_zono_as_box_at_layer_k(optimizer, specLB, specUB, label, k)

                        if(is_verified):
                            epsilon = eps_cur
                            break 
                    if(eps_cur < least_count):
                        epsilon = 0
            
            logging.info('%s is the selected max epsion.', epsilon)

            specLB, specUB = util.get_spec_patch_eps(
                image_start, i, j, i + plen, j + pwid, dataset, epsilon)
            
            # Getting the approximating box of the zonotope
            # We store the template as box approximation, since checking the containment is 
            # faster in the box 
            _, analyzer = init_nn(model, specLB, specUB)
            element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(k)
            element2 = elina_zonotope_to_box_zono(man, element)
            
            lb = nlb[-1]
            ub = nub[-1]

            templates.append(Template([lb], [ub], element2, man, k))

            j += pwid   
        i += plen

    return templates

def get_box_generated_templates(model, netname, image_start, label, k, plen, pwid, dataset, least_count, template_search):
    templates = []
    optimizer = get_optimizer(model)

    if dataset == 'mnist':
       width = 28
    else:
        width = 32     
    import time
    i = 0
    while i <= width - plen:
        j = 0
        while j <= width - pwid:
            
            # Find the maximum verifiable epsilon for 
            eps_min = 0
            eps_max = 1

            if template_search == 'bs':
                start_bs = time.time()
                # Standard binary search:
                while(eps_max - eps_min > least_count):
                    alpha = 0.5
                    eps_med = alpha*eps_min + (1-alpha)*eps_max

                    specLB, specUB = util.get_spec_patch_eps(
                    image_start, i, j, i + plen, j + pwid, dataset, eps_med)

                    is_verified = propagate_box(model, specLB, specUB, label)

                    if is_verified:
                        eps_min = eps_med
                    else:
                        eps_max = eps_med

                
                if eps_max == 1:
                    epsilon = eps_max
                else:
                    epsilon = eps_min

                logging.info("Binary search time: %s Epsilon: %s", time.time()-start_bs, epsilon)

            else:
                
                if dataset=='mnist':
                    eps_cur = 0.25
                else:
                    eps_cur = 0.03125

                specLB, specUB = util.get_spec_patch_eps(
                    image_start, i, j, i + plen, j + pwid, dataset, eps_cur)
                
                is_verified = propagate_box(model, specLB, specUB, label)

                if(is_verified):
                    while(eps_cur < 1):
                        eps_cur *= 2
                        specLB, specUB = util.get_spec_patch_eps(
                            image_start, i, j, i + plen, j + pwid, dataset, eps_cur)

                        is_verified = propagate_box(model, specLB, specUB, label)

                        # logging.info(is_verified)

                        if(not is_verified):
                            epsilon = eps_cur/2
                            break
                    
                    if(eps_cur >= 1):
                        epsilon = 1
                else:
                    while(eps_cur > least_count):
                        eps_cur /= 2
                        specLB, specUB = util.get_spec_patch_eps(
                            image_start, i, j, i + plen, j + pwid, dataset, eps_cur)

                        is_verified = propagate_box(model, specLB, specUB, label)

                        if(is_verified):
                            epsilon = eps_cur
                            break 
                    if(eps_cur < least_count):
                        epsilon = 0
            
            logging.info('%s is the selected max epsion.', epsilon)

            specLB, specUB = util.get_spec_patch_eps(
                image_start, i, j, i + plen, j + pwid, dataset, epsilon)
            
            # Just for getting man
            _, analyzer = init_nn(model, specLB, specUB)
            _, man, _, _, _ = analyzer.get_abstract0_at_layer(1)

            # Getting the box generated templates 
            # Fix k-2 thing
            lb, ub = propagate_box_at_layer_k(model, specLB, specUB, label, k-2)
            
            element2 = zonotope_from_network_input(man, 0, len(lb), np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64))

            templates.append(Template([lb], [ub], element2, man, k))

            j += pwid   
        i += plen

    return templates


# Need to train networks robust at particular layer
def get_box_eps_templates(model, netname, image_start, label, k, plen, pwid, dataset, least_count, template_search):
    templates = []
    optimizer = get_optimizer(model)

    specLB = np.clip(image_start, 0, 1)
    specUB = np.clip(image_start, 0, 1)
    util.normalize(specLB, dataset)
    util.normalize(specUB, dataset)

    nn, analyzer = init_nn(model, specLB, specUB)
    element, man, nlb, nub, _ = analyzer.get_abstract0_at_layer(k)

    eps_cur = 1
    while(eps_cur > least_count):
        print(eps_cur)
        eps_cur /= 2
        lb = []
        ub = []

        for ii in range(len(nlb[-1])):
            lb.append(nlb[-1][ii]-eps_cur)
            ub.append(nub[-1][ii]+eps_cur)

        element2 = zonotope_from_network_input(man, 0, len(lb), np.array(lb), np.array(ub))

        nn, analyzer = init_nn_at_layer_with_optimizer(optimizer, specLB, specUB, k)

        nlb2 = []
        nub2 = []
        _ = analyzer.get_abstract0_from_layer(k, element2, nlb2, nub2, nn)

        is_verified = util.verify(nlb2[-1], nub2[-1], label)

        if(is_verified):
            epsilon = eps_cur
            break 
    if(eps_cur < least_count):
        epsilon = 0
    
    lb = []
    ub = []

    for ii in range(len(nlb[-1])):
            lb.append(nlb[-1][ii]-epsilon)
            ub.append(nub[-1][ii]+epsilon)
    
    element2 = zonotope_from_network_input(man, 0, len(lb), np.array(lb), np.array(ub))
    
    templates.append(Template([lb], [ub], element2, man, k))
    
    return templates

def propagate_zono_as_box_at_layer_k(optimizer, specLB, specUB, label, k):
    nn, analyzer = init_nn_with_optimizer(optimizer, specLB, specUB)

    element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(k)
    element2 = zonotope_from_network_input(man, 0, len(nlb[-1]), np.array(nlb[-1]), np.array(nub[-1]))
    _ = analyzer.get_abstract0_from_layer(k, element2, nlb, nub, nn)
    is_verified = util.verify(nlb[-1], nub[-1], label)

    return is_verified

def propagate_box(model, specLB, specUB, label):
    num_layers = len(model.graph.node)
                    
    cur_layer = 0
    model_name_to_val_dict = {init_vals.name: torch.from_numpy(numpy_helper.to_array(init_vals)) for init_vals in model.graph.initializer} 
    cur_low = torch.from_numpy(specLB).float()
    cur_up = torch.from_numpy(specUB).float()

    prev_res = [(cur_low, cur_up)]
    cur_k = 0

    for cur_layer in range(num_layers):

        operation = model.graph.node[cur_layer].op_type
        nd_inps  = model.graph.node[cur_layer].input
        # nd_ops = model.graph.node[cur_layer].output

        # print(operation)

        if operation == 'Gemm':
            cur_k += 1
            inp_low = prev_res[cur_layer][0]
            inp_up = prev_res[cur_layer][1]
            
            wt = model_name_to_val_dict[nd_inps[1]].t()
            bias = model_name_to_val_dict[nd_inps[2]].t()

            wt_pos = F.relu(wt)
            wt_neg = -F.relu(-wt)

            cur_low = inp_low @ wt_pos + inp_up @ wt_neg + bias
            cur_up = inp_up @ wt_pos + inp_low @ wt_neg + bias

        elif operation == 'Add':
            inp_low = prev_res[cur_layer][0]
            inp_up = prev_res[cur_layer][1]

            inp2 = model_name_to_val_dict[nd_inps[1]]
            
            cur_low = inp_low + inp2
            cur_up = inp_up + inp2

        elif operation == 'Relu':
            cur_k += 1
            inp_low = prev_res[cur_layer][0]
            inp_up = prev_res[cur_layer][1]
            
            cur_low = F.relu(inp_low)
            cur_up = F.relu(inp_up)

        else:
            # no-op
            # print(cur_layer)
            # print(len(prev_res))

            cur_low = prev_res[cur_layer][0]
            cur_up = prev_res[cur_layer][1]

        prev_res.append((cur_low, cur_up))

        # if (cur_k-1 == k):
        #     break
    
    # print(cur_low)
    # print(cur_up)

    is_verified = ((cur_low[label] >= cur_up).sum() == 9).item()
    # print((cur_low[label] >= cur_up).sum())

    return is_verified

def propagate_box_at_layer_k(model, specLB, specUB, label, k):
    num_layers = len(model.graph.node)
                    
    cur_layer = 0
    model_name_to_val_dict = {init_vals.name: torch.from_numpy(numpy_helper.to_array(init_vals)) for init_vals in model.graph.initializer} 
    cur_low = torch.from_numpy(specLB).float()
    cur_up = torch.from_numpy(specUB).float()

    prev_res = [(cur_low, cur_up)]
    cur_k = 0

    for cur_layer in range(num_layers):

        operation = model.graph.node[cur_layer].op_type
        nd_inps  = model.graph.node[cur_layer].input
        # nd_ops = model.graph.node[cur_layer].output

        # print(operation)

        if operation == 'Gemm':
            cur_k += 1
            inp_low = prev_res[cur_layer][0]
            inp_up = prev_res[cur_layer][1]
            
            wt = model_name_to_val_dict[nd_inps[1]].t()
            bias = model_name_to_val_dict[nd_inps[2]].t()

            wt_pos = F.relu(wt)
            wt_neg = -F.relu(-wt)

            cur_low = inp_low @ wt_pos + inp_up @ wt_neg + bias
            cur_up = inp_up @ wt_pos + inp_low @ wt_neg + bias

        elif operation == 'Add':
            inp_low = prev_res[cur_layer][0]
            inp_up = prev_res[cur_layer][1]

            inp2 = model_name_to_val_dict[nd_inps[1]]
            
            cur_low = inp_low + inp2
            cur_up = inp_up + inp2

        elif operation == 'Relu':
            cur_k += 1
            inp_low = prev_res[cur_layer][0]
            inp_up = prev_res[cur_layer][1]
            
            cur_low = F.relu(inp_low)
            cur_up = F.relu(inp_up)

        else:
            # no-op
            # print(cur_layer)
            # print(len(prev_res))

            cur_low = prev_res[cur_layer][0]
            cur_up = prev_res[cur_layer][1]

        prev_res.append((cur_low, cur_up))

        if (cur_k-1 == k):
            break
    
    # print(cur_low)
    # print(cur_up)

    # is_verified = ((cur_low[label] >= cur_up).sum() == 9).item()
    # print((cur_low[label] >= cur_up).sum())

    return cur_low.numpy(), cur_up.numpy()

def get_Loo_template(model, image_start, label, k, dataset):
    # Find the maximum verifiable epsilon for L_oo
    eps_min = 0
    eps_max = 0.1

    import time

    total_pre = 0
    total_ver = 0
    cnt_iteration = 0

    while(eps_max - eps_min > 0.0001):
        cnt_iteration += 1    
        start_it = time.time()

        eps_med = (eps_min + eps_max) / 2

        # logging.info(image_start.shape)
        # logging.info(label)
        # eps_med = 0.2

        specLB = util.preprocess_spec(image_start - eps_med, dataset)
        specUB = util.preprocess_spec(image_start + eps_med, dataset)

        # logging.info(len(specUB))

        nn, analyzer = init_nn(model, specLB, specUB)

        pre_it = time.time()
        total_pre += (pre_it - start_it)

        # element, nlb, nub = analyzer.get_abstract0()

        element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(k)

        mid_it = time.time()

        element2, man2, nlb, nub = analyzer.get_abstract0_from_layer(k, element, nlb, nub, nn)

        
        end_it = time.time()


        is_verified = util.verify(nlb[-1], nub[-1], label)


        # logging.info(end_it - mid_it)    

        total_ver += (end_it - mid_it)

        if is_verified:
            eps_min = eps_med
        else:
            eps_max = eps_med

    epsilon = eps_min
    logging.info('%s is the selected max epsion.', epsilon)

    logging.info('total_pre: %s',total_pre)  
    logging.info('total_ver: %s',total_ver)  
    logging.info('cnt_iteration: %s',cnt_iteration)  

    specLB = util.preprocess_spec(image_start - epsilon, dataset)
    specUB = util.preprocess_spec(image_start + epsilon, dataset)

    nn, analyzer = init_nn(model, specLB, specUB)

    element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(k)
    # element_cp = copy.deepcopy(element)

    return Template(nlb, nub, element, man, k)

def adjust_templates(approx_model, old_templates, image, dataset, lc1, lc):
    templates = []

    for template in old_templates:
        specLB = np.clip(image, 0, 1)
        specUB = np.clip(image, 0, 1)
        util.normalize(specLB, dataset)
        util.normalize(specUB, dataset)

        nn, analyzer = init_nn(approx_model, specLB, specUB)
        element, man, nlb, nub, _ = analyzer.get_abstract0_at_layer(template.k)

        lb = []
        ub = []

        for ii in range(len(nlb[-1])):
            wid = template.nub[-1][ii] - template.nlb[-1][ii]
            if nlb[-1][ii]!=0 and nlb[-1][ii] - lc1*wid < template.nlb[-1][ii]:
                lbi = nlb[-1][ii]-lc*wid-lc1*wid
            else: 
                lbi = template.nlb[-1][ii]
            
            if nub[-1][ii]!=0 and template.nub[-1][ii] < nub[-1][ii]+lc1*wid:
                ubi = nub[-1][ii] + lc*wid + lc1*wid
            else:
                ubi = template.nub[-1][ii]
            # lbi = nlb[-1][ii] - wid
            # ubi = nub[-1][ii] + wid

            lb.append(lbi)
            ub.append(ubi)
            # lb.append(min(nlb[-1][ii], template.nlb[-1][ii]) - lc)
            # ub.append(max(nub[-1][ii], template.nub[-1][ii]) + lc)

        element2 = zonotope_from_network_input(man, 0, len(lb), np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64))

        templates.append(Template([lb], [ub], element2, man, template.k))
    
    return templates

