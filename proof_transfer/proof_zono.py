"""
Module for transfering zonotope proofs
"""

import sys

sys.path.insert(0, '../ELINA/python_interface/')
# sys.path.insert(0, '../../deepg/code/')
sys.path.insert(0, '../tf_verify')

import os
import numpy as np
import argparse
import util_pt
import random

from deepzono_nodes import *
from onnx_translator import *
from optimizer import *
from analyzer import *
from eran import ERAN


# from ctypes import *
# from ctypes.util import *

cpu_affinity = os.sched_getaffinity(0)

imn = 1
domain = 'deepzono'
is_conv = False


def main(run_configs):
    print(run_configs)
    # We assume ONNX input for networks
    netname = run_configs[0]['network']
    # epsilon = config.epsilon

    dataset = run_configs[0]['dataset']

    model, _ = read_onnx_net(netname)

    is_onnx = True

    eran = ERAN(model, is_onnx=is_onnx)
    tests = util_pt.get_tests(dataset, False)
    test_li = [test for i, test in enumerate(tests)]
    image_start = np.float64(test_li[imn][1:len(test_li[0])]) / np.float64(255)

    specLB = np.copy(image_start)
    specUB = np.copy(image_start)

    util_pt.normalize(specLB, dataset)
    util_pt.normalize(specUB, dataset)

    _, nn, nlb, nub, _, _ = eran.analyze_box(
        specLB, specUB, 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
    # print("concrete ", nlb[-1])

    # Proof for image patches
    specLBs, specUBs = util_pt.get_spec_set(image_start, 2, 2, dataset)

    # Number of neurons in the paused layer
    num_neurons = 256

    # Number of templates that are created
    temp_cnt = 10

    # k=5
    # # for k in range(2, 11):
    # calculate_union_statistics_multi(
    #     test_li,
    #     specLBs,
    #     specUBs,
    #     model,
    #     k,
    #     temp_cnt,
    #     num_neurons)

    # for k in range(2, 11):
    k = 5
    check_overlap_for_Loo_template(
        test_li,
        specLBs,
        specUBs,
        model,
        k,
        num_neurons,
        image_start,
        dataset)


'''
Calculate the statistics such as average overlap and full overlaps between templates and intermediate proof shape.
'''


def calculate_union_statistics_multi(
        test_li,
        specLBs,
        specUBs,
        approx_model,
        k,
        temp_cnt,
        num_neurons):
    for i in range(4, 5):
        repeat = 10

        cnt_unverified = 0
        max_avg_ol = 0
        max_full_ol = 0
        tot_avg_ol = 0
        tot_full_ol = 0

        for j in range(repeat):

            # i here is used as the count of shapes that are unioned to
            # generate the template
            avg_ol, full_ol = check_overlap_union_multi(
                test_li, specLBs, specUBs, approx_model, k, i, temp_cnt, num_neurons)

            max_full_ol = max(max_full_ol, full_ol)
            max_avg_ol = max(max_avg_ol, avg_ol)
            tot_avg_ol += avg_ol
            tot_full_ol += full_ol

            if full_ol == 0:
                cnt_unverified += 1

        print('\n')
        print('For k = ', k)
        print('For random union of size: ', i)

        print('Max avergae ol: ', max_avg_ol)
        print('Max full ol: ', max_full_ol)

        if repeat > cnt_unverified:
            print('Average ol: ', tot_avg_ol / (repeat - cnt_unverified))
            print('Average full ol: ', tot_full_ol / (repeat - cnt_unverified))

        print('Total unverified:', cnt_unverified)


'''
Use L_oo input spec to create the template.
'''


def check_overlap_for_Loo_template(
        test_li,
        specLBs,
        specUBs,
        model,
        k,
        num_neurons,
        image_start,
        dataset):
    label = int(test_li[imn][0])

    # Pause a proof and rerun
    # Find the maximum verifiable epsilon for L_oo
    eps_min = 0
    eps_max = 1

    while(eps_max - eps_min > 0.0001):
        eps_med = (eps_min + eps_max) / 2

        # print(image_start.shape)
        # print(label)
        specLB = util_pt.preprocess_spec(image_start - eps_med, dataset)
        specUB = util_pt.preprocess_spec(image_start + eps_med, dataset)

        # print(len(specUB))

        nn, analyzer = init_nn(model, specLB, specUB)
        element, nlb, nub = analyzer.get_abstract0()
        is_verified = util_pt.verify(nlb[-1], nub[-1], label)
        if is_verified:
            eps_min = eps_med
        else:
            eps_max = eps_med

    epsilon = eps_min
    print(epsilon, ' is the selected max epsion.')

    specLB = util_pt.preprocess_spec(image_start - epsilon, dataset)
    specUB = util_pt.preprocess_spec(image_start + epsilon, dataset)

    # Expanding spec at first layer
    # Doesn't give much benefit
    # specLB, specUB = expand_spec_rand_search(label, model, specLB, specUB)

    nn, analyzer = init_nn(model, specLB, specUB)

    element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(k)

    # Expanding is actually making it worse!
    # element, _ = expand_template_at_layer_rand_search(k, element, man, label, num_neurons, model, np.clip(image_start - epsilon,0,1), np.clip(image_start + epsilon,0,1))

    nlbt = nlb[-1]
    nubt = nub[-1]

    _, _, nlb2, nub2 = analyzer.get_abstract0_from_layer(
        k, element, nlb, nub, nn)

    # print(nlbt)
    # print(nubt)

    is_verified = util_pt.verify(nlb2[-1], nub2[-1], label)

    if not is_verified:
        print('L_oo template not verified!')
        print('label:', label)
        print('nlb', nlb2[-1])
        print('nub', nub2[-1])
        return

    all_lbi = []
    all_ubi = []

    for i in range(len(specUBs)):
        specLB = specLBs[i]
        specUB = specUBs[i]

        nn, analyzer = init_nn(model, specLB, specUB)
        element, _, nlb, nub, nn = analyzer.get_abstract0_at_layer(k)

        all_lbi.append(nlb[-1])
        all_ubi.append(nub[-1])

    # print(all_lbi[0])
    # print(all_ubi[0])

    total_overlap = 0
    sum_overlap = 0
    cnt_full_overlap = 0

    for i in range(len(all_lbi)):
        # calculate the overlap
        cs = contain_score_multi(all_lbi[i], all_ubi[i], [nlbt], [nubt])
        total_overlap += 1
        sum_overlap += cs
        print(cs, num_neurons)
        cnt_full_overlap += (cs == num_neurons)

    print("For k = ", k)
    print("Average overlap:" + str(sum_overlap / total_overlap))
    print("Full overlap: " + str(cnt_full_overlap) + "/" + str(total_overlap))

    return sum_overlap / total_overlap, cnt_full_overlap


'''
@size is the count of shapes used to union to generate a template.
@temp_cnt is the number of templates that are created.
'''


def check_overlap_union_multi(
        test_li,
        specLBs,
        specUBs,
        model,
        k,
        size,
        temp_cnt,
        num_neurons):
    label = int(test_li[imn][0])

    temp_lbs = []
    temp_ubs = []

    choose = [i for i in range(0, 728)]

    while(len(temp_lbs) < temp_cnt):
        nn, analyzer = init_nn_at_layer(model, specLBs[0], specUBs[0], k)

        # Generate a template taking union of @size random shapes
        element, man = get_random_union_element_from_list(
            size, specLBs, specUBs, model, k, label, choose)

        # Expamd the generated template even more
        # This is not working correctly yet
        # element, man = expand_template_at_layer_rand_search(k, element, man, label, num_neurons, model, specLBs[0], specUBs[0])

        lb, ub = get_bound_from_element(man, element, num_neurons)

        # print(analyzer.nn.calc_layerno())
        _, _, nlb2, nub2 = analyzer.get_abstract0_from_layer(
            k, element, [], [], nn)
        # print(analyzer.nn.calc_layerno())
        is_verified = util_pt.verify(nlb2[-1], nub2[-1], label)

        # print(is_verified, len(temp_lbs), temp_cnt)

        if is_verified:
            temp_lbs.append(lb)
            temp_ubs.append(ub)

    all_lbi = []
    all_ubi = []

    for i in range(len(specUBs)):
        specLB = specLBs[i]
        specUB = specUBs[i]

        nn, analyzer = init_nn(model, specLB, specUB)
        element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(k)

        all_lbi.append(nlb[-1])
        all_ubi.append(nub[-1])

    total_overlap = 0
    sum_overlap = 0
    cnt_full_overlap = 0

    for i in range(len(all_lbi)):
        # calculate the overlap
        cs = contain_score_multi(all_lbi[i], all_ubi[i], temp_lbs, temp_ubs)
        total_overlap += 1
        sum_overlap += cs
        cnt_full_overlap += (cs == num_neurons)

    print("Average overlap:" + str(sum_overlap / total_overlap))
    print("Full overlap: " + str(cnt_full_overlap) + "/" + str(total_overlap))

    return sum_overlap / total_overlap, cnt_full_overlap


'''
Chooses m random patche specs and takes theie union
'''


def get_random_union_element_from_list(
        m, specLBs, specUBs, model, k, label, choose):

    specLB_list = []
    specUB_list = []

    for i in range(m):
        pi = random.randint(0, len(choose) - 1)
        p = choose[pi]
        # choose.remove(p)
        # print('Chosen patch: ', p)
        specLB_list.append(specLBs[p])
        specUB_list.append(specUBs[p])

    return get_multi_union_element(specLB_list, specUB_list, model, k, label)


'''
Gets the union of two zonotopes
'''


def get_multi_union_element(specLB_list, specUB_list, model, k, label):
    m = len(specLB_list)

    _, analyzer = init_nn(model, specLB_list[0], specUB_list[0])
    element_out, man, _, _, _ = analyzer.get_abstract0_at_layer(k)

    for i in range(m - 1):
        _, analyzer2 = init_nn(model, specLB_list[i + 1], specUB_list[i + 1])

        # print(analyzer2.nn.calc_layerno())
        element2, man, _, _, _ = analyzer2.get_abstract0_at_layer(k)
        # print(analyzer2.nn.calc_layerno())

        element_out = zonotope_union(man, element_out, element2)

    return element_out, man


def contain_score_multi(lbi, ubi, lbs, ubs):
    total = len(lbi)
    contain = 0

    temps = len(lbs)

    for t in range(len(lbs)):
        lbt = lbs[t]
        ubt = ubs[t]
        contain_t = 0

        # print(total, len(lbt), len(lbi))

        for i in range(total):
            # print(lbt[i], )
            if lbt[i] <= lbi[i] and ubi[i] <= ubt[i]:
                contain_t += 1
            # else:
            #     print(i, '.', lbt[i],lbi[i],ubi[i],ubt[i])
        contain = max(contain, contain_t)
    return contain


'''
 Returns the lb and ub for an abstract element of zonotope
'''


def get_bound_from_element(man, element, num_neurons):
    bounds = elina_abstract0_to_box(man, element)
    lbi = []
    ubi = []
    # print('layerno ',layerno)
    num_out_pixels = num_neurons
    for i in range(num_out_pixels):
        inf = bounds[i].contents.inf
        sup = bounds[i].contents.sup

        lbi.append(inf.contents.val.dbl)
        ubi.append(sup.contents.val.dbl)
    return lbi, ubi


'''
Randomly picks dimensions and keeps expanding it until it is verified
'''


def expand_template_at_layer_rand_search(
        k,
        element,
        man,
        label,
        num_neurons,
        model,
        specLB,
        specUB):

    alphas = []

    nn, analyzer = init_nn_at_layer(model, specLB, specUB, k)
    _, _, nlb2, nub2 = analyzer.get_abstract0_from_layer(
        k, element, [], [], nn)

    is_verified = verify(nlb2[-1], nub2[-1], label)

    print("Initial verification: ", is_verified)
    # copy
    element_dm = elina_zonotope_expand_dim_one_dir(man, element, 0, 0, 0)

    # Converted into box and again converted back to zonotope after the
    # expansion

    nn, analyzer = init_nn_at_layer(model, specLB, specUB, k)
    _, _, nlb2, nub2 = analyzer.get_abstract0_from_layer(
        k, element_dm, [], [], nn)

    is_verified = util.verify(nlb2[-1], nub2[-1], label)

    print("Copied verification: ", is_verified)

    fail_update = 0

    max_fail_update = 10
    delta_change = 0.5

    alphas = [[0, 0] for i in range(num_neurons)]

    while(fail_update < max_fail_update):

        dim_val = random.randint(0, num_neurons - 1)
        side = random.randint(0, 1)

        # print(dim_val)
        # print_box(element_dm, man)

        element_tm = elina_zonotope_expand_dim_one_dir(
            man, element_dm, delta_change, dim_val, side)

        # print("Changed to:")
        # print_box(element_tm, man)

        # After this element_tm is modified
        # is_verified = analyze_with_box_from_layer(k, nn, element_tm, man, [], [], label)
        # print(analyzer.nn.calc_layerno())
        nn, analyzer = init_nn_at_layer(model, specLB, specUB, k)

        _, _, nlb2, nub2 = analyzer.get_abstract0_from_layer(
            k, element_tm, [], [], nn)

        is_verified = verify(nlb2[-1], nub2[-1], label)

        # print(is_verified)

        if(is_verified):
            # print_box(element_tm, man)
            # copy
            element_dm = elina_zonotope_expand_dim_one_dir(
                man, element_dm, delta_change, dim_val, side)
            # print_box(element_dm, man)
            alphas[dim_val][side] += delta_change
            # print(alphas)
        else:
            fail_update += 1
    print('Final alpha value: ', alphas)
    return element_dm, man


'''
Randomly picks dimensions and keeps expanding it until it is verified
'''


def expand_spec_rand_search(
        label,
        model,
        specLB,
        specUB):

    alphas = []
    spec_len = len(specLB)

    nn, analyzer = init_nn(model, specLB, specUB)

    _, nlb2, nub2 = analyzer.get_abstract0()

    is_verified = verify(nlb2[-1], nub2[-1], label)

    print("Initial verification: ", is_verified)

    fail_update = 0

    max_fail_update = 100
    delta_change = 0.005

    alphas = [[0, 0] for i in range(spec_len)]

    while(fail_update < max_fail_update):

        dim_val = random.randint(0, spec_len - 1)
        side = random.randint(0, 1)

        if side == 0:
            specLB[dim_val] -= delta_change
        else:
            specUB[dim_val] += delta_change

        nn, analyzer = init_nn(model, specLB, specUB)

        _, nlb2, nub2 = analyzer.get_abstract0()

        is_verified = verify(nlb2[-1], nub2[-1], label)

        # print(is_verified)

        if(is_verified):
            # print_box(element_dm, man)
            alphas[dim_val][side] += delta_change
            # print(alphas)
        else:
            if side == 0:
                specLB[dim_val] += delta_change
            else:
                specUB[dim_val] -= delta_change

            fail_update += 1
    print('Final alpha value: ', alphas)
    return specLB, specUB


def get_optimizer(model):
    translator = ONNXTranslator(model, False)
    operations, resources = translator.translate()
    optimizer = Optimizer(operations, resources)
    return optimizer


def init_nn(model, specLB, specUB):
    label = 1 # does not matter, remove
    nn = layers()
    optimizer = get_optimizer(model)
    execute_list, _ = optimizer.get_deepzono(nn, specLB, specUB)
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

    return nn, analyzer

def init_nn_get_optimizer(model, specLB, specUB):
    label = 1
    nn = layers()
    optimizer = get_optimizer(model)
    execute_list, _ = optimizer.get_deepzono(nn, specLB, specUB)
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
    return nn, analyzer, optimizer


def init_nn_with_optimizer(optimizer, specLB, specUB):
    label = 1
    nn = layers()
    # optimizer = get_optimizer(model)
    execute_list, _ = optimizer.get_deepzono(nn, specLB, specUB)
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
    return nn, analyzer

# def init_nn_dp(model, specLB, specUB):
#     label = 1
#     nn = layers()
#     optimizer = get_optimizer(model)

#     execute_list, _ = optimizer.get_deeppoly(nn, specLB, specUB, None, None, None, None, None, None, 0, None)
#     analyzer = Analyzer(
#         execute_list,
#         nn,
#         'deeppoly',
#         False,
#         None,
#         None,
#         False,
#         label,
#         None,
#         False)

#     # for i in execute_list:
#     #     print(i)
#     return nn, analyzer

def init_nn_at_layer(model, specLB, specUB, k):
    label = 1
    nn = layers()
    optimizer = get_optimizer(model)
    execute_list, _ = optimizer.get_deepzono(nn, specLB, specUB)
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

    for i in range(k):
        nd = execute_list[i]
        # print(type(nd))
        if(isinstance(nd, DeepzonoRelu)):
            analyzer.nn.activation_counter += 1
            nn.activation_counter += 1
        if(isinstance(nd, DeepzonoAffine)):
            analyzer.nn.ffn_counter += 1
            nn.ffn_counter += 1

    # print('Haw ', analyzer.nn.calc_layerno())

    return nn, analyzer

def init_nn_at_layer_with_optimizer(optimizer, specLB, specUB, k):
    label = 1
    nn = layers()
    # optimizer = get_optimizer(model)
    execute_list, _ = optimizer.get_deepzono(nn, specLB, specUB)
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

    for i in range(k):
        nd = execute_list[i]
        # print(type(nd))
        if(isinstance(nd, DeepzonoRelu)):
            analyzer.nn.activation_counter += 1
            nn.activation_counter += 1
        if(isinstance(nd, DeepzonoAffine)):
            analyzer.nn.ffn_counter += 1
            nn.ffn_counter += 1

    # print('Haw ', analyzer.nn.calc_layerno())

    return nn, analyzer

if __name__ == '__main__':
    import yaml

    with open("config.yml", 'r') as stream:
        try:
            run_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(run_configs)
