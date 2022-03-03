# Try to see the radius of the box obtained at the k'th layer generated from the patch attack 

import sys

sys.path.insert(0, '../../ELINA/python_interface/')
sys.path.insert(0, '../../deepg/code/')
sys.path.insert(0, '../')

from eran import ERAN
from read_net_file import read_onnx_net
import random
import copy
import time
import template as tp
import argparse
import numpy as np
import os
import util_pt
import attack
from proof_zono import init_nn, init_nn_at_layer, init_nn_with_optimizer, get_optimizer
from elina_abstract0 import elina_abstract0_free
from zonoml import zonotope_from_network_input
from tabulate import tabulate

import logging
import re
import pathlib

cpu_affinity = os.sched_getaffinity(0)

domain = 'deepzono'
is_conv = False


def approx_transfer(run_config):

    netname = run_config['network']
    approx_netname = run_config['anetwork']
    dataset = run_config['dataset']
    attack_type = run_config['attack']

    model, _ = read_onnx_net(netname)
    approx_model, _ = read_onnx_net(approx_netname)

    tests = util_pt.get_tests(dataset, False)
    test_li = [test for i, test in enumerate(tests)]

    iterations = run_config['iterations']
    strategies = run_config['strategy']

    # set default template search strategy
    if 'template_search' not in run_config: run_config['template_search'] = 'bs'

    if strategies == None:
        strategies = []

    for strategy in strategies:
        strategy['avg_time'] = 0

    itr = 0
    total_its = 0

    while itr < iterations:
        total_its += 1
        imn = random.randint(0, 90)

        logging.info('Iteration on image: %s', imn)

        image_start = np.float64(
            test_li[imn][1:len(test_li[imn])]) / np.float64(255)
        label = int(test_li[imn][0])

        # chek if the label matches the predicted class
        nlb = util_pt.get_model_output(model, image_start, dataset)

        is_verified = util_pt.verify(nlb, nlb, label)
        
        if(not is_verified):
            logging.info('Skipping as the image is not classified correctly!')
            continue

        if attack_type == 'patch':
            specLBs, specUBs = attack.get_patch_spec_set(image_start, 2, 2, dataset)
        elif attack_type == 'l0':
            specLBs, specUBs = attack.get_l0_spec_set(image_start, 3, dataset, 20, model)
        elif attack_type == 'l0_center':
            specLBs, specUBs = attack.get_l0_center_spec_set(image_start, 5, dataset, 16, model)
        elif attack_type == 'l0_random':
            specLBs, specUBs = attack.get_l0_random_spec_set(image_start, 5, dataset, 1000)
        elif attack_type == 'bright':
            specLBs, specUBs = attack.get_bright_spec_set(image_start, 0.05, dataset, 8)
        elif attack_type == 'linf':
            specLBs, specUBs = attack.get_linf_spec_set(image_start, 0.05, dataset)

        itr += 1
        
        kk = 3

        olb = util_pt.get_model_output_at_layer(approx_model, image_start, dataset, kk)

        for ii in range(len(specLBs)):
            specLB = specLBs[ii]
            specUB = specUBs[ii]

            nn, analyzer = init_nn(approx_model, specLB, specUB)

            element, man, nlb, nub, _ = analyzer.get_abstract0_at_layer(kk)

            rd = get_radius(olb, nlb[-1], nub[-1])
            print('norm:', get_norm(olb))
            print('radius:', rd)

def get_radius(olb, nlb, nub):
    ma = 0
    assert len(olb) == len(nlb)
    for i in range(len(olb)):

        # print(i, ' : ', nlb[i], olb[i], nub[i])
        # assert nlb[i] <= olb[i]
        # assert nub[i] >= olb[i]

        ma = max(ma, olb[i]-nlb[i], nub[i]-olb[i])
    return ma

def get_norm(olb):
    ma = 0
    for i in range(len(olb)):
        ma = max(ma, abs(olb[i]))
    return ma

if __name__ == '__main__':
    import yaml
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="specify a config file to run", default='config.yml')
    parser.add_argument("--attack", help="specify a config file to run", default=None)
    parser.add_argument("--anetname", help="specify a config file to run", default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            run_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for run_config in run_configs:
        # CL arguments can override config arguments
        if(args.attack is not None):
            run_config['attack'] = args.attack

        if(args.anetname is not None):
            run_config['anetwork'] = args.anetname      

        if('lc1' not in run_config):
            run_config['lc1'] = 0
        
        if('lc1' not in run_config):
            run_config['lc'] = 0

        get_name = lambda name: re.split(r'[\.,/]', name)[-2]

        path = "paper_configs/" + run_config['attack'] + "/" + get_name(run_config['network']) 

        if('tuning_mode' in run_config):
            logfile = path + "/" + get_name(run_config['network']) + "_tune.txt"
        else:
            logfile = path + "/" +  get_name(run_config['anetwork']) + ".txt"

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)     

        logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
        logging.info("hello")    

        if('tuning_mode' in run_config):
            tuning_transfer(run_config)
        else:
            approx_transfer(run_config)

