"""
We try 2 strategies for proving the patch safety of an approximate network:
1. Just use deepzono proof directly (baseline)
2. Create templates for original network and use these templates for the proof of approximate network

TODO: Implement your own zonotope analyzer based on pytorch! This will save you lots of time. 
"""

import sys
 
# TODO: remove these dependencies
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../tf_verify')

# from eran import ERAN
import random
import copy
import time
import template as tp
import argparse
import numpy as np
import os
import logging
import re
import pathlib
import util_pt
import attack

from proof_zono import init_nn, init_nn_at_layer, init_nn_with_optimizer, get_optimizer
from elina_abstract0 import elina_abstract0_free
from tabulate import tabulate
from read_net_file import read_onnx_net

cpu_affinity = os.sched_getaffinity(0)

domain = 'deepzono'
is_conv = False

'''
Runs the experiment for proof transfer between original network and a single approximate network.
'''
def approx_transfer(run_config):

    netname = run_config['network']
    approx_netname = run_config['anetwork']
    dataset = run_config['dataset']
    attack_type = run_config['attack']

    model, _ = read_onnx_net(netname)
    approx_model, _ = read_onnx_net(approx_netname)

    tests = util_pt.get_tests(dataset, False)
    test_li = [test for i, test in enumerate(tests)]

    strategies = run_config['strategy']

    # set default template search strategy
    if 'template_search' not in run_config: run_config['template_search'] = 'bs'

    if strategies == None:
        strategies = []

    for strategy in strategies:
        if 'tot_time' not in strategy:
            strategy['tot_time'] = 0

        while True:
            imn = random.randint(1, 99)

            logging.info('Iteration on image: %s', imn)

            image_start = np.float64(
                test_li[imn][1:len(test_li[imn])]) / np.float64(255)
            label = int(test_li[imn][0])

            # chek if the label matches the predicted class
            output, _ = util_pt.get_onnx_model_output(netname, image_start, dataset)
            
            print(output, label)

            if(output != label):
                logging.info('Skipping as the image is not classified correctly!')
                continue
            else:
                break
        
        # Get the specifications for the corresponding attack
        specLBs, specUBs = attack.get_specs(attack_type, imn, image_start, model, dataset)

        for strategy in strategies:

            if strategy['type'] == 1:
                logging.info('Strategy 1:')
                time1, count1 = verify_spec_brute(
                    approx_model, specLBs, specUBs, label)
                logging.info('%s %s', time1, count1)
                strategy['tot_time'] += time1

            if strategy['type'] == 2 or strategy['type'] == 3 or strategy['type'] == 4:
                logging.info('Strategy %s:', strategy['type'])

                if not 'templates' in strategy:
                        strategy['templates'] = []
                        template_start_time = time.time()

                        for val in strategy['layer']:
                            k1 = 2*val + 1
                            strategy['templates'] += tp.get_patch_eps_templates(
                                model, netname, image_start, label, k1, run_config['template']['length'], run_config['template']['width'], dataset, run_config['least_count'], run_config['template_search'])

                        strategy['template_creation_time'] = time.time() - template_start_time
                logging.info('Size of the template: %s', len(strategy['templates']))

                time3, count31, count32 = verify_spec_given_template(
                    approx_model, specLBs, specUBs, label, strategy['templates'], image_start, dataset, lc1 = run_config['lc1'], lc = run_config['lc'], adjust = (strategy['type'] == 3 or strategy['type'] == 4))
                logging.info('%s %s %s', time3, count31, count32)
                strategy['tot_time'] += time3
            
            logging.info('\n\n')
    
        # clear templates for new image
        for strategy in strategies:
            if 'templates' in strategy:
                del strategy['templates']
                
    for strategy in strategies:
        
        for key,val in strategy.items():
            logging.info('%s : %s', key, val) 
        logging.info('\n')

    logging.info("Config: %s", run_config)

'''
Runs the experiment for proof transfer between original network and a multiple approximate networks. Currently, it is set up to work with different pruned networks. 
'''
def tuning_transfer(run_config):

    netname = run_config['network']
    # approx_netname = run_config['anetwork']
    dataset = run_config['dataset']
    attack_type = run_config['attack']
    tuning_iterations = run_config['tuning_mode']['iterations']
    tuning_net_prefix = run_config['tuning_mode']['anetwork']

    model, _ = read_onnx_net(netname)
    # approx_model, _ = read_onnx_net(approx_netname)

    tests = util_pt.get_tests(dataset, False)
    test_li = [test for i, test in enumerate(tests)]

    strategies = run_config['strategy']

    if strategies == None:
        strategies = []

    for strategy in strategies:
        strategy['tot_time'] = {}

    while True:
        imn = random.randint(0, 99)

        logging.info('Iteration on image: %s', imn)

        image_start = np.float64(
            test_li[imn][1:len(test_li[imn])]) / np.float64(255)
        label = int(test_li[imn][0])

        output, _ = util_pt.get_onnx_model_output(netname, image_start, dataset)
                
        if(output != label):
            logging.info('Skipping as the image is not classified correctly!')
            continue
        else:
            break
    
    # Get the specifications for the corresponding attack
    specLBs, specUBs = attack.get_specs(attack_type, imn, image_start, model, dataset)
    
    # clear templates for new image
    for strategy in strategies:
        if 'templates' in strategy:
            del strategy['templates']

    for tune_i in range(tuning_iterations):
        logging.info('On the tuned network: %s', tune_i)

        approx_netname = tuning_net_prefix + '_tune' + str(tune_i) + '.onnx'
        approx_model, _ = read_onnx_net(approx_netname)

        for strategy in strategies:
            if not tune_i in strategy['tot_time']:
                    strategy['tot_time'][tune_i] = 0

            if strategy['type'] == 1:
                logging.info('Strategy 1:')
                time1, count1 = verify_spec_brute(
                    approx_model, specLBs, specUBs, label)
                logging.info("%s %s", time1, count1)
                strategy['tot_time'][tune_i] += time1
            
            if strategy['type'] == 2 or strategy['type'] == 3 or strategy['type'] == 4:
                logging.info('Strategy %s:', strategy['type'])

                should_create_templete = not 'templates' in strategy
                
                if strategy['type'] == 4 and tune_i == 6:
                    should_create_templete = True
                    # Use the current approximate model to create templates
                    model = approx_model

                # Create templates
                if should_create_templete:
                    strategy['templates'] = []
                    template_start_time = time.time()

                    for val in strategy['layer']:
                        k1 = 2*val + 1
                        strategy['templates'] += tp.get_patch_eps_templates(
                            model, netname, image_start, label, k1, run_config['template']['length'], run_config['template']['width'], dataset, run_config['least_count'], run_config['template_search'])

                    strategy['template_creation_time'] = time.time() - template_start_time
                    
                logging.info('Size of the template: %s', len(strategy['templates']))
                logging.info('Template creation time is: %s', strategy['template_creation_time'])

                # Verify the approximate model
                time3, count31, count32 = verify_spec_given_template(
                    approx_model, specLBs, specUBs, label, strategy['templates'], image_start, dataset, lc1 = run_config['lc1'], lc = run_config['lc'], adjust = (strategy['type'] == 3 or strategy['type'] == 4))
                logging.info("%s %s %s", time3, count31, count32)
                # tp.free_template_memory(templates)
                strategy['tot_time'][tune_i] += time3
            
            logging.info('\n\n')
    
    table = [['Type'] + [x for x in range(tuning_iterations)] + ['Total time']]

    for strategy in strategies:
        strategy['total_time'] = 0
        for tune_i in range(tuning_iterations):
            strategy['total_time'] += strategy['tot_time'][tune_i]

        table.append([strategy['type']] + [str(round(strategy['tot_time'][tune_i], 2)) for tune_i in range(tuning_iterations)] + [str(round(strategy['total_time'], 2))])

        for key,val in strategy.items():
            logging.info("%s : %s", key, val) 
        logging.info('\n')
    logging.info('\n')

    print(tabulate(table))
    logging.info("%s", tabulate(util_pt.invert_table(table), tablefmt='latex'))

    logging.info("Config: %s", run_config)
    
'''
 baseline verification
'''
def verify_spec_brute(approx_model, specLBs, specUBs, label):

    start_time = time.time()
    count = 0

    for i in range(len(specLBs)):
        specLB = specLBs[i]
        specUB = specUBs[i]
        nn, analyzer = init_nn(approx_model, specLB, specUB)

        element, nlb2, nub2 = analyzer.get_abstract0()

        elina_abstract0_free(analyzer.man, element)    

        is_verified = util_pt.verify(nlb2[-1], nub2[-1], label)
        
        if(is_verified):
            count += 1

    end_time = time.time()

    return end_time - start_time, count


'''
Verifies the specs given the templates. Adjusts the templates using the template transformer.
'''
def verify_spec_given_template(
        approx_model,
        specLBs,
        specUBs,
        label,
        old_templates,
        image,
        dataset,
        lc1 = 0, 
        lc = 0, 
        adjust = False):

    start_time = time.time()
    count1 = {}
    count2 = 0

    # prove the template
    logging.info('Try to prove the templates first:')
    layer_to_bound_map = {}

    # Template transformer: Adjust old templates for the new model
    if adjust:
        templates = tp.adjust_templates(approx_model, old_templates, image, dataset, lc1, lc)
    else:
        templates = old_templates

    for template in templates:

        bound = Bound(template.nlb[-1], template.nub[-1])

        nn, analyzer = init_nn_at_layer(
            approx_model, specLBs[0], specUBs[0], template.k)
       
        element1, _, nlb, nub = analyzer.get_abstract0_from_layer(
            template.k, template.element, [], [], nn)
        
        # elina_abstract0_free(man1, element1)
        template.is_verified = util_pt.verify(nlb[-1], nub[-1], label)

        if(template.is_verified):
            logging.info('The template at layer %s is verified on the approx model!', str(template.k))

            if(template.k not in layer_to_bound_map.keys()):
                layer_to_bound_map[template.k] = []

            layer_to_bound_map[template.k].append(bound)
        else:
            logging.info('The template at layer %s is not verified on the approx model! :(', str(template.k))

    logging.info('Template verification at: %s', time.time()-start_time)

    time_prop = 0
    time_contain = 0
    time_verification = 0
    prever = 0    
    
    optimizer = get_optimizer(approx_model)

    for i in range(len(specLBs)):
        specLB = specLBs[i]
        specUB = specUBs[i]

        nn, analyzer = init_nn_with_optimizer(optimizer, specLB, specUB)

        skip = False
        can_continue = False

        start_pre_ver = time.time()

        for k in layer_to_bound_map.keys():
            
            start_it = time.time()

            if can_continue:
                element1, man1, nlb2, nub2, nn = analyzer.get_abstract0_between_layer(prev_k, k, element1,nlb2, nub2, nn)
            else:    
                element1, man1, nlb2, nub2, nn = analyzer.get_abstract0_at_layer(k)

            cs = contain_score(nlb2[-1], nub2[-1], layer_to_bound_map[k])
            prev_k = k
            can_continue = True
            
            num_neurons_kth_layer = len(nlb2[-1])

            start_contain = time.time()
            time_prop += (start_contain - start_it)

            is_contained_bool = is_contained(nlb2[-1], nub2[-1], layer_to_bound_map[k], num_neurons_kth_layer)

            time_contain += (time.time() - start_contain)

            if(is_contained_bool):
                if k not in count1.keys():
                    count1[k] = 0

                count1[k] += 1
                skip = True
                break
        
        start_ver = time.time()
        prever += (time.time() - start_pre_ver)      

        if skip:
            continue

        if(can_continue):
            k = templates[-1].k
            nn, analyzer = init_nn_at_layer(approx_model, specLB, specUB, k)

            element2, man2, nlb2, nub2 = analyzer.get_abstract0_from_layer(
                k, element1, nlb2, nub2, nn)

            is_verified = util_pt.verify(nlb2[-1], nub2[-1], label)
        else:
            nn, analyzer = init_nn(approx_model, specLB, specUB)
            element2, nlb2, nub2 = analyzer.get_abstract0()

            is_verified = util_pt.verify(nlb2[-1], nub2[-1], label)

        time_verification += (time.time() - start_ver)

        if(is_verified):
            count2 += 1

    end_time = time.time()

    logging.info('Time other: %s', prever)
    logging.info('Time Prop: %s', time_prop)
    logging.info('Time contain: %s', time_contain)
    logging.info('Time verification: %s', time_verification)

    return end_time - start_time, count1, count2


class Bound:
    def __init__(self, nlb, nub):
        self.nlb = nlb
        self.nub = nub


def contain_score(lbi, ubi, bounds):
    total = len(lbi)
    contain = 0

    for t in range(len(bounds)):
        lbt = bounds[t].nlb
        ubt = bounds[t].nub
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

def is_contained(lbi, ubi, bounds, num_neurons):
    total = len(lbi)

    # Just for stats
    # for t in range(len(bounds)):
    #     lbt = bounds[t].nlb
    #     ubt = bounds[t].nub
    #     contain_t = 0

    #     # print(total, len(lbt), len(lbi))
    #     cnt = {}

    #     for i in range(total):
    #         # print(lbt[i], )
    #         if lbt[i] <= lbi[i] and ubi[i] <= ubt[i]:
    #             contain_t += 1
    #         else: 
    #             if i not in cnt:
    #                 cnt[i] = 0
    #             cnt[i] = cnt[i] + 1
    #             logging.info("%s ,%s %s, %s %s", i, lbt[i], lbi[i], ubi[i], ubt[i])
    #         # else:
    #         #     break
    #         # else:
    #         #     print(i, '.', lbt[i],lbi[i],ubi[i],ubt[i])
    #     logging.info("Contain: %s %s", contain_t, num_neurons)
    #     logging.info(cnt)
    #     if (contain_t == num_neurons):
    #         return True

    for t in range(len(bounds)):
        lbt = bounds[t].nlb
        ubt = bounds[t].nub
        contain_t = 0

        # print(total, len(lbt), len(lbi))

        for i in range(total):
            # print(lbt[i], )
            if lbt[i] <= lbi[i] and ubi[i] <= ubt[i]:
                contain_t += 1
            else:
                break
            # else:
            #     print(i, '.', lbt[i],lbi[i],ubi[i],ubt[i])
        if (contain_t == num_neurons):
            return True
        
    return False


if __name__ == '__main__':
    import yaml
    import argparse
    import pathlib

    get_name = lambda name: re.split(r'[\.,/]', name)[-2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="specify a config file to run", default='config.yml')
    parser.add_argument("--attack", help="specify a config file to run", default=None)
    parser.add_argument("--netname", help="specify a config file to run", default=None)
    parser.add_argument("--anetname", help="specify a config file to run", default=None)
    parser.add_argument("--iteration", help="specify a config file to run", default=1, type=int)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            run_configs = yaml.safe_load(stream)
            run_config = run_configs[get_name(args.netname)]
        except yaml.YAMLError as exc:
            print(exc)

    if(args.anetname is not None):
        run_config['anetwork'] = args.anetname
    
    # CL arguments can override config arguments
    if(args.attack is not None):
        run_config['attack'] = args.attack

    path = "paper_configs/" + run_config['attack'] + "/" + get_name(run_config['network']) + "/" + args.config

    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

    if('tuning_mode' in run_config):
        runfile = path + "/" + get_name(run_config['network']) + "_tune.yml"
    else:
        runfile = path + "/" +  get_name(run_config['anetwork']) + ".yml"
    
    if args.iteration > 1:
        with open(runfile, 'r') as stream:
            try:
                run_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)            

    if('lc1' not in run_config):
        run_config['lc1'] = 0
    
    if('lc1' not in run_config):
        run_config['lc'] = 0

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
    
    with open(runfile, 'w') as stream:
        yaml.dump(run_config, stream)
        


