import logging
import util_pt
import random
from read_net_file import read_onnx_net
import numpy as np
import template as tp
from proof_zono import init_nn, init_nn_at_layer, init_nn_with_optimizer, get_optimizer
import attack
from tabulate import tabulate
import re

networks = ['nets/fcnn7', 'nets/fconv4', 'nets/fcnn7_cifar', 'nets/fconv4_cifar']
# approxs = ['float16', 'quant16', 'quant8', 'quant4']
prune_mode = True

# networks = ['nets/fcnn7_cifar']
# approxs = ['quant8']

def tt():
    logfile = 'paper_configs/template_transfer.txt'
    logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("hello")  

    if prune_mode:
        approxs = ['prune_tune' + str(x) for x in range(10)]

    for network in networks:
        for approx in approxs:

            netname = network + '.onnx'

            if prune_mode:
                get_name = lambda name: re.split(r'[\.,/]', name)[-1]
                approx_netname = network + '_prune/' + get_name(network) + '_' + approx + '.onnx' 
            else:
                approx_netname = network + '_' + approx + '.onnx'

            if('cifar' in netname):
                dataset = 'cifar10'
            else: 
                dataset = 'mnist'

            if network == 'nets/fcnn7':
                lent = 14
                widt = 14
                k = 2
                search = 'ms'
            elif network == 'nets/fconv4':
                lent = 14
                widt = 14
                k = 4
                search = 'ms'
            elif network == 'nets/fcnn7_cifar':
                lent = 16
                widt = 16
                k = 2
                search = 'ms'
            else:
                lent = 16
                widt = 16
                k = 5 
                search = 'ms'

            model, _ = read_onnx_net(netname)
            approx_model, _ = read_onnx_net(approx_netname)

            tests = util_pt.get_tests(dataset, False)
            test_li = [test for i, test in enumerate(tests)]

            count_verified = 0
            count_total = 0
            k = 2*k + 1

            for imn in range(25, 50):
                # imn = random.randint(0, 99)

                logging.info('Iteration on image: %s', imn)
                

                image_start = np.float64(
                    test_li[imn][1:len(test_li[imn])]) / np.float64(255)
                label = int(test_li[imn][0])
                
                specLBs, specUBs = attack.get_patch_spec_set(image_start, 2, 2, dataset)
                
                templates = tp.get_patch_eps_templates(
                            model, netname, image_start, label, k, lent, widt, dataset, 0.005, search)

                logging.info('Try to prove the templates first:')

                # template.nlb gets modified after the proof
                num_neurons = len(templates[0].nlb[-1])
                logging.info('Number of neurons in the template layer: %s', num_neurons)


                for template in templates:
                    count_total += 1

                    nn, analyzer = init_nn_at_layer(
                        approx_model, specLBs[0], specUBs[0], template.k)
                
                    element1, man1, nlb, nub = analyzer.get_abstract0_from_layer(
                        template.k, template.element, [], [], nn)
                    
                    # elina_abstract0_free(man1, element1)
                    template.is_verified = util_pt.verify(nlb[-1], nub[-1], label)

                    if(template.is_verified):
                        count_verified += 1
                        logging.info('The template at layer %s is verified on the approx model!', str(template.k))

                    else:
                        logging.info('The template at layer %s is not verified on the approx model! :(', str(template.k))
                        # logging.info()
                
            logging.info('Network: %s', netname)
            logging.info('Approximation: %s', approx)
            logging.info('Verified templates: %s   Total templates %s:  ratio: %s', count_verified, count_total, count_verified/count_total)

def cal_diff(approxs):
    logfile = 'paper_configs/network_diff.txt'
    logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("hello") 

    for network in networks:
        
        netname = network + '.onnx'

        model, _ = read_onnx_net(netname)

        if('cifar' in netname):
            dataset = 'cifar10'
        else: 
            dataset = 'mnist'
        
        tests = util_pt.get_tests(dataset, False)
        test_li = [test for i, test in enumerate(tests)]

        # calculate total layers
        image_ex = np.float64(test_li[0][1:len(test_li[0])]) / np.float64(255)
        layer_no = util_pt.layer_count(model, image_ex)

        table = [[''] + [x for x  in range(1, layer_no//2)]] 
        
        if prune_mode:
            approxs = ['prune_tune' + str(x) for x in range(10)]

        for approx in approxs:

            if prune_mode:
                get_name = lambda name: re.split(r'[\.,/]', name)[-1]
                approx_netname = network + '_prune/' + get_name(network) + '_' + approx + '.onnx' 
            else:
                approx_netname = network + '_' + approx + '.onnx'     

            approx_model, _ = read_onnx_net(approx_netname)

            dis_l2_mp = {}
            dis_linf_mp = {}

            total = 10

            for imn in range(total):

                logging.info('Iteration on image: %s', imn)

                image_start = np.float64(
                    test_li[imn][1:len(test_li[imn])]) / np.float64(255)
                label = int(test_li[imn][0])


                for ii in range(1, layer_no//2):
                    layer = 2*ii+1

                    y1 = util_pt.get_model_output_at_layer(model, image_start, dataset, layer)
                    y2 = util_pt.get_model_output_at_layer(approx_model, image_start, dataset, layer)

                    dis_l2 = util_pt.l2_norm(y1, y2)
                    dis_linf = util_pt.linf_norm(y1, y2)
                    
                    if layer not in dis_l2_mp:
                        dis_l2_mp[ii] = dis_l2/total
                    else:
                        dis_l2_mp[ii] += dis_l2/total
                    
                    if layer not in dis_linf_mp:
                        dis_linf_mp[ii] = dis_linf/total
                    else:
                        dis_linf_mp[ii] += dis_linf/total
                    
                    logging.info("layer: %s, dis_l2: %s, dis_linf: %s", ii, dis_l2, dis_linf) 

            table.append([approx] + [dis_linf_mp[x] for x in dis_linf_mp.keys()])

            logging.info('Network: %s', netname)
            logging.info('Approximation: %s', approx)

            logging.info('Average distance mapping l2: %s linf: %s',dis_l2_mp, dis_linf_mp)
        
        print(tabulate(table))


if __name__ == '__main__':
    tt()
    # cal_diff([])