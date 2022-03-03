import sys
sys.path.insert(0, '../')
import csv
import numpy as np
import torch
from onnx import numpy_helper
from torch.nn import functional as F
import util_pt
import attack

from read_net_file import read_onnx_net


def propagate_box_at_layer_k(model, specLB, specUB, label, k):
    num_layers = len(model.graph.node)
                    
    cur_layer = 0
    model_name_to_val_dict = {init_vals.name: torch.from_numpy(numpy_helper.to_array(init_vals)) for init_vals in model.graph.initializer} 
    cur_low = torch.from_numpy(specLB).t()
    cur_up = torch.from_numpy(specUB)

    prev_res = [(cur_low, cur_up)]
    cur_k = 0

    for cur_layer in range(num_layers):

        operation = model.graph.node[cur_layer].op_type
        nd_inps  = model.graph.node[cur_layer].input
        # nd_ops = model.graph.node[cur_layer].output

        print(operation)

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
    
    print(cur_low)
    print(cur_up)

    is_verified = ((cur_low[label] >= cur_up).sum() == 9).item()
    # print((cur_low[label] >= cur_up).sum())

    return is_verified

if __name__ == '__main__':
    netname = 'nets/fcnn7.onnx'
    model, _ = read_onnx_net(netname)
    dataset = "mnist"
    tests = util_pt.get_tests(dataset, False)
    test_li = [test for i, test in enumerate(tests)]
    image_start = np.float32(test_li[0][1:len(test_li[0])]) / np.float32(255)
    specLBs, specUBs = attack.get_patch_spec_set(image_start, 2, 2, dataset)
    label = int(test_li[0][0])

    spec = util_pt.preprocess_spec(image_start, dataset)


    out = util_pt.get_onnx_model_output(netname, image_start, dataset)
    print(out)

    isv = propagate_box_at_layer_k(model, specLBs[0], specUBs[0], label, 7)
    print(isv)


