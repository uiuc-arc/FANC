import sys

sys.path.insert(0, '../')
sys.path.insert(0, '../../ELINA/python_interface/')
sys.path.insert(0, '../../deepg/code/')

import numpy as np
import util_pt
from read_net_file import read_onnx_net
from proof_zono import init_nn, contain_score_multi
from approx_transfer import contain_score

def test1(image, model):
    k = 3

    specLB, specUB = util_pt.get_spec_patch(image, 0, 0, 4, 4, 'mnist')

    _, analyzer = init_nn(model, specLB, specUB)

    _, _, nlb, nub, _ = analyzer.get_abstract0_at_layer(k)

    for i in range(0, 2):
        for j in range(0, 2):
            specLB2, specUB2 = util_pt.get_spec_patch(image, i, j, i+2, j+2, 'mnist')

            _, analyzer = init_nn(model, specLB2, specUB2)

            _, _, nlb2, nub2, _ = analyzer.get_abstract0_at_layer(k)

            sz = len(nlb[-1])

            print(sz)
            print(contain_score_multi(nlb2[-1], nub2[-1], [nlb[-1]], [nub[-1]]))
    

if __name__ == '__main__':
    tests = util_pt.get_tests('mnist', False)
    test_li = [test for i, test in enumerate(tests)]
    imn = 0
    image = np.float64(
            test_li[imn][1:len(test_li[imn])]) / np.float64(255)

    model, _ = read_onnx_net('certifiedpatchdefense/fconv4.onnx')
    test1(image, model)