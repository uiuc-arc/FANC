"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import numpy as np
from zonoml import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from functools import reduce
from ai_milp import *
from krelu import encode_kactivation_cons
from config import *
if config.device == Device.CPU:
    from fppoly import *
else:
    from fppoly_gpu import *


def refine_activation_with_solver_bounds(nn, self, man, element, nlb, nub, relu_groups, timeout_lp, timeout_milp, use_default_heuristic, domain):
    """
    refines the relu transformer

    Arguments
    ---------
    self : Object
        will be a DeepzonoNode or DeeppolyNode
    man : ElinaManagerPtr
        manager which is responsible for element
    element : ElinaAbstract0Ptr
        the element in which the results after affine transformation are stored
    nlb: list of list of doubles
        contains the lower bounds for all neurons upto layer layerno
    nub: list of list of doubles
        contains the upper bounds for all neurons upto layer layerno
    use_milp: bool array
        whether to use milp or lp for refinement
    Return
    ------
     the updated abstract element
    """
    layerno = nn.calc_layerno()
    predecessor_index = nn.predecessors[layerno+1][0] - 1
    if domain == 'deepzono':
        offset, length = self.abstract_information
    else:
        offset = 0
        length = get_num_neurons_in_layer(man, element, predecessor_index)
    lbi = nlb[predecessor_index]
    ubi = nub[predecessor_index]
    first_FC = -1
    timeout = timeout_milp
    for i in range(nn.numlayer):
        if nn.layertypes[i] == 'FC':
            first_FC = i
            break

    if nn.activation_counter==0:
        if domain=='deepzono':
            encode_kactivation_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi,
                                    relu_groups, False, 'refinepoly', nn.layertypes[layerno])
            if nn.layertypes[layerno] == 'ReLU':
                element = relu_zono_layerwise(man,True,element,offset, length, use_default_heuristic)
            elif nn.layertypes[layerno] == 'Sigmoid':
                element = sigmoid_zono_layerwise(man, True, element, offset, length)
            else:
                element = tanh_zono_layerwise(man, True, element, offset, length)
            return element
        else:
            encode_kactivation_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi,
                                    relu_groups, False, 'refinepoly', nn.layertypes[layerno])
            if nn.layertypes[layerno] == 'ReLU':
                handle_relu_layer(*self.get_arguments(man, element), use_default_heuristic)
            elif nn.layertypes[layerno] == 'Sigmoid':
                handle_sigmoid_layer(*self.get_arguments(man, element), use_default_heuristic)
            else:
                handle_tanh_layer(*self.get_arguments(man, element), use_default_heuristic)

    else:
        if predecessor_index==first_FC:
            use_milp = 1
        else:
            use_milp = 0
            timeout = timeout_lp
        use_milp = use_milp and config.use_milp
        candidate_vars = []
        for i in range(length):
            if((lbi[i]<0 and ubi[i]>0) or (lbi[i]>0)):
                 candidate_vars.append(i)
        #TODO handle residual layers here
        if config.refine_neurons==True:
            resl, resu, indices = get_bounds_for_layer_with_milp(nn, nn.specLB, nn.specUB, predecessor_index, predecessor_index, length, nlb, nub, relu_groups, use_milp,  candidate_vars, timeout)
            nlb[predecessor_index] = resl
            nub[predecessor_index] = resu

        lbi = nlb[predecessor_index]
        ubi = nub[predecessor_index]

        if domain == 'deepzono':
            encode_activation_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi, relu_groups, False, 'refinezono', nn.layertypes[layerno])
            if config.refine_neurons==True:
                j = 0
                for i in range(length):
                    if((j < len(indices)) and (i==indices[j])):
                        element = relu_zono_refined(man,True, element,i+offset, resl[i],resu[i])
                        j=j+1
                    else:
                        element = relu_zono(man,True,element,i+offset)
                return element

            else:

                element = relu_zono_layerwise(man,True,element,offset, length, use_default_heuristic)
                return element
        else:
            if config.refine_neurons:
                for j in indices:
                    update_bounds_for_neuron(man,element,predecessor_index,j,resl[j],resu[j])
            encode_kactivation_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi,
                                    relu_groups, False, 'refinepoly', nn.layertypes[layerno])
            if nn.layertypes[layerno] == 'ReLU':
                handle_relu_layer(*self.get_arguments(man, element), use_default_heuristic)
            elif nn.layertypes[layerno] == 'Sigmoid':
                handle_sigmoid_layer(*self.get_arguments(man, element), use_default_heuristic)
            else:
                handle_tanh_layer(*self.get_arguments(man, element), use_default_heuristic)
