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


from gurobipy import *
from fconv import *
import numpy as np
from config import config
import multiprocessing
import math
import sys


def milp_callback(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if obj_bound > 0:
            model.terminate()


def handle_conv(model, var_list,start_counter, filters,biases,filter_size,input_shape, strides, out_shape, pad_top, pad_left, lbi, ubi, use_milp, is_nchw=False):

    num_out_neurons = np.prod(out_shape)
    num_in_neurons = np.prod(input_shape)#input_shape[0]*input_shape[1]*input_shape[2]
    #print("filters", filters.shape, filter_size, input_shape, strides, out_shape, pad_top, pad_left)
    start = len(var_list)
    for j in range(num_out_neurons):
        var_name = "x" + str(start+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub =ubi[j], name=var_name)
        var_list.append(var)

    #print("OUT SHAPE ", out_shape, input_shape, filter_size, filters.shape, biases.shape)
    if is_nchw:
        for out_z in range(out_shape[1]):
            for out_x in range(out_shape[2]):
                for out_y in range(out_shape[3]):
                
                    dst_ind = out_z*out_shape[2]*out_shape[3] + out_x*out_shape[3] + out_y
                    expr = LinExpr()
                    #print("dst ind ", dst_ind)
                    expr += -1*var_list[start+dst_ind]
                    
                    for inp_z in range(input_shape[0]):
                        for x_shift in range(filter_size[0]):
                            for y_shift in range(filter_size[1]):
                                x_val = out_x*strides[0]+x_shift-pad_top
                                y_val = out_y*strides[1]+y_shift-pad_left
                                if(y_val<0 or y_val >= input_shape[2]):
                                    continue
                                if(x_val<0 or x_val >= input_shape[1]):
                                    continue
                                mat_offset = x_val*input_shape[2] + y_val + inp_z*input_shape[1]*input_shape[2]
                                if(mat_offset>=num_in_neurons):
                                    continue 
                                src_ind = start_counter + mat_offset
                                #print("src ind ", mat_offset)
                                #filter_index = x_shift*filter_size[1]*input_shape[0]*out_shape[1] + y_shift*input_shape[0]*out_shape[1] + inp_z*out_shape[1] + out_z
                                expr.addTerms(filters[out_z][inp_z][x_shift][y_shift],var_list[src_ind])
                                                           
                    expr.addConstant(biases[out_z])
                    
                    model.addConstr(expr, GRB.EQUAL, 0)  
                      
    else:
        for out_x in range(out_shape[1]):
            for out_y in range(out_shape[2]):
                for out_z in range(out_shape[3]):
                    dst_ind = out_x*out_shape[2]*out_shape[3] + out_y*out_shape[3] + out_z
                    expr = LinExpr()
                    expr += -1*var_list[start+dst_ind]
                    for inp_z in range(input_shape[2]):
                        for x_shift in range(filter_size[0]):
                            for y_shift in range(filter_size[1]):
                                x_val = out_x*strides[0]+x_shift-pad_top
                                y_val = out_y*strides[1]+y_shift-pad_left

                                if(y_val<0 or y_val >= input_shape[1]):
                                    continue

                                if(x_val<0 or x_val >= input_shape[0]):
                                    continue

                                mat_offset = x_val*input_shape[1]*input_shape[2] + y_val*input_shape[2] + inp_z
                                if(mat_offset>=num_in_neurons):
                                    continue
                                src_ind = start_counter + mat_offset
                                #filter_index = x_shift*filter_size[1]*input_shape[2]*out_shape[3] + y_shift*input_shape[2]*out_shape[3] + inp_z*out_shape[3] + out_z
                             #expr.addTerms(filters[filter_index],var_list[src_ind])
                                expr.addTerms(filters[x_shift][y_shift][inp_z][out_z],var_list[src_ind])

                    expr.addConstant(biases[out_z])
                    model.addConstr(expr, GRB.EQUAL, 0)
    return start


def handle_maxpool(model, var_list, layerno, src_counter, pool_size, input_shape, strides, output_shape, pad_top, pad_left, lbi, ubi, lbi_prev, ubi_prev, use_milp):

    use_milp = use_milp and config.use_milp
    start = len(var_list)
    num_neurons = np.prod(input_shape)#input_shape[0]*input_shape[1]*input_shape[2]
    binary_counter = start
    maxpool_counter = start
    if(use_milp==1):
        maxpool_counter = start + num_neurons
        for j in range(num_neurons):
            var_name = "x" + str(start+j)
            var = model.addVar(vtype=GRB.BINARY, name=var_name)

            var_list.append(var)
    o1 = output_shape[1]
    o2 = output_shape[2]
    o3 = output_shape[3]
    output_size = o1*o2*o3
    i12 = input_shape[1]*input_shape[2]
    o12 = output_shape[2]*output_shape[3]
    #print("strides ", strides, pad_top, pad_left)
    for j in range(output_size):
        var_name = "x" + str(maxpool_counter+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb = lbi[j], ub=ubi[j],  name=var_name)
        var_list.append(var)

    for out_pos in range(output_size):
        out_x = int(out_pos / o12)
        out_y = int((out_pos-out_x*o12) / output_shape[3])
        out_z = int(out_pos-out_x*o12 - out_y*output_shape[3])
        inp_z = out_z
               
        max_u = float("-inf")
        max_l = float("-inf")
        sum_l = 0.0
        max_l_var = 0.0
        max_u_var = 0.0
        pool_map = []
        l = 0
        for x_shift in range(pool_size[0]):
            for y_shift in range(pool_size[1]):
                x_val = out_x*strides[0] + x_shift - pad_top
                if(x_val<0 or x_val>=input_shape[0]):
                    continue
                y_val = out_y*strides[1] + y_shift - pad_left
                if(y_val < 0 or y_val>=input_shape[1]):
                    continue
                pool_cur_dim = x_val*i12 + y_val*input_shape[2] + inp_z
                if pool_cur_dim >= num_neurons:
                    
                    continue    
                pool_map.append(pool_cur_dim)
                lb = lbi_prev[pool_cur_dim] 
                ub = ubi_prev[pool_cur_dim]
                sum_l = sum_l + lb       
                if ub>max_u:
                    max_u = ub
                    max_u_var = pool_cur_dim
                if lb > max_l:   
                    max_l = lb
                    max_l_var = pool_cur_dim
                l = l + 1     
        dst_index = maxpool_counter+out_pos
                
        if use_milp==1:
            binary_expr = LinExpr()
            for l in range(len(pool_map)):
                src_index = pool_map[l]
                src_var = src_index + src_counter
                binary_var = src_index + binary_counter
                if(ubi_prev[src_index]<max_l):
                    continue

                # y >= x

                expr = var_list[dst_index] -  var_list[src_var]
                model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                # y <= x + (1-a)*(u_{rest}-l)
                max_u_rest = float("-inf")
                for j in range(len(pool_map)):
                    if j==l:
                        continue
                    if(sup[j]>max_u_rest):
                        max_u_rest = sup[j]

                cst = max_u_rest-inf[l]

                expr = var_list[dst_index] - var_list[src_var] + cst*var_list[binary_var]
                model.addConstr(expr, GRB.LESS_EQUAL, cst)

	        # indicator constraints
                model.addGenConstrIndicator(var_list[binary_var], True, var_list[dst_index]-var_list[src_var], GRB.EQUAL, 0.0)

                binary_expr+=var_list[binary_var]

            model.addConstr(binary_expr, GRB.EQUAL, 1)

        else:
            flag = True
            for l in range(len(pool_map)):
                if pool_map[l] == max_l_var:
                    continue
                ub = ubi_prev[pool_map[l]]
                if ub >= max_l:
                   
                   flag = False
                   break
            if flag==True:
                src_var = max_l_var + src_counter
                expr = var_list[dst_index] - var_list[src_var]
                model.addConstr(expr, GRB.EQUAL, 0)
            else:
                add_expr = LinExpr()
                add_expr+=-1*var_list[dst_index]
                for l in range(len(pool_map)):
                    src_index = pool_map[l]
                    src_var = src_index + src_counter
                    # y >= x
                    expr = var_list[dst_index] - var_list[src_var]
                    model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                    add_expr+=var_list[src_var]
                model.addConstr(add_expr, GRB.GREATER_EQUAL, sum_l - max_l)


    return maxpool_counter


def handle_affine(model,var_list,counter,weights,biases, lbi, ubi):
    num_neurons_affine = len(weights)
    start = len(var_list)

    # output of matmult
    for j in range(num_neurons_affine):
        var_name = "x" + str(start+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub =ubi[j], name=var_name)
        var_list.append(var)

    for j in range(num_neurons_affine):
        num_in_neurons = len(weights[j])

        expr = LinExpr()
        expr += -1*var_list[start+j]
        # matmult constraints
        for k in range(num_in_neurons):
            expr.addTerms(weights[j][k],var_list[counter+k])
        expr.addConstant(biases[j])
        model.addConstr(expr, GRB.EQUAL, 0)
    return start


def handle_residual(model, var_list, branch1_counter, branch2_counter, lbi, ubi):
    num_neurons_affine = len(lbi)
    start = len(var_list)

    # output of matmult
    for j in range(num_neurons_affine):
        var_name = "x" + str(start + j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub=ubi[j], name=var_name)
        var_list.append(var)

    for j in range(num_neurons_affine):
        # num_in_neurons = len(weights[j])

        expr = LinExpr()
        expr += -1 * var_list[start + j]
        # matmult constraints
        # for k in range(num_in_neurons):
        expr += var_list[branch1_counter + j]
        expr += var_list[branch2_counter + j]
        expr.addConstant(0)
        model.addConstr(expr, GRB.EQUAL, 0)
    return start


def _add_kactivation_constraints(model, var_list, constraint_groups, x_counter, y_counter):
    #print("start here ")
    for inst in constraint_groups:
        for row in inst.cons:
            k = len(inst.varsid)
            expr = LinExpr()
            expr.addConstant(row[0])
            for i, x in enumerate(inst.varsid):
                expr.addTerms(row[1 + i], var_list[x_counter + x])
                expr.addTerms(row[1 + k + i], var_list[y_counter + x])
            #print("row ", row)
            model.addConstr(expr >= 0)


def handle_relu(model,var_list, affine_counter, num_neurons, lbi, ubi, relu_groupsi, use_milp):
    use_milp = use_milp and config.use_milp
    #print("relu groups ")
    start= len(var_list)
    binary_counter = start
    relu_counter = start
    #print("neurons ", num_neurons)
    if(use_milp==1):
    #if num_neurons <= 1000:
        #indicator variables
        relu_counter = start + num_neurons
        for j in range(num_neurons):
           var_name = "x" + str(start+j)
           var = model.addVar(vtype=GRB.BINARY, name=var_name)
           var_list.append(var)

    # relu variables
    for j in range(num_neurons):
        var_name = "x" + str(relu_counter+j)
        upper_bound = max(0,ubi[j])
        var = model.addVar(vtype=GRB.CONTINUOUS, lb = 0.0, ub=upper_bound,  name=var_name)
        var_list.append(var)


    if(use_milp==1):
        #print("MILP here")
    #if num_neurons <= 1000:
        for j in range(num_neurons):
            if(ubi[j]<=0):
               expr = var_list[relu_counter+j]
               model.addConstr(expr, GRB.EQUAL, 0)
            elif(lbi[j]>=0):
               expr = var_list[relu_counter+j] - var_list[affine_counter+j]
               model.addConstr(expr, GRB.EQUAL, 0)
            else:
               # y <= x - l(1-a)
               expr = var_list[relu_counter+j] - var_list[affine_counter+j] - lbi[j]*var_list[binary_counter+j]
               model.addConstr(expr, GRB.LESS_EQUAL, -lbi[j])

               # y >= x
               expr = var_list[relu_counter+j] -  var_list[affine_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)

               # y <= u.a
               expr = var_list[relu_counter+j] - ubi[j]*var_list[binary_counter+j]
               model.addConstr(expr, GRB.LESS_EQUAL, 0)

               # y >= 0
               expr = var_list[relu_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)

               # indicator constraint
               model.addGenConstrIndicator(var_list[binary_counter+j], True, var_list[affine_counter+j], GRB.GREATER_EQUAL, 0.0)
    if len(relu_groupsi)>0:
        if use_milp==0:
            for j in range(num_neurons):
                if ubi[j] <= 0:
                    expr = var_list[relu_counter+j]
                    model.addConstr(expr, GRB.EQUAL, 0)
                elif lbi[j] >= 0:
                    expr = var_list[relu_counter+j] - var_list[affine_counter+j]
                    model.addConstr(expr, GRB.EQUAL, 0)
        _add_kactivation_constraints(model, var_list, relu_groupsi, affine_counter, relu_counter)

    return relu_counter



def handle_sign(model,var_list, affine_counter, num_neurons, lbi, ubi):
    start= len(var_list)
    binary_counter = start
    sign_counter = start + num_neurons
    for j in range(num_neurons):
        var_name = "x" + str(start+j)
        var = model.addVar(vtype=GRB.BINARY, name=var_name)
        var_list.append(var)

    # sign variables
    for j in range(num_neurons):
        var_name = "x" + str(sign_counter+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb = 0.0, ub=1.0,  name=var_name)
        var_list.append(var)


    for j in range(num_neurons):
        if(ubi[j]<=0):
            expr = var_list[sign_counter+j]
            model.addConstr(expr, GRB.EQUAL, 0)
        elif(lbi[j]>=0):
            expr = var_list[sign_counter+j] - var_list[affine_counter+j]
            model.addConstr(expr, GRB.EQUAL, 1)
        else:
            # x >= l(1-a)
            expr = - var_list[affine_counter+j] - lbi[j]*var_list[binary_counter+j]
            model.addConstr(expr, GRB.LESS_EQUAL, -lbi[j])


            # x <= u.a
            expr = var_list[affine_counter+j] - ubi[j]*var_list[binary_counter+j]
            model.addConstr(expr, GRB.LESS_EQUAL, 0)

            # y = a
            expr = var_list[sign_counter+j]
            model.addConstr(expr, GRB.GREATER_EQUAL, var_list[binary_counter+j])

            # indicator constraint
            model.addGenConstrIndicator(var_list[binary_counter+j], True, var_list[affine_counter+j], GRB.GREATER_EQUAL, 0.0)

    return sign_counter

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def handle_tanh_sigmoid(model, var_list, affine_counter, num_neurons, lbi, ubi,
                        constraint_groups, activation_type):
    assert activation_type in ["Tanh", "Sigmoid"]
    y_counter = len(var_list)

    for j in range(num_neurons):
        var_name = "x" + str(y_counter + j)
        x_lb = lbi[j]
        x_ub = ubi[j]
        if activation_type == "Tanh":
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=math.tanh(x_lb), ub=math.tanh(x_ub), name=var_name)
        else:
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=sigmoid(x_lb), ub=sigmoid(x_ub), name=var_name)
        var_list.append(var)

    _add_kactivation_constraints(model, var_list, constraint_groups, affine_counter, y_counter)

    return y_counter


def create_model(nn, LB_N0, UB_N0, nlb, nub, relu_groups, numlayer, use_milp, is_nchw=False):
    use_milp = use_milp and config.use_milp

    model = Model("milp")

    model.setParam("OutputFlag",0)
    model.setParam(GRB.Param.FeasibilityTol, 1e-5)

    num_pixels = len(LB_N0)
    #output_counter = num_pixels
    ffn_counter = nn.ffn_counter
    conv_counter = nn.conv_counter
    residual_counter = nn.residual_counter
    pool_counter = nn.pool_counter
    activation_counter = nn.activation_counter
    
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.residual_couter = 0
    nn.pool_counter = 0
    nn.activation_counter = 0
    var_list = []
    counter = 0
    # TODO zonotope
    if len(UB_N0)==0:
        num_pixels = nn.zonotope.shape[0]
        num_error_terms = nn.zonotope.shape[1]
        for j in range(num_error_terms-1):
            var_name = "x" + str(j)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb = -1, ub=1, name=var_name)
            var_list.append(var)
        counter = num_error_terms-1
        for i in range(num_pixels):
            lower_bound = nn.zonotope[i][0]
            upper_bound = lower_bound
            for j in range(1,num_error_terms):
                lower_bound = lower_bound - abs(nn.zonotope[i][j])
                upper_bound = upper_bound + abs(nn.zonotope[i][j])
            var_name = "x" + str(counter+i)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb = lower_bound, ub=upper_bound, name=var_name)
            var_list.append(var)
            expr = LinExpr()
            expr += -1 * var_list[counter + i]
            for j in range(num_error_terms-1):
                expr.addTerms(nn.zonotope[i][j+1],var_list[j])

            expr.addConstant(nn.zonotope[i][0])
            model.addConstr(expr, GRB.EQUAL, 0)

    else:
        for i in range(num_pixels):
            var_name = "x" + str(i)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb = LB_N0[i], ub=UB_N0[i], name=var_name)
            var_list.append(var)

    #for i in range(numlayer):
    #    if(nn.layertypes[i]=='SkipNet1'):
    #        start = i+1
            #break
    #    elif(nn.layertypes[i]=='SkipNet2'):
    #        start = i+1
            #break

    #for i in range(start):
    #    if(nn.layertypes[i] in ['ReLU','Affine']):
    #        nn.ffn_counter+=1
    #    elif(nn.layertypes[i]=='Conv2D'):
    #        nn.conv_counter+=1
    #    elif(nn.layertypes[i]=='MaxPooling2D'):
    #        nn.maxpool_counter+=1

    start_counter = []
    start_counter.append(counter)
    for i in range(numlayer):
        #count = nn.ffn_counter + nn.conv_counter
        if(nn.layertypes[i] in ['SkipCat']):
            continue
        elif nn.layertypes[i] in ['FC']:
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter+nn.conv_counter]
            index = nn.predecessors[i+1][0]
            #print("index ", index, start_counter,i, len(relu_groups))
            counter = start_counter[index]
            counter = handle_affine(model,var_list,counter,weights,biases,nlb[i],nub[i])
            nn.ffn_counter+=1
            start_counter.append(counter)

        elif(nn.layertypes[i]=='ReLU'):
            index = nn.predecessors[i+1][0]
            if relu_groups is None:
                counter = handle_relu(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1], [], use_milp)
            else:
                counter = handle_relu(model,var_list, counter,len(nlb[i]),nlb[index-1],nub[index-1], relu_groups[nn.activation_counter], use_milp)
            nn.activation_counter += 1
            start_counter.append(counter)

        elif nn.layertypes[i] == 'Sigmoid' or nn.layertypes[i] == 'Tanh':
            index = nn.predecessors[i+1][0]
            if relu_groups is None:
                counter = handle_tanh_sigmoid(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1],
                                          [], nn.layertypes[i])
            else:
                counter = handle_tanh_sigmoid(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1],
                                          relu_groups[nn.activation_counter], nn.layertypes[i])

            nn.activation_counter += 1
            start_counter.append(counter)
            
        elif nn.layertypes[i] == 'Sign':
            index = nn.predecessors[i+1][0]
            counter = handle_sign(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1])
            nn.activation_counter += 1
            start_counter.append(counter)

        elif nn.layertypes[i] in ['Conv']:
            filters = nn.filters[nn.conv_counter]
            biases = nn.biases[nn.ffn_counter+nn.conv_counter]
            filter_size = nn.filter_size[nn.conv_counter]
            numfilters = nn.numfilters[nn.conv_counter]
            out_shape = nn.out_shapes[nn.conv_counter + nn.pool_counter]
            padding = nn.padding[nn.conv_counter + nn.pool_counter]
            strides = nn.strides[nn.conv_counter + nn.pool_counter]
            input_shape = nn.input_shape[nn.conv_counter +nn.pool_counter]
            num_neurons = np.prod(out_shape)

            index = nn.predecessors[i+1][0]
            counter = start_counter[index]
            counter = handle_conv(model, var_list, counter, filters, biases, filter_size, input_shape, strides, out_shape, padding[0], padding[1], nlb[i],nub[i],use_milp, is_nchw=is_nchw)

            start_counter.append(counter)

            nn.conv_counter+=1


        elif(nn.layertypes[i]=='Maxpool'):
            pool_size = nn.pool_size[nn.pool_counter]
            input_shape = nn.input_shape[nn.conv_counter + nn.pool_counter]
            out_shape = nn.out_shapes[nn.conv_counter + nn.pool_counter]
            padding = nn.padding[nn.conv_counter + nn.pool_counter]
            strides = nn.strides[nn.conv_counter + nn.pool_counter]
            index = nn.predecessors[i+1][0]
            counter = start_counter[index]
            counter = handle_maxpool(model,var_list,i,counter,pool_size, input_shape, strides, out_shape, padding[0], padding[1], nlb[i],nub[i], nlb[i-1], nub[i-1],use_milp)
            start_counter.append(counter)
            nn.pool_counter+=1

        elif nn.layertypes[i] in ['Resadd']:
            index1 = nn.predecessors[i+1][0]
            index2 = nn.predecessors[i+1][1]
            counter1 = start_counter[index1]
            counter2 = start_counter[index2]
            counter = handle_residual(model,var_list,counter1,counter2,nlb[i],nub[i])
            start_counter.append(counter)
            nn.residual_counter +=1


        else:
            assert 0, 'layertype:' + nn.layertypes[i] + 'not supported for refine'
    nn.ffn_counter = ffn_counter
    nn.conv_counter = conv_counter
    nn.residual_counter = residual_counter
    nn.pool_counter = pool_counter
    nn.activation_counter = activation_counter
    #model.write("model_refinepoly.lp")
    np.set_printoptions(threshold=sys.maxsize)
    #print("NLB ", nlb[-4], len(nlb[-4]))
    return counter, var_list, model


class Cache:
    model = None
    output_counter = None
    lbi = None
    ubi = None


def solver_call(ind):
    model = Cache.model.copy()
    runtime = 0

    obj = LinExpr()
    obj += model.getVars()[Cache.output_counter+ind]
    #print (f"{ind} {model.getVars()[Cache.output_counter+ind].VarName}")

    model.setObjective(obj, GRB.MINIMIZE)
    model.reset()
    model.optimize()
    runtime += model.RunTime
    soll = Cache.lbi[ind] if model.SolCount==0 else model.objbound
    #print (f"{ind} {model.status} lb ({Cache.lbi[ind]}, {soll}) {model.RunTime}s")
    sys.stdout.flush()

    model.setObjective(obj, GRB.MAXIMIZE)
    model.reset()
    model.optimize()
    runtime += model.RunTime
    solu = Cache.ubi[ind] if model.SolCount==0 else model.objbound
    #print (f"{ind} {model.status} ub ({Cache.ubi[ind]}, {solu}) {model.RunTime}s")
    sys.stdout.flush()

    soll = max(soll, Cache.lbi[ind])
    solu = min(solu, Cache.ubi[ind])

    addtoindices = (soll > Cache.lbi[ind]) or (solu < Cache.ubi[ind])

    return soll, solu, addtoindices, runtime


def get_bounds_for_layer_with_milp(nn, LB_N0, UB_N0, layerno, abs_layer_count, output_size, nlb, nub, relu_groups, use_milp, candidate_vars, timeout):
    lbi = nlb[abs_layer_count]
    ubi = nub[abs_layer_count]
    #numlayer = nn.numlayer

    candidate_length = len(candidate_vars)
    widths = np.zeros(candidate_length)
    #avg_weight = np.zeros(candidate_length)
    #next_layer = nn.calc_layerno() + 1

    # HEURISTIC 2
    # in case of relu, the gradients are wrt to neurons after relu
    #logits_diff = np.asarray([nn.logits[config.label]-l for l in nn.logits])
    # subtract c-th column from each column, where c is correct class label
    #grad_diff = nn.grads[layerno][:,config.label][:, np.newaxis] - nn.grads[layerno]
    #deltas = np.nanmin(np.abs(logits_diff[np.newaxis, :] / grad_diff), axis=1)
    widths = [ubi[j]-lbi[j] for j in range(len(lbi))]

    candidate_vars = sorted(candidate_vars, key=lambda k: widths[k])
    counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, relu_groups, layerno+1, use_milp)
    resl = [0]*len(lbi)
    resu = [0]*len(ubi)
    indices = []

    num_candidates = len(candidate_vars)
    if nn.layertypes[layerno] == 'Conv2D':
        num_candidates = num_candidates
    else:
        if(abs_layer_count<=3):
            #num_candidates = int(len(candidate_vars)/math.pow(5,abs_layer_count-1))
            num_candidates = len(candidate_vars)
        else:
            #int(len(candidate_vars)/math.pow(2,abs_layer_count-4))
            num_candidates = len(candidate_vars)
            #num_candidates = num_candidates

    #print("Refine layer ", abs_layer_count, nn.layertypes[abs_layer_count])
    neuron_map = [0]*len(lbi)

    model.setParam(GRB.Param.TimeLimit, timeout)
    model.setParam(GRB.Param.Threads, 2)
    output_counter = counter

    model.update()
    model.reset()

    NUMPROCESSES = config.numproc
    Cache.model = model
    Cache.output_counter = output_counter
    Cache.lbi = lbi
    Cache.ubi = ubi

    refined = [False]*len(lbi)

    var_idxs = candidate_vars[:num_candidates]
    for v in var_idxs:
        refined[v] = True
        #print (f"{v} {deltas[v]} {widths[v]} {deltas[v]/widths[v]}")
    with multiprocessing.Pool(NUMPROCESSES) as pool:
       solver_result = pool.map(solver_call, var_idxs)
    solvetime = 0
    for (l, u, addtoindices, runtime), ind in zip(solver_result, var_idxs):
        resl[ind] = l
        resu[ind] = u
        if addtoindices:
            indices.append(ind)
        solvetime += runtime

    avg_solvetime = (solvetime+1)/(2*num_candidates+1)

    model.setParam('TimeLimit', avg_solvetime/2)
    model.update()
    model.reset()

    var_idxs = candidate_vars[num_candidates:]
    if nn.layertypes[layerno] == 'Conv2D':
        if len(var_idxs) >= num_candidates//3:
            var_idxs = var_idxs[:(num_candidates//3)]
    else:
        if len(var_idxs) >= 50:
            var_idxs = var_idxs[:50]
    for v in var_idxs:
        refined[v] = True
        #print (f"{v} {deltas[v]} {widths[v]} {deltas[v]/widths[v]}")

    with multiprocessing.Pool(NUMPROCESSES) as pool:
       solver_result = pool.map(solver_call, var_idxs)
    solvetime = 0
    for (l, u, addtoindices, runtime), ind in zip(solver_result, var_idxs):
        resl[ind] = l
        resu[ind] = u
        if addtoindices:
            indices.append(ind)
        solvetime += runtime

    for i, flag in enumerate(refined):
        if not flag:
            resl[i] = lbi[i]
            resu[i] = ubi[i]

    avg_solvetime = solvetime/(2*len(var_idxs)) if len(var_idxs) else 0.0

    for i in range(abs_layer_count):
        for j in range(len(nlb[i])):
            if(nlb[i][j]>nub[i][j]):
                print("fp unsoundness detected ", nlb[i][j],nub[i][j],i,j)
    for j in range(len(resl)):
        if (resl[j]>resu[j]):
            print (f"unsound {j}")
            resl[j], resu[j] = lbi[j], ubi[j]

    return resl, resu, sorted(indices)


def add_spatial_constraints(model, spatial_constraints, var_list, input_size):

    delta = spatial_constraints.get('delta')
    gamma = spatial_constraints.get('gamma')
    channels = spatial_constraints.get('channels')

    lower_planes = spatial_constraints.get('lower_planes')
    upper_planes = spatial_constraints.get('upper_planes')

    add_norm_constraints = spatial_constraints.get('add_norm_constraints')
    neighboring_indices = spatial_constraints.get('neighboring_indices')

    vector_field = list()

    for idx in range(input_size // channels):
        vx = model.addVar(lb=-delta, ub=delta)
        vy = model.addVar(lb=-delta, ub=delta)
        add_norm_constraints(model, vx, vy)
        vector_field.append({'vx': vx, 'vy': vy})

    if (lower_planes is not None) and (upper_planes is not None):

        for idx, vector in enumerate(vector_field):
            for channel in range(channels):
                index = channels * idx + channel
                var = var_list[index]
                lb_a, ub_a = lower_planes[0][index], upper_planes[0][index]
                lb_b, ub_b = lower_planes[1][index], upper_planes[1][index]
                lb_c, ub_c = lower_planes[2][index], upper_planes[2][index]

                model.addConstr(
                    var >= lb_a + lb_b * vector['vx'] + lb_c * vector['vy']
                )
                model.addConstr(
                    var <= ub_a + ub_b * vector['vx'] + ub_c * vector['vy']
                )

    if (gamma is not None) and (gamma < float('inf')):

        indices = neighboring_indices['indices'][::channels] // channels
        neighbors = neighboring_indices['neighbors'][::channels] // channels

        for idx, nbr in zip(indices.tolist(), neighbors.tolist()):
            model.addConstr(
                vector_field[idx]['vx'] - vector_field[nbr]['vx'] <= gamma
            )
            model.addConstr(
                vector_field[nbr]['vx'] - vector_field[idx]['vx'] <= gamma
            )
            model.addConstr(
                vector_field[idx]['vy'] - vector_field[nbr]['vy'] <= gamma
            )
            model.addConstr(
                vector_field[nbr]['vy'] - vector_field[idx]['vy'] <= gamma
            )

def verify_network_with_milp(nn, LB_N0, UB_N0, nlb, nub, constraints, spatial_constraints=None):
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.residual_counter = 0
    nn.maxpool_counter = 0
    numlayer = nn.numlayer
    input_size = len(LB_N0)
    counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, None, numlayer, True)
    #print("timeout ", config.timeout_milp)
    model.setParam(GRB.Param.TimeLimit, config.timeout_complete)
    
    if spatial_constraints is not None:
        add_spatial_constraints(
            model, spatial_constraints, var_list, input_size
        )
                
    adv_examples = []
    non_adv_examples = []
    for or_list in constraints:
        or_result = False
        for (i, j, k) in or_list:
            obj = LinExpr()
            if j== -1:
                obj += float(k)- 1*var_list[counter + i]
                model.setObjective(obj,GRB.MINIMIZE)
                model.optimize(milp_callback)
                #status.append(model.SolCount>0)
                if model.objbound > 0:
                    or_result = True
                    #print("objbound ", model.objbound)
                    if model.solcount > 0:
                        non_adv_examples.append(model.x[0:input_size])
                    break
                elif model.solcount > 0:
                    adv_examples.append(model.x[0:input_size])

            else:
                if i!=j:
                    obj += 1*var_list[counter + i]
                    obj += -1*var_list[counter + j]
                    model.setObjective(obj,GRB.MINIMIZE)
                    model.optimize(milp_callback)
                    #status.append(model.solcount>0)
                    #print("status ", model.status, model.objbound)                    
                    if model.objbound > 0:
                        or_result = True
                        #print("objbound ", model.objbound)
                        if model.solcount > 0:
                            non_adv_examples.append(model.x[0:input_size])
                        break
                    elif model.solcount > 0:
                        adv_examples.append(model.x[0:input_size])
        if not or_result:
            if len(adv_examples) > 0:
                return False, adv_examples
            else:
                return False, None
    if len(non_adv_examples) > 0:
        return True, non_adv_examples
    else:
        return True, None

