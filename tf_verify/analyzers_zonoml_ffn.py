import sys
sys.path.insert(0, '../ELINA/python_interface/')


import numpy as np
import re
import csv
from zonoml import *
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
# from zonoml_milp import *
from gurobipy import *
import time


from ctypes import *
from ctypes.util import *   


libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')

# class layers:
#     def __init__(self):
#         self.layertypes = []
#         self.weights = []
#         self.biases = []
#         self.filters = []
#         self.numfilters = []
#         self.filter_size = [] 
#         self.input_shape = []
#         self.strides = []
#         self.padding = []
#         self.pool_size = []
#         self.numlayer = 0
#         self.ffn_counter = 0
#         self.conv_counter = 0
#         self.maxpool_counter = 0
#         self.maxpool_lb = []
#         self.maxpool_ub = []

def runRepl(arg, repl):
    for a in repl:
        arg = arg.replace(a+"=", "'"+a+"':")
    return eval("{"+arg+"}")


def extract_mean(text):
    mean = ''
    m = re.search('mean=\[(.+?)\]', text)
    
    if m:
        means = m.group(1)
    mean_str = means.split(',')
    num_means = len(mean_str)
    mean_array = np.zeros(num_means)
    for i in range(num_means):
         mean_array[i] = np.float64(mean_str[i])
    return mean_array

def extract_std(text):
    std = ''
    m = re.search('std=\[(.+?)\]', text)
    if m:
        stds = m.group(1)
    std_str =stds.split(',')
    num_std = len(std_str)
    std_array = np.zeros(num_std)
    for i in range(num_std):
        std_array[i] = np.float64(std_str[i])
    return std_array


def parse_conv_param(text):
    
    line = text
    
    args = None
    start = 0
    if("ReLU" in line):
        start = 5
    elif("Sigmoid" in line):
        start = 8
    elif("Tanh" in line):
        start = 5
    
    if 'padding' in line:
         args =  runRepl(line[start:], ["filters", "input_shape", "kernel_size", "stride", "padding"])
    else:
         args = runRepl(line[start:], ["filters", "input_shape", "kernel_size"])
        
        
    if("padding" in line):
        if(args["padding"]==1):
            padding_arg = False
        else:
            padding_arg = True
    else:
        padding_arg = False

    if("stride" in line):
        stride_arg = args["stride"] 
    else:
        stride_arg = [1,1]
               
    return args["filters"], args["kernel_size"], args["input_shape"], stride_arg, padding_arg

def parse_maxpool_param(text):
    pool_index = text.index("pool_size")
    pool_size_str = text[pool_index+11:pool_index+15]
    pool_size = pool_size_str.split(",")
    pool_size.append('1')
    pool_size = list(map(int, pool_size))
    #print(pool_size)
    #y = np.array(pool_size)
    #y = y.astype(np.uintp)
    input_shape_str = text[text.index("input_shape")+13:len(text)-1]
    input_shape = input_shape_str.split(",")
    #x = np.array(input_shape)
    #x = x.astype(np.uintp)
    input_shape = list(map(int, input_shape))
    return pool_size, input_shape




def parse_filter(text):
    line = text.replace('[', '')
    line = line.replace(']', '')
    line = line.split(',')
    x = np.array(line)
    x = x.astype(np.double)
    return x

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    v = parse_bias(text)
    return v.reshape((v.size,1))
    #return v



def parse_matrix(text):
    text = text.lstrip()
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))]) 

def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def transpose(W, h,w,c):
    #print("W",W)
    col = h*w*c
    rows = W.shape[0]
    res = np.zeros((rows,col))
    for i in range(rows):
        for j in range(c):
            for k in range(h):
                for l in range(w):
                    res[i][k*w*c + l*c + j] = W[i][j*h*w + k*w + l]
    #print("res", res)
    return res

def parse_net(net_file):
    net = open(net_file,'r')
    text = net.read()
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    needs_normalization = False
    mean = 0 
    std = 0
    last_layer = None
    h,w,c = None, None, None
    is_conv = False
    
    while i < len(lines):
        if 'Normalize' in lines[i]:
            mean = extract_mean(lines[i])
            std = extract_std(lines[i])
            print("mean ", mean)
            print("std ", std)
            needs_normalization = True
            last_layer = "Normalize"
            i+= 1
        elif 'SkipNet1' in lines[i]:
            last_layer = "SkipNet1"
            res.layertypes.append(lines[i])
            res.numlayer+=1
            i+=1
        elif 'SkipNet2' in lines[i]:
            last_layer = "SkipNet2"
            res.layertypes.append(lines[i])
            res.numlayer+=1
            i+=1
        elif 'SkipCat' in lines[i]:
            last_layer = "SkipCat"
            res.layertypes.append('SkipCat')
            res.numlayer+=1
            i+=1
        elif(lines[i]=='Conv2D'):
            last_layer = "Conv2D"
            nf, fs, inp_s, strides, padding = parse_conv_param(lines[i+1])
            res.numfilters.append(nf)
            res.filter_size.append(fs)
            res.input_shape.append(inp_s)
            res.strides.append(strides)
            res.padding.append(padding)
            res.filters.append(parse_filter(lines[i+2]))
            b = parse_bias(lines[i+3])
            res.biases.append(b)
            res.layertypes.append(lines[i])
            is_conv = True
            res.numlayer+=1
            c = nf
            if(padding==True):
                h = int(np.ceil((inp_s[0] - fs[0]+1) / strides[0]))
                w = int(np.ceil((inp_s[1] - fs[1]+1) / strides[1]))
            else:
                h = int(np.ceil(inp_s[0]/ strides[0]))
                w = int(np.ceil(inp_s[1]/ strides[1]))
            i += 4
        #elif lines[i]=='MaxPooling2D':
        #    ps, inp_s = parse_maxpool_param(lines[i+1])
        #    res.pool_size.append(ps)
        #    res.input_shape.append(inp_s)
        #    res.layertypes.append(lines[i])
        #    res.numlayer+=1
        #    i += 2
        elif lines[i] in ['ReLU', 'Affine', 'Sigmoid', 'Tanh']:
            if last_layer == "Conv2D" and needs_normalization:
                W = transpose(parse_matrix(lines[i+1]),h,w,c)
                #print("W", W.shape)
                #print("h ", h, "w ", w, "c ", c)
            else:
                W = parse_matrix(lines[i+1])
            last_layer = lines[i]
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            print("curr line ",lines[i])
            raise Exception('parse error: '+ lines[i])
        #print("last layer ", last_layer)
    return res, is_conv, needs_normalization, mean, std

   
def handle_affine_ai2(man,element,weights, biases,nlb,nub):
    
    dims = elina_abstract0_dimension(man,element)
    num_in_pixels = dims.intdim + dims.realdim
    num_out_pixels = len(weights)
    #print("in pixels ",num_in_pixels," out pixels ",num_out_pixels)
    dimadd = elina_dimchange_alloc(0,num_out_pixels)    
    for i in range(num_out_pixels):
        dimadd.contents.dim[i] = num_in_pixels
    elina_abstract0_add_dimensions(man, True, element, dimadd, False)
    elina_dimchange_free(dimadd)
    np.ascontiguousarray(weights, dtype=np.double)
    xpp = (weights.__array_interface__['data'][0]+ np.arange(weights.shape[0])*weights.strides[0]).astype(np.uintp) 
        
    np.ascontiguousarray(biases, dtype=np.double)
    
    element = ffn_matmult_zono(man, True, element, num_in_pixels, xpp, biases, num_out_pixels, 0, num_in_pixels)
    
    dimrem = elina_dimchange_alloc(0,num_in_pixels)
    for i in range(num_in_pixels):
        dimrem.contents.dim[i] = i
    elina_abstract0_remove_dimensions(man, True, element, dimrem)
    elina_dimchange_free(dimrem)
    bounds = elina_abstract0_to_box(man,element)
    lbi = []
    ubi = []
    #print('layerno ',layerno)
    for i in range(num_out_pixels):
        inf = bounds[i].contents.inf
        sup = bounds[i].contents.sup
        #print('i ',i)
        #elina_interval_fprint(cstdout,bounds[i])
        #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
        lbi.append(inf.contents.val.dbl)
        ubi.append(sup.contents.val.dbl)
    
    nlb.append(lbi)
    nub.append(ubi) 

    elina_interval_array_free(bounds,num_out_pixels) 
   
    return element


def handle_conv_ai2(man,element,filters,biases,filter_size, numfilters,input_shape,nlb,nub, strides, padding):
    #print("padding ", padding)
    dims = elina_abstract0_dimension(man,element)
    num_in_pixels = dims.intdim + dims.realdim
    if(padding==True):
       o1 = int(np.ceil((input_shape[0] - filter_size[0]+1)/strides[0]))
       o2 = int(np.ceil((input_shape[1] - filter_size[1]+1)/strides[1]))
    else:
       o1 = int(np.ceil(input_shape[0] / strides[0]))
       o2 = int(np.ceil(input_shape[1] / strides[1]))
    o3 = numfilters    
    num_out_pixels = o1*o2*o3
    dimadd = elina_dimchange_alloc(0,num_out_pixels)
    for i in range(num_out_pixels):
        dimadd.contents.dim[i] = num_in_pixels
    elina_abstract0_add_dimensions(man, True, element, dimadd, False)
    elina_dimchange_free(dimadd)
    np.ascontiguousarray(biases, dtype=np.double)
    
    element = conv_matmult_zono(man, True, element, num_in_pixels, filters, biases, input_shape, 0, filter_size, numfilters, strides, padding, True)
     
    dimrem = elina_dimchange_alloc(0,num_in_pixels)
    for i in range(num_in_pixels):
        dimrem.contents.dim[i] = i
    elina_abstract0_remove_dimensions(man, True, element, dimrem)
    elina_dimchange_free(dimrem)
    bounds = elina_abstract0_to_box(man,element)
    lbi = []
    ubi = []
    
    #print('layerno ',layerno)
    for i in range(num_out_pixels):
        inf = bounds[i].contents.inf
        sup = bounds[i].contents.sup
        
        #print('i ',i)
        #elina_interval_fprint(cstdout,bounds[i])
        #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
        lbi.append(inf.contents.val.dbl)
        ubi.append(sup.contents.val.dbl)
    
    nlb.append(lbi)
    nub.append(ubi) 
    
    elina_interval_array_free(bounds,num_out_pixels)
    return element

def handle_maxpool_ai2(man,element,input_shape, pool_size, nlb, nub):
    dims = elina_abstract0_dimension(man,element)
    num_in_pixels = dims.intdim + dims.realdim
    o1 = int(input_shape[0]/pool_size[0])
    o2 = int(input_shape[1]/pool_size[1])
    o3 = int(input_shape[2]/pool_size[2])
    num_out_pixels = o1*o2*o3
    tmp2 = [2,2]
    maxpool_lb = []
    maxpool_ub = []
    strides = (ctypes.c_size_t * len(tmp2))(*tmp2) 
    bounds = elina_abstract0_to_box(man,element)
    for i in range(num_in_pixels):
        inf = bounds[i].contents.inf
        sup = bounds[i].contents.sup
        #print('i ',i)
        #elina_interval_fprint(cstdout,bounds[i])
        #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
        maxpool_lb.append(inf.contents.val.dbl)
        maxpool_ub.append(sup.contents.val.dbl)
    elina_interval_array_free(bounds,num_in_pixels)   
    element = maxpool_zono(man, True, element, pool_size, input_shape, 0, strides, 3, num_in_pixels, True)
    dims = elina_abstract0_dimension(man,element)
    #print('maxpool output ',dims.intdim + dims.realdim)
    dimrem = elina_dimchange_alloc(0,num_in_pixels)
    for i in range(num_in_pixels):
        dimrem.contents.dim[i] = i
    elina_abstract0_remove_dimensions(man, True, element, dimrem)
    elina_dimchange_free(dimrem)
    bounds = elina_abstract0_to_box(man,element)
    lbi = []
    ubi = []
    #print('layerno ',layerno)
    for i in range(num_out_pixels):
        inf = bounds[i].contents.inf
        sup = bounds[i].contents.sup
        #print('i ',i)
        #elina_interval_fprint(cstdout,bounds[i])
        #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
        lbi.append(inf.contents.val.dbl)
        ubi.append(sup.contents.val.dbl)
    
    nlb.append(lbi)
    nub.append(ubi) 

    elina_interval_array_free(bounds,num_out_pixels)
    return element, maxpool_lb, maxpool_ub


def analyze_layer(nn,layerno, man,element, element1, nlb, nub, label, use_label):
    #print('counter ',nn.ffn_counter,' numlayer ', nn.numlayer, ' layerno ', layerno)
    
    if(layerno==nn.numlayer):
       output_size = len(nn.weights[nn.ffn_counter-1])
       
       #bounds = elina_abstract0_to_box(man,element)
       #for i in range(output_size):
       #    elina_interval_fprint(cstdout,bounds[i])
       #elina_interval_array_free(bounds,output_size)
       verified_flag = False
       #label = 0
       if(use_label):
           flag = True
           for j in range(output_size):
               if((j!=label) and not is_greater_zono(man,element,label,j)):
                   print('label ',label,'j ',j)
                   flag = False
                   break
           if(flag):   
               verified_flag = True
        
           return label, verified_flag  
       else:
           for i in range(output_size):
               flag = True
               for j in range(output_size):
                   if((i!=j) and not is_greater_zono(man,element,i,j)):
                       flag = False
                       break
               if(flag):
                   label = i
                   verified_flag = True
                   break
        
           return label, verified_flag 
    #label = 0

    
    #print('layertype ',nn.layertypes[layerno])
    #bounds = elina_abstract0_to_box(man,element)
    #num_in_pixels = elina_abstract0_dimension(man,element).realdim
    #for i in range(num_in_pixels):
    #    print('layername ',nn.layertypes[layerno],'layerno ',layerno,'i ', i, 'lb ',bounds[i].contents.inf.contents.val.dbl, ' ub ',bounds[i].contents.sup.contents.val.dbl)
    if(nn.layertypes[layerno]=='SkipNet1'):
        #print("skip1")
        nlb.append([])
        nub.append([])
    elif(nn.layertypes[layerno]=='SkipNet2'):
        #print("skip2")
        tmp = element
        element = element1
        element1 = tmp
        nlb.append([])
        nub.append([])
    elif(nn.layertypes[layerno]=='SkipCat'):
        
        zono_concat_element(man,element1,element)
        #print("skipcat", elina_abstract0_dimension(man,element1).realdim,elina_abstract0_dimension(man,element).realdim)
        tmp = element
        element = element1
        element1 = tmp
        nlb.append([])
        nub.append([])
    elif(nn.layertypes[layerno] in ['ReLU', 'Affine', 'Sigmoid', 'Tanh']):
        weights = nn.weights[nn.ffn_counter]
        biases = nn.biases[nn.ffn_counter+nn.conv_counter]
        #print("input")
        #elina_abstract0_fprint(cstdout,man, element, None)
        element = handle_affine_ai2(man,element,weights, biases,nlb,nub)
        num_out_pixels = len(nlb[layerno])
        if(nn.layertypes[layerno]=='ReLU'):
            element = relu_zono_layerwise(man,True,element,0, num_out_pixels)
        elif(nn.layertypes[layerno]=='Sigmoid'):
            element = sigmoid_zono_layerwise(man,True,element,0, num_out_pixels)
        elif(nn.layertypes[layerno]=='Tanh'):
            element = tanh_zono_layerwise(man,True,element,0, num_out_pixels)
        nn.ffn_counter+=1


    elif(nn.layertypes[layerno]=='Conv2D'):
        tmp = nn.input_shape[nn.conv_counter+nn.maxpool_counter]
        #tmp = tmp.astype(np.uintp)
        tmp1 = nn.filter_size[nn.conv_counter]
        tmp2 = nn.strides[nn.conv_counter]
        numfilters = nn.numfilters[nn.conv_counter]
        #print('num_in_pixels ',num_in_pixels,' num_out_pixels ',num_out_pixels)
        input_shape = (ctypes.c_size_t * len(tmp))(*tmp)
        filter_size = (ctypes.c_size_t * len(tmp1))(*tmp1)
        strides = (ctypes.c_size_t * len(tmp2))(*tmp2)
        padding = nn.padding[nn.conv_counter]
        #padding = True
        #if(padding==1):
        #    padding = False
        filters = nn.filters[nn.conv_counter]
        biases = nn.biases[nn.ffn_counter+nn.conv_counter]
        #print(filters)
        element = handle_conv_ai2(man,element,filters,biases,filter_size, numfilters,input_shape,nlb,nub, strides, padding)
        num_out_pixels = len(nlb[layerno])
        element = relu_zono_layerwise(man,True,element,0, num_out_pixels)
        nn.conv_counter+=1

    elif(nn.layertypes[layerno]=='MaxPooling2D'):
        tmp = nn.input_shape[nn.conv_counter+nn.maxpool_counter]
        tmp1 = nn.pool_size[nn.maxpool_counter]
        input_shape = (ctypes.c_size_t * len(tmp))(*tmp)
        pool_size = (ctypes.c_size_t * len(tmp1))(*tmp1)
        element,_,_ = handle_maxpool_ai2(man,element,input_shape,pool_size,nlb,nub)
        nn.maxpool_counter+=1
    else:
         print("Invalid layertype")
         return
  
    label, verified_flag = analyze_layer(nn,layerno+1, man, element, element1, nlb, nub, label,use_label)
       
    if(not verified_flag):
        #elina_interval_array_free(bounds,num_out_pixels)
        #elina_abstract0_free(man,element)
        return label, False
    
    
    #
    return label, verified_flag    


def get_perturbed_image(x, epsilon):
    num_pixels = len(x)
    LB_N0 = x - epsilon
    UB_N0 = x + epsilon
    
    num_pixels = len(LB_N0) 
    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0

def analyze_with_zono(nn, LB_N0, UB_N0, nlb, nub, label, use_label):
    # contains numlayer+1 arrays, each corresponding to a lower/upper bound
    #print(UB_N0)
    #np.ascontiguousarray(LB_N0, dtype=np.double)
    #np.ascontiguousarray(UB_N0, dtype=np.double)
    
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.maxpool_counter = 0
    num_pixels = len(LB_N0)
    man = zonoml_manager_alloc()
    ## construct input abstraction
    element = zonotope_from_network_input(man, 0, num_pixels, LB_N0, UB_N0)
    #for i in range(nn.numlayer):
    #    if(nn.layertypes[i]=='SkipNet1'):
    #          is_skip = True
    #          break
    #element1 = None
    #if(is_skip):
    element1 = zonotope_from_network_input(man, 0, num_pixels, LB_N0, UB_N0)
    label, verified_flag = analyze_layer(nn,0,man,element, element1, nlb,nub, label,use_label)
    
    
    if(LB_N0[0]!=UB_N0[0]):
       #for i in range(nn.numlayer):
           #print("i ",i)
           print(nlb[nn.numlayer-1])
           print(nub[nn.numlayer-1])
    elina_abstract0_free(man,element)
    elina_abstract0_free(man,element1)
    elina_manager_free(man)

    return label, verified_flag



def analyze_with_box(nn, LB_N0, UB_N0, nlb, nub, label):   
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.maxpool_counter = 0
    numlayer = nn.numlayer 

    # print("Here? ")

    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    
    # print("Here2? ")

    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    # print("Here3? ")

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)

    # print("Here4? ", numlayer)

    for layerno in range(numlayer):

        # print("Here5? ", layerno)

        # if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
        if(nn.layertypes[layerno] in ['FC']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter+nn.conv_counter]
           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           num_out_pixels = len(weights)
           #print("in pixels ",num_in_pixels," out pixels ",num_out_pixels)
           dimadd = elina_dimchange_alloc(0,num_out_pixels)    
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels
           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)
           bounds = elina_abstract0_to_box(man,element)
           lbi = []
           ubi = []
           # print('layerno ',layerno)
           for i in range(num_out_pixels):
               inf = bounds[i].contents.inf
               sup = bounds[i].contents.sup
               #print('i ',i)
               #elina_interval_fprint(cstdout,bounds[i])
               #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
               lbi.append(inf.contents.val.dbl)
               ubi.append(sup.contents.val.dbl)
    
           nlb.append(lbi)
           nub.append(ubi) 

           elina_interval_array_free(bounds,num_out_pixels) 
        elif(nn.layertypes[layerno]=='ReLU'):
            element = relu_box_layerwise(man,True,element,0, num_out_pixels)
              #bounds = elina_abstract0_to_box(man,element)
              #for i in range(num_out_pixels):
              #    elina_interval_fprint(cstdout,bounds[i])
              #elina_interval_array_free(bounds,num_out_pixels)
            nn.ffn_counter+=1 

        else:
           print(nn.layertypes[layerno] + ' net type not supported')
           #return
        #
    # print(nlb[-1])    
    # print(nub[-1]) 

    # Get dominant class from the resulting box if it is label
    verified = True
    # print('label :' + str(label))

    for i in range(len(nlb[-1])):
        if i!=label and nub[-1][i] > nlb[-1][label]:
            verified = False
            # print(i, ' counters the ', label)
            # print(nub[-1][i], ' and ', nlb[-1][label])
            # print('in ', )
            break

    #output_size = len(nlb[numlayer-1])
    #bounds = elina_abstract0_to_box(man,element)
    #for i in range(output_size):
    #    elina_interval_fprint(cstdout,bounds[i])
    #elina_interval_array_free(bounds,output_size)
    #for i in range(numlayer):
    #    print("i ",i) 
    #    print(nlb[i])
    #    print(nub[i])
    #verified_flag = True   
    #for j in range(output_size):
    #    if((j!=label) and not is_greater_zono(man,element,label,j)):
            #print("Failed")
            
    #        verified_flag = False
    #        break
    elina_abstract0_free(man,element)
    
    # elina_manager_free(man)        
    return verified  
    #return verified_flag


def analyze_with_box_till_layer(k, nn, LB_N0, UB_N0, nlb, nub, label):   
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.maxpool_counter = 0
    numlayer = nn.numlayer 

    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)

    for layerno in range(k):

        if(nn.layertypes[layerno] in ['FC']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter+nn.conv_counter]
           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           num_out_pixels = len(weights)
           #print("in pixels ",num_in_pixels," out pixels ",num_out_pixels)
           dimadd = elina_dimchange_alloc(0,num_out_pixels)    
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels
           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)
           bounds = elina_abstract0_to_box(man,element)
           lbi = []
           ubi = []
           # print('layerno ',layerno)
           for i in range(num_out_pixels):
               inf = bounds[i].contents.inf
               sup = bounds[i].contents.sup
               #print('i ',i)
               #elina_interval_fprint(cstdout,bounds[i])
               #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
               lbi.append(inf.contents.val.dbl)
               ubi.append(sup.contents.val.dbl)
    
           nlb.append(lbi)
           nub.append(ubi) 

           elina_interval_array_free(bounds,num_out_pixels) 
        elif(nn.layertypes[layerno]=='ReLU'):
            element = relu_box_layerwise(man,True,element,0, num_out_pixels)
              #bounds = elina_abstract0_to_box(man,element)
              #for i in range(num_out_pixels):
              #    elina_interval_fprint(cstdout,bounds[i])
              #elina_interval_array_free(bounds,num_out_pixels)
            nn.ffn_counter+=1 

        else:
           print(nn.layertypes[layerno] + ' net type not supported')
           #return

    # libc = CDLL(find_library('c'))
    # cstdout = c_void_p.in_dll(libc, 'stdout')       
    # elina_box_fprint(cstdout, man, element, None)       

    # element = elina_box_expand(man, element, 1.4)

    # elina_box_fprint(cstdout, man, element, None) 

    return element, man


def analyze_with_box_from_layer(k, nn, element, man, nlb, nub, label):   
    # num_pixels = 
    nn.ffn_counter = int(k/2)
    # nn.conv_counter = 0
    # nn.maxpool_counter = 0
    numlayer = nn.numlayer 


    for layerno in range(k, numlayer):
        # print("layerno in from: ", layerno)
        # print(nn.ffn_counter, nn.layertypes[layerno])
        weights = nn.weights[nn.ffn_counter]
        num_out_pixels = len(weights)

        if(nn.layertypes[layerno] in ['FC']):
           
           biases = nn.biases[nn.ffn_counter+nn.conv_counter]
           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           
           #print("in pixels ",num_in_pixels," out pixels ",num_out_pixels)
           dimadd = elina_dimchange_alloc(0,num_out_pixels)    
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels
           # print(num_in_pixels, num_out_pixels)
           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               # print(i, '--->', nn.ffn_counter)
               # print(weights[i])
               # print(k, numlayer)
               # print(weights)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)
           bounds = elina_abstract0_to_box(man,element)
           lbi = []
           ubi = []
           # print('layerno ',layerno)
           for i in range(num_out_pixels):
               inf = bounds[i].contents.inf
               sup = bounds[i].contents.sup
               #print('i ',i)
               #elina_interval_fprint(cstdout,bounds[i])
               #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
               lbi.append(inf.contents.val.dbl)
               ubi.append(sup.contents.val.dbl)
    
           nlb.append(lbi)
           nub.append(ubi) 

           elina_interval_array_free(bounds,num_out_pixels) 
        elif(nn.layertypes[layerno]=='ReLU'):
            element = relu_box_layerwise(man,True,element,0, num_out_pixels)
              #bounds = elina_abstract0_to_box(man,element)
              #for i in range(num_out_pixels):
              #    elina_interval_fprint(cstdout,bounds[i])
              #elina_interval_array_free(bounds,num_out_pixels)
            nn.ffn_counter+=1 

        else:
           print(nn.layertypes[layerno] + ' net type not supported')
           #return

    verified = True
    for i in range(len(nlb[-1])):
        if i!=label and nub[-1][i] > nlb[-1][label]:
            verified = False
            # print(i, ' counters the ', label)
            # print(nub[-1][i], ' and ', nlb[-1][label])
            # print('in ', )
            break

    return verified
    

def analyze_with_LP(nn, LB_N0, UB_N0, nlb, nub, label):   
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.maxpool_counter = 0
    numlayer = nn.numlayer
    use_milp = []
    
    for i in range(numlayer):
        use_milp.append(0)
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    
    elina_interval_array_free(itv,num_pixels)


    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter+nn.conv_counter]
           num_out_pixels = len(weights)
           if(layerno==0):

               dims = elina_abstract0_dimension(man,element)
               num_in_pixels = dims.intdim + dims.realdim
               
               #print("in pixels ",num_in_pixels," out pixels ",num_out_pixels)
               dimadd = elina_dimchange_alloc(0,num_out_pixels)    
               for i in range(num_out_pixels):
                   dimadd.contents.dim[i] = num_in_pixels
               elina_abstract0_add_dimensions(man, True, element, dimadd, False)
               elina_dimchange_free(dimadd)
               np.ascontiguousarray(weights, dtype=np.double)
               np.ascontiguousarray(biases, dtype=np.double)
               var = num_in_pixels
               for i in range(num_out_pixels):
                   tdim= ElinaDim(var)
                   linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
                   element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
                   var+=1
               dimrem = elina_dimchange_alloc(0,num_in_pixels)
               for i in range(num_in_pixels):
                   dimrem.contents.dim[i] = i
               elina_abstract0_remove_dimensions(man, True, element, dimrem)
               elina_dimchange_free(dimrem)
               bounds = elina_abstract0_to_box(man,element)
               lbi = []
               ubi = []
               #print('layerno ',layerno)
               for i in range(num_out_pixels):
                   inf = bounds[i].contents.inf
                   sup = bounds[i].contents.sup
                   #print('i ',i)
                   #elina_interval_fprint(cstdout,bounds[i])
                   #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
                   lbi.append(inf.contents.val.dbl)
                   ubi.append(sup.contents.val.dbl)
    
               nlb.append(lbi)
               nub.append(ubi) 

               elina_interval_array_free(bounds,num_out_pixels) 
           else:
               relu_needed = [0]*(layerno+1)
               for i in range(layerno):
                   relu_needed[i] = 1
               lbi = [0]*num_out_pixels
               ubi = [0]*num_out_pixels
               for i in range(num_out_pixels):
                   lbi[i] = float("-inf")
                   ubi[i] = float("inf")
               nlb.append(lbi)
               nub.append(ubi)
               #print("layerno ", layerno, "numlayer ", numlayer)
               counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, layerno+1, use_milp, relu_needed)
               for i in range(num_out_pixels):
                   obj = LinExpr()
                   obj += var_list[counter+i]
                   model.setObjective(obj,GRB.MINIMIZE)
                   model.optimize()
                   lbi[i] = model.objval
                   model.setObjective(obj,GRB.MAXIMIZE)
                   model.optimize()
                   ubi[i] = model.objval
               nlb.pop()
               nub.pop()
               nlb.append(lbi)
               nub.append(ubi)
           nn.ffn_counter+=1 

        else:
           print(' net type not supported')
           #return
        #
        
    output_size = len(nlb[numlayer-1])
    #bounds = elina_abstract0_to_box(man,element)
    #for i in range(output_size):
    #    elina_interval_fprint(cstdout,bounds[i])
    #elina_interval_array_free(bounds,output_size)
    #for i in range(numlayer):
    #    print("i ",i) 
    #    print(nlb[i])
    #    print(nub[i])
    verified_flag = True   
    relu_needed = []
    for i in range(numlayer):
        relu_needed.append(1)
    counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, numlayer, use_milp, relu_needed)
    for j in range(output_size):
        if((j!=label)):
            #print("Failed")
            #obj = LinExpr()
            obj = var_list[counter+label] - var_list[counter+j]
            model.setObjective(obj,GRB.MINIMIZE)
            model.optimize()
            if(model.objval<=0):
                verified_flag = False
                break
    elina_abstract0_free(man,element)
    elina_manager_free(man)        
    return verified_flag



def analyze_with_box_and_full_milp(nn, LB_N0, UB_N0, label):
    nlb = []
    nub = []
    analyze_with_box(nn, LB_N0, UB_N0, nlb, nub, label,)
    verified_flag = verify_network_with_milp(nn,LB_N0,UB_N0,label, nlb, nub)
    return verified_flag  

def analyze_with_LP_and_full_milp(nn, LB_N0, UB_N0, label):
    nlb = []
    nub = []
    verified_flag = analyze_with_LP(nn, LB_N0, UB_N0, nlb, nub, label)
    if(verified_flag):
       print("Verified by LP", label)
    else:
       #get_bounds_for_layer_with_milp(weights,biases,LB_N0,UB_N0,2,nlb,nub)
       #print("Failed")
       #print(nlb)
       #print(nub)
       verified_flag = verify_network_with_milp(nn,LB_N0,UB_N0,label, nlb, nub)
    return verified_flag  

def analyze_with_zono_and_full_milp(nn, LB_N0, UB_N0, label):
    nlb = []
    nub = []
    use_label = True
    nlabel,verified_flag = analyze_with_zono(nn, LB_N0, UB_N0, nlb, nub, label, use_label)
    if(verified_flag):
       print("Verified by AI2", nlabel)
    else:
       #get_bounds_for_layer_with_milp(weights,biases,LB_N0,UB_N0,2,nlb,nub)
       #print("Failed")
       #print(nlb)
       #print(nub)
       verified_flag = verify_network_with_milp(nn,LB_N0,UB_N0,label, nlb, nub)
    return verified_flag  

def refine_relu_with_milp_bounds(man,element, num_neurons, resl, resu, indices):
     j = 0
     
     #needs_refinement = np.full((num_neurons), False, dtype=bool)
     #new_inf = np.zeros(num_neurons)
     #new_sup = np.zeros(num_neurons)
     #np.ascontiguousarray(new_inf, dtype=np.double)
     #np.ascontiguousarray(new_sup, dtype=np.double)
     #np.ascontiguousarray(needs_refinement, dtype=np.bool)
     #for i in range(num_neurons):
     #    if((j < len(indices)) and (i==indices[j])):
     #        needs_refinement[i] = True
     #        j=j+1
     #    new_inf[i] = resl[i]
     #    new_sup[i] = resu[i]
     
     #element = relu_zono_refine_layerwise(man,True,element,0,num_neurons,new_inf,new_sup,needs_refinement)
     for i in range(num_neurons):
         if((j < len(indices)) and (i==indices[j])):
             
             element = relu_zono_refined(man,True, element,i, resl[i],resu[i])
             j=j+1
         else:
             element = relu_zono(man,True,element,i)

     
     return element


def refine_maxpool_with_milp_bounds(man,element, num_neurons, resl, resu, indices):
     j = 0
     for i in range(num_neurons):
         if((j < len(indices)) and (i==indices[j])):
             element = maxpool_zono_refined(man,True, element,i, resl[i],resu[i])
             j=j+1
     return element

def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0




def analyze_with_zono_and_part_milp(nn, LB_N0, UB_N0, label):
    nlb = []
    nub = []    
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.maxpool_counter = 0
    numlayer = nn.numlayer 
    man = zonoml_manager_alloc()
    ## construct input abstraction
    element = zonotope_from_network_input(man, 0, num_pixels, LB_N0, UB_N0)
    element1 = zonotope_from_network_input(man, 0, num_pixels, LB_N0, UB_N0)
    use_milp = []
    is_conv = len(nn.filters)>0
    for i in range(numlayer):       
        if(is_conv):
           if(i<=1):
              use_milp.append(0)
           else:
              use_milp.append(0)
        else:
           if(i<=3):
              use_milp.append(1)
           else:
              use_milp.append(0)
    abs_layer_count = 0
    for layerno in range(numlayer):
        
        
        if(nn.layertypes[layerno]=='SkipNet1'):
            #print("skip1")
            abs_layer_count = 0
        elif(nn.layertypes[layerno]=='SkipNet2'):
            #print("skip2")
            tmp = element
            element = element1
            element1 = tmp
            nlb = []
            nub = []
            abs_layer_count = 0
        elif(nn.layertypes[layerno]=='SkipCat'):
            zono_concat_element(man,element1,element)
            #print("skipcat", elina_abstract0_dimension(man,element1).realdim,elina_abstract0_dimension(man,element).realdim)
            tmp = element
            element = element1
            element1 = tmp
            #nlb.append([])
            #nub.append([])
            #print("layertype ",nn.layertypes[layerno])
        elif(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter+nn.conv_counter]
           element = handle_affine_ai2(man,element,weights,biases,nlb,nub)
           if(nn.layertypes[layerno]=='ReLU'):
              num_out_pixels = len(weights)
              if(abs_layer_count==0):
                 #print("coming here")
                 element = relu_zono_layerwise(man,True,element,0, num_out_pixels)
              else :
                 candidate_vars = []
                 lbi = nlb[abs_layer_count]
                 ubi = nub[abs_layer_count]
                 for i in range(num_out_pixels):
                    if((lbi[i]<0 and ubi[i]>0) or (lbi[i]>0)):
                       candidate_vars.append(i)
                 resl, resu, indices = get_bounds_for_layer_with_milp(nn, LB_N0, UB_N0, layerno, abs_layer_count, num_out_pixels, nlb, nub, use_milp,  candidate_vars)
                 nlb.pop()
                 nub.pop()
                 nlb.append(resl)
                 nub.append(resu)
                 element = refine_relu_with_milp_bounds(man, element, num_out_pixels, resl, resu, indices)
           abs_layer_count+=1
           nn.ffn_counter+=1 


        elif(nn.layertypes[layerno]=='Conv2D'):
           tmp = nn.input_shape[nn.conv_counter+nn.maxpool_counter]
           #tmp = tmp.astype(np.uintp)
           tmp1 = nn.filter_size[nn.conv_counter]
           tmp2 = nn.strides[nn.conv_counter]
           numfilters = nn.numfilters[nn.conv_counter]
           #print('num_in_pixels ',num_in_pixels,' num_out_pixels ',num_out_pixels)
           input_shape = (ctypes.c_size_t * len(tmp))(*tmp)
           filter_size = (ctypes.c_size_t * len(tmp1))(*tmp1)
           strides = (ctypes.c_size_t * len(tmp2))(*tmp2)
           filters = nn.filters[nn.conv_counter]
           biases = nn.biases[nn.ffn_counter+nn.conv_counter]
           padding = nn.padding[nn.conv_counter]
           #padding = True
           #if(padding=="SAME"):
           #    padding = False
           element = handle_conv_ai2(man,element,filters,biases,filter_size, numfilters,input_shape,nlb,nub,strides, padding)
           num_out_pixels = len(nlb[abs_layer_count])
           output_size = elina_abstract0_dimension(man,element).realdim
           #print("output size ", output_size)
           if(abs_layer_count==0):
              element = relu_zono_layerwise(man,True,element,0, num_out_pixels)
           else:
              candidate_vars = []
              lbi = nlb[abs_layer_count]
              ubi = nub[abs_layer_count]
              for i in range(num_out_pixels):
                 #if((lbi[i]<0 and ubi[i]>0) or (lbi[i]>0)):
                 #   candidate_vars.append(i)
                 if(lbi[i]<0 and ubi[i]>0):
                      candidate_vars.append(i)
              resl, resu, indices = get_bounds_for_layer_with_milp(nn, LB_N0, UB_N0, layerno, abs_layer_count, num_out_pixels, nlb, nub, use_milp, candidate_vars)
              nlb.pop()
              nub.pop()
              nlb.append(resl)
              nub.append(resu)
              element = refine_relu_with_milp_bounds(man, element, num_out_pixels, resl, resu, indices)
           abs_layer_count+=1
           nn.conv_counter+=1


        elif(nn.layertypes[layerno]=='MaxPooling2D'):
           tmp = nn.input_shape[nn.conv_counter+nn.maxpool_counter]
           tmp1 = nn.pool_size[nn.maxpool_counter]
           input_shape = (ctypes.c_size_t * len(tmp))(*tmp)
           pool_size = (ctypes.c_size_t * len(tmp1))(*tmp1)
           element, lb, ub = handle_maxpool_ai2(man,element,input_shape,pool_size,nlb,nub)
           nn.maxpool_lb.append(lb)
           nn.maxpool_ub.append(ub)
           candidate_vars = []
           num_out_pixels = len(nlb[layerno])
           for i in range(num_out_pixels):
               if(affine_form_is_box(man,element,i)):
                  candidate_vars.append(i)
           #print('before ')
           #print(nlb[layerno])
           #print(nub[layerno])
           resl, resu, indices = get_bounds_for_layer_with_milp(nn, LB_N0, UB_N0, layerno, abs_layer_count, num_out_pixels, nlb, nub, use_milp, candidate_vars)
           #print('after ')
           #print(resl)
           #print(resu)
           nlb.pop()
           nub.pop()
           nlb.append(resl)
           nub.append(resu)
           element = refine_maxpool_with_milp_bounds(man, element, num_out_pixels, resl, resu, indices)
           nn.maxpool_counter+=1
        else:
           print('Parsing error, net type not supported')
           #return
        #
        
    output_size = 10#len(nlb[numlayer-1])
    bounds = elina_abstract0_to_box(man,element)
    for i in range(output_size):
        elina_interval_fprint(cstdout,bounds[i])
    elina_interval_array_free(bounds,output_size)
    #for i in range(numlayer):
    #    print("i ",i) 
    #print(nlb[numlayer-1])
    #print(nub[numlayer-1])
    verified_flag = True   
    for j in range(output_size):
        if((j!=label) and not is_greater_zono(man,element,label,j)):
            #print("Failed")
            verified_flag = False
            break
    for i in range(len(nn.maxpool_lb)):
        nn.maxpool_lb.pop()
        nn.maxpool_ub.pop()
    elina_abstract0_free(man,element)
    elina_abstract0_free(man,element1)
    elina_manager_free(man)        
    return verified_flag


def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
   
    #result = np.array([*map(lambda x: parse_bias(x.strip()), filter(lambda x: len(x) != 0, text.split('\n')))])
    #low, high=result[:,0], result[:,1]
    #print("high" ,high)
    #print("low", low)
    #return low.reshape((low.size,1)), high.reshape((high.size,1))
    return low,high


