from optimizer import *
from krelu import *

def refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, true_label, labels_to_be_verified):
    relu_groups = []
    nlb = []
    nub = []
    #print("INPUT SIZE ", network._lib.getOutputSize(network._nn, 0))
    layerno = 2
    new_relu_layers = []
    for l in range(nn.numlayer):
        num_neurons = network._lib.getOutputSize(network._nn, layerno)
        #print("num neurons ", num_neurons)
        if layerno in relu_layers:
            pre_lbi = nlb[len(nlb)-1]
            pre_ubi = nub[len(nub)-1]
            lbi = np.zeros(num_neurons)
            ubi = np.zeros(num_neurons)
            for j in range(num_neurons):
                lbi[j] = max(0,pre_lbi[j])
                ubi[j] = max(0,pre_ubi[j])
            layerno =  layerno+2
            new_relu_layers.append(len(nlb))
            #print("RELU ")
        else:
            #print("COMING HERE")
            #A = np.zeros((num_neurons,num_neurons), dtype=np.double)
            #print("FINISHED ", num_neurons)
            #for j in range(num_neurons):
            #    A[j][j] = 1
            bounds = network.evalAffineExpr(layer=layerno)
            #print("num neurons", num_neurons)
            lbi = bounds[:,0]
            ubi = bounds[:,1]
            layerno = layerno+1
        nlb.append(lbi)
        nub.append(ubi)
       
    index = 0 
    for l in relu_layers:
        gpu_layer = l - 1
        layerno = new_relu_layers[index]
        index = index+1
        lbi = nlb[layerno-1]
        ubi = nub[layerno-1]
        #print("LBI ", lbi, "UBI ", ubi, "specLB")
        num_neurons = len(lbi)
        kact_args = sparse_heuristic_with_cutoff(num_neurons, lbi, ubi)
        kact_cons = []
        total_size = 0
        for varsid in kact_args:
            size = 3**len(varsid) - 1
            total_size = total_size + size
        #print("total size ", total_size, kact_args)
        A = np.zeros((total_size, num_neurons), dtype=np.double)
        i = 0
        #print("total_size ", total_size)
        for varsid in kact_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                for j in range(len(varsid)):
                    A[i][varsid[j]] = coeffs[j] 
               
                i = i + 1
        bounds = network.evalAffineExpr(A, layer=gpu_layer, back_substitute=network.FULL_BACKSUBSTITUTION, dtype=np.double)
        upper_bound = bounds[:,1]
        i=0
        input_hrep_array = []
        for varsid in kact_args:
            input_hrep = []
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                input_hrep.append([upper_bound[i]] + [-c for c in coeffs])
                i = i + 1
            input_hrep_array.append(input_hrep)
        KAct.type = "ReLU"
        with multiprocessing.Pool(config.numproc) as pool:
            kact_results = pool.map(make_kactivation_obj, input_hrep_array)
        gid = 0
        for inst in kact_results:
            varsid = kact_args[gid]
            inst.varsid = varsid
            kact_cons.append(inst)
            gid = gid+1
        relu_groups.append(kact_cons)
    counter, var_list, model = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups, nn.numlayer, config.complete==True, is_nchw=True)
    model.setParam(GRB.Param.TimeLimit, config.timeout_lp)
    num_var = len(var_list)
    #output_size = num_var - counter
    #print("TIMEOUT ", config.timeout_lp)
    flag = True
    x = None
    for label in labels_to_be_verified:
        obj = LinExpr()
        #obj += 1*var_list[785]
        obj += 1*var_list[counter + true_label]
        obj += -1*var_list[counter + label]
        model.setObjective(obj,GRB.MINIMIZE)
        model.optimize()
        #model.computeIIS()
        #model.write("model_refinegpupo.ilp")
        print("objval ", label, model.Status, model.objval)
        if model.Status!=2:
            print("model was not successful status is", model.Status)
            model.write("final.mps")
            flag = False
            break                       
        elif model.objval < 0:
                               
            flag = False
            if model.objval != math.inf:
                  x = model.x[0:len(nn.specLB)]
            break
    return flag, x
                
