
import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import numpy as np
import argparse
import csv
import time
import copy
import random

from eran import ERAN
from onnx_translator import *
from optimizer import *
from analyzer import *
from onnx_translator import *
from ctypes import *
from ctypes.util import *	
from analyzers_zonoml_ffn import *
from graphviz import Graph


alpha_sum = 0
alpha_nz = 0

def main():
	parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
	parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
	parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')
	parser.add_argument('--anetname', type=isnetworkfile, default=config.anetname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')

	args = parser.parse_args()
	for k, v in vars(args).items():
		setattr(config, k, v)
	config.json = vars(args)

	assert config.netname, 'a network has to be provided for analysis.'
	# We assume ONNX input for networks
	netname = config.netname
	epsilon = config.epsilon
	approx_netname = config.anetname

	dataset = config.dataset
	assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

	# Assuming MNIST for now
	num_pixels = 784   
	model, is_conv = read_onnx_net(netname)
	
	# The thing crashes if called immediately?
	approx_model, approx_is_conv = read_onnx_net(approx_netname)

	is_onnx = True

	if dataset == 'mnist':
		means = [0]
		stds = [1]

	eran = ERAN(model, is_onnx=is_onnx)
	# Do binary search and expand the template in the element
	k = 4
	tests = get_tests(dataset, False)
	# tests = get_tests(dataset, config.geometric)

	approx_model, approx_is_conv = read_onnx_net(approx_netname) 
	verified_cnt = 0
	total = 0

	test_li = [test for i, test in enumerate(tests)]

	sum_overlap = 0
	total_overlap = 0	


	image_start = np.float64(test_li[0][1:len(test_li[0])])/np.float64(255)
	

	def rand_perturb_image(image, num_perturb):
		image2 = copy.deepcopy(image)
		for i in range(num_perturb):
			pix = random.randint(0,783)
			val = random.uniform(0, 1)
			image2[pix] = val 
		return image2	
		
	def generate_perturb_set(image, size):
		per_set = [image]
		for i in range(size):
			per_image = rand_perturb_image(image, 4)
			per_set.append(per_image)
		return per_set	
			
	per_set = generate_perturb_set(image_start, 100)	

	# for i in range(len(test_li)):
	# 	# SHUBHAM: JUST DO IT FOR ONE TEST NOW
	#	 # if(i>0):
	#	 #	 break
	#	 test = test_li[i] 

	#	 image= np.float64(test[1:len(test)])/np.float64(255)

	#	 # normalize(specLB, means, stds, dataset)
	#	 # normalize(specUB, means, stds, dataset)	

	#	 # Finding the actual label
	#	 # label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB,'deeppoly', None, None, False)
	#	 label = int(test[0])
	#	 specLB = np.clip(image - epsilon,0,1)
	#	 specUB = np.clip(image + epsilon,0,1)
	#	 prop = -1 # target	


		
	#	 element, man = get_box_template_at_layer_rand_search(approx_model, k, specLB, specUB, label, prop)

	#	 lb,ub = get_box_from_element(man, element)

	#	 for i in range(len(all_lbi)):
	#	 	# print(test_li[i][0], label, test_li[i][0] == label, print(type(test_li[i][0])), print(type(label)))
	#	 	if(int(test_li[i][0]) == label):
	# 			cs = contain_score(all_lbi[i], all_ubi[i], lb, ub)
	# 			total_overlap += 1
	# 			sum_overlap += cs
	# 			print(i,'. ', cs)

		
	#	 approx_nn = init_nn(approx_model, specLB, specUB)
		 
	#	 # verify if the template is true for the approx model
	#	 is_verified = analyze_with_box_from_layer(k, approx_nn, element, man, specLB, specUB, [], [], label)

	#	 print('is verified?', is_verified)

	#	 if(is_verified):
	#	 	verified_cnt += 1
	#	 total += 1	
	total_ver = 0
	union_ver = 0
	total_same_label = 0
	dot = Graph()

	# for i in range(len(test_li)):
	# 	dot.node(str(i)) 
		
	# for i in range(len(test_li)):
	# 	for j in range(i, len(test_li)):
	# 		test1 = test_li[i] 
	# 		test2 = test_li[j]

	# 		if(i == j):
	# 			continue

	# 		if(test1[0] != test2[0]):
	# 			continue

	# 		image1 = np.float64(test1[1:len(test1)])/np.float64(255)
	# 		image2 = np.float64(test2[1:len(test2)])/np.float64(255)

	# 		label = int(test1[0])
	# 		specLB1 = np.clip(image1 - epsilon,0,1)
	# 		specUB1 = np.clip(image1 + epsilon,0,1)

	# 		specLB2 = np.clip(image2 - epsilon,0,1)
	# 		specUB2 = np.clip(image2 + epsilon,0,1)

	# 		prop = -1 # target

	# 		nn1 = init_nn(approx_model, specLB1, specUB1)
	# 		nn2 = init_nn(approx_model, specLB2, specUB2)

	# 		element1, man = analyze_with_box_till_layer(k, nn1, specLB1, specUB1, [], [], label)
	# 		element2, man = analyze_with_box_till_layer(k, nn1, specLB2, specUB2, [], [], label)

	# 		element3 = elina_box_union(man, element1, element2)

	# 		is_verified1 = analyze_with_box_from_layer(k, nn1, element1, man, specLB1, specUB1, [], [], label)
	# 		is_verified2 = analyze_with_box_from_layer(k, nn1, element2, man, specLB2, specUB2, [], [], label)
	# 		is_verified3 = analyze_with_box_from_layer(k, nn1, element3, man, specLB1, specLB1, [], [], label)

	# 		if(is_verified1):
	# 			dot.node(str(i))

	# 		total_same_label += 1
	# 		if (is_verified1 and is_verified2):
	# 			total_ver += 1
	# 			if(is_verified3):
	# 				union_ver += 1
	# 				print('i:',i,' j:',j)
	# 				dot.edge(str(i), str(j))
	# 				print('verified_labe:', label)

	# 		print(is_verified1, is_verified2, is_verified3)

	for i in range(len(per_set)):
		for j in range(i+1, len(per_set)):

			image1 = per_set[i]
			image2 = per_set[j]

			label = int(test_li[0][0])
			specLB1 = np.clip(image1 - epsilon,0,1)
			specUB1 = np.clip(image1 + epsilon,0,1)

			specLB2 = np.clip(image2 - epsilon,0,1)
			specUB2 = np.clip(image2 + epsilon,0,1)

			prop = -1 # target

			nn1 = init_nn(approx_model, specLB1, specUB1)
			nn2 = init_nn(approx_model, specLB2, specUB2)

			element1, man = analyze_with_box_till_layer(k, nn1, specLB1, specUB1, [], [], label)
			element2, man = analyze_with_box_till_layer(k, nn1, specLB2, specUB2, [], [], label)

			element3 = elina_box_union(man, element1, element2)

			is_verified1 = analyze_with_box_from_layer(k, nn1, element1, man, specLB1, specUB1, [], [], label)
			is_verified2 = analyze_with_box_from_layer(k, nn1, element2, man, specLB2, specUB2, [], [], label)
			is_verified3 = analyze_with_box_from_layer(k, nn1, element3, man, specLB1, specLB1, [], [], label)

			if(is_verified1):
				dot.node(str(i))

			total_same_label += 1
			if (is_verified1 and is_verified2):
				total_ver += 1
				if(is_verified3):
					union_ver += 1
					print('i:',i,' j:',j)
					dot.edge(str(i), str(j))
					print('verified_labe:', label)

			print(is_verified1, is_verified2, is_verified3)	

	# print('>1 alpha:', alpha_nz)
	# print('alpha_sum:', alpha_sum/100)   
	print('total: ', total_ver, 'union_ver: ', union_ver, 'total_same_label:', total_same_label)
	dot.render('tmp.gv', view=True)

	# print("Average overlap:" + str(sum_overlap/total_overlap)) 
	print(verified_cnt,'/',total,' images verified!')	

def get_optimizer(model):
	translator = ONNXTranslator(model, False)
	operations, resources = translator.translate()
	optimizer  = Optimizer(operations, resources)	
	return optimizer

def init_nn(model, specUB, specLB):
	label = -1
	prop = -1
	nn = layers()
	optimizer = get_optimizer(model)
	execute_list, output_info = optimizer.get_deeppoly(nn, specLB, specUB, None, None, None, None, None, None, 0, None)
	analyzer = Analyzer(execute_list, nn, 'deeppoly', False, None, None, False, label, prop, False)
	return nn


def isnetworkfile(fname):
	_, ext = os.path.splitext(fname)
	if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
		raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
	return fname

def get_tests(dataset, geometric):
	if geometric:
		csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
	else:
		if config.subset == None:
			csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
		else:
			filename = '../data/'+ dataset+ '_test_' + config.subset + '.csv'
			csvfile = open(filename, 'r')
	tests = csv.reader(csvfile, delimiter=',')

	return tests

def save_element(element, man,filename):
	libc = CDLL(find_library('c'))
	cstdout = c_void_p.in_dll(libc, 'stdout')
	fopen = libc.fopen
	fopen.argtypes = ctypes.c_char_p, ctypes.c_char_p,
	fopen.restype = ctypes.c_void_p

	file = (ctypes.c_char_p) ((filename).encode('utf-8'))
	edit_type = (ctypes.c_char_p) (('w').encode('utf-8'))
	fp = fopen(file, edit_type)
	elina_abstract0_fprint(fp, man, element, None)

	fclose = libc.fclose
	fclose.argtypes = ctypes.c_void_p,
	fclose.restype = ctypes.c_int
	fclose(fp)



def load_element(man, filename, read_filename):
	libc = CDLL(find_library('c'))
	cstdout = c_void_p.in_dll(libc, 'stdout')
	fopen = libc.fopen
	fopen.argtypes = ctypes.c_char_p, ctypes.c_char_p,
	fopen.restype = ctypes.c_void_p

	file = (ctypes.c_char_p) ((read_filename).encode('utf-8'))
	read_type = (ctypes.c_char_p) (('r').encode('utf-8'))
	fp = fopen(file, read_type)

	file2 = (ctypes.c_char_p) ((filename).encode('utf-8'))
	edit_type2 = (ctypes.c_char_p) (('w').encode('utf-8'))
	fp2 = fopen(file2, edit_type2)
	element = fppoly_fread(fp2, fp, man, None)

	print('LOADED!!!')
	fclose = libc.fclose
	fclose.argtypes = ctypes.c_void_p,
	fclose.restype = ctypes.c_int
	
	fclose(fp)
	fclose(fp2)	
	return element

def normalize(image, means, stds, dataset):
	# normalization taken out of the network
	if len(means) == len(image):
		for i in range(len(image)):
			image[i] -= means[i]
			if stds!=None:
				image[i] /= stds[i]
	elif dataset == 'mnist'  or dataset == 'fashion':
		for i in range(len(image)):
			image[i] = (image[i] - means[0])/stds[0]
	elif(dataset=='cifar10'):
		count = 0
		tmp = np.zeros(3072)
		for i in range(1024):
			tmp[count] = (image[count] - means[0])/stds[0]
			count = count + 1
			tmp[count] = (image[count] - means[1])/stds[1]
			count = count + 1
			tmp[count] = (image[count] - means[2])/stds[2]
			count = count + 1

		
		is_gpupoly = (domain=='gpupoly' or domain=='refinegpupoly')
		if is_conv and not is_gpupoly:
			for i in range(3072):
				image[i] = tmp[i]
			#for i in range(1024):
			#	image[i*3] = tmp[i]
			#	image[i*3+1] = tmp[i+1024]
			#	image[i*3+2] = tmp[i+2048]
		else:
			count = 0
			for i in range(1024):
				image[i] = tmp[count]
				count = count+1
				image[i+1024] = tmp[count]
				count = count+1
				image[i+2048] = tmp[count]
				count = count+1

def verify_template(analyzer, element, nlb, nub, label):
		dominant_class = get_dominant_class(analyzer, element, nlb, nub, label)

		print(nlb[-1])
		print('dominant class and label:', dominant_class, label)

		if dominant_class == label:
			return True
		else:
			return False

# Should be refactored with the overlapping code in analyzer.analyze			
def get_dominant_class(analyzer, element, nlb, nub, label):
	output_size = 0

	output_size = analyzer.ir_list[-1].output_length #reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)

	dominant_class = -1

	label_failed = []
	x = None
	   
	candidate_labels = []
	if label == -1:
		for i in range(output_size):
			candidate_labels.append(i)
	else:
		candidate_labels.append(analyzer.label)
	adv_labels = []

	if analyzer.prop == -1:
		for i in range(output_size):
			# print("Bub ", i)
			adv_labels.append(i)
	else:
		adv_labels.append(analyzer.prop)   

	for i in candidate_labels:
		flag = True
		label = i
		# print("Maims", adv_labels)

		for j in adv_labels:
			if label!=j and not analyzer.is_greater(analyzer.man, element, label, j, analyzer.use_default_heuristic):
				#linexpr = get_output_uexpr_defined_over_previous_layers(self.man, element, 7, 0)
				#elina_linexpr0_print(linexpr,None)				
				if(analyzer.domain=='refinepoly'):
					obj = LinExpr()
					obj += 1*var_list[counter+label]
					obj += -1*var_list[counter + j]
					model.setObjective(obj,GRB.MINIMIZE)
					if config.complete == True:
						model.optimize(milp_callback)
						if not hasattr(model,"objbound") or model.objbound <= 0:
							flag = False
							if analyzer.label!=-1:
								label_failed.append(j)
							if model.solcount > 0:
								x = model.x[0:len(analyzer.nn.specLB)]
							break	
					else:
						model.optimize()
						print("objval ", j, model.objval)
						if model.Status!=2:
							print("model was not successful status is", model.Status)
							model.write("final.mps")
							flag = False
							break
						elif model.objval < 0:
				   
							flag = False
							if model.objval != math.inf:
								x = model.x[0:len(analyzer.nn.specLB)]
							break

				else:
					flag = False
					if analyzer.label!=-1:
						label_failed.append(j)
					if config.complete == False:
						break


		if flag:
			dominant_class = i
			break

	return dominant_class

def get_alpha_max(lb, ub):
	return 100

def expand_last_layer_by_factor(alpha, nlb, nub):
	nlb_tm = copy.deepcopy(nlb)
	nub_tm = copy.deepcopy(nub)

	alpha_prime = 1 + ((alpha-1)/2)

	for i in range(len(nlb[-1])):
		nlb_tm[-1][i] = nlb[-1][i] + alpha_prime*(nub[-1][i]-nlb[-1][i])
		nub_tm[-1][i] = nub[-1][i] - alpha_prime*(nub[-1][i]-nlb[-1][i])
	
	return nlb_tm, nub_tm	


# TODO: backsubstitution can make this unsound
def get_deeppoly_template_at_layer(model, k, specLB, specUB, label, prop):
	optimizer = get_optimizer(model)
	nn = layers()
	execute_list, output_info = optimizer.get_deeppoly(nn, specLB, specUB, None, None, None, None, None, None, 0, None)
	analyzer = Analyzer(execute_list, nn, 'deeppoly', False, None, None, False, label, prop, False)

	element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(3)

	# perform binary search to expand nlb and nub
	alpha_min = 1
	alpha_max = get_alpha_max(nlb[-1], nub[-1])

	least_err = 0.001
	while(alpha_max - alpha_min > least_err):
		alpha_mid = (alpha_min + alpha_max)/2
		print("alpha mid value:", alpha_mid)

		nlb_tm, nub_tm = expand_last_layer_by_factor(alpha_mid, nlb, nub)

		print("nlb ", nlb[-1])
		print("nlb_tm ", nlb_tm[-1])
		print(nlb.shas)

		# two things
		# nlb_tm had no effect on the analysis at all
		# the expand by factor function is buggy too

		element2, man2, nlb2, nub2 = analyzer.get_abstract0_from_layer(3, element, nlb_tm, nub_tm, nn)


		is_verified = verify_template(analyzer, element2, nlb2, nub2, label)

		if is_verified :
			alpha_min = alpha_mid
		else :
			alpha_max = alpha_mid	

	return element, nlb, nub

all_lbi = []
all_ubi = []

def get_box_template_at_layer(model, k, specLB, specUB, label, prop):
	nn = init_nn(model, specLB, specUB)

	element, man = analyze_with_box_till_layer(k, nn, specLB, specUB, [], [], label)

	lbi, ubi = get_box_from_element(man, element)

	all_lbi.append(lbi)
	all_ubi.append(ubi)

	# perform binary search to expand nlb and nub
	least_err = 0.001
	
	dims = elina_abstract0_dimension(man,element)
	
	dim_cnt = dims.intdim + dims.realdim
	alphas = []
	element_dm = element

	for dim_val in range(0, dim_cnt):

		element_tm = None

		alpha_min = 0
		alpha_max = 100

		while(alpha_max - alpha_min > least_err):
			alpha_mid = (alpha_min + alpha_max)/2
			# print("alpha mid value:", alpha_mid)

			element_tm = elina_box_expand_dim(man, element_dm, alpha_mid, dim_val)
			nn.ffn_counter = 0
			is_verified = analyze_with_box_from_layer(k, nn, element_tm, man, specLB, specUB, [], [], label)

			if is_verified :
				alpha_min = alpha_mid
			else :
				alpha_max = alpha_mid	

		# fin_alpha = 0.9*alpha_min
		fin_alpha = 0.8*alpha_min		
		#copy		
		element_dm = elina_box_expand_dim(man, element_dm, fin_alpha, dim_val)

		alphas.append(fin_alpha)		
		

	element_tm = elina_box_expand_dim(man, element, 0, 0)
	# lbi, ubi = get_box_from_element(man, element)


	is_verified = analyze_with_box_from_layer(k, nn, element_tm, man, specLB, specUB, [], [], label)		
	print('For no expand: ' ,is_verified)

	print('Final alpha value: ', alphas)		
	# libc = CDLL(find_library('c'))
	# cstdout = c_void_p.in_dll(libc, 'stdout')
	# elina_box_fprint(cstdout, man, element, None)
	# elina_box_fprint(cstdout, man, elina_box_expand(man, element, alpha_min), None)

	global alpha_sum
	global alpha_nz
	alpha_sum += alpha_min
	alpha_nz += (alpha_min != 1)

	return element_dm, man

def get_box_template_at_layer_rand_search(model, k, specLB, specUB, label, prop):
	nn = init_nn(model, specLB, specUB)

	element, man = analyze_with_box_till_layer(k, nn, specLB, specUB, [], [], label)

	lbi, ubi = get_box_from_element(man, element)

	all_lbi.append(lbi)
	all_ubi.append(ubi)
	
	dims = elina_abstract0_dimension(man,element)
	
	dim_cnt = dims.intdim + dims.realdim
	alphas = []

	#copy
	element_dm = elina_box_expand_dim(man, element, 0, 0)

	fail_update = 0

	max_fail_update = 100
	delta_change = 0.5

	alphas = [[0,0] for i in range(10)]

	while(fail_update < max_fail_update):

		dim_val = random.randint(0,9)
		side = random.randint(0,1)
		
		# print(dim_val)
		# print_box(element_dm, man)

		element_tm = elina_box_expand_dim_one_dir(man, element_dm, delta_change, dim_val, side)

		# print("Changed to:")
		# print_box(element_tm, man)

		# After thsi element_tm is modified
		is_verified = analyze_with_box_from_layer(k, nn, element_tm, man, specLB, specUB, [], [], label)


		if(is_verified):
			# print_box(element_tm, man)
			#copy		
			element_dm = elina_box_expand_dim_one_dir(man, element_dm, delta_change, dim_val, side)
			# print_box(element_dm, man)
			alphas[dim_val][side] += delta_change
			# print(alphas)
		else:
			fail_update += 1


	is_verified = analyze_with_box_from_layer(k, nn, element, man, specLB, specUB, [], [], label)		
	print('For no expand: ' ,is_verified)

	print('Final alpha value: ', alphas)		
	# libc = CDLL(find_library('c'))
	# cstdout = c_void_p.in_dll(libc, 'stdout')
	# elina_box_fprint(cstdout, man, element, None)
	# elina_box_fprint(cstdout, man, elina_box_expand(man, element, alpha_min), None)

	return element_dm, man


def print_box(element, man):
	libc = CDLL(find_library('c'))
	cstdout = c_void_p.in_dll(libc, 'stdout')
	elina_box_fprint(cstdout, man, element, None)

def get_box_from_element(man, element):
	bounds = elina_abstract0_to_box(man,element)
	lbi = []
	ubi = []
	# print('layerno ',layerno)
	num_out_pixels = 10
	for i in range(num_out_pixels):
	   inf = bounds[i].contents.inf
	   sup = bounds[i].contents.sup
	   #print('i ',i)
	   #elina_interval_fprint(cstdout,bounds[i])
	   #print('[',inf.contents.val.dbl, ',',sup.contents.val.dbl,']')
	   lbi.append(inf.contents.val.dbl)
	   ubi.append(sup.contents.val.dbl)
	return lbi, ubi   

def contain_score(lbi, ubi, lbt, ubt):
	total = len(lbi)
	contain = 0

	for i in range(total):
		if lbt[i]<=lbi[i] and ubi[i]<=ubt[i]:
			contain += 1
		# else:
		# 	print(i, '.', lbt[i],lbi[i],ubi[i],ubt[i])

	return contain		

if __name__ == '__main__':
	main()


# for i, test in enumerate(tests):
		
	# SHUBHAM: JUST DO IT FOR ONE TEST NOW
	# if(i>1):
	#	 break
	# if(i<1):
	# 	continue	

	# image= np.float64(test[1:len(test)])/np.float64(255)
	# specLB = np.copy(image)
	# specUB = np.copy(image)	

	# normalize(specLB, means, stds, dataset)
	# normalize(specUB, means, stds, dataset)	

	# is_correctly_classified = False

	# Analysis with original image. Maybe we don't run it now?
	# label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
	# label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, 'deeppoly', False, None, False)
	# approx_optimizer = get_optimizer(approx_model)

	# specLB = np.reshape(specLB, (-1,))
	# specUB = np.reshape(specUB, (-1,))
	
	# approx_nn = layers()
	# approx_nn.specLB = specLB
	# approx_nn.specUB = specUB

	# Finding the actual label
	# label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB,'deeppoly', None, None, False)
	
	# specLB = np.clip(image - epsilon,0,1)
	# specUB = np.clip(image + epsilon,0,1)

	# print("concrete ", nlb[-1])

	# if label == int(test[0]):
	# 	is_correctly_classified = True

		
	# nn = layers()
	# # label = 2 # actually find this using separate run
	# prop = -1 # target	

	# optimizer = get_optimizer(model)
	# print('This network has ' + str(optimizer.get_neuron_count()) + ' neurons.')

	# execute_list, output_info = optimizer.get_deeppoly(nn, np.clip(image - epsilon,0,1), np.clip(image + epsilon,0,1), None, None, None, None, None, None, 0, None)
	# analyzer = Analyzer(execute_list, nn, 'deeppoly', False, None, None, False, label, prop, False)
	# # dominant_class, nlb, nub, failed_labels, x = analyzer.analyze()

	# dominant_class, nlb, nub, failed_labels, x = analyzer.analyze()
	# print("nlb ", nlb[-1])
	# print("nub ", nub[-1])

	# save_element(element, man, 'abstract_element.txt')
	# element2 = load_element(man, 'abstract_element2.txt', 'abstract_element.txt')

	# Get the analyzer for the approximate model
	# approx_optimizer = get_optimizer(approx_model)
	# approx_execute_list, approx_output_info = approx_optimizer.get_deeppoly(approx_nn, np.clip(image - epsilon,0,1), np.clip(image + epsilon,0,1), None, None, None, None, None, None, 0, None)
	# approx_analyzer = Analyzer(approx_execute_list, approx_nn, 'deeppoly', False, None, None, False, label, None, False)

	# element, man, nlb, nub, nn = analyzer.get_abstract0_at_layer(3)

	# # save_element(element, man, 'abstract_element.txt')
	# # element2 = load_element(man, 'abstract_element2.txt', 'abstract_element.txt')

	# element, man, nlb, nub = approx_analyzer.get_abstract0_from_layer(3, element2, nlb, nub, nn)

	
	# element, nlb, nub = get_template_at_layer(model, k, specLB, specUB, label, prop)
		
	# is_template_verified = verify_template(analyzer, element, nlb, nub, label)		
		
	# if is_template_verified:
	# 	approx_optimizer = get_optimizer(approx_model)
	# 	approx_execute_list, approx_output_info = approx_optimizer.get_deeppoly(approx_nn, np.clip(image - epsilon,0,1), np.clip(image + epsilon,0,1), None, None, None, None, None, None, 0, None)
	# 	approx_analyzer = Analyzer(approx_execute_list, approx_nn, 'deeppoly', False, None, None, False, label, None, False)
	# 	element_ap, man_ap, nlb_ap, nub_ap, nn_ap = approx_analyzer.get_abstract0_at_layer(3)

	# 	is_contained = check_contains(nlb_ap[-1], nub_ap[-1], nlb[-1], nub[-1])
	# 	if is_contained:
	# 		print('approx model verified using the template')

	# start = time.time()

	# perturbed_label, _, nlb, nub,failed_labels, x = eran.analyze_box(specLB, specUB, 'deeppoly', False, None, False, label=label, prop=prop)
#	 print("nlb ", nlb[-1], " nub ", nub[-1])
# print(verified_cnt,'/',total,' images verified!')   