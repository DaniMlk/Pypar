import numpy
import math
import pyparticleest.models.nlg as nlg
import pyparticleest.simulator as simulator
import matplotlib.pyplot as plt
import pdb
import scipy.special
import random 
import networkx as nx
from itertools import islice
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
import datetime
import os
import scipy.io
# from sklearn.metrics import mean_squared_error
nodes = 10
l_dataset = 20 

now = datetime.datetime.now()
if not os.path.exists("date_"+str(now)):
	os.makedirs("date_"+str(now))

def k_shortest_paths(G, source, target, k, weight=None):
	return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

dict = {}
# pos = {i: (numpy.random.uniform(), numpy.random.uniform()) for i in range(nodes)}
# pos = {i: (numpy.random.poisson(), numpy.random.poisson()) for i in range(nodes)}
# pdb.set_trace()
# G = nx.random_geometric_graph(nodes,.5,dim=2,pos=pos)
G = nx.random_geometric_graph(nodes,.85)
adj = nx.adjacency_matrix(G)
scipy.io.savemat('./Graph.mat',{'adj':adj})
# pdb.set_trace()

pos=nx.get_node_attributes(G,'pos')




dmin=1
ncenter=0
for n in pos:
    x,y=pos[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin:
        ncenter=n
        dmin=d

# color by path length from node near center
p=nx.single_source_shortest_path_length(G,ncenter)
# pdb.set_trace()
plt.figure(figsize=(8,8))
nx.draw_networkx_edges(G,pos,nodelist=[ncenter],alpha=1,label="Number of Nodes "+str(nodes))
nx.draw_networkx_nodes(G,pos,
                       node_size=80,
                       node_color= p.values(),
                       cmap=plt.cm.Reds_r)

# pdb.set_trace()
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.legend()
plt.axis('on')
plt.savefig('./date_'+str(now)+'/random_geometric_graph_'+str(nodes)+'.svg')
# plt.show()



# nx.draw_networkx(G,pos=pos,with_labels=False)  
# plt.show()
plt.close()
# pdb.set_trace()
p = 0
for i in range(nodes):
	for j in range(nodes):
		if i != j:
			try:
				path = k_shortest_paths(G, i, j, 5)
				dict.update({p:path}) # t is defined here; t= k_shortest_path
			except nx.exception.NetworkXNoPath:
				print(p,"not found")
		p += 1

def generate_dataset(steps, P0, Q, R, l_dataset,h_dataset, h_counter):
	h = numpy.zeros((nodes,steps + 1,))
	y = numpy.zeros((nodes,steps + 1,))
	h[0] = numpy.random.multivariate_normal((0.0,), P0)
	y[0] = (0.05 * h[0] ** 2 + numpy.random.multivariate_normal((0.0,), R))
	for k in range(0, steps):
		for i in range(nodes):
			# pdb.set_trace()
			# sigma_1 = (sum(x[i,step] for step in range(l)))
			# x[i,k+1] = scipy.special.jn(0,2*math.pi*l*10000)* sigma_1 + 8 * math.cos(1.2 * k) + numpy.random.multivariate_normal((0.0,), Q)
			# x[i,k+1] = scipy.special.jn(0,2*math.pi*l*10000)* sigma_1 + 8 * math.cos(1.2 * k) + numpy.random.multivariate_normal((0.0,), Q)
			# x[i,k+1] = scipy.special.jn(0,2*math.pi*l*.4)* sigma_1 + numpy.random.multivariate_normal((0.0,), Q)
			for ll in range(1,l_dataset):
				h[i,k+1] = scipy.special.jn(0,2*math.pi*ll*.4) * h[i,ll] +2 * math.cos(1.2 * k)+ .4 * numpy.random.multivariate_normal((0.0,),Q)
			# pdb.set_trace()
			if h_dataset[h_counter] ==True:
				y[i,k + 1] = (0.05 * h[i,k + 1] ** 2 + numpy.random.multivariate_normal((0.0,), R))
			else:
				y[i,k + 1] = numpy.random.multivariate_normal((0.0,), R)
	return (h, y)

class StdNonLin(nlg.NonlinearGaussianInitialGaussian):
	# x_{k+1} = 0.5*x_k + 25.0*x_k/(1+x_k**2) +
	#           8*math.cos(1.2*k) + v_k = f(x_k) + v:
	# y_k = 0.05*x_k**2 + e_k = g(x_k) + e_k,
	# x(0) ~ N(0,P0), v_k ~ N(0,Q), e_k ~ N(0,R)

	def __init__(self, P0, Q, R, l, l_counter):
	# Set covariances in the constructor since they
	# are constant
		super(StdNonLin, self).__init__(Px0=P0, Q=Q, R=R)

	def calc_g(self, particles, t):
	# Calculate value of g(\xi_t,t)
		return 0.05 * particles ** 2

	def calc_f(self, particles, u, t):
	# Calculate value of f(xi_t,t)
		for ll in range(1,l[l_counter]):
			output = scipy.special.jn(0,2*math.pi*ll*.4) * particles + 2 * math.cos(1.2 * t)
		return output

T = 100
P0 = 5.0 * numpy.eye(1)
Q = 1 * numpy.eye(1)
R = .1 * numpy.eye(1)

# Forward particles
N = 200
# Backward trajectories
M = 50
# model = StdNonLin(P0, Q, R)
h_dataset = [True, False]  #Option to change
for h_counter in range(len(h_dataset)):
	numpy.random.seed(0)
	(x, y) = generate_dataset(T, P0, Q, R, l_dataset,h_dataset, h_counter)
	scipy.io.savemat('./h.mat',{'x':x})
	scipy.io.savemat('./y.mat',{'y':y})
	l = [5,10]  # Option to change
	for l_counter in range(len(l)):
		first_time = 0
		error_x_y = abs(x-y) ################error between x and y #####################
		for t in range(1,len(dict)):
			print("counter = {}/{}".format(t,len(dict)))
			# mean_smooth_tmp_all_routes= []
			error_estimation_all_routes=[]
			# mean_smooth_min_all_routes=[]
			# mean_smooth_max_all_routes=[]
			# mean_smooth_mean_all_routes=[]
			# mean_filt_tmp_all_routes=[]
			# error_estimation_node_all_routes=[]

			best = numpy.zeros((len(dict[t])))
			mean_smooth_min = numpy.zeros((len(dict[t]),T+1,1))
			mean_smooth_max = numpy.zeros((len(dict[t]),T+1,1))
			mean_smooth_mean = numpy.zeros((len(dict[t]),T+1,1))

			mean_filt_min = numpy.zeros((len(dict[t]),T+1,1))
			mean_filt_max = numpy.zeros((len(dict[t]),T+1,1))
			mean_filt_mean = numpy.zeros((len(dict[t]),T+1,1))
			counter_i = 0
			for i in dict[t]:
				mean_smooth_tmp = numpy.zeros((len(i),T+1,1))
				mean_filt_tmp = numpy.zeros((len(i),T+1,1))
				x_mean = numpy.zeros((len(i),T+1))
				counter = 0
				for j in i:
					model = StdNonLin(P0, Q, R, l, l_counter)
					sim = simulator.Simulator(model, u=None, y=y[j])
					sim.simulate(N, M, filter='PF', smoother='rsas', meas_first=True)

					(est_filt, w_filt) = sim.get_filtered_estimates()

					mean_filt_tmp[counter] = sim.get_filtered_mean()

					est_smooth = sim.get_smoothed_estimates()
					mean_smooth_tmp[counter] = sim.get_smoothed_mean()
					x_mean[counter] = x[j]
					counter += 1
					# pdb.set_trace()
					error_estimation_node = numpy.sqrt((x[j] - mean_smooth_tmp[counter-1].reshape(T+1))**2)
					# error_estimation_node = error_estimation_node/numpy.max(error_estimation_node)
					plt.plot(range(T+1),error_estimation_node,label='Error Estimation of representaion node'+str(counter))
					plt.legend()
					plt.xlabel("Number of Nodes")
					plt.ylabel("Mean Squared Error")
					# plt.show()
					# pdb.set_trace()
					counter
					plt.savefig('./date_'+str(now)+'/Err_sd_'+str(t)+'node_'+str(counter)+'_'+str(j)+'_of_'+str(counter_i)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.svg')
					plt.close()
					# scipy.io.savemat('./err_node_'+str(counter)+'_'+str(j)+'_'+str(counter_i)+'.mat',{'error_estimation_node':error_estimation_node})
					# pdb.set_trace()
					if first_time ==0:
						error_estimation_node_all_routes = error_estimation_node
					else:
						error_estimation_node_all_routes = numpy.append(error_estimation_node_all_routes,error_estimation_node,axis=0)
				error_estimation = numpy.sqrt((x_mean - mean_smooth_tmp[:,:,0])**2).mean(axis=0)
				error_estimation = error_estimation/numpy.max(error_estimation)
				plt.plot(range(T+1),error_estimation, label = 'Error Estimation of representaion Route '+str(counter_i))
				plt.legend()
				# plt.show()
				plt.ylabel("Mean Squared Error")
				plt.xlabel("Number of Nodes")
				plt.ylim(0,5)
				plt.savefig('./date_'+str(now)+'/Err_sd_'+str(t)+'Route_'+str(counter_i)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.svg')
				plt.close()
				# scipy.io.savemat('./err_route'+str(counter_i)+'.mat',{'error_estimation':error_estimation})
				if first_time == 0:
					error_estimation_all_routes = error_estimation
				else:
					error_estimation_all_routes = numpy.append(error_estimation_all_routes,error_estimation,axis=0)
				for e in range(len(i)):
					place = e + 1
					plt.subplot(len(i),1,place)
					plt.plot(range(T+1),x_mean[e], label='Real Value')
					plt.plot(range(T+1),mean_smooth_tmp[e],label='Estimation')
					for tt in range(1, T + 1):
						plt.plot((tt,) * N, est_filt[tt, :, 0].ravel(),'k.', markersize=0.5)
				# scipy.io.savemat('./estimation.mat',{'mean_smooth_tmp':mean_smooth_tmp})
				# pdb.set_trace()
				if first_time ==0:
					mean_smooth_tmp_all_routes = mean_smooth_tmp
				else:
					mean_smooth_tmp_all_routes = numpy.append(mean_smooth_tmp_all_routes,mean_smooth_tmp,axis=0)
				# scipy.io.savemat('./mean_filt.mat',{'mean_filt_tmp':mean_filt_tmp})
				if first_time ==0:
					mean_filt_tmp_all_routes = mean_filt_tmp
				else:
					mean_filt_tmp_all_routes = numpy.append(mean_filt_tmp_all_routes,mean_filt_tmp,axis=0)

				plt.legend()
				# plt.show()
				plt.xlabel("Number of Nodes")
				plt.ylabel("Interference")
				plt.savefig('./date_'+str(now)+'/Estimation_sd_'+str(t)+'_'+str(counter_i)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.svg')
				plt.close()
				mean_smooth_min[counter_i] = numpy.min(mean_smooth_tmp,axis=0)
				plt.plot(range(T+1),mean_smooth_min[counter_i],label='Min_nodes_in_route_'+str(counter_i))
				plt.legend()
				plt.xlabel("Number of Nodes")
				plt.ylabel("Min on smoothed")
				plt.savefig('./date_'+str(now)+'/Min_of_'+str(t)+'_nodes_in_route_'+str(counter_i)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.svg')
				plt.close()
				# scipy.io.savemat('./Min_of_nodes_in_route_'+str(counter_i)+'.mat',{'minn_smooth_min':mean_smooth_min[counter_i]})
				if first_time ==0:
					mean_smooth_min_all_routes = mean_smooth_min[counter_i]
				else:
					mean_smooth_min_all_routes = numpy.append(mean_smooth_min_all_routes,mean_smooth_min[counter_i],axis=0)
				mean_smooth_max[counter_i] = numpy.max(mean_smooth_tmp,axis=0)
				plt.plot(range(T+1),mean_smooth_max[counter_i],label='Max_nodes_in_route_'+str(counter_i))
				plt.legend()
				plt.xlabel("Number of Nodes")
				plt.ylabel("Max on smooothed")
				plt.savefig('./date_'+str(now)+'/Max_sd'+str(t)+'_of_nodes_in_route_'+str(counter_i)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.svg')
				plt.close()
				# scipy.io.savemat('./Mean_of_nodes_in_route_'+str(counter_i)+'.mat',{'max_smooth_min':mean_smooth_max[counter_i]})
				if first_time ==0:
					mean_smooth_max_all_routes = mean_smooth_max[counter_i]
				else:
					mean_smooth_max_all_routes = numpy.append(mean_smooth_max_all_routes,mean_smooth_max[counter_i],axis=0)
				mean_smooth_mean[counter_i] = numpy.mean(mean_smooth_tmp,axis=0)
				plt.plot(range(T+1),mean_smooth_mean[counter_i],label='Mean_nodes_in_route_'+str(counter_i))
				plt.legend()
				plt.xlabel("Number of Nodes")
				plt.ylabel("Mean on smoothed")
				plt.savefig('./date_'+str(now)+'/Mean_sd_'+str(t)+'of_nodes_in_route_'+str(counter_i)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.svg')
				plt.close()
				# scipy.io.savemat('./Mean_of_nodes_in_route_'+str(counter_i)+'.mat',{'mean_smooth_mean':mean_smooth_mean[counter_i]})
				if first_time == 0:
					mean_smooth_mean_all_routes = mean_smooth_mean[counter_i]
				else:
					mean_smooth_mean_all_routes = numpy.append(mean_smooth_mean_all_routes,mean_smooth_mean[counter_i],axis=1)
				# pdb.set_trace()
				x_mean_total = numpy.mean(x_mean,axis=0)

				mean_filt_min[counter_i] = numpy.min(mean_filt_tmp,axis=0)
				mean_filt_max[counter_i] = numpy.max(mean_filt_tmp,axis=0)
				mean_filt_mean[counter_i] = numpy.mean(mean_filt_tmp,axis=0)

				counter_i = counter_i + 1

				first_time = 1
			# mean_smooth_tmp_all_routes.reshape(T+1,-1)
			scipy.io.savemat('./date_'+str(now)+'/estimation'+str(t)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.mat',{'mean_smooth_tmp_all_routes':mean_smooth_tmp_all_routes})
			scipy.io.savemat('./date_'+str(now)+'/err_route'+str(t)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.mat',{'error_estimation_all_routes':error_estimation_all_routes})
			scipy.io.savemat('./date_'+str(now)+'/Min_of_nodes_in_route_'+str(t)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.mat',{'min_smooth_min_all_routes':mean_smooth_min_all_routes})
			scipy.io.savemat('./date_'+str(now)+'/Mean_of_nodes_in_route_'+str(t)+'_l'+str(l[l_counter])+'.mat',{'max_smooth_min':mean_smooth_max_all_routes})
			scipy.io.savemat('./date_'+str(now)+'/mean_filt'+str(t)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.mat',{'mean_filt_tmp':mean_filt_tmp_all_routes})
			scipy.io.savemat('./date_'+str(now)+'/err_node_'+str(t)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.mat',{'error_estimation_node':error_estimation_node_all_routes})
			
			mu = numpy.zeros((counter_i))
			std = numpy.zeros((counter_i))
			for i in range(counter_i):
				mu[i], std[i] = norm.fit(mean_smooth_min[i])
			std_min_arg = numpy.argmin(std,axis=0)
			mean_smooth_min_arg = numpy.argmin(mean_smooth_min,axis=0)
			mean_smooth_min_hop = numpy.min(mean_smooth_min,axis=0)
			mu_hop, std_hop = norm.fit(mean_smooth_min_hop)

			mean_smooth_max_arg = numpy.argmax(mean_smooth_max,axis=0)

			mean_filt_min_arg = numpy.argmin(mean_filt_min,axis=0)
			mean_filt_max_arg = numpy.argmax(mean_filt_max,axis=0)


			plot_test = numpy.zeros((len(dict[t]),T+1))
			for i in range(T+1):
				plot_test[mean_smooth_min_arg[i],i]=1
			for i in range(len(dict[t])):
				best[i] = numpy.sum(plot_test[i])
			plot_test = plot_test.astype(int)
			for e in range(len(dict[t])):
				if(numpy.argmax(best)==e):
					place = e + 1
					plt.subplot(len(dict[t]),1,place)
					plt.step(range(T+1),plot_test[e],label='hop_to_hop_selection')
				else:
					place = e + 1
					plt.subplot(len(dict[t]),1,place)
					plt.step(range(T+1),plot_test[e],color='green',label='hop_to_hop_selection')

			scipy.io.savemat('./hop_sd_'+str(t)+'.mat',{'plot_test':plot_test})
			# plt.show()
			plt.legend()
			plt.xlabel("Number of Nodes")
			plt.ylabel("Jump")
			plt.savefig('./date_'+str(now)+'/Selection_Route_sd_'+str(t)+'_'+str(counter_i)+'_l'+str(l[l_counter])+'_h'+str(h_dataset[h_counter])+'.svg')

			# plt.plot(range(51),mean_smooth_min_arg)
			# plt.show()
			# for i in range(51):
			# 	read_argmin = mean_smooth_min[mean_smooth_min_arg[i],...]

				# pdb.set_trace()
				# plt.plot(range(T + 1), x_mean_total, 'r-', linewidth=2.0, label='True')
				# plt.plot((0,) * N, est_filt[0, :, 0].ravel(), 'k.',
				# 		markersize=0.5, label='Particles')
				# for t in xrange(1, T + 1):
				# 	plt.plot((t,) * N, est_filt[t, :, 0].ravel(),
				# 			 'k.', markersize=0.5)

				# plt.plot(range(T + 1), mean_filt_min[:, 0], 'g--',
				# 		 linewidth=2.0, label='Filter mean_min')
				# plt.plot(range(T + 1), mean_filt_max[:, 0], 'b--',
				# 		 linewidth=2.0, label='Filter mean_max')
				# plt.plot(range(T + 1), mean_filt_mean[:, 0], 'm--',
				# 		 linewidth=2.0, label='Filter mean_mean')


				# # plt.plot(range(T + 1), mean_smooth_min[:, 0], 'b--',
				# # 		 linewidth=2.0, label='Smoother mean')
				# # plt.plot(range(T + 1), mean_smooth_max[:, 0], 'b:',
				# # 		 linewidth=2.0, label='Smoother mean')
				# # plt.plot(range(T + 1), mean_smooth_mean[:, 0], 'b-.',
				# # 		 linewidth=2.0, label='Smoother mean')

				# plt.xlabel('t')
				# plt.ylabel('x')
				# plt.legend()
				# plt.show()