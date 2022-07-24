import sys
sys.path.insert(0,'cnm-main')
sys.path.insert(0,'cnm-main/examples')
from cnm import Clustering, TransitionProperties, Propagation
import numpy as np
from helper import create_roessler_data
from sklearn.cluster import KMeans

def run_wormpose():

	# CNM parameters:
	# ---------------
	K = 100 # Number of clusters
	L = 20 # Model order

	# Create the Lorenz data
	case_data = np.load('../Wormpose/data/opensource_data/x_all.npy') # (12, 33600, 5)
	#data, dt = case_data['data'], case_data['dt']
	data = case_data[0] # (n_time, n_dim)
	dt = 1 / 32
	t = np.arange(data.shape[0]) * dt

	# Clustering
	# ----------
	cluster_config = {
			'data': data,
			'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10),
			'dataset': 'wormpose',
			}

	clustering = Clustering(**cluster_config)

	# Transition properties
	# ---------------------
	transition_config = {
			'clustering': clustering,
			'dt': dt,
			'K': K,
			'L': L,
			}

	transition_properties = TransitionProperties(**transition_config)

	# Propagation
	# -----------
	propagation_config = {
			'transition_properties': transition_properties,
			}

	ic = 1        # Index of the centroid to start in
	t_total = 300 # Total simulation time
	dt_hat = dt   # To spline-interpolate the centroid-to-centroid trajectory

	propagation = Propagation(**propagation_config)
	t_hat, x_hat = propagation.run(t_total,ic,dt_hat)

	# Plot the results
	# ----------------
	from helper import (plot_phase_space, plot_time_series,plot_cpd,
						plot_autocorrelation)

	# phase space
	n_dim = 5
	plot_phase_space(data,clustering.centroids,clustering.labels,n_dim=n_dim)

	# time series
	time_range = (0,30)
	n_dim = 5
	plot_label = ['e_1','e_2','e_3','e_4','e_5']
	plot_time_series(t,data,t_hat,x_hat,time_range,plot_label,n_dim=n_dim)

	# cluster probability distribution
	plot_cpd(data,x_hat)

	# autocorrelation function
	time_blocks = t_hat[-1]
	time_range = [0,30]
	#method = 'dot'
	plot_autocorrelation(t,data,t_hat,x_hat,time_blocks,time_range)

if __name__== '__main__':
	run_wormpose()
