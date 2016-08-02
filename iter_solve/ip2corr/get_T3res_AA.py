
# coding: utf-8

# AA MSM from Qhel directly via rama-plot
# -------
# with nearest neighbor correlations
# ====

# In[1]:

import pyemma
pyemma.__version__

import numpy as np

# In[2]:

import os
#get_ipython().magic(u'pylab inline')
#matplotlib.rcParams.update({'font.size': 12})


# In[3]:

import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import msmbuilder
from msmbuilder.msm.ratematrix import ContinuousTimeMSM
import scipy
from msmtools.analysis.dense.decomposition import eigenvectors, eigenvalues
import operator


# Read in the dtrajs
# ------

# In[4]:

indir = '/usr/data/rudzinski/Systems/Biomolecules/ALA7/AA/'

dtraj_rama_2 = []
dtraj_rama_3 = []
dtraj_rama_4 = []
dtraj_rama_5 = []
dtraj_rama_6 = []
dtraj_rama_2.append( np.genfromtxt(indir+'ala7_phipsi_1ps.dat')[:,0:2] )
dtraj_rama_3.append( np.genfromtxt(indir+'ala7_phipsi_1ps.dat')[:,2:4] )
dtraj_rama_4.append( np.genfromtxt(indir+'ala7_phipsi_1ps.dat')[:,4:6] )
dtraj_rama_5.append( np.genfromtxt(indir+'ala7_phipsi_1ps.dat')[:,6:8] )
dtraj_rama_6.append( np.genfromtxt(indir+'ala7_phipsi_1ps.dat')[:,8:10] )


# In[5]:

for i in range( len(dtraj_rama_2) ):
    dtraj_rama_2[i][np.where(dtraj_rama_2[i][:,1] < -125)[0],1] += 360  
    dtraj_rama_3[i][np.where(dtraj_rama_3[i][:,1] < -125)[0],1] += 360
    dtraj_rama_4[i][np.where(dtraj_rama_4[i][:,1] < -125)[0],1] += 360
    dtraj_rama_5[i][np.where(dtraj_rama_5[i][:,1] < -125)[0],1] += 360
    dtraj_rama_6[i][np.where(dtraj_rama_6[i][:,1] < -125)[0],1] += 360


# In[6]:

dtraj_phi_2 = []
dtraj_phi_3 = []
dtraj_phi_4 = []
dtraj_phi_5 = []
dtraj_phi_6 = []
for i in range( len(dtraj_rama_2) ):
    dtraj_phi_2.append(dtraj_rama_2[i][:,1])
    dtraj_phi_3.append(dtraj_rama_3[i][:,1])
    dtraj_phi_4.append(dtraj_rama_4[i][:,1])
    dtraj_phi_5.append(dtraj_rama_5[i][:,1])
    dtraj_phi_6.append(dtraj_rama_6[i][:,1])


# **simple clustering along psi only for discretization**

# In[7]:

n_clusters = 2     # number of k-means clusters


# In[8]:

clustering_rama_2 = coor.cluster_kmeans(dtraj_phi_2,k=n_clusters,max_iter=100, tolerance=1e-12, fixed_seed=True)
clustering_rama_3 = coor.cluster_kmeans(dtraj_phi_3,k=n_clusters,max_iter=100, tolerance=1e-12, fixed_seed=True)
clustering_rama_4 = coor.cluster_kmeans(dtraj_phi_4,k=n_clusters,max_iter=100, tolerance=1e-12, fixed_seed=True)
clustering_rama_5 = coor.cluster_kmeans(dtraj_phi_5,k=n_clusters,max_iter=100, tolerance=1e-12, fixed_seed=True)
clustering_rama_6 = coor.cluster_kmeans(dtraj_phi_6,k=n_clusters,max_iter=100, tolerance=1e-12, fixed_seed=True)


# In[9]:

cc_rama_2 = clustering_rama_2.clustercenters[:,0]
cc_rama_3 = clustering_rama_3.clustercenters[:,0]
cc_rama_4 = clustering_rama_4.clustercenters[:,0]
cc_rama_5 = clustering_rama_5.clustercenters[:,0]
cc_rama_6 = clustering_rama_6.clustercenters[:,0]


# In[10]:

dtrajs_rama_2 = clustering_rama_2.dtrajs
dtrajs_rama_3 = clustering_rama_3.dtrajs
dtrajs_rama_4 = clustering_rama_4.dtrajs
dtrajs_rama_5 = clustering_rama_5.dtrajs
dtrajs_rama_6 = clustering_rama_6.dtrajs


# In[11]:

if ( dtrajs_rama_2[0][np.where(dtraj_phi_2[0] < 0)[0][0]] != 0 ):
    dtrajs_rama_2[0][np.where(dtrajs_rama_2[0] == 0)[0]] -= 1
    dtrajs_rama_2[0][np.where(dtrajs_rama_2[0] == 1)[0]] -= 1
    dtrajs_rama_2[0][np.where(dtrajs_rama_2[0] == -1)[0]] += 2
if ( dtrajs_rama_3[0][np.where(dtraj_phi_3[0] < 0)[0][0]] != 0 ):
    dtrajs_rama_3[0][np.where(dtrajs_rama_3[0] == 0)[0]] -= 1
    dtrajs_rama_3[0][np.where(dtrajs_rama_3[0] == 1)[0]] -= 1
    dtrajs_rama_3[0][np.where(dtrajs_rama_3[0] == -1)[0]] += 2
if ( dtrajs_rama_4[0][np.where(dtraj_phi_4[0] < 0)[0][0]] != 0 ):
    dtrajs_rama_4[0][np.where(dtrajs_rama_4[0] == 0)[0]] -= 1
    dtrajs_rama_4[0][np.where(dtrajs_rama_4[0] == 1)[0]] -= 1
    dtrajs_rama_4[0][np.where(dtrajs_rama_4[0] == -1)[0]] += 2
if ( dtrajs_rama_5[0][np.where(dtraj_phi_5[0] < 0)[0][0]] != 0 ):
    dtrajs_rama_5[0][np.where(dtrajs_rama_5[0] == 0)[0]] -= 1
    dtrajs_rama_5[0][np.where(dtrajs_rama_5[0] == 1)[0]] -= 1
    dtrajs_rama_5[0][np.where(dtrajs_rama_5[0] == -1)[0]] += 2
if ( dtrajs_rama_6[0][np.where(dtraj_phi_6[0] < 0)[0][0]] != 0 ):
    dtrajs_rama_6[0][np.where(dtrajs_rama_6[0] == 0)[0]] -= 1
    dtrajs_rama_6[0][np.where(dtrajs_rama_6[0] == 1)[0]] -= 1
    dtrajs_rama_6[0][np.where(dtrajs_rama_6[0] == -1)[0]] += 2


# In[17]:

dtrajs_nnn_234 = []
dtrajs_nnn_345 = []
dtrajs_nnn_456 = []
for i in range( len(dtraj_rama_2) ):
    dtrajs_nnn_234.append( np.vstack( (dtrajs_rama_2[i], dtrajs_rama_3[i], dtrajs_rama_4[i]) ).T )
    dtrajs_nnn_234[i].astype('int64')
    dtrajs_nnn_345.append( np.vstack( (dtrajs_rama_3[i], dtrajs_rama_4[i], dtrajs_rama_5[i]) ).T )
    dtrajs_nnn_345[i].astype('int64')
    dtrajs_nnn_456.append( np.vstack( (dtrajs_rama_4[i], dtrajs_rama_5[i], dtrajs_rama_6[i]) ).T )
    dtrajs_nnn_456[i].astype('int64')


# In[18]:

n_clusters = 8
clustering_nnn_234 = coor.cluster_regspace(dtrajs_nnn_234,max_centers=n_clusters,dmin=0.5)
clustering_nnn_345 = coor.cluster_regspace(dtrajs_nnn_345,max_centers=n_clusters,dmin=0.5)
clustering_nnn_456 = coor.cluster_regspace(dtrajs_nnn_456,max_centers=n_clusters,dmin=0.5)

# In[19]:

dtrajs_1D_234 = clustering_nnn_234.dtrajs
dtrajs_1D_345 = clustering_nnn_345.dtrajs
dtrajs_1D_456 = clustering_nnn_456.dtrajs

# In[20]:

# shift the cluster indices so they are all consistent
cc_234 = clustering_nnn_234.clustercenters[:]
cc_345 = clustering_nnn_345.clustercenters[:]
cc_456 = clustering_nnn_456.clustercenters[:]

# needed more complex sorting method for ipm, m>1
# calculate the number of adjacent and almost adjacent alphas for each state
cc_nn_234 = np.zeros(len(cc_234))
cc_nnn_234 = np.zeros(len(cc_234))
for i in range(len(cc_234)):
    if ( np.where(cc_234[i]==1)[0].shape[0] == 2):
        d1 = np.where(cc_234[i]==1)[0][1] - np.where(cc_234[i]==1)[0][0]
        if (d1==1):
            cc_nn_234[i] += 1
        elif (d1==2):
            cc_nnn_234[i] += 1
    elif ( np.where(cc_234[i]==1)[0].shape[0] == 3):
        d1 = np.where(cc_234[i]==1)[0][1] - np.where(cc_234[i]==1)[0][0]
        d2 = np.where(cc_234[i]==1)[0][2] - np.where(cc_234[i]==1)[0][1]
        if (d1==1):
            cc_nn_234[i] += 1
        elif (d1==2):
            cc_nnn_234[i] += 1
        if (d2==1):
            cc_nn_234[i] += 1
        elif (d2==2):
            cc_nnn_234[i] += 1
    elif ( np.where(cc_234[i]==1)[0].shape[0] == 4):
        d1 = np.where(cc_234[i]==1)[0][1] - np.where(cc_234[i]==1)[0][0]
        d2 = np.where(cc_234[i]==1)[0][2] - np.where(cc_234[i]==1)[0][1]
        d3 = np.where(cc_234[i]==1)[0][3] - np.where(cc_234[i]==1)[0][2]
        if (d1==1):
            cc_nn_234[i] += 1
        elif (d1==2):
            cc_nnn_234[i] += 1
        if (d2==1):
            cc_nn_234[i] += 1
        elif (d2==2):
            cc_nnn_234[i] += 1
        if (d3==1):
            cc_nn_234[i] += 1
        elif (d3==2):
            cc_nnn_234[i] += 1
#
cc_nn_345 = np.zeros(len(cc_345))
cc_nnn_345 = np.zeros(len(cc_345))
for i in range(len(cc_345)):
    if ( np.where(cc_345[i]==1)[0].shape[0] == 2):
        d1 = np.where(cc_345[i]==1)[0][1] - np.where(cc_345[i]==1)[0][0]
        if (d1==1):
            cc_nn_345[i] += 1
        elif (d1==2):
            cc_nnn_345[i] += 1
    elif ( np.where(cc_345[i]==1)[0].shape[0] == 3):
        d1 = np.where(cc_345[i]==1)[0][1] - np.where(cc_345[i]==1)[0][0]
        d2 = np.where(cc_345[i]==1)[0][2] - np.where(cc_345[i]==1)[0][1]
        if (d1==1):
            cc_nn_345[i] += 1
        elif (d1==2):
            cc_nnn_345[i] += 1
        if (d2==1):
            cc_nn_345[i] += 1
        elif (d2==2):
            cc_nnn_345[i] += 1
    elif ( np.where(cc_345[i]==1)[0].shape[0] == 4):
        d1 = np.where(cc_345[i]==1)[0][1] - np.where(cc_345[i]==1)[0][0]
        d2 = np.where(cc_345[i]==1)[0][2] - np.where(cc_345[i]==1)[0][1]
        d3 = np.where(cc_345[i]==1)[0][3] - np.where(cc_345[i]==1)[0][2]
        if (d1==1):
            cc_nn_345[i] += 1
        elif (d1==2):
            cc_nnn_345[i] += 1
        if (d2==1):
            cc_nn_345[i] += 1
        elif (d2==2):
            cc_nnn_345[i] += 1
        if (d3==1):
            cc_nn_345[i] += 1
        elif (d3==2):
            cc_nnn_345[i] += 1
#
cc_nn_456 = np.zeros(len(cc_456))
cc_nnn_456 = np.zeros(len(cc_456))
for i in range(len(cc_456)):
    if ( np.where(cc_456[i]==1)[0].shape[0] == 2):
        d1 = np.where(cc_456[i]==1)[0][1] - np.where(cc_456[i]==1)[0][0]
        if (d1==1):
            cc_nn_456[i] += 1
        elif (d1==2):
            cc_nnn_456[i] += 1
    elif ( np.where(cc_456[i]==1)[0].shape[0] == 3):
        d1 = np.where(cc_456[i]==1)[0][1] - np.where(cc_456[i]==1)[0][0]
        d2 = np.where(cc_456[i]==1)[0][2] - np.where(cc_456[i]==1)[0][1]
        if (d1==1):
            cc_nn_456[i] += 1
        elif (d1==2):
            cc_nnn_456[i] += 1
        if (d2==1):
            cc_nn_456[i] += 1
        elif (d2==2):
            cc_nnn_456[i] += 1
    elif ( np.where(cc_456[i]==1)[0].shape[0] == 4):
        d1 = np.where(cc_456[i]==1)[0][1] - np.where(cc_456[i]==1)[0][0]
        d2 = np.where(cc_456[i]==1)[0][2] - np.where(cc_456[i]==1)[0][1]
        d3 = np.where(cc_456[i]==1)[0][3] - np.where(cc_456[i]==1)[0][2]
        if (d1==1):
            cc_nn_456[i] += 1
        elif (d1==2):
            cc_nnn_456[i] += 1
        if (d2==1):
            cc_nn_456[i] += 1
        elif (d2==2):
            cc_nnn_456[i] += 1
        if (d3==1):
            cc_nn_456[i] += 1
        elif (d3==2):
            cc_nnn_456[i] += 1

# In[21]:

lcc = np.arange(8)
cc_stack_234 = []
cc_stack_345 = []
cc_stack_456 = []
for i in range(len(cc_234)):
    cc_stack_234.append(np.hstack((np.sum(cc_234,axis=1)[i],cc_nn_234[i],cc_nnn_234[i],cc_234[i],lcc[i])))
    cc_stack_345.append(np.hstack((np.sum(cc_345,axis=1)[i],cc_nn_345[i],cc_nnn_345[i],cc_345[i],lcc[i])))
    cc_stack_456.append(np.hstack((np.sum(cc_456,axis=1)[i],cc_nn_456[i],cc_nnn_456[i],cc_456[i],lcc[i])))

# In[22]:

sorted_list_234 = sorted(cc_stack_234, key=operator.itemgetter(0,1,2,4,5,3,6))
sorted_list_345 = sorted(cc_stack_345, key=operator.itemgetter(0,1,2,4,5,3,6))
sorted_list_456 = sorted(cc_stack_456, key=operator.itemgetter(0,1,2,4,5,3,6))
cc_sorted_234 = np.array(sorted_list_234)[:,3:6]
cc_sorted_345 = np.array(sorted_list_345)[:,3:6]
cc_sorted_456 = np.array(sorted_list_456)[:,3:6]
cc_sorted_234 = cc_sorted_234.tolist()
cc_sorted_345 = cc_sorted_345.tolist()
cc_sorted_456 = cc_sorted_456.tolist()
for i in range(len(cc_234)):
    cc_sorted_234[i] = map(int,cc_sorted_234[i])
    cc_sorted_345[i] = map(int,cc_sorted_345[i])
    cc_sorted_456[i] = map(int,cc_sorted_456[i])

# In[23]:

lcc_sorted_234 = np.array(sorted_list_234)[:,6]
lcc_sorted_345 = np.array(sorted_list_345)[:,6]
lcc_sorted_456 = np.array(sorted_list_456)[:,6]
lcc_sorted_234 = lcc_sorted_234.tolist()
lcc_sorted_345 = lcc_sorted_345.tolist()
lcc_sorted_456 = lcc_sorted_456.tolist()
lcc_sorted_234 = map(int, lcc_sorted_234)
lcc_sorted_345 = map(int, lcc_sorted_345)
lcc_sorted_456 = map(int, lcc_sorted_456)

# In[24]:

dtrajs_1D_234_sorted = []
dtrajs_1D_345_sorted = []
dtrajs_1D_456_sorted = []
for i in range( dtrajs_1D_234[0].shape[0] ):
    dtrajs_1D_234_sorted.append(lcc_sorted_234[dtrajs_1D_234[0][i]])
    dtrajs_1D_345_sorted.append(lcc_sorted_345[dtrajs_1D_345[0][i]])
    dtrajs_1D_456_sorted.append(lcc_sorted_456[dtrajs_1D_456[0][i]])

# In[25]:

lags = np.linspace(1,1000,200,dtype='int')
its_234 = msm.timescales_msm(dtrajs_1D_234_sorted, lags=lags, nits=n_clusters)
its_345 = msm.timescales_msm(dtrajs_1D_345_sorted, lags=lags, nits=n_clusters)
its_456 = msm.timescales_msm(dtrajs_1D_456_sorted, lags=lags, nits=n_clusters)

# In[27]:

tau = 400
Cmat_234 = pyemma.msm.estimation.count_matrix(dtrajs_1D_234_sorted, tau, sliding=True, sparse_return=False)
Cmat_345 = pyemma.msm.estimation.count_matrix(dtrajs_1D_345_sorted, tau, sliding=True, sparse_return=False)
Cmat_456 = pyemma.msm.estimation.count_matrix(dtrajs_1D_456_sorted, tau, sliding=True, sparse_return=False)


# In[28]:
Cmat_totind = Cmat_234+Cmat_345+Cmat_456
Cmat_totind


# In[29]:

lcc_totind = pyemma.msm.estimation.largest_connected_set(Cmat_totind, directed=True)
Cmat_totind = pyemma.msm.estimation.largest_connected_submatrix(Cmat_totind, directed=True, lcc=lcc_totind)
Tmle_totind = pyemma.msm.estimation.transition_matrix(Cmat_totind, reversible=True)


# In[30]:

Tmle_totind


# In[31]:

mle_totind = pyemma.msm.markov_model(Tmle_totind)
evals_mle_totind = mle_totind.eigenvalues(k=n_clusters)
evecs_mle_totind = mle_totind.eigenvectors_left(k=n_clusters)


# In[32]:

dtrajs = []
for i in range( len(dtraj_rama_2) ):
    dtrajs.append( np.vstack( (dtrajs_rama_2[i], dtrajs_rama_3[i], dtrajs_rama_4[i], dtrajs_rama_5[i], dtrajs_rama_6[i]) ).T )
    dtrajs[i].astype('int64')


# In[33]:

dtrajs_sum = []
for i in range( len(dtrajs) ):
    dtrajs_sum.append( np.sum(dtrajs[i],axis=1) )
    dtrajs_sum[i].astype('int64')


# In[34]:

# we need a single dimensional identifier of the microstate, can we cluster to automize?
n_clusters = 6
clustering = coor.cluster_regspace(dtrajs_sum,max_centers=n_clusters,dmin=0.5)


# In[35]:

dtrajs_1D = clustering.dtrajs


# In[36]:

its = msm.timescales_msm(dtrajs_1D, lags=1200, nits=n_clusters)

# In[38]:

tau = 400
nts = n_clusters
Cmat = pyemma.msm.estimation.count_matrix(dtrajs_1D, tau, sliding=True, sparse_return=False)
lcc = pyemma.msm.estimation.largest_connected_set(Cmat, directed=True)
Cmat = pyemma.msm.estimation.largest_connected_submatrix(Cmat, directed=True, lcc=lcc)
Tmle = pyemma.msm.estimation.transition_matrix(Cmat, reversible=True)

# In[39]:

mle = pyemma.msm.markov_model(Tmle)
evals_mle = mle.eigenvalues(k=n_clusters)
evecs_mle = mle.eigenvectors_left(k=n_clusters)


# In[40]:

cc = clustering.clustercenters[:]


# In[41]:

cc_labels = []
for i in range(len(cc)):
    cc_labels.append(str(round((n_clusters-1-i)/float(n_clusters-1),2))+'a')
cc_labels = np.array(cc_labels)


# In[43]:

sorted_list = sorted(cc)
np.array(sorted_list).shape


# In[44]:

cc_sorted = np.array(sorted_list)[:]
cc_sorted = cc_sorted.tolist()
for i in range(len(cc)):
    cc_sorted[i] = map(int,cc_sorted[i])
cc_sorted


# In[45]:

#lcc_sorted = np.array(sorted_list)[:,0]
#lcc_sorted = lcc_sorted.tolist()
#lcc_sorted = map(int, lcc_sorted)
#lcc_sorted = cc[:,0]
#lcc_sorted
#cc.shape
lcc_sorted = []
for i in range(cc.shape[0]):
    lcc_sorted.append( np.where(cc==lcc[i])[0][0] )
lcc_sorted


# In[46]:

from copy import deepcopy
T_sorted = deepcopy(mle.transition_matrix)
for i in range(len(lcc_sorted)):
    for j in range(len(lcc_sorted)):
        T_sorted[i,j] = mle.transition_matrix[lcc_sorted[i],lcc_sorted[j]]


# In[48]:

def nck(n,k):
    return scipy.misc.factorial(n) / (scipy.misc.factorial(k) * scipy.misc.factorial(n-k))


# In[49]:

# we need a single dimensional identifier of the microstate, can we cluster to automize?
n_clusters_full = 2**5
clustering_full = coor.cluster_regspace(dtrajs,max_centers=n_clusters_full,dmin=0.5)


# In[50]:

cc_full = clustering_full.clustercenters[:]

# let's get the full model

dtrajs_1D_full = clustering_full.dtrajs

tau = 400
nts = n_clusters_full
Cmat_full = pyemma.msm.estimation.count_matrix(dtrajs_1D_full, tau, sliding=True, sparse_return=False)
lcc_full = pyemma.msm.estimation.largest_connected_set(Cmat_full, directed=True)
Cmat_full = pyemma.msm.estimation.largest_connected_submatrix(Cmat_full, directed=True, lcc=lcc)
Tmle_full = pyemma.msm.estimation.transition_matrix(Cmat_full, reversible=True)

mle_full = pyemma.msm.markov_model(Tmle_full)
evals_mle_full = mle.eigenvalues(k=n_clusters_full)
evecs_mle_full = mle.eigenvectors_left(k=n_clusters_full)

cc_labels_full = []
for i in range(len(cc_full)):
    cc_labels_full.append([])
    cc_labels_full[i] = ''
    for j in range(len(cc_full[0])):
        if (cc_full[i,j]==0):
            cc_labels_full[i] += 'a'
        else:
            cc_labels_full[i] += 'b'
cc_labels_full = np.array(cc_labels_full)


# calculate the number of adjacent and almost adjacent alphas for each state
cc_nn = np.zeros(len(cc_full))
cc_nnn = np.zeros(len(cc_full))
for i in range(len(cc_full)):
    if ( np.where(cc_full[i]==1)[0].shape[0] == 2):
        d1 = np.where(cc_full[i]==1)[0][1] - np.where(cc_full[i]==1)[0][0]
        if (d1==1):
            cc_nn[i] += 1
        elif (d1==2):
            cc_nnn[i] += 1
    elif ( np.where(cc_full[i]==1)[0].shape[0] == 3):
        d1 = np.where(cc_full[i]==1)[0][1] - np.where(cc_full[i]==1)[0][0]
        d2 = np.where(cc_full[i]==1)[0][2] - np.where(cc_full[i]==1)[0][1]
        if (d1==1):
            cc_nn[i] += 1
        elif (d1==2):
            cc_nnn[i] += 1
        if (d2==1):
            cc_nn[i] += 1
        elif (d2==2):
            cc_nnn[i] += 1
    elif ( np.where(cc_full[i]==1)[0].shape[0] == 4):
        d1 = np.where(cc_full[i]==1)[0][1] - np.where(cc_full[i]==1)[0][0]
        d2 = np.where(cc_full[i]==1)[0][2] - np.where(cc_full[i]==1)[0][1]
        d3 = np.where(cc_full[i]==1)[0][3] - np.where(cc_full[i]==1)[0][2]
        if (d1==1):
            cc_nn[i] += 1
        elif (d1==2):
            cc_nnn[i] += 1
        if (d2==1):
            cc_nn[i] += 1
        elif (d2==2):
            cc_nnn[i] += 1
        if (d3==1):
            cc_nn[i] += 1
        elif (d3==2):
            cc_nnn[i] += 1

cc_stack_full = []
for i in range(len(cc_full)):
    cc_stack_full.append(np.hstack((np.sum(cc_full,axis=1)[i],cc_nn[i],cc_nnn[i],cc_full[i],lcc_full[i])))

sorted_list_full = sorted(cc_stack_full, key=operator.itemgetter(0,1,2,5,4,6))

cc_sorted_full = np.array(sorted_list_full)[:,3:8]
cc_sorted_full = cc_sorted_full.tolist()
for i in range(len(cc_full)):
    cc_sorted_full[i] = map(int,cc_sorted_full[i])

lcc_sorted_full = np.array(sorted_list_full)[:,8]
lcc_sorted_full = lcc_sorted_full.tolist()
lcc_sorted_full = map(int, lcc_sorted_full)


# Now save all the data
# Qhel
np.save('lccQhel',lcc_sorted)
np.save('ccQhel',cc_sorted)
np.save('TQhel',T_sorted)
# 3res
np.save('lcc3res',lcc_sorted_234)
np.save('cc3res',cc_234)
np.save('T3res',Tmle_totind)
# full
np.save('lcc_full',lcc_sorted_full)
np.save('cc_full',cc_sorted_full)
np.save('Tfull',Tmle_full)
