
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

dtrajs_nn_23 = []
dtrajs_nn_34 = []
dtrajs_nn_45 = []
dtrajs_nn_56 = []
for i in range( len(dtraj_rama_2) ):
    dtrajs_nn_23.append( np.vstack( (dtrajs_rama_2[i], dtrajs_rama_3[i]) ).T )
    dtrajs_nn_23[i].astype('int64')
    dtrajs_nn_34.append( np.vstack( (dtrajs_rama_3[i], dtrajs_rama_4[i]) ).T )
    dtrajs_nn_34[i].astype('int64')
    dtrajs_nn_45.append( np.vstack( (dtrajs_rama_4[i], dtrajs_rama_5[i]) ).T )
    dtrajs_nn_45[i].astype('int64')
    dtrajs_nn_56.append( np.vstack( (dtrajs_rama_5[i], dtrajs_rama_6[i]) ).T )
    dtrajs_nn_56[i].astype('int64')


# In[18]:

n_clusters = 4
clustering_nn_23 = coor.cluster_regspace(dtrajs_nn_23,max_centers=n_clusters,dmin=0.5)
clustering_nn_34 = coor.cluster_regspace(dtrajs_nn_34,max_centers=n_clusters,dmin=0.5)
clustering_nn_45 = coor.cluster_regspace(dtrajs_nn_45,max_centers=n_clusters,dmin=0.5)
clustering_nn_56 = coor.cluster_regspace(dtrajs_nn_56,max_centers=n_clusters,dmin=0.5)


# In[19]:

dtrajs_1D_23 = clustering_nn_23.dtrajs
dtrajs_1D_34 = clustering_nn_34.dtrajs
dtrajs_1D_45 = clustering_nn_45.dtrajs
dtrajs_1D_56 = clustering_nn_56.dtrajs


# In[20]:

# shift the cluster indices so they are all consistent
cc_23 = clustering_nn_23.clustercenters[:]
cc_34 = clustering_nn_34.clustercenters[:]
cc_45 = clustering_nn_45.clustercenters[:]
cc_56 = clustering_nn_56.clustercenters[:]


# In[21]:

lcc = np.arange(4)
cc_stack_23 = []
cc_stack_34 = []
cc_stack_45 = []
cc_stack_56 = []
for i in range(len(cc_23)):
    cc_stack_23.append(np.hstack((cc_23[i],lcc[i])))
    cc_stack_34.append(np.hstack((cc_34[i],lcc[i])))
    cc_stack_45.append(np.hstack((cc_45[i],lcc[i])))
    cc_stack_56.append(np.hstack((cc_56[i],lcc[i])))


# In[22]:

sorted_list_23 = sorted(cc_stack_23, key=operator.itemgetter(0,1))
sorted_list_34 = sorted(cc_stack_34, key=operator.itemgetter(0,1))
sorted_list_45 = sorted(cc_stack_45, key=operator.itemgetter(0,1))
sorted_list_56 = sorted(cc_stack_56, key=operator.itemgetter(0,1))
cc_sorted_23 = np.array(sorted_list_23)[:,:2]
cc_sorted_34 = np.array(sorted_list_34)[:,:2]
cc_sorted_45 = np.array(sorted_list_45)[:,:2]
cc_sorted_56 = np.array(sorted_list_56)[:,:2]
cc_sorted_23 = cc_sorted_23.tolist()
cc_sorted_34 = cc_sorted_34.tolist()
cc_sorted_45 = cc_sorted_45.tolist()
cc_sorted_56 = cc_sorted_56.tolist()
for i in range(len(cc_23)):
    cc_sorted_23[i] = map(int,cc_sorted_23[i])
    cc_sorted_34[i] = map(int,cc_sorted_34[i])
    cc_sorted_45[i] = map(int,cc_sorted_45[i])
    cc_sorted_56[i] = map(int,cc_sorted_56[i])


# In[23]:

lcc_sorted_23 = np.array(sorted_list_23)[:,2]
lcc_sorted_34 = np.array(sorted_list_34)[:,2]
lcc_sorted_45 = np.array(sorted_list_45)[:,2]
lcc_sorted_56 = np.array(sorted_list_56)[:,2]
lcc_sorted_23 = lcc_sorted_23.tolist()
lcc_sorted_34 = lcc_sorted_34.tolist()
lcc_sorted_45 = lcc_sorted_45.tolist()
lcc_sorted_56 = lcc_sorted_56.tolist()
lcc_sorted_23 = map(int, lcc_sorted_23)
lcc_sorted_34 = map(int, lcc_sorted_34)
lcc_sorted_45 = map(int, lcc_sorted_45)
lcc_sorted_56 = map(int, lcc_sorted_56)


# In[24]:

dtrajs_1D_23_sorted = []
dtrajs_1D_34_sorted = []
dtrajs_1D_45_sorted = []
dtrajs_1D_56_sorted = []
for i in range( dtrajs_1D_23[0].shape[0] ):
    dtrajs_1D_23_sorted.append(lcc_sorted_23[dtrajs_1D_23[0][i]])
    dtrajs_1D_34_sorted.append(lcc_sorted_34[dtrajs_1D_34[0][i]])
    dtrajs_1D_45_sorted.append(lcc_sorted_45[dtrajs_1D_45[0][i]])
    dtrajs_1D_56_sorted.append(lcc_sorted_56[dtrajs_1D_56[0][i]])


# In[25]:

lags = np.linspace(1,1000,200,dtype='int')
its_23 = msm.timescales_msm(dtrajs_1D_23_sorted, lags=lags, nits=n_clusters)
its_34 = msm.timescales_msm(dtrajs_1D_34_sorted, lags=lags, nits=n_clusters)
its_45 = msm.timescales_msm(dtrajs_1D_45_sorted, lags=lags, nits=n_clusters)
its_56 = msm.timescales_msm(dtrajs_1D_56_sorted, lags=lags, nits=n_clusters)


# In[27]:

tau = 400
Cmat_23 = pyemma.msm.estimation.count_matrix(dtrajs_1D_23_sorted, tau, sliding=True, sparse_return=False)
Cmat_34 = pyemma.msm.estimation.count_matrix(dtrajs_1D_34_sorted, tau, sliding=True, sparse_return=False)
Cmat_45 = pyemma.msm.estimation.count_matrix(dtrajs_1D_45_sorted, tau, sliding=True, sparse_return=False)
Cmat_56 = pyemma.msm.estimation.count_matrix(dtrajs_1D_56_sorted, tau, sliding=True, sparse_return=False)


# In[28]:

# Treat the pairs independently now
#Cmat_totind = Cmat_23+Cmat_34+Cmat_45+Cmat_56
#Cmat_totind


# In[29]:

lcc_cc_23 = pyemma.msm.estimation.largest_connected_set(Cmat_23, directed=True)
Cmat_cc_23 = pyemma.msm.estimation.largest_connected_submatrix(Cmat_23, directed=True, lcc=lcc_cc_23)
Tmle_cc_23 = pyemma.msm.estimation.transition_matrix(Cmat_cc_23, reversible=True)
lcc_cc_34 = pyemma.msm.estimation.largest_connected_set(Cmat_34, directed=True)
Cmat_cc_34 = pyemma.msm.estimation.largest_connected_submatrix(Cmat_34, directed=True, lcc=lcc_cc_34)
Tmle_cc_34 = pyemma.msm.estimation.transition_matrix(Cmat_cc_34, reversible=True)
lcc_cc_45 = pyemma.msm.estimation.largest_connected_set(Cmat_45, directed=True)
Cmat_cc_45 = pyemma.msm.estimation.largest_connected_submatrix(Cmat_45, directed=True, lcc=lcc_cc_45)
Tmle_cc_45 = pyemma.msm.estimation.transition_matrix(Cmat_cc_45, reversible=True)
lcc_cc_56 = pyemma.msm.estimation.largest_connected_set(Cmat_56, directed=True)
Cmat_cc_56 = pyemma.msm.estimation.largest_connected_submatrix(Cmat_56, directed=True, lcc=lcc_cc_56)
Tmle_cc_56 = pyemma.msm.estimation.transition_matrix(Cmat_cc_56, reversible=True)

# In[31]:

#mle_totind = pyemma.msm.markov_model(Tmle_totind)
#evals_mle_totind = mle_totind.eigenvalues(k=n_clusters)
#evecs_mle_totind = mle_totind.eigenvectors_left(k=n_clusters)


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
# 2res
np.save('lcc2res',lcc_sorted_23)
np.save('cc2res',cc_23)
np.save('T2res_0',Tmle_cc_23)
np.save('T2res_1',Tmle_cc_34)
np.save('T2res_2',Tmle_cc_45)
np.save('T2res_3',Tmle_cc_56)
# full
np.save('lcc_full',lcc_sorted_full)
np.save('cc_full',cc_sorted_full)
np.save('Tfull',Tmle_full)
