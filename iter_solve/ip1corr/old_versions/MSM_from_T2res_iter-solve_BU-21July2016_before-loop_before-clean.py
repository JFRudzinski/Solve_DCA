# imports
import numpy as np
from sympy import *
import pyemma
pyemma.__version__
import os
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import msmbuilder
from msmbuilder.msm.ratematrix import ContinuousTimeMSM
import scipy
from msmtools.analysis.dense.decomposition import eigenvectors, eigenvalues
import operator
from scipy.optimize import fsolve
from copy import deepcopy

# Read in system-specific quantities
P2res_inp = np.load('T2res.npy')
cc2res = np.load('cc2res.npy')
cc_full = np.load('cc_full.npy')

# variables
S2 = [0,1]
Nres = 5
# first, get all the possible states of the full system
states_Nres = cc_full.astype(int)

# now, get all 2 res states
states_2res = cc2res.astype(int)

# now, define the Nres variables
PN = []
for s1 in range(len(states_Nres)):
    PN.append([])
    seq1 = ''.join(str(s) for s in states_Nres[s1])
    for s2 in range(len(states_Nres)):
        seq2 = ''.join(str(s) for s in states_Nres[s2])
        PN[s1].append(Symbol('P_'+seq2+'g'+seq1, positive=True))

# now, define the 2res variables
P2res = []
for res in range(Nres-1):
    P2res.append([])
    for s1 in range(len(states_2res)):
        P2res[res].append([])
        seq1 = ''.join(str(s) for s in states_2res[s1])
        for s2 in range(len(states_2res)):
            seq2 = ''.join(str(s) for s in states_2res[s2])
            P2res[res][s1].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1, positive=True))
# also define the boundaries
P2res_end = []
for res in range(2):
    P2res_end.append([])
    for s1 in range(len(states_2res)):
        P2res_end[res].append([])
        seq1 = ''.join(str(s) for s in states_2res[s1])
        for s2 in range(len(states_2res)):
            seq2 = ''.join(str(s) for s in states_2res[s2])
            if ( res == 0 ):
                P2res_end[res][s1].append( Symbol('P_'+'b'+str(0)+'_'+seq2+'g'+seq1, positive=True) )
            else:
                P2res_end[res][s1].append( Symbol('P_'+str(Nres-1)+'e'+'_'+seq2+'g'+seq1, positive=True) )

# now, the conditional prob of jumping for each res
P1res = []
for res in range(Nres):
    P1res.append([])
    for s1 in range(len(S2)):
        P1res[res].append([])
        seq1 = str(S2[s1])
        for s2 in range(len(S2)):
            seq2 = str(S2[s2])
            P1res[res][s1].append(Symbol('P_'+str(res)+'_'+seq2+'g'+seq1, positive=True))
# also define the boundaries
P1res_end = []
for res in range(2):
    P1res_end.append([])
    for s1 in range(len(S2)):
        P1res_end[res].append([])
        seq1 = str(S2[s1])
        for s2 in range(len(S2)):
            seq2 = str(S2[s2])
            if ( res == 0 ):
                P1res_end[res][s1].append(Symbol('P_'+'b'+'_'+seq2+'g'+seq1, positive=True))
            else:
                P1res_end[res][s1].append(Symbol('P_'+'e'+'_'+seq2+'g'+seq1, positive=True))


# finally, the static probabilties of the pairs and singlets
P2res_stat = []
for res in range(Nres-1):
    P2res_stat.append([])
    for s1 in range(len(states_2res)):
        seq1 = ''.join(str(s) for s in states_2res[s1])
        P2res_stat[res].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq1, positive=True))
# also define the boundaries
P2res_stat_end = []
for res in range(2):
    P2res_stat_end.append([])
    for s1 in range(len(states_2res)):
        seq1 = ''.join(str(s) for s in states_2res[s1])
        if ( res == 0 ):
            P2res_stat_end[res].append(Symbol('P_'+'b'+str(0)+'_'+seq1, positive=True))
        else:
            P2res_stat_end[res].append(Symbol('P_'+str(Nres-1)+'e'+'_'+seq1, positive=True))

P1res_stat = []
for res in range(Nres):
    P1res_stat.append([])
    for s1 in range(len(S2)):
        seq1 = str(S2[s1])
        P1res_stat[res].append(Symbol('P_'+str(res)+'_'+seq1, positive=True))
# also define the boundaries
P1res_stat_end = []
for res in range(2):
    P1res_stat_end.append([])
    for s1 in range(len(S2)):
        seq1 = str(S2[s1])
        if ( res == 0 ):
            P1res_stat_end[res].append(Symbol('P_'+'b'+'_'+seq1, positive=True))
        else:
            P1res_stat_end[res].append(Symbol('P_'+'e'+'_'+seq1, positive=True))

# put together the equations for each global prob
eqns_PN = []
for s in range(len(states_Nres)):
    for sp in range(len(states_Nres)):
        tmp = 1.
        for res in range(Nres-1):
            statep = states_Nres[sp][res:res+2]
            indp = np.where( np.all(states_2res==np.array(statep),axis=1) == True )[0][0]
            state = states_Nres[s][res:res+2]
            ind = np.where( np.all(states_2res==np.array(state),axis=1) == True )[0][0]
            tmp *= P2res[res][ind][indp]
        for res in range(Nres-2):
            statep = states_Nres[sp][res+1]
            state = states_Nres[s][res+1]
            tmp /= P1res[res][state][statep]
        tmp *= -1
        tmp += PN[s][sp]
        eqns_PN.append(tmp)

# now, add the definitions of the 1res conditional and static probabilties
eqns_P1res = []
eqns_P1res_stat = []
for res in range(Nres):
    for si in range(len(S2)):
        Nxxp = [0,0]
        for sip in range(len(S2)):
            for sim1p in range(len(S2)):
                for sip1p in range(len(S2)):
                    for sim1 in range(len(S2)):
                        for sip1 in range(len(S2)):
                            state1p = [sim1p,sip]
                            ind1p = np.where( np.all(states_2res==np.array(state1p),axis=1) == True )[0][0]
                            state1 = [sim1,si]
                            ind1 = np.where( np.all(states_2res==np.array(state1),axis=1) == True )[0][0]
                            state2p = [sip,sip1p]
                            ind2p = np.where( np.all(states_2res==np.array(state2p),axis=1) == True )[0][0]
                            state2 = [si,sip1]
                            ind2 = np.where( np.all(states_2res==np.array(state2),axis=1) == True )[0][0]
                            if ( res == 0 ):
                                Nxxp[sip] += P2res_end[0][ind1][ind1p]*P2res[res][ind2][ind2p]*P2res_stat_end[0][ind1]*P2res_stat[res][ind2]
                            elif ( res == Nres-1 ):
                                Nxxp[sip] += P2res[res-1][ind1][ind1p]*P2res_end[1][ind2][ind2p]*P2res_stat[res-1][ind1]*P2res_stat_end[1][ind2]
		            else:
                                Nxxp[sip] += P2res[res-1][ind1][ind1p]*P2res[res][ind2][ind2p]*P2res_stat[res-1][ind1]*P2res_stat[res][ind2]
        # for each res and particular set of states, but input in pairs of states
        den = np.sum(Nxxp)
        eqns_P1res.append( P1res[res][si][0] - (Nxxp[0] / den)**(0.5) )
        eqns_P1res.append( P1res[res][si][1] - (Nxxp[1] / den)**(0.5) )
        # also add the 1 res static definitions for each res
        #eqns_P1res_stat.append( P1res_stat[res][si] - np.sum(pxpx) ) # I don't think I actually need this for anything

# definitions of the 2res static probabilties --- Not using this at the moment, inputing guess and solving iteratively
eqns_P2res_stat = []
for res in range(Nres-1):
    for si in range(len(S2)):
        for sip1 in range(len(S2)):

            for sim1 in range(len(S2)):
                for sim1p in range(len(S2)):
                    for sip in range(len(S2)):
                        for sip1p in range(len(S2)):
                            for sip2 in range(len(S2)):
                                for sip2p in range(len(S2)):
                                    state1p = [sim1p,sip]
                                    ind1p = np.where( np.all(states_2res==np.array(state1p),axis=1) == True )[0][0]
                                    state1 = [sim1,si]
                                    ind1 = np.where( np.all(states_2res==np.array(state1),axis=1) == True )[0][0]
                                    state2p = [sip,sip1p]
                                    ind2p = np.where( np.all(states_2res==np.array(state2p),axis=1) == True )[0][0]
                                    state2 = [si,sip1]
                                    ind2 = np.where( np.all(states_2res==np.array(state2),axis=1) == True )[0][0]
                                    state3p = [sip1p,sip2p]
                                    ind3p = np.where( np.all(states_2res==np.array(state3p),axis=1) == True )[0][0]
                                    state3 = [sip1,sip2]
                                    ind3 = np.where( np.all(states_2res==np.array(state3),axis=1) == True )[0][0]
                                    if ( res == 0 ):
                                        tmp1 = P2res_end[0][ind1][ind1p]*P2res_stat_end[0][ind1]
                                    else:
                                        tmp1 = P2res[res-1][ind1][ind1p]*P2res_stat[res-1][ind1]
                                    if ( res == Nres-2 ):
                                        tmp3 = P2res_end[1][ind3][ind3p]*P2res_stat_end[1][ind3]
                                    else:
                                        tmp3 = P2res[res+1][ind3][ind3p]*P2res_stat[res+1][ind3]
                                    den = P1res[res][si][sip]*P1res[res+1][sip1][sip1p]*P1res_stat[res][si]*P1res_stat[res+1][sip1]
                                    tmp2 = ( P2res[res][ind2][ind2p]*P2res_stat[res][ind2] ) / den
                                    tmp += tmp1*tmp2*tmp3
                # for each pair of residues and particular set of states
                eqns_P2res_stat.append( P2res_stat[res][si] - tmp )

# add the normalization conditions
# TN norm
eqns_norm_PN = []
for s1 in range(len(states_Nres)):
    eqns_norm_PN.append( np.sum(PN,axis=1)[s1] - 1 )
# 1 res cond norm
eqns_norm_P1res = []
for res in range(Nres):
    for si in range(len(S2)):
        eqns_norm_P1res.append( P1res[res][si][0] + P1res[res][si][1] - 1 )
# 2 res stat norm
eqns_norm_P2res_stat = []
for res in range(Nres-1):
    tmp = 0
    for state in range(len(states_2res)):
        tmp += P2res_stat[res][state]
    eqns_norm_P2res_stat.append( tmp - 1 )
# 1 res stat norm
eqns_norm_P1res_stat = []
for res in range(Nres):
    tmp = 0
    for si in range(len(S2)):
        tmp += P1res_stat[res][si]
    eqns_norm_P1res_stat.append( tmp - 1 )

# set the T2res as input, same for each res
# not sure if eqns or direct substitutions is better -- ok, direct substitutions are better but we will leave this another round
eqns_T2res_inp = []
for res in range(-1,Nres):
    for s1 in range(len(states_2res)):
        seq1 = ''.join(str(s) for s in states_2res[s1])
        for s2 in range(len(states_2res)):
            seq2 = ''.join(str(s) for s in states_2res[s2])
            if (res == -1):
                var = Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1, positive=True)
            elif (res == Nres-1):
                var = Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1, positive=True)
            else:
                var = P2res[res][s1][s2]
            eqns_T2res_inp.append( var - P2res_inp[s1][s2] )

# specify the boundary conditions
eqns_bndry_cond = []
eqns_bndry_cond_inp = []
# set the 1res static prob to 1/0
for res in range(2):
    for s1 in range(len(S2)):
        var = P1res_stat_end[res][s1]
        if ( s1 == 0 ):
            eqns_bndry_cond_inp.append( var )
        else:
            eqns_bndry_cond_inp.append( var - 1 )
# now, relate the 2res static prop
for res in range(2):
    for s1 in range(len(states_2res)):
        var = P2res_stat_end[res][s1]
        if ( res == 0 ):
            eqns_bndry_cond.append( var - P1res_stat_end[0][states_2res[s1][0]]*P1res_stat[0][states_2res[s1][1]] )
        else:
            eqns_bndry_cond.append( var - P1res_stat[Nres-1][states_2res[s1][0]]*P1res_stat_end[1][states_2res[s1][1]] )
# finally, the 1 res cond prob
for res in range(2):
    for si in range(len(S2)):
        for sip in range(len(S2)):
            var = P1res_end[res][si][sip]
            if ( sip == 1 ):
                val = 1.
            else:
                val = 0.
            eqns_bndry_cond_inp.append( var - val )


# initialize the model by assuming that the 2res stat prob correspond to the 2res model (i.e., no correlations beyond 2 res)
mle_tmp = pyemma.msm.markov_model(P2res_inp)
mu_eff_2res = mle_tmp.eigenvectors_left(k=1)[0]

# not using this now, but we'll leave it one more round
eqns_P2res_stat_inp = []
for res in range(Nres-1):
    for s1 in range(len(states_2res)):
        eqns_P2res_stat_inp.append( P2res_stat[res][s1] - mu_eff_2res[s1] )

## solve for the conditional probabilities
### first collect the relevant equations
#eqns = eqns_P1res+eqns_norm_P1res+eqns_bndry_cond # the boundary condition eqns are not right here
eqns = deepcopy(eqns_P1res)
### set up a dictionary for the inputs
inp_var = {}
# P2res
for res in range(-1,Nres):
    for s1 in range(len(states_2res)):
        for s2 in range(len(states_2res)):
            if (res == -1):
                var = P2res_end[0][s1][s2]
            elif (res == Nres-1):
                var = P2res_end[1][s1][s2]
            else:
                var = P2res[res][s1][s2]
            inp_var[var] = P2res_inp[s1][s2]
# P2res_stat
for res in range(-1,Nres):
    for state in range(len(states_2res)):
        if ( res == -1 ):
            var = P2res_stat_end[0][state]
            if ( states_2res[state][0] == 0 ):
                inp_var[var] = 0.
            else:
                inp_var[var] = 1.
        elif ( res == Nres-1 ):
            var = P2res_stat_end[1][state]
            if ( states_2res[state][1] == 0 ):
                inp_var[var] = 0.
            else:
                inp_var[var] = 1.            
        else:
            var = P2res_stat[res][state]
            inp_var[var] = mu_eff_2res[state]
# P1res, boundary only
for res in range(2):
    for si in range(len(S2)):
        for sip in range(len(S2)):
            var = P1res_end[res][si][sip]
            if ( sip == 1 ):
                val = 1.
            else:
                val = 0.
            inp_var[var] = val

# substitute the inputs into the eqns
for eqn in range(len(eqns)):
    eqns[eqn] = eqns[eqn].subs(inp_var)

# let's first try to solve just with sympy
#P1res_soln = solve( eqns, np.concatenate(np.concatenate(P1res)).tolist() )
# this gives no solution, possibly due to numerics
# instead let's convert to scipy and use a root finding algorithm

var_live = np.concatenate(np.concatenate(P1res)).tolist()
fncs = [lambdify((var_live), eqn, modules='numpy') for eqn in eqns]

def f_v(zz):
    return np.array([fnc(*zz) for fnc in fncs]) 

zz0 = 0.5*np.ones(len(var_live))
P1res_soln = fsolve(f_v, zz0)

# check accuracy of the solution and then normalization
soln_var = {}
for var in range(len(var_live)):
    soln_var[var_live[var]] = P1res_soln[var]

print 'checking the accuracy of the solution...'
sumsq = 0.
for eqn in range(len(eqns)):
    sumsq += eqns[eqn].subs(soln_var)**2
print 'sumsq = '+str(sumsq)

print 'checking the normalization...'
sumsq = 0
for eqn in range(len(eqns_norm_P1res)):
    sumsq += eqns_norm_P1res[eqn].subs(soln_var)**2
print 'sumsq = '+str(sumsq)

# forget the inconsistency in the normalization for now
# calculate the full probabilities
tot_var = dict(inp_var.items()+soln_var.items())
eqns = deepcopy(eqns_PN)
for eqn in range(len(eqns)):
    eqns[eqn] = eqns[eqn].subs(tot_var)

var_live = np.concatenate(PN).tolist()
fncs = [lambdify((var_live), eqn, modules='numpy') for eqn in eqns]
def f_v(zz):
    return np.array([fnc(*zz) for fnc in fncs])
zz0 = 0.5*np.ones(len(var_live))
PN_soln = fsolve(f_v, zz0)

# check accuracy of the solution and then normalization
'''
soln_var = {}
for var in range(len(var_live)):
    soln_var[var_live[var]] = PN_soln[var]
print 'checking the accuracy of the solution...'
sumsq = 0.
for eqn in range(len(eqns)):
    sumsq += eqns[eqn].subs(soln_var)**2
print 'sumsq = '+str(sumsq)
print 'checking the normalization...'
sumsq = 0
for eqn in range(len(eqns_norm_PN)):
    sumsq += eqns_norm_PN[eqn].subs(soln_var)**2
print 'sumsq = '+str(sumsq)
'''

# set the matrix values
TN = np.zeros(shape=(len(states_Nres),len(states_Nres)))
ctr = 0
for s1 in range(len(states_Nres)):
    for s2 in range(len(states_Nres)):
        TN[s1][s2] = PN_soln[ctr]
        ctr += 1
# artificially fix the normalization
TN = TN / np.sum(TN,axis=1,dtype=float,keepdims=True)

# now, get the stat dist for this matrix
mle_tmp = pyemma.msm.markov_model(TN)
mu_N = mle_tmp.eigenvectors_left(k=1)[0]

# calculate the 2res stationary dist directly from mu_N
# P2res_stat
for res in range(Nres-1):
    for s1 in range(len(states_2res)):
        states = np.where( np.all(states_Nres[:,res:res+2]==states_2res[s1],axis=1) == True )[0]
        tot = 0.
        for state in states:
            tot += mu_N[state]
        var = P2res_stat[res][s1]
        inp_var[var] = tot


'''
import numpy as np
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify
import sympy as sy

sy.init_printing()  # LaTeX like pretty printing for IPython


x1, x2, x3, x4 = sy.symbols('x1 x2 x3 x4')
xx = (x1, x2, x3, x4)
f = -2*x1**2*x3+6*x1**2*x4+13*x1**2-3*x1*x2**2+x1*x2+3*x1*x3**2-3*x4+103
f_n = lambdify(xx, f, modules='numpy')

# Build Jacobian:
jac_f = [f.diff(x) for x in xx]
jac_fn = [lambdify(xx, jf, modules='numpy') for jf in jac_f]


def f_v(zz):
    """ Helper for receiving vector parameters """
    return f_n(zz[0], zz[1], zz[2], zz[3])


def jac_v(zz):
    """ Jacobian Helper for receiving vector parameters """
    return np.array([jfn(zz[0], zz[1], zz[2], zz[3]) for jfn in jac_fn])


bnds = ((-1, 1), (-1, 1), (-1, 1), (-1, 1))
zz0 = np.array([1, 1, 1, 1])

rslts = minimize(f_v, zz0, method='SLSQP', jac=jac_v, bounds=bnds)
print(rslts)
'''



import pickle
def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#save_object('P1res_soln.pkl', P1res_soln)


# list of equations
#eqns_PN+eqns_norm_PN
#eqns_P1res+eqns_norm_P1res
#eqns_P1res_stat+eqns_norm_P1res_stat
#eqns_P2res_stat+eqns_norm_P2res_stat
#eqns_bndry_cond
#eqns_bndry_cond_inp
#eqns_T2res_inp
#eqns_P2res_stat_inp
