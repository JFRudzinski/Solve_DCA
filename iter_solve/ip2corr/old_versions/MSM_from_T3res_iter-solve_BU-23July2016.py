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
from scipy.optimize import fsolve, minimize
from copy import deepcopy

# Read in system-specific quantities
Pkp1res_inp = np.load('T3res.npy')
cckp1res = np.load('cc3res.npy')
cckres = np.load('cc2res.npy')
cc_full = np.load('cc_full.npy')

# variables
S2 = [0,1]
Nres = 5
Ncorr = 2 # == k
# first, get all the possible states of the full system
states_Nres = cc_full.astype(int)

# now, get all 3 res states
states_kp1res = cckp1res.astype(int)
# and 2 res states
states_kres = cckres.astype(int)

# now, define the Nres variables
PN = []
for s1 in range(len(states_Nres)):
    PN.append([])
    seq1 = ''.join(str(s) for s in states_Nres[s1])
    for s2 in range(len(states_Nres)):
        seq2 = ''.join(str(s) for s in states_Nres[s2])
        PN[s1].append(Symbol('P_'+seq2+'g'+seq1, positive=True))

# now, define the 3res variables
Pkp1res = []
for res in range(Nres-Ncorr):
    Pkp1res.append([])
    for s1 in range(len(states_kp1res)):
        Pkp1res[res].append([])
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        for s2 in range(len(states_kp1res)):
            seq2 = ''.join(str(s) for s in states_kp1res[s2])
            Pkp1res[res][s1].append(Symbol('P_'+str(res)+str(res+1)+str(res+2)+'_'+seq2+'g'+seq1, positive=True))
# also define the boundaries
Pkp1res_end = []
for res in range(2*Ncorr):
    Pkp1res_end.append([])
    for s1 in range(len(states_kp1res)):
        Pkp1res_end[res].append([])
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        for s2 in range(len(states_kp1res)):
            seq2 = ''.join(str(s) for s in states_kp1res[s2])
            if ( res == 0 ):
                Pkp1res_end[res][s1].append( Symbol('P_'+'b1'+'b0'+str(0)+'_'+seq2+'g'+seq1, positive=True) )
            elif ( res == 1 ):
                Pkp1res_end[res][s1].append( Symbol('P_'+'b0'+str(0)+str(1)+'_'+seq2+'g'+seq1, positive=True) )
            elif ( res == 2 ):
                Pkp1res_end[res][s1].append( Symbol('P_'+str(Nres-2)+str(Nres-1)+'e0'+'_'+seq2+'g'+seq1, positive=True) )
            else:
                Pkp1res_end[res][s1].append( Symbol('P_'+str(Nres-1)+'e0'+'e1'+'_'+seq2+'g'+seq1, positive=True) )

# now, the conditional prob of jumping for each pair of res
Pkres = []
for res in range(Nres-(Ncorr-1)):
    Pkres.append([])
    for s1 in range(len(states_kres)):
        Pkres[res].append([])
        seq1 = ''.join(str(s) for s in states_kres[s1])
        for s2 in range(len(states_kres)):
            seq2 = ''.join(str(s) for s in states_kres[s2])
            Pkres[res][s1].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1, positive=True))
# also define the boundaries
Pkres_end = []
for res in range(2*Ncorr):
    Pkres_end.append([])
    for s1 in range(len(states_kres)):
        Pkres_end[res].append([])
        seq1 = ''.join(str(s) for s in states_kres[s1])
        for s2 in range(len(states_kres)):
            seq2 = ''.join(str(s) for s in states_kres[s2])
            if ( res == 0 ):
                P1res_end[res][s1].append(Symbol('P_'+'b1'+'b0'+'_'+seq2+'g'+seq1, positive=True))
            elif ( res == 1 ):
                P1res_end[res][s1].append(Symbol('P_'+'b0'+str(0)+'_'+seq2+'g'+seq1, positive=True))
            elif ( res == 2 ):
                P1res_end[res][s1].append(Symbol('P_'+str(Nres-1)+'e'+'_'+seq2+'g'+seq1, positive=True))
            else:
                P1res_end[res][s1].append(Symbol('P_'+'e0'+'e1'+'_'+seq2+'g'+seq1, positive=True))


# finally, the static probabilties of the kp1's and k's
Pkp1res_stat = []
for res in range(Nres-Ncorr):
    Pkp1res_stat.append([])
    for s1 in range(len(states_kp1res)):
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        Pkp1res_stat[res].append(Symbol('P_'+str(res)+str(res+1)+str(res+2)+'_'+seq1, positive=True))
# also define the boundaries
Pkp1res_stat_end = []
for res in range(2*Ncorr):
    Pkp1res_stat_end.append([])
    for s1 in range(len(states_kp1res)):
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        if ( res == 0 ):
            P2res_stat_end[res].append(Symbol('P_'+'b1'+'b0'+str(0)+'_'+seq1, positive=True))
        elif ( res == 1 ):
            P2res_stat_end[res].append(Symbol('P_'+'b0'+str(0)+str(1)+'_'+seq1, positive=True))
        elif ( res == 2 ):
            P2res_stat_end[res].append(Symbol('P_'+str(Nres-2)+str(Nres-1)+'e0'+'_'+seq1, positive=True))
        else:
            P2res_stat_end[res].append(Symbol('P_'+str(Nres-1)+'e0'+'e1'+'_'+seq1, positive=True))

Pkres_stat = []
for res in range(Nres-(Ncorr-1)):
    P1res_stat.append([])
    for s1 in range(len(states_kres)):
        seq1 = ''.join(str(s) for s in states_kres[s1])
        Pkres_stat[res].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq1, positive=True))
# also define the boundaries
Pkres_stat_end = []
for res in range(2*Ncorr):
    Pkres_stat_end.append([])
    for s1 in range(len(states_kres)):
        seq1 = ''.join(str(s) for s in states_kres[s1])
        if ( res == 0 ):
            Pkres_stat_end[res].append(Symbol('P_'+'b1'+'b0'+'_'+seq1, positive=True))
        elif ( res == 0 ):
            Pkres_stat_end[res].append(Symbol('P_'+'b0'+str(0)+'_'+seq1, positive=True))
        elif:
            Pkres_stat_end[res].append(Symbol('P_'+str(Nres-1)+'e0'+'_'+seq1, positive=True))
        else:
            Pkres_stat_end[res].append(Symbol('P_'+'e0'+'e1'+'_'+seq1, positive=True))

# put together the equations for each global prob
eqns_PN = []
for s in range(len(states_Nres)):
    for sp in range(len(states_Nres)):
        tmp = 1.
        for res in range(Nres-Ncorr):
            statep = states_Nres[sp][res:res+Ncorr]
            indp = np.where( np.all(states_kp1res==np.array(statep),axis=1) == True )[0][0]
            state = states_Nres[s][res:res+Ncorr]
            ind = np.where( np.all(states_kp1res==np.array(state),axis=1) == True )[0][0]
            tmp *= Pkp1res[res][ind][indp]
        for res in range(Nres-Ncorr-1):
            statep = states_Nres[sp][res+1]
            state = states_Nres[s][res+1]
            tmp /= Pkres[res][state][statep]
        tmp *= -1
        tmp += PN[s][sp]
        eqns_PN.append(tmp)

# now, add the definitions of the kres conditional and static probabilties
eqns_Pkres = []
for res in range(Nres-(Ncorr-1)):
    for si in range(len(S2)):
        for sip1 in range(len(S2)):
            for sip in range(len(S2)):
                for sip1p in range(len(S2)):
                    Nxxp = 0.
                    for sim2p in range(len(S2)):
                        for sim1p in range(len(S2)):
                            for sip2p in range(len(S2)):
                                for sip3p in range(len(S2)):
                                    for sim2 in range(len(S2)):
                                        for sim1 in range(len(S2)):
                                            for sip2 in range(len(S2)):
                                                for sip3 in range(len(S2)):
                                                    state1p = [sim2p,sim1p,sip]
                                                    ind1p = np.where( np.all(states_kp1res==np.array(state1p),axis=1) == True )[0][0]
                                                    state1 = [sim2,sim1,si]
                                                    ind1 = np.where( np.all(states_kp1res==np.array(state1),axis=1) == True )[0][0]
                                                    state2p = [sim1p,sip,sip1p]
                                                    ind2p = np.where( np.all(states_kp1res==np.array(state2p),axis=1) == True )[0][0]
                                                    state2 = [sim1,si,sip1]
                                                    ind2 = np.where( np.all(states_kp1res==np.array(state2),axis=1) == True )[0][0]
                                                    state3p = [sip,sip1p,sip2p]
                                                    ind3p = np.where( np.all(states_kp1res==np.array(state3p),axis=1) == True )[0][0]
                                                    state3 = [si,sip1,sip2]
                                                    ind3 = np.where( np.all(states_kp1res==np.array(state3),axis=1) == True )[0][0]
                                                    state4p = [sip1p,sip2p,sip3p]
                                                    ind4p = np.where( np.all(states_kp1res==np.array(state4p),axis=1) == True )[0][0]
                                                    state4 = [sip1,sip2,sip3]
                                                    ind4 = np.where( np.all(states_kp1res==np.array(state4),axis=1) == True )[0][0]
                                                    if ( res == 0 ):
                                                        Nxxp += Pkp1res_end[0][ind1][ind1p]*Pkp1res_end[1][ind2][ind2p]*Pkp1res[res][ind3][ind3p]*Pkp1res[res+1][ind4][ind4p]*Pkp1res_stat_end[0][ind1]*P2res_stat_end[1][ind2]*P2res_stat[res][ind3]*P2res_stat[res+1][ind4]
                                                    elif ( res == 1 ):
                                                        Nxxp += Pkp1res_end[1][ind1][ind1p]*Pkp1res[res-1][ind2][ind2p]*Pkp1res[res][ind3][ind3p]*Pkp1res[res+1][ind4][ind4p]*Pkp1res_stat_end[1][ind1]*P2res_stat[res-1][ind2]*P2res_stat[res][ind3]*P2res_stat[res+1][ind4]
                                                    elif ( res == Nres-3 ):
                                                        Nxxp += Pkp1res[res-2][ind1][ind1p]*Pkp1res[res-1][ind2][ind2p]*Pkp1res[res][ind3][ind3p]*Pkp1res_end[2][ind4][ind4p]*Pkp1res_stat[res-2][ind1]*P2res_stat[res-1][ind2]*P2res_stat[res][ind3]*P2res_stat_end[2][ind4]
                                                    elif ( res == Nres-2 ):
                                                        Nxxp += Pkp1res[res-2][ind1][ind1p]*Pkp1res[res-1][ind2][ind2p]*Pkp1res_end[2][ind3][ind3p]*Pkp1res_end[3][ind4][ind4p]*Pkp1res_stat[res-2][ind1]*P2res_stat[res-1][ind2]*P2res_stat_end[2][ind3]*P2res_stat_end[3][ind4]
                      		                    else:
                                                        Nxxp += Pkp1res[res-2][ind1][ind1p]*Pkp1res[res-1][ind2][ind2p]*Pkp1res[res][ind3][ind3p]*Pkp1res[res+1][ind4][ind4p]*Pkp1res_stat[res-2][ind1]*P2res_stat[res-1][ind2]*P2res_stat[res][ind3]*P2res_stat[res+1][ind4]
                    # for each res and particular set of states, but input in pairs of states
                    state = [si,sip1]
                    ind = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                    statep = [sip,sip1p]
                    indp = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
                    den = Pkres_stat[res][ind]
                    eqns_Pkres.append( Pkres[res][ind][indp] - (Nxxp**(0.5))/den )

# add the normalization conditions
# TN norm
eqns_norm_PN = []
for s1 in range(len(states_Nres)):
    eqns_norm_PN.append( np.sum(PN,axis=1)[s1] - 1 )
# k res cond norm
eqns_norm_Pkres = []
for res in range(Nres-(Ncorr-1)):
    for s1 in range(len(states_kres)):
        eqn_tmp = 0.
        for s2 in range(len(states_kres)):
            eqn_tmp += P1res[res][s1][s2]
        eqns_norm_Pkres.append( eqn_tmp - 1 )
# kp1 res stat norm
eqns_norm_Pkp1res_stat = []
for res in range(Nres-Ncorr):
    tmp = 0
    for state in range(len(states_kp1res)):
        tmp += Pkp1res_stat[res][state]
    eqns_norm_Pkp1res_stat.append( tmp - 1 )
# 1 res stat norm
eqns_norm_Pkres_stat = []
for res in range(Nres-(Ncorr-1)):
    tmp = 0
    for s1 in range(len(states_kres)):
        tmp += P1res_stat[res][s1]
    eqns_norm_P1res_stat.append( tmp - 1 )


# specify the boundary conditions
eqns_bndry_cond = []
eqns_bndry_cond_inp = []
# set the kres static prob to 1/0
for res in range(2*Ncorr):
    for s1 in range(len(states_kres)):
        var = P1res_stat_end[res][s1]
        if ( (states_kres[s1][0] == 0) or (states_kres[s1][1] == 0) ):
            eqns_bndry_cond_inp.append( var )
        else:
            eqns_bndry_cond_inp.append( var - 1 )
# now, relate the kp1res static prop ## JFR - This is quite confusing, I should double check it
for res in range(2*Ncorr):
    for s1 in range(len(states_kp1res)):
        var = Pkp1res_stat_end[res][s1]
        if ( res == 0 ):
            tmp = 0.
            for y1 in range(len(S2)):
                state = [states_kp1res[s1][2],y1]
                ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                tmp += Pkres_stat[0][ind]
            if ( states_kp1res[s1][0] == 0 ):
                P1_b1 = 0
            else:
                P1_b1 = 1
            if ( states_kp1res[s1][1] == 0 ):
                P1_b0 = 0
            else:
               P1_b0 = 1
            eqns_bndry_cond.append( var - P1_b1*P1_b0*tmp )
        elif ( res == 1 ):
            ind = np.where( np.all(states_kres==states_kp1res[s1][1:],axis=1) == True )[0][0]
            if ( states_kp1res[s1][0] == 0 ):
                P1_b0 = 0
            else:
                P1_b0 = 1
            eqns_bndry_cond.append( var - P1_b0*Pkres_stat[0][ind] )
        elif ( res == 2 ):
            ind = np.where( np.all(states_kres==states_kp1res[s1][:2],axis=1) == True )[0][0]
            if ( states_kp1res[s1][2] == 0 ):
                P1_e0 = 0
            else:
                P1_e0 = 1
            eqns_bndry_cond.append( var - Pkres_stat[Nres-2][ind]*P1_e0 )
        else:
            tmp = 0.
            for ynm1 in range(len(S2)):
                state = [ynm1,states_kp1res[s1][0]]
                ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                tmp += Pkres_stat[Nres-2][ind]
            if ( states_kp1res[s1][2] == 0 ):
                P1_e1 = 0
            else:
                P1_e1 = 1
            if ( states_kp1res[s1][1] == 0 ):
                P1_e0 = 0
            else:
                P1_e0 = 1
            eqns_bndry_cond.append( var - tmp*P1_e0*P1_e1 )
# and the kp1 res cond prob
for res in range(2*Ncorr):
    for s1 in range(len(states_kp1res)):
        for s2 in range(len(states_kp1res)):
            var = P2res_end[res][s1][s2]
            if ( res == 0 ):
            tmp = 0.
            for y1 in range(len(S2)):
                state = [states_kp1res[s1][2],y1]
                ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                for y1p in range(len(S2)):
                    statep = [states_kp1res[s2][2],y1p]
                    indp  = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
                    tmp += Pkres[0][ind][indp]
            if ( states_kp1res[s2][0] == 0 ):
                P1_b1 = 0
            else:
                P1_b1 = 1
            if ( states_kp1res[s2][1] == 0 ):
                P1_b0 = 0
            else:
                P1_b0 = 1
            eqns_bndry_cond.append( var - P1_b1*P1_b0*tmp )
            elif ( res == 1 ):
                ind1 = np.where( np.all(states_kres==states_kp1res[s1][1:],axis=1) == True )[0][0]
                ind2 = np.where( np.all(states_kres==states_kp1res[s2][1:],axis=1) == True )[0][0]
                if ( states_kp1res[s2][0] == 0 ):
                    P1_b0 = 0
                else:
                    P1_b0 = 1
                eqns_bndry_cond.append( var - P1_b0*Pkres[0][ind1][ind2] )
            elif ( res == 2 ):
                ind1 = np.where( np.all(states_kres==states_kp1res[s1][:2],axis=1) == True )[0][0]
                ind2 = np.where( np.all(states_kres==states_kp1res[s2][:2],axis=1) == True )[0][0]
                if ( states_kp1res[s2][2] == 0 ):
                    P1_e0 = 0
                else:
                    P1_e0 = 1
                eqns_bndry_cond.append( var - P1res[Nres-2][ind1][ind2]*P1_e0 )
            else:
                tmp = 0.
                for ynm1 in range(len(S2)):
                    state = [ynm1,states_kp1res[s1][0]]
                    ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                    for ynm1p in range(len(S2)):
                        statep = [ynm1p,states_kp1res[s2][0]]
                        indp  = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
                        tmp += Pkres[Nres-2][ind][indp]
                if ( states_kp1res[s2][2] == 0 ):
                    P1_e1 = 0
                else:
                    P1_e1 = 1
                if ( states_kp1res[s2][1] == 0 ):
                    P1_e0 = 0
                else:
                    P1_e0 = 1
                eqns_bndry_cond.append( var - tmp*P1_e0*P1_e1 )
# finally, the k res cond prob
for res in range(2*Ncorr):
    for state in range(len(states_kres)):
        for statep in range(len(states_kres)):
            var = Pkres_end[res][state][statep]
            if ( statep[0] == 0 or statep[1] == 0 ):
                val = 0.
            else:
                val = 1.
            eqns_bndry_cond_inp.append( var - val )

# initialize the model by assuming that the kp1res stat prob correspond to the kp1res model (i.e., no correlations beyond kp1 res)
mle_tmp = pyemma.msm.markov_model(Pkp1res_inp)
mu_eff_kp1res = mle_tmp.eigenvectors_left(k=1)[0]
# also initialize the 1res stat prob, from mu_eff_2res
mu_eff_kres = np.zeros(2**(Ncorr))
for kstate in range(len(states_kres)):
    tmp = 1.
    for kp1state in range(len(states_kp1res)):
        flag1 = np.all( states_kp1res[kp1state][:2] == states_kres[kstate])
        flag2 = np.all( states_kp1res[kp1state][0:] == states_kres[kstate])
        if ( flag1 and flag2 ):
            mu_eff_kres[kstate] += mu_eff_kp1res[kp1state]
        elif ( flag1 or flag2 ):
            tmp *= mu_eff_kp1res[kp1state]
    tmp = np.sqrt(tmp)
    mu_eff_kres[kstate] += tmp

## JFR - Here I am, but I really needed to check everything above!!

## solve for the conditional probabilities
### set up a dictionary for the inputs
inp_var = {} # nb - part of this dictionary is permanent
# P2res
for res in range(Nres-1):
    for s1 in range(len(states_2res)):
        for s2 in range(len(states_2res)):
            var = P2res[res][s1][s2]
            inp_var[var] = P2res_inp[s1][s2]
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
# P1res stat, boundary only
for res in range(2):
    for si in range(len(S2)):
        var = P1res_stat_end[res][si]
        if ( si == 1 ):
            val = 1.
        else:
            val = 0.
        inp_var[var] = val

# about to start the iterations, define all the necessary functions
def add_P2res_stat_dict_vals( dic, P2res_stat, P2res_stat_val, states_2res ):
    for res in range(Nres-1):
        for state in range(len(states_2res)):
            var = P2res_stat[res][state]
            dic[var] = P2res_stat_val[state]
    return dic
def sub_inp( eqns, dic ):
    for eqn in range(len(eqns)):
        eqns[eqn] = eqns[eqn].subs(dic)
    return eqns

def lambdify_vec( eqns, var ):
    fncs = [lambdify((var), eqn, modules='numpy') for eqn in eqns]
    return fncs

def gen_fv( fncs ):
    return lambda zz: np.array([fnc(*zz) for fnc in fncs])

def init_soln( N ):
    return 0.5*np.ones(N)

def sumsq_eqns( eqns, var ):
    sumsq = 0.
    for eqn in range(len(eqns)):
        sumsq += eqns[eqn].subs(var)**2
    return sumsq


# before starting, solve the boundary conditions in terms of P2res_stat_end and P2res_end
var_live = np.concatenate(P2res_stat_end).tolist()+np.concatenate(np.concatenate(P2res_end)).tolist()
bndry_soln = solve(eqns_bndry_cond,var_live)
# replace solns in the P1res eqns
eqns_P1res = sub_inp( eqns_P1res, bndry_soln )
# also replace permanent inputs
eqns_P1res = sub_inp( eqns_P1res, inp_var )
# replace both inputs in the P1stat eqns
#eqns_P1res_stat = sub_inp( eqns_P1res_stat, bndry_soln )
#eqns_P1res_stat = sub_inp( eqns_P1res_stat, inp_var )
# now, solve for P1res stat
#var_live = np.concatenate(P1res_stat).tolist()
#P1res_stat_soln = solve(eqns_P1res_stat,var_live,dict=True)
# replace in P1res eqns, now everything should be in terms of P1res and P2res_stat, without the boundaries
#eqns_P1res = sub_inp( eqns_P1res, P1res_stat_soln[0] )
# nb PN eqns already only depend on P1res and P2res, so we can wait to substitute

print 'starting the iterations...'

# initialize the P2res_stat inputs
inp_tmp = add_P2res_stat_dict_vals( {}, P2res_stat, mu_eff_2res, states_2res ) # this is temporary!
# add the P1res_stat inputs
for res in range(Nres):
    for state in range(2):
        var = P1res_stat[res][state]
        inp_tmp[var] = mu_eff_1res[state]

np.save('mu_eff_2res_0',mu_eff_2res)
np.save('mu_eff_1res_0',mu_eff_1res)

flag_conv = False
sumsq_P1res_norm = []
sumsq_PN_norm = []
sumsq_P1res_opt = []
iter_ctr = 0
tol = 1e-3
max_steps = 100
while (not flag_conv):

    # substitute the inputs into the eqns
    eqns = deepcopy(eqns_P1res)
    eqns = sub_inp( eqns, inp_tmp )

    # solve the equations with scipy
    var_live = np.concatenate(np.concatenate(P1res)).tolist()
    fncs = lambdify_vec( eqns, var_live )
    f_v = gen_fv( fncs )
    zz0 = init_soln(len(var_live))
    P1res_soln = fsolve(f_v, zz0)

    # check accuracy of the normalization
    soln_var = {}
    for var in range(len(var_live)):
        soln_var[var_live[var]] = P1res_soln[var]
    sumsq_P1res_norm.append(sumsq_eqns(eqns_norm_P1res,soln_var))

    # instead of what I did below, we can do something simpler
    # solve the constraints for 1 variable, plug in the answer for the 2nd, to obtain a ``biased'' estimate
    # or we could solve each set of eqns seperately as below, which might help

    if ( iter_ctr == 0 ):  # only necessary for the first step, when the P1res prob are inconsistent
        P1res_opt = np.zeros(len(var_live))
        for pair in range(len(var_live)/2):
            ind1 = 2*pair
            ind2 = ind1+1
            p0 = soln_var[var_live[ind1]]
            p1 = soln_var[var_live[ind2]]
            #P1res_opt[ind2] = (p1 - p0 + 1.)/2. # min sumsq
            P1res_opt[ind2] = p1 / ( p1 + p0 ) # min KLD
            P1res_opt[ind1] = 1. - P1res_opt[ind2]
        sumsq_P1res_opt.append( np.sum( (P1res_opt )**2 ) )
        P1res_opt_old = deepcopy(P1res_opt)
    else:
        P1res_opt = P1res_soln
        sumsq_P1res_opt.append( np.sum( (P1res_opt - P1res_opt_old)**2 ) )
        P1res_opt_old = deepcopy(P1res_opt)

    # replace the old values with the optimized ones
    for var in range(len(var_live)):
        soln_var[var_live[var]] = P1res_opt[var]

    # let's try to adjust the P1res probs to satisfy the proper constraints
    #eqns_opt_P1res = []
    #KLD = 0.
    #for var in var_live:
    #    #KLD += soln_var[var]*log(soln_var[var]) - soln_var[var]*log(var)
    #    KLD += (soln_var[var] - var)**2
    #eqns_opt_P1res.append(KLD)
    ## set up the eqns
    #fncs = lambdify_vec( eqns_opt_P1res, var_live )
    #f_v = gen_fv( fncs )
    ## build the Jacobian for help
    ## Build Jacobian:
    #jac_fv = [KLD.diff(var) for var in var_live]
    #jac_fncs = lambdify_vec( jac_fv, var_live )
    #jac_v = gen_fv( jac_fncs )
    #const_dics = []
    #bounds = np.ones(shape=(len(var_live),2))
    #bounds[:,0] *= 0.0
    #for con in range(len(eqns_norm_P1res)):
    #    fncs_const = lambdify((var_live), eqns_norm_P1res[con], modules='numpy')
    #    #f_v_const = gen_fv( fncs_const )
    #    f_v_const = lambda zz: [fncs_const(*zz)]
    #    const_dics.append( {'type':'eq','fun': f_v_const} )
    #zz0 = deepcopy(P1res_soln)
    #zz0 = init_soln(len(var_live))
    #P1_res_opt = minimize(f_v, zz0, constraints=const_dics, bounds=bounds, jac=jac_v )
    #sumsq_P1res_opt.append( np.sum( (P1res_opt - P1res_soln)**2 ) )
    #print P1res_soln
    #print P1res_opt

    # calculate the full probabilities
    tot_var = dict(inp_tmp.items()+soln_var.items()+inp_var.items())
    eqns = deepcopy(eqns_PN)
    eqns = sub_inp( eqns, tot_var )

    var_live = np.concatenate(PN).tolist()
    fncs = lambdify_vec( eqns, var_live )
    f_v = gen_fv( fncs )
    zz0 = init_soln(len(var_live))
    PN_soln = fsolve(f_v, zz0)

    # check accuracy of the normalization
    soln_var = {}
    for var in range(len(var_live)):
        soln_var[var_live[var]] = PN_soln[var]
    sumsq_PN_norm.append(sumsq_eqns(eqns_norm_PN,soln_var))

    # set the matrix values
    TN = np.zeros(shape=(len(states_Nres),len(states_Nres)))
    ctr = 0
    for s1 in range(len(states_Nres)): # should use reshape instead!
        for s2 in range(len(states_Nres)):
            TN[s1][s2] = PN_soln[ctr]
            ctr += 1
     
    # artificially fix the normalization
    TN = TN / np.sum(TN,axis=1,dtype=float,keepdims=True)
    if ( iter_ctr == 0 ):
        np.save('TN_0',TN)

    # now, get the stat dist for this matrix
    mle_tmp = pyemma.msm.markov_model(TN)
    mu_N = mle_tmp.eigenvectors_left(k=1)[0]

    # calculate the 2res stationary dist directly from mu_N, and replace the values in the dictionary
    # P2res_stat
    inp_tmp = {}
    for res in range(Nres-1):
        for s1 in range(len(states_2res)):
            states = np.where( np.all(states_Nres[:,res:res+2]==states_2res[s1],axis=1) == True )[0]
            tot = 0.
            for state in states:
                tot += mu_N[state]
            var = P2res_stat[res][s1]
            inp_tmp[var] = tot
    for res in range(Nres):
        for s1 in range(2):
            states = np.where( np.all(states_Nres[:,res:res+1]==s1,axis=1) == True )[0]
            tot = 0.
            for state in states:
                tot += mu_N[state]
            var = P1res_stat[res][s1]
            inp_tmp[var] = tot

    # convergence
    if ( (sumsq_P1res_opt[iter_ctr] < tol) and (sumsq_P1res_norm[iter_ctr] < tol) and (sumsq_PN_norm[iter_ctr] < tol) ):
        flag_conv = True

    print 'just finished iteration '+str(iter_ctr)
    print 'err P1res opt = '+str(sumsq_P1res_opt[iter_ctr])
    print 'err P1res norm = '+str(sumsq_P1res_norm[iter_ctr])
    print 'err PN norm = '+str(sumsq_PN_norm[iter_ctr])
    iter_ctr += 1

    if ( iter_ctr > max_steps ):
        flag_conv = True


# save the results
np.save('sumsq_P1res_opt',sumsq_P1res_opt)
np.save('sumsq_P1res_norm',sumsq_P1res_norm)
np.save('sumsq_PN_norm',sumsq_PN_norm)
np.save('TN_opt',TN)
np.save('P1res_opt',P1res_soln)

mu_eff_2res_f = []
for res in range(Nres-1):
    for s1 in range(len(states_2res)):
        states = np.where( np.all(states_Nres[:,res:res+2]==states_2res[s1],axis=1) == True )[0]
        tot = 0.
        for state in states:
            tot += mu_N[state]
        mu_eff_2res_f.append(tot)
np.save('mu_eff_2res_f',mu_eff_2res_f)
mu_eff_1res_f = []
for res in range(Nres):
    for s1 in range(2):
        states = np.where( np.all(states_Nres[:,res:res+1]==s1,axis=1) == True )[0]
        tot = 0.
        for state in states:
            tot += mu_N[state]
        mu_eff_1res_f.append(tot)
np.save('mu_eff_1res_f',mu_eff_1res_f)

'''
# let's check if the 2res prob are consistent
P2res_out = []
for res in range(Nres-1):
    P2res_out.append([])
    for s1 in range(len(states_2res)):
        states1 = np.where( np.all(states_Nres[:,res:res+2]==states_2res[s1],axis=1) == True )[0]
        P2res_out[res].append([])
        for s2 in range(len(states_2res)):
            states2 = np.where( np.all(states_Nres[:,res:res+2]==states_2res[s2],axis=1) == True )[0]
            tot = 0.
            for state1 in states1:
                tot_tmp = 0.
                for state2 in states2:
                    tot_tmp += TN[state1][state2]
                ind = np.where( np.all( states_2res == states_2res[s1], axis=1) == True )[0][0]
                tot_tmp /= mu_eff_2res_f[ind]
                tot += tot_tmp
            P2res_out[res][s1].append(tot/len(states1))
'''




