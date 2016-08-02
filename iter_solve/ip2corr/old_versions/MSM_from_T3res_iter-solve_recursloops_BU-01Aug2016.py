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
        PN[s1].append(Symbol('P_'+seq2+'g'+seq1, nonnegative=True))

# now, define the 3res variables
Pkp1res = []
for res in range(Nres-Ncorr):
    Pkp1res.append([])
    for s1 in range(len(states_kp1res)):
        Pkp1res[res].append([])
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        for s2 in range(len(states_kp1res)):
            seq2 = ''.join(str(s) for s in states_kp1res[s2])
            Pkp1res[res][s1].append(Symbol('P_'+str(res)+str(res+1)+str(res+2)+'_'+seq2+'g'+seq1, nonnegative=True))
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
                Pkp1res_end[res][s1].append( Symbol('P_'+'b1'+'b0'+str(0)+'_'+seq2+'g'+seq1, nonnegative=True) )
            elif ( res == 1 ):
                Pkp1res_end[res][s1].append( Symbol('P_'+'b0'+str(0)+str(1)+'_'+seq2+'g'+seq1, nonnegative=True) )
            elif ( res == 2 ):
                Pkp1res_end[res][s1].append( Symbol('P_'+str(Nres-2)+str(Nres-1)+'e0'+'_'+seq2+'g'+seq1, nonnegative=True) )
            else:
                Pkp1res_end[res][s1].append( Symbol('P_'+str(Nres-1)+'e0'+'e1'+'_'+seq2+'g'+seq1, nonnegative=True) )

# now, the conditional prob of jumping for each pair of res
Pkres = []
for res in range(Nres-(Ncorr-1)):
    Pkres.append([])
    for s1 in range(len(states_kres)):
        Pkres[res].append([])
        seq1 = ''.join(str(s) for s in states_kres[s1])
        for s2 in range(len(states_kres)):
            seq2 = ''.join(str(s) for s in states_kres[s2])
            Pkres[res][s1].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1, nonnegative=True))
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
                Pkres_end[res][s1].append(Symbol('P_'+'b1'+'b0'+'_'+seq2+'g'+seq1, nonnegative=True))
            elif ( res == 1 ):
                Pkres_end[res][s1].append(Symbol('P_'+'b0'+str(0)+'_'+seq2+'g'+seq1, nonnegative=True))
            elif ( res == 2 ):
                Pkres_end[res][s1].append(Symbol('P_'+str(Nres-1)+'e'+'_'+seq2+'g'+seq1, nonnegative=True))
            else:
                Pkres_end[res][s1].append(Symbol('P_'+'e0'+'e1'+'_'+seq2+'g'+seq1, nonnegative=True))


# finally, the static probabilties of the kp1's and k's
Pkp1res_stat = []
for res in range(Nres-Ncorr):
    Pkp1res_stat.append([])
    for s1 in range(len(states_kp1res)):
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        Pkp1res_stat[res].append(Symbol('P_'+str(res)+str(res+1)+str(res+2)+'_'+seq1, nonnegative=True))
# also define the boundaries
Pkp1res_stat_end = []
for res in range(2*Ncorr):
    Pkp1res_stat_end.append([])
    for s1 in range(len(states_kp1res)):
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        if ( res == 0 ):
            Pkp1res_stat_end[res].append(Symbol('P_'+'b1'+'b0'+str(0)+'_'+seq1, nonnegative=True))
        elif ( res == 1 ):
            Pkp1res_stat_end[res].append(Symbol('P_'+'b0'+str(0)+str(1)+'_'+seq1, nonnegative=True))
        elif ( res == 2 ):
            Pkp1res_stat_end[res].append(Symbol('P_'+str(Nres-2)+str(Nres-1)+'e0'+'_'+seq1, nonnegative=True))
        else:
            Pkp1res_stat_end[res].append(Symbol('P_'+str(Nres-1)+'e0'+'e1'+'_'+seq1, nonnegative=True))

Pkres_stat = []
for res in range(Nres-(Ncorr-1)):
    Pkres_stat.append([])
    for s1 in range(len(states_kres)):
        seq1 = ''.join(str(s) for s in states_kres[s1])
        Pkres_stat[res].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq1, nonnegative=True))
# also define the boundaries
Pkres_stat_end = []
for res in range(2*Ncorr):
    Pkres_stat_end.append([])
    for s1 in range(len(states_kres)):
        seq1 = ''.join(str(s) for s in states_kres[s1])
        if ( res == 0 ):
            Pkres_stat_end[res].append(Symbol('P_'+'b1'+'b0'+'_'+seq1, nonnegative=True))
        elif ( res == 1 ):
            Pkres_stat_end[res].append(Symbol('P_'+'b0'+str(0)+'_'+seq1, nonnegative=True))
        elif ( res == 2 ):
            Pkres_stat_end[res].append(Symbol('P_'+str(Nres-1)+'e0'+'_'+seq1, nonnegative=True))
        else:
            Pkres_stat_end[res].append(Symbol('P_'+'e0'+'e1'+'_'+seq1, nonnegative=True))

# put together the equations for each global prob
eqns_PN = []
for s in range(len(states_Nres)):
    for sp in range(len(states_Nres)):
        tmp = 1.
        for res in range(Nres-Ncorr):
            statep = states_Nres[sp][res:res+Ncorr+1]
            indp = np.where( np.all(states_kp1res==np.array(statep),axis=1) == True )[0][0]
            state = states_Nres[s][res:res+Ncorr+1]
            ind = np.where( np.all(states_kp1res==np.array(state),axis=1) == True )[0][0]
            tmp *= Pkp1res[res][ind][indp]
        for res in range(Nres-Ncorr-1):
            statep = states_Nres[sp][res+1]
            state = states_Nres[s][res+1]
            tmp /= Pkres[res+1][state][statep]
        tmp *= -1
        tmp += PN[s][sp]
        eqns_PN.append(tmp)

'''
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
'''

# let's do the same thing without all the loops
def get_poss_states( Nres ):
    states = []
    state = [0]*Nres
    for gstate in range(2**Nres):
        states.append(state[:])
        state[0] = (state[0]+1) % 2
        for res in range(1,Nres):
            if ( (state[res-1]+1) % 2 == 1 ):
                state[res] = (state[res]+1) % 2
            else:
                break
    return states

eqns_Pkres = []
Nres_corr = Ncorr+2*Ncorr
for res in range(Nres-(Ncorr-1)):
    Nxxp = [[0.]*len(states_kres) for _ in range(len(states_kres))]
    states = get_poss_states(Nres_corr)
    for state in states:
        statesp = get_poss_states(Nres_corr)
        for statep in statesp:
            tmp = 1.
            for kset in range(Nres_corr-Ncorr):
                indp = np.where( np.all(states_kp1res==statep[kset:kset+Ncorr+1],axis=1) == True )[0][0]
                ind = np.where( np.all(states_kp1res==state[kset:kset+Ncorr+1],axis=1) == True )[0][0]
                # need to take care of the boundary conditions, Ncond will grow, so I should figure out how to automate this
                if ( (res==0) and (kset==0) ):
                    tmp *= Pkp1res_end[0][ind][indp]*Pkp1res_stat_end[0][ind]
                elif ( (res==0) and (kset==1) ):
                    tmp *= Pkp1res_end[1][ind][indp]*Pkp1res_stat_end[1][ind]
                elif ( (res==1) and (kset==0) ):
                    tmp *= Pkp1res_end[1][ind][indp]*Pkp1res_stat_end[1][ind]
                elif ( (res==Nres-Ncorr-1) and (kset==Nres_corr-Ncorr-1) ):
                    tmp *= Pkp1res_end[2][ind][indp]*Pkp1res_stat_end[2][ind]
                elif ( (res==Nres-Ncorr) and (kset==Nres_corr-Ncorr-1) ):
                    tmp *= Pkp1res_end[3][ind][indp]*Pkp1res_stat_end[3][ind]
                elif ( (res==Nres-Ncorr) and (kset==Nres_corr-Ncorr-2) ):
                    tmp *= Pkp1res_end[2][ind][indp]*Pkp1res_stat_end[2][ind]
                else:
                    tmp *= Pkp1res[res-2+kset][ind][indp]*Pkp1res_stat[res-2+kset][ind]   
            ind = np.where( np.all(states_kres==state[Ncorr:Ncorr+Ncorr],axis=1) == True )[0][0]
            indp = np.where( np.all(states_kres==statep[Ncorr:Ncorr+Ncorr],axis=1) == True )[0][0]
            Nxxp[ind][indp] += tmp
    for kstate in range(len(states_kres)):
        den = Pkres_stat[res][kstate]
        for kstatep in range(len(states_kres)):
            eqns_Pkres.append( Pkres[res][kstate][kstatep] - (Nxxp[kstate][kstatep]**(0.5))/den )

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
            eqn_tmp += Pkres[res][s1][s2]
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
        tmp += Pkres_stat[res][s1]
    eqns_norm_Pkres_stat.append( tmp - 1 )


# specify the boundary conditions
eqns_bndry_cond = []
inp_bndry_cond = {} # try using a dic for the bndry instead of solving the equations
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
                P1_b1 = 0.
            else:
                P1_b1 = 1.
            if ( states_kp1res[s1][1] == 0 ):
                P1_b0 = 0.
            else:
                P1_b0 = 1.
            eqns_bndry_cond.append( var - P1_b1*P1_b0*tmp )
            inp_bndry_cond[var] = P1_b1*P1_b0*tmp
        elif ( res == 1 ):
            ind = np.where( np.all(states_kres==states_kp1res[s1][1:],axis=1) == True )[0][0]
            if ( states_kp1res[s1][0] == 0 ):
                P1_b0 = 0.
            else:
                P1_b0 = 1.
            eqns_bndry_cond.append( var - P1_b0*Pkres_stat[0][ind] )
            inp_bndry_cond[var] = P1_b0*Pkres_stat[0][ind]
        elif ( res == 2 ):
            ind = np.where( np.all(states_kres==states_kp1res[s1][:2],axis=1) == True )[0][0]
            if ( states_kp1res[s1][2] == 0 ):
                P1_e0 = 0.
            else:
                P1_e0 = 1.
            eqns_bndry_cond.append( var - Pkres_stat[Nres-(Ncorr-1)-1][ind]*P1_e0 )
            inp_bndry_cond[var] = Pkres_stat[Nres-(Ncorr-1)-1][ind]*P1_e0
        else:
            tmp = 0.
            for ynm1 in range(len(S2)):
                state = [ynm1,states_kp1res[s1][0]]
                ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                tmp += Pkres_stat[Nres-(Ncorr-1)-1][ind]
            if ( states_kp1res[s1][2] == 0 ):
                P1_e1 = 0.
            else:
                P1_e1 = 1.
            if ( states_kp1res[s1][1] == 0 ):
                P1_e0 = 0.
            else:
                P1_e0 = 1.
            eqns_bndry_cond.append( var - tmp*P1_e0*P1_e1 )
            inp_bndry_cond[var] = tmp*P1_e0*P1_e1

# and the kp1 res cond prob
for res in range(2*Ncorr):
    for s1 in range(len(states_kp1res)):
        for s2 in range(len(states_kp1res)):
            var = Pkp1res_end[res][s1][s2]
            if ( res == 0 ):
                tmp = 0.
                tmp_stat = 0.
                for y1 in range(len(S2)):
                    state = [states_kp1res[s1][2],y1]
                    ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                    # we need to also recalculate the single res prob for normalization
                    tmp_stat += Pkres_stat[0][ind]
                    for y1p in range(len(S2)):
                        statep = [states_kp1res[s2][2],y1p]
                        indp  = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
                        tmp += Pkres[0][ind][indp]*Pkres_stat[0][ind]
                if ( states_kp1res[s2][0] == 0 ):
                    P1_b1 = 0
                else:
                    P1_b1 = 1
                if ( states_kp1res[s2][1] == 0 ):
                    P1_b0 = 0
                else:
                    P1_b0 = 1
                eqns_bndry_cond.append( var - P1_b1*P1_b0*tmp/tmp_stat )
                inp_bndry_cond[var] = P1_b1*P1_b0*tmp/tmp_stat
            elif ( res == 1 ):
                ind1 = np.where( np.all(states_kres==states_kp1res[s1][1:],axis=1) == True )[0][0]
                ind2 = np.where( np.all(states_kres==states_kp1res[s2][1:],axis=1) == True )[0][0]
                if ( states_kp1res[s2][0] == 0 ):
                    P1_b0 = 0
                else:
                    P1_b0 = 1
                eqns_bndry_cond.append( var - P1_b0*Pkres[0][ind1][ind2] )
                inp_bndry_cond[var] = P1_b0*Pkres[0][ind1][ind2]
            elif ( res == 2 ):
                ind1 = np.where( np.all(states_kres==states_kp1res[s1][:2],axis=1) == True )[0][0]
                ind2 = np.where( np.all(states_kres==states_kp1res[s2][:2],axis=1) == True )[0][0]
                if ( states_kp1res[s2][2] == 0 ):
                    P1_e0 = 0
                else:
                    P1_e0 = 1
                eqns_bndry_cond.append( var - Pkres[Nres-2][ind1][ind2]*P1_e0 )
                inp_bndry_cond[var] = Pkres[Nres-2][ind1][ind2]*P1_e0
            else:
                tmp = 0.
                tmp_stat = 0.
                for ynm1 in range(len(S2)):
                    state = [ynm1,states_kp1res[s1][0]]
                    ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                    # we need to also recalculate the single res prob for normalization
                    tmp_stat += Pkres_stat[Nres-2][ind]
                    for ynm1p in range(len(S2)):
                        statep = [ynm1p,states_kp1res[s2][0]]
                        indp  = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
                        tmp += Pkres[Nres-2][ind][indp]*Pkres_stat[Nres-2][ind]
                if ( states_kp1res[s2][2] == 0 ):
                    P1_e1 = 0
                else:
                    P1_e1 = 1
                if ( states_kp1res[s2][1] == 0 ):
                    P1_e0 = 0
                else:
                    P1_e0 = 1
                eqns_bndry_cond.append( var - tmp*P1_e0*P1_e1/tmp_stat )
                inp_bndry_cond[var] = tmp*P1_e0*P1_e1/tmp_stat

# initialize the model by assuming that the kp1res stat prob correspond to the kp1res model (i.e., no correlations beyond kp1 res)
mle_tmp = pyemma.msm.markov_model(Pkp1res_inp)
mu_eff_kp1res = mle_tmp.eigenvectors_left(k=1)[0]
# also initialize the 1res stat prob, from mu_eff_2res
mu_eff_kres = np.zeros(2**(Ncorr))
for kp1state in range(len(states_kp1res)):
    ind1 = np.where( np.all( states_kres==states_kp1res[kp1state][:2], axis=1 ) )[0][0]
    ind2 = np.where( np.all( states_kres==states_kp1res[kp1state][1:], axis=1 ) )[0][0]
    mu_eff_kres[ind1] += 0.5*mu_eff_kp1res[kp1state]
    mu_eff_kres[ind2] += 0.5*mu_eff_kp1res[kp1state]


## solve for the conditional probabilities
### set up a dictionary for the inputs
inp_var = {} # nb - part of this dictionary is permanent
# Pkp1res
for res in range(Nres-Ncorr):
    for s1 in range(len(states_kp1res)):
        for s2 in range(len(states_kp1res)):
            var = Pkp1res[res][s1][s2]
            inp_var[var] = Pkp1res_inp[s1][s2]
# Pkres, boundary only
for res in range(2*Ncorr):
    for state in range(len(states_kres)):
        for statep in range(len(states_kres)):
            var = Pkres_end[res][state][statep]
            if ( states_kres[statep][0] == 0 or states_kres[statep][1] == 0 ):
                val = 0.
            else:
                val = 1.
            inp_var[var] = val
# Pkres stat, boundary only
for res in range(2*Ncorr):
    for s1 in range(len(states_kres)):
        var = Pkres_stat_end[res][s1]
        if ( (states_kres[s1][0] == 0) or (states_kres[s1][1] == 0) ):
            val = 0.
        else:
            val = 1.
        inp_var[var] = val


# about to start the iterations, define all the necessary functions
def add_Pkp1res_stat_dict_vals( dic, Pkp1res_stat, Pkp1res_stat_val, states_kp1res ):
    for res in range(Nres-Ncorr):
        for state in range(len(states_kp1res)):
            var = Pkp1res_stat[res][state]
            dic[var] = Pkp1res_stat_val[state]
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

def gen_jv( fncs ):
    return lambda zz: np.array([ [ifnc(*zz) for ifnc in jfncs] for jfncs in fncs ])

def init_soln( N ):
    return 0.5*np.ones(N)

def sumsq_eqns( eqns, var ):
    sumsq = 0.
    for eqn in range(len(eqns)):
        sumsq += eqns[eqn].subs(var)**2
    return sumsq


# before starting, solve the boundary conditions in terms of Pkp1res_stat_end and Pkp1res_end
var_live = np.concatenate(Pkp1res_stat_end).tolist()+np.concatenate(np.concatenate(Pkp1res_end)).tolist()
#bndry_soln = solve(eqns_bndry_cond,var_live)
# replace solns in the Pkres eqns
#eqns_Pkres = sub_inp( eqns_Pkres, bndry_soln )
eqns_Pkres = sub_inp( eqns_Pkres, inp_bndry_cond )
# also replace permanent inputs
eqns_Pkres = sub_inp( eqns_Pkres, inp_var )

print 'starting the iterations...'

# initialize the Pkp1res_stat inputs
inp_tmp = add_Pkp1res_stat_dict_vals( {}, Pkp1res_stat, mu_eff_kp1res, states_kp1res ) # this is temporary!
# add the Pkres_stat inputs
for res in range(Nres-(Ncorr-1)):
    for state in range(len(states_kres)):
        var = Pkres_stat[res][state]
        inp_tmp[var] = mu_eff_kres[state]

np.save('mu_eff_kp1res_0',mu_eff_kp1res)
np.save('mu_eff_kres_0',mu_eff_kres)

eqns_Pkres[10000]

flag_conv = False
sumsq_Pkres_norm = []
sumsq_PN_norm = []
sumsq_Pkres_opt = []
iter_ctr = 0
tol = 1e-3
max_steps = 3
'''
>>> const_dics = []
>>> bounds = np.ones(shape=(len(var_live),2))
>>> bounds[:,0] *= 0.0
>>> for con in range(len(eqns_norm_Pkres)):
...     fncs_const = lambdify((var_live), eqns_norm_Pkres[con], modules='numpy')
...     f_v_const = lambda zz: [fncs_const(*zz)]
...     const_dics.append( {'type':'eq','fun': f_v_const} )


>>> def LS_f_v( f_v, zz ):
...     sqrs = f_v(zz)
...     sqrs = sqrs**2
...     sqrs = np.sum(sqrs)
...     return sqrs
... 
>>> scalar_f_v = lambda zz: LS_f_v(f_v, zz)
>>> P1_res_opt = minimize(scalar_f_v, zz0, constraints=const_dics, bounds=bounds )
>>> P1_res_opt
     fun: 13.477573358377448


'''


while (not flag_conv):

    # substitute the inputs into the eqns
    eqns = deepcopy(eqns_Pkres)
    eqns = sub_inp( eqns, inp_tmp )

    # solve the equations with scipy
    var_live = np.concatenate(np.concatenate(Pkres)).tolist()
    fncs = lambdify_vec( eqns, var_live )
    f_v = gen_fv( fncs )
    # try adding the Jacobian for help
    jac_fv = []
    jac_fncs = []
    jac_v = []
    for eqn in range(len(eqns)):
        jac_fv.append( [eqns[eqn].diff(var) for var in var_live] )
        jac_fncs.append(lambdify_vec( jac_fv[eqn], var_live ) )
    jac_v = gen_jv( jac_fncs )
    # and maybe bounds
    bounds = np.ones(shape=(len(var_live),2))
    bounds[:,0] *= 0.0
    #
    zz0 = init_soln(len(var_live))
    if ( iter_ctr == 0 ): # all variables are equal for some reason
        Pkres_soln = scipy.optimize.least_squares(f_v, zz0, bounds=[0.25,1]).x
    else:
        Pkres_soln = fsolve(f_v, zz0, fprime=jac_v) # work with the log instead to ignore negative posibilities?!


    # check accuracy of the normalization
    soln_var = {}
    for var in range(len(var_live)):
        soln_var[var_live[var]] = Pkres_soln[var]
    sumsq_Pkres_norm.append(sumsq_eqns(eqns_norm_Pkres,soln_var))

    # instead of what I did below, we can do something simpler
    # solve the constraints for 1 variable, plug in the answer for the 2nd, to obtain a ``biased'' estimate
    # or we could solve each set of eqns seperately as below, which might help

    if ( iter_ctr == 0 ):  # only necessary for the first step, when the Pkres prob are inconsistent
        # analytic soln is harder to come by now, just rescale each set of probs
        Pkres_opt = np.zeros(len(var_live))
        gsize = len(states_kres)
        for group in range(len(var_live)/gsize):
            indi = (group*gsize)
            indf = (group*gsize)+gsize
            norm = np.sum(Pkres_soln[indi:indf+1])
            Pkres_opt[indi:indf+1] = Pkres_soln[indi:indf+1] / norm
        sumsq_Pkres_opt.append( np.sum( (Pkres_opt )**2 ) )
        Pkres_opt_old = deepcopy(Pkres_opt)
    else:
        Pkres_opt = Pkres_soln
        sumsq_Pkres_opt.append( np.sum( (Pkres_opt - Pkres_opt_old)**2 ) )
        Pkres_opt_old = deepcopy(Pkres_opt)

    # replace the old values with the optimized ones
    #for var in range(len(var_live)):
    #    soln_var[var_live[var]] = Pkres_opt[var]

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
    # Pkp1res_stat
    inp_tmp = {}
    for res in range(Nres-Ncorr):
        for s1 in range(len(states_kp1res)):
            states = np.where( np.all(states_Nres[:,res:res+Ncorr+1]==states_kp1res[s1],axis=1) == True )[0]
            tot = 0.
            for state in states:
                tot += mu_N[state]
            var = Pkp1res_stat[res][s1]
            inp_tmp[var] = tot
    for res in range(Nres-(Ncorr-1)):
        for s1 in range(len(states_kres)):
            states = np.where( np.all(states_Nres[:,res:res+Ncorr]==states_kres[s1],axis=1) == True )[0]
            tot = 0.
            for state in states:
                tot += mu_N[state]
            var = Pkres_stat[res][s1]
            inp_tmp[var] = tot

    # convergence
    if ( (sumsq_Pkres_opt[iter_ctr] < tol) and (sumsq_Pkres_norm[iter_ctr] < tol) and (sumsq_PN_norm[iter_ctr] < tol) ):
        flag_conv = True

    print 'just finished iteration '+str(iter_ctr)
    print 'err Pkres opt = '+str(sumsq_Pkres_opt[iter_ctr])
    print 'err Pkres norm = '+str(sumsq_Pkres_norm[iter_ctr])
    print 'err PN norm = '+str(sumsq_PN_norm[iter_ctr])
    iter_ctr += 1

    if ( iter_ctr > max_steps ):
        flag_conv = True


# save the results
np.save('sumsq_Pkres_opt',sumsq_Pkres_opt)
np.save('sumsq_Pkres_norm',sumsq_Pkres_norm)
np.save('sumsq_PN_norm',sumsq_PN_norm)
np.save('TN_opt',TN)
np.save('Pkres_opt',Pkres_soln)

mu_eff_kp1res_f = []
for res in range(Nres-Ncorr):
    for s1 in range(len(states_kp1res)):
        states = np.where( np.all(states_Nres[:,res:res+Ncorr+1]==states_kp1res[s1],axis=1) == True )[0]
        tot = 0.
        for state in states:
            tot += mu_N[state]
        mu_eff_kp1res_f.append(tot)
np.save('mu_eff_kp1res_f',mu_eff_kp1res_f)
mu_eff_kres_f = []
for res in range(Nres):
    for s1 in range(len(states_kres)):
        states = np.where( np.all(states_Nres[:,res:res+Ncorr]==states_kres[s1],axis=1) == True )[0]
        tot = 0.
        for state in states:
            tot += mu_N[state]
        mu_eff_kres_f.append(tot)
np.save('mu_eff_kres_f',mu_eff_kres_f)





