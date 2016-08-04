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

# for saving objects
import pickle
def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# user input
Nres = 5
Ncorr = 2 # == k
T_basenm = 'T3res'
flag_seqdep = True
flag_bndry_cond = False
bndry_cond_path = ''
tol = 1e-2
max_steps = 10

# fixed variables
S2 = [0,1]

# Read in system-specific quantities
Pkp1res_inp = []
Nsets = Nres-Ncorr
for group in range(Nsets):
    if ( flag_seqdep ):
        Pkp1res_inp.append(np.load(T_basenm+'_'+str(group)+'.npy'))
    else:
        Pkp1res_inp.append(np.load(T_basenm+'.npy'))
cckp1res = np.load('cc3res.npy')
cckres = np.load('cc2res.npy')
cc_full = np.load('cc_full.npy')


# first, get all the possible states of the full system
states_Nres = cc_full.astype(int)

# now, get all kp1 res states
states_kp1res = cckp1res.astype(int)
# and k res states
states_kres = cckres.astype(int)

# now, define the Nres variables
PN = []
for s1 in range(len(states_Nres)):
    PN.append([])
    seq1 = ''.join(str(s) for s in states_Nres[s1])
    for s2 in range(len(states_Nres)):
        seq2 = ''.join(str(s) for s in states_Nres[s2])
        PN[s1].append(Symbol('P_'+seq2+'g'+seq1, nonnegative=True))

# now, define the kp1res variables
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
            poss = ''
            if ( res < Ncorr ):
                for pos in range(0,Ncorr-res):
                    poss += 'b'+str(Ncorr-1-res-pos)
                for pos in range(Ncorr-res,Ncorr+1):
                    poss += str(pos-(Ncorr-res))
            else:
                for pos in range(0,2*Ncorr-res):
                    poss += str(Nres-Ncorr+res-Ncorr+pos) 
                for pos in range(2*Ncorr-res,2*Ncorr-1):
                    poss += 'e'+str(pos-(2*Ncorr-res))   
            Pkp1res_end[res][s1].append( Symbol('P_'+poss+'_'+seq2+'g'+seq1, nonnegative=True) )
'''
Pkp1res_end_old = []
for res in range(2*Ncorr):
    Pkp1res_end_old.append([])
    for s1 in range(len(states_kp1res)):
        Pkp1res_end_old[res].append([])
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        for s2 in range(len(states_kp1res)):
            seq2 = ''.join(str(s) for s in states_kp1res[s2])
            if ( res == 0 ):
                Pkp1res_end_old[res][s1].append( Symbol('P_'+'b1'+'b0'+str(0)+'_'+seq2+'g'+seq1, nonnegative=True) )
            elif ( res == 1 ):
                Pkp1res_end_old[res][s1].append( Symbol('P_'+'b0'+str(0)+str(1)+'_'+seq2+'g'+seq1, nonnegative=True) )
            elif ( res == 2 ):
                Pkp1res_end_old[res][s1].append( Symbol('P_'+str(Nres-2)+str(Nres-1)+'e0'+'_'+seq2+'g'+seq1, nonnegative=True) )
            else:
                Pkp1res_end_old[res][s1].append( Symbol('P_'+str(Nres-1)+'e0'+'e1'+'_'+seq2+'g'+seq1, nonnegative=True) )
'''

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
            poss = ''
            if ( res < Ncorr ):
                for pos in range(0,Ncorr-res):
                    poss += 'b'+str(Ncorr-1-res-pos)
                for pos in range(Ncorr-res,Ncorr):
                    poss += str(pos-(Ncorr-res))
            else:
                for pos in range(1,2*Ncorr-res):
                    poss += str(Nres-Ncorr+res-Ncorr+pos)
                for pos in range(2*Ncorr-res,2*Ncorr-1):
                    poss += 'e'+str(pos-(2*Ncorr-res))
            Pkres_end[res][s1].append(Symbol('P_'+poss+'_'+seq2+'g'+seq1, nonnegative=True))
'''
Pkres_end_old = []
for res in range(2*Ncorr):
    Pkres_end_old.append([])
    for s1 in range(len(states_kres)):
        Pkres_end_old[res].append([])
        seq1 = ''.join(str(s) for s in states_kres[s1])
        for s2 in range(len(states_kres)):
            seq2 = ''.join(str(s) for s in states_kres[s2])
            if ( res == 0 ):
                Pkres_end_old[res][s1].append(Symbol('P_'+'b1'+'b0'+'_'+seq2+'g'+seq1, nonnegative=True))
            elif ( res == 1 ):
                Pkres_end_old[res][s1].append(Symbol('P_'+'b0'+str(0)+'_'+seq2+'g'+seq1, nonnegative=True))
            elif ( res == 2 ):
                Pkres_end_old[res][s1].append(Symbol('P_'+str(Nres-1)+'e0'+'_'+seq2+'g'+seq1, nonnegative=True))
            else:
                Pkres_end_old[res][s1].append(Symbol('P_'+'e0'+'e1'+'_'+seq2+'g'+seq1, nonnegative=True))
'''

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
        poss = ''
        if ( res < Ncorr ):
            for pos in range(0,Ncorr-res):
                poss += 'b'+str(Ncorr-1-res-pos)
            for pos in range(Ncorr-res,Ncorr+1):
                poss += str(pos-(Ncorr-res))
        else:
            for pos in range(0,2*Ncorr-res):
                poss += str(Nres-Ncorr+res-Ncorr+pos)
            for pos in range(2*Ncorr-res,2*Ncorr-1):
                poss += 'e'+str(pos-(2*Ncorr-res))
        Pkp1res_stat_end[res].append(Symbol('P_'+poss+'_'+seq1, nonnegative=True))
'''
Pkp1res_stat_end_old = []
for res in range(2*Ncorr):
    Pkp1res_stat_end_old.append([])
    for s1 in range(len(states_kp1res)):
        seq1 = ''.join(str(s) for s in states_kp1res[s1])
        if ( res == 0 ):
            Pkp1res_stat_end_old[res].append(Symbol('P_'+'b1'+'b0'+str(0)+'_'+seq1, nonnegative=True))
        elif ( res == 1 ):
            Pkp1res_stat_end_old[res].append(Symbol('P_'+'b0'+str(0)+str(1)+'_'+seq1, nonnegative=True))
        elif ( res == 2 ):
            Pkp1res_stat_end_old[res].append(Symbol('P_'+str(Nres-2)+str(Nres-1)+'e0'+'_'+seq1, nonnegative=True))
        else:
            Pkp1res_stat_end_old[res].append(Symbol('P_'+str(Nres-1)+'e0'+'e1'+'_'+seq1, nonnegative=True))
'''

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
        poss = ''
        if ( res < Ncorr ):
            for pos in range(0,Ncorr-res):
                poss += 'b'+str(Ncorr-1-res-pos)
            for pos in range(Ncorr-res,Ncorr):
                poss += str(pos-(Ncorr-res))
        else:
            for pos in range(1,2*Ncorr-res):
                poss += str(Nres-Ncorr+res-Ncorr+pos)
            for pos in range(2*Ncorr-res,2*Ncorr-1):
                poss += 'e'+str(pos-(2*Ncorr-res))
        Pkres_stat_end[res].append(Symbol('P_'+poss+'_'+seq1, nonnegative=True))
'''
Pkres_stat_end_old = []
for res in range(2*Ncorr):
    Pkres_stat_end_old.append([])
    for s1 in range(len(states_kres)):
        seq1 = ''.join(str(s) for s in states_kres[s1])
        if ( res == 0 ):
            Pkres_stat_end_old[res].append(Symbol('P_'+'b1'+'b0'+'_'+seq1, nonnegative=True))
        elif ( res == 1 ):
            Pkres_stat_end_old[res].append(Symbol('P_'+'b0'+str(0)+'_'+seq1, nonnegative=True))
        elif ( res == 2 ):
            Pkres_stat_end_old[res].append(Symbol('P_'+str(Nres-1)+'e0'+'_'+seq1, nonnegative=True))
        else:
            Pkres_stat_end_old[res].append(Symbol('P_'+'e0'+'e1'+'_'+seq1, nonnegative=True))
'''

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
            statep = states_Nres[sp][res+1:res+1+Ncorr]
            indp = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
            state = states_Nres[s][res+1:res+1+Ncorr]
            ind = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
            tmp /= Pkres[res+1][ind][indp]
        tmp *= -1
        tmp += PN[s][sp]
        eqns_PN.append(tmp)

# Set up the equations for the kres cond probs
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
Nk = 2*Ncorr
Nres_corr = Ncorr+Nk
for res in range(Nres-(Ncorr-1)):
    Nxxp = [[0.]*len(states_kres) for _ in range(len(states_kres))]
    states = get_poss_states(Nres_corr)
    for state in states:
        statesp = get_poss_states(Nres_corr)
        for statep in statesp:
            tmp = 1.
            for kset in range(Nk):
                indp = np.where( np.all(states_kp1res==statep[kset:kset+Ncorr+1],axis=1) == True )[0][0]
                ind = np.where( np.all(states_kp1res==state[kset:kset+Ncorr+1],axis=1) == True )[0][0]
                ind2p = np.where( np.all(states_kres==statep[kset:kset+Ncorr],axis=1) == True )[0][0]
                ind2 = np.where( np.all(states_kres==state[kset:kset+Ncorr],axis=1) == True )[0][0]
                if ( (res<Ncorr) and (kset<Ncorr-res) ):
                    tmp *= Pkp1res_end[res+kset][ind][indp]*Pkp1res_stat_end[res+kset][ind]
                    if ( (kset > 0) and (kset != Ncorr) ):
                        tmp /= (Pkres_end[res+kset][ind2][ind2p]*Pkres_stat_end[res+kset][ind2])
                elif ( (res>Nres-Nk) and (kset>(Nk-1)-( res-(Nres-Nk+1)+1 )) ): # nb - the second term is kmax-Nksets  
                    # some useful quantities - old!
                    # pos_ind = pos_start + kset - kmin
                    # Nksets = res-(Nres-Ncorr-1)+1, 
                    # pos_start = Nres-1-Ncorr
                    # dkset = kset - ksetmax, posmax = Nres-Ncorr, dposmax = posmax-res
                    # ksetmax = Nk-1, kmin = kmax-Nksets+1
                    # some useful quantities - generalized for > 2 corr
                    # pos_start = Nres-Nk+1
                    # Nksets = res - pos_start + 1
                    # pos_ind = pos_start + kset - kmin
                    # dkset = kset - ksetmax, posmax = Nres-Ncorr, dposmax = posmax-res
                    # ksetmax = Nk-1, kmin = kmax-Nksets+1
                    # some useful quantities - generalized for > 2 corr
                    pos_start = (Nres-Nk+1)
                    Nksets = (res-pos_start+1)
                    kmax = (Nk-1)
                    endind_start = (kmax+1-Ncorr)
                    kmin = ( kmax - Nksets + 1 )
                    kmid = Ncorr
                    pos_ind = endind_start + kset - kmin 
                    tmp *= Pkp1res_end[pos_ind][ind][indp]*Pkp1res_stat_end[pos_ind][ind] 
                    if ( kset != kmid ):
                        if ( res+kset > Nres ):
                            tmp /= (Pkres_end[pos_ind-1][ind2][ind2p]*Pkres_stat_end[pos_ind-1][ind2]) # I am pretty sure about the pos_ind-1, but should check!
                        else:
                            tmp /= (Pkres[res-Ncorr+kset][ind2][ind2p]*Pkres_stat[res-Ncorr+kset][ind2])
                else:
                    tmp *= Pkp1res[res-Ncorr+kset][ind][indp]*Pkp1res_stat[res-Ncorr+kset][ind]
                    if ( (kset!=0) and (kset!=Ncorr) ):
                        tmp /= (Pkres[res-Ncorr+kset][ind2][ind2p]*Pkres_stat[res-Ncorr+kset][ind2])
           
                # need to take care of the boundary conditions, Ncond will grow, so I should figure out how to automate this
                #if ( (res==0) and (kset==0) ):
                #    tmp *= Pkp1res_end[0][ind][indp]*Pkp1res_stat_end[0][ind]
                #elif ( (res==0) and (kset==1) ):
                #    tmp *= Pkp1res_end[1][ind][indp]*Pkp1res_stat_end[1][ind]
                #    tmp /= (Pkres_end[1][ind2][ind2p]*Pkres_stat_end[1][ind2])
                #elif ( (res==1) and (kset==0) ):
                #    tmp *= Pkp1res_end[1][ind][indp]*Pkp1res_stat_end[1][ind]
                #elif ( (res==Nres-Ncorr-1) and (kset==Nres_corr-Ncorr-1) ):
                #    tmp *= Pkp1res_end[2][ind][indp]*Pkp1res_stat_end[2][ind]
                #    tmp /= (Pkres[res-2+kset][ind2][ind2p]*Pkres_stat[res-2+kset][ind2])
                #elif ( (res==Nres-Ncorr) and (kset==Nres_corr-Ncorr-1) ):
                #    tmp *= Pkp1res_end[3][ind][indp]*Pkp1res_stat_end[3][ind]
                #    tmp /= (Pkres_end[2][ind2][ind2p]*Pkres_stat_end[2][ind2])
                #elif ( (res==Nres-Ncorr) and (kset==Nres_corr-Ncorr-2) ):
                #    tmp *= Pkp1res_end[2][ind][indp]*Pkp1res_stat_end[2][ind]
                #else:
                #    tmp *= Pkp1res[res-2+kset][ind][indp]*Pkp1res_stat[res-2+kset][ind] 
                #    if ( (kset == 1) or (kset == Nres_corr-Ncorr-1) ):
                #        tmp /= (Pkres[res-2+kset][ind2][ind2p]*Pkres_stat[res-2+kset][ind2])

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
inp_bndry_cond_stat = {} # try using a dic for the bndry instead of solving the equations
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
            inp_bndry_cond_stat[var] = P1_b1*P1_b0*tmp
        elif ( res == 1 ):
            ind = np.where( np.all(states_kres==states_kp1res[s1][1:],axis=1) == True )[0][0]
            if ( states_kp1res[s1][0] == 0 ):
                P1_b0 = 0.
            else:
                P1_b0 = 1.
            eqns_bndry_cond.append( var - P1_b0*Pkres_stat[0][ind] )
            inp_bndry_cond_stat[var] = P1_b0*Pkres_stat[0][ind]
        elif ( res == 2 ):
            ind = np.where( np.all(states_kres==states_kp1res[s1][:2],axis=1) == True )[0][0]
            if ( states_kp1res[s1][2] == 0 ):
                P1_e0 = 0.
            else:
                P1_e0 = 1.
            eqns_bndry_cond.append( var - Pkres_stat[Nres-(Ncorr-1)-1][ind]*P1_e0 )
            inp_bndry_cond_stat[var] = Pkres_stat[Nres-(Ncorr-1)-1][ind]*P1_e0
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
            inp_bndry_cond_stat[var] = tmp*P1_e0*P1_e1

# and the kp1 res cond prob
inp_bndry_cond = {} # separate the cond prob from the static prob boundary conditions
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

# and the kres static prop
# Pkres stat, boundary only
inp_bndry_cond_kres_stat = {}
for res in range(2*Ncorr):
    for s1 in range(len(states_kres)):
        var = Pkres_stat_end[res][s1]
        if ( (res == 0) or (res==3) ):
            if ( (states_kres[s1][0] == 0) or (states_kres[s1][1] == 0) ):
                val = 0.
            else:
                val = 1.
            inp_bndry_cond_kres_stat[var] = val
        elif ( res == 1 ):
            tmp = 0.
            for y1 in range(len(S2)):
                state = [states_kres[s1][1],y1]
                ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                tmp += Pkres_stat[0][ind]
            if ( states_kp1res[s1][0] == 0 ):
                P1_b0 = 0.
            else:
                P1_b0 = 1.
            inp_bndry_cond_kres_stat[var] = P1_b0*tmp
        elif ( res == 2 ):
            tmp = 0.
            for ynm1 in range(len(S2)):
                state = [ynm1,states_kres[s1][0]]
                ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                tmp += Pkres_stat[Nres-(Ncorr-1)-1][ind]
            if ( states_kp1res[s1][1] == 0 ):
                P1_e0 = 0.
            else:
                P1_e0 = 1.
            inp_bndry_cond_kres_stat[var] = tmp*P1_e0

# and the kres cond prob
inp_bndry_cond_kres = {} # separate the cond prob from the static prob boundary conditions
for res in range(2*Ncorr):
    for s1 in range(len(states_kres)):
        for s2 in range(len(states_kres)):
            var = Pkres_end[res][s1][s2]
            if ( (res == 0) or (res == 3) ):
                if ( states_kres[s2][0] == 0 or states_kres[s2][1] == 0 ):
                    val = 0.
                else:
                    val = 1.
                inp_bndry_cond_kres[var] = val
            elif ( res == 1 ):
                tmp = 0.
                tmp_stat = 0.
                for y1 in range(len(S2)):
                    state = [states_kres[s1][1],y1]
                    ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                    # we need to also recalculate the single res prob for normalization
                    tmp_stat += Pkres_stat[0][ind]
                    for y1p in range(len(S2)):
                        statep = [states_kres[s2][1],y1p]
                        indp  = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
                        tmp += Pkres[0][ind][indp]*Pkres_stat[0][ind]
                if ( states_kres[s2][0] == 0 ):
                    P1_b0 = 0.
                else:
                    P1_b0 = 1.
                inp_bndry_cond_kres[var] = P1_b0*tmp/tmp_stat
            elif ( res == 2 ):
                tmp = 0.
                tmp_stat = 0.
                for ynm1 in range(len(S2)):
                    state = [ynm1,states_kres[s1][0]]
                    ind  = np.where( np.all(states_kres==np.array(state),axis=1) == True )[0][0]
                    # we need to also recalculate the single res prob for normalization
                    tmp_stat += Pkres_stat[Nres-2][ind]
                    for ynm1p in range(len(S2)):
                        statep = [ynm1p,states_kres[s2][0]]
                        indp  = np.where( np.all(states_kres==np.array(statep),axis=1) == True )[0][0]
                        tmp += Pkres[Nres-2][ind][indp]*Pkres_stat[Nres-2][ind]
                if ( states_kres[s2][1] == 0 ):
                    P1_e0 = 0.
                else:
                    P1_e0 = 1.
                inp_bndry_cond_kres[var] = tmp*P1_e0/tmp_stat


# initialize the model by assuming that the kp1res stat prob correspond to the kp1res model (i.e., no correlations beyond kp1 res)
mu_eff_kp1res = []
mu_eff_kres = []
for group in range(Nsets):
    mle_tmp = pyemma.msm.markov_model(Pkp1res_inp[group])
    mu_eff_kp1res.append(mle_tmp.eigenvectors_left(k=1)[0])
# also initialize the 1res stat prob, from mu_eff_2res
for res in range(Nres-(Ncorr-1)):
    mu_eff_kres.append(np.zeros(2**(Ncorr)))
    for kp1state in range(len(states_kp1res)):
        ind1 = np.where( np.all( states_kres==states_kp1res[kp1state][:2], axis=1 ) )[0][0]
        ind2 = np.where( np.all( states_kres==states_kp1res[kp1state][1:], axis=1 ) )[0][0]
        if ( res == 0 ):
            mu_eff_kres[res][ind1] += mu_eff_kp1res[res][kp1state]
        elif ( res == Nres-(Ncorr-1)-1 ):
            mu_eff_kres[res][ind2] += mu_eff_kp1res[res-1][kp1state]
        else:
            mu_eff_kres[res][ind1] += 0.5*mu_eff_kp1res[res][kp1state]
            mu_eff_kres[res][ind2] += 0.5*mu_eff_kp1res[res-1][kp1state]


## solve for the conditional probabilities
### set up a dictionary for the inputs
inp_var = {} # nb - part of this dictionary is permanent
# Pkp1res
for res in range(Nres-Ncorr):
    for s1 in range(len(states_kp1res)):
        for s2 in range(len(states_kp1res)):
            var = Pkp1res[res][s1][s2]
            inp_var[var] = Pkp1res_inp[res][s1][s2]


# about to start the iterations, define all the necessary functions
def add_Pkp1res_stat_dict_vals( dic, Pkp1res_stat, Pkp1res_stat_val, states_kp1res ):
    for res in range(Nres-Ncorr):
        for state in range(len(states_kp1res)):
            var = Pkp1res_stat[res][state]
            dic[var] = Pkp1res_stat_val[res][state]
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

# before starting, input the boundary relationships and values
# this is expensive, if already done, just read in the saved eqns
if ( flag_bndry_cond ):
    with open(bndry_cond_path+'eqns_Pkres.pkl', 'rb') as f:
        eqns_Pkres = pickle.load(f)
else:
    eqns_Pkres = sub_inp( eqns_Pkres, inp_bndry_cond )
    eqns_Pkres = sub_inp( eqns_Pkres, inp_bndry_cond_stat )
    eqns_Pkres = sub_inp( eqns_Pkres, inp_bndry_cond_kres )
    eqns_Pkres = sub_inp( eqns_Pkres, inp_bndry_cond_kres_stat )
    save_object('eqns_Pkres.pkl', eqns_Pkres)

# also replace permanent (i.e., not updated during the iterations) inputs
eqns_Pkres = sub_inp( eqns_Pkres, inp_var )

eqns_Pkres[1000]

print 'starting the iterations...'

# initialize the Pkp1res_stat inputs
inp_tmp = add_Pkp1res_stat_dict_vals( {}, Pkp1res_stat, mu_eff_kp1res, states_kp1res ) # this is temporary!
# add the Pkres_stat inputs
for res in range(Nres-(Ncorr-1)):
    for state in range(len(states_kres)):
        var = Pkres_stat[res][state]
        inp_tmp[var] = mu_eff_kres[res][state]

np.save('mu_eff_kp1res_0',mu_eff_kp1res)
np.save('mu_eff_kres_0',mu_eff_kres)


flag_conv = False
sumsq_Pkres_norm = []
sumsq_PN_norm = []
sumsq_Pkres_opt = []
iter_ctr = 0
while (not flag_conv):

    # substitute the inputs into the eqns
    eqns = deepcopy(eqns_Pkres)
    eqns = sub_inp( eqns, inp_tmp )

    # solve the equations with scipy
    var_live = np.concatenate(np.concatenate(Pkres)).tolist()
    fncs = lambdify_vec( eqns, var_live )
    f_v = gen_fv( fncs )
    zz0 = init_soln(len(var_live))
    Pkres_soln = fsolve(f_v, zz0)

    # check accuracy of the normalization
    soln_var = {}
    for var in range(len(var_live)):
        soln_var[var_live[var]] = Pkres_soln[var]
    sumsq_Pkres_norm.append(sumsq_eqns(eqns_norm_Pkres,soln_var))

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
    for var in range(len(var_live)):
        soln_var[var_live[var]] = Pkres_opt[var]

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





