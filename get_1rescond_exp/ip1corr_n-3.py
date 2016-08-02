import numpy as np
from sympy import *

# variables
S2 = [0,1]
Nres = 3
# first, get all the possible states of the full system
states_Nres = []
state1 = 0
state2 = 0
state3 = 0
for gstate in range(2**Nres):
    states_Nres.append([state1,state2,state3])
    state3 = (state3+1)%2
    if ( (state3+1) % 2 == 1 ):
        state2 = (state2+1)%2
        if ( (state2+1) % 2 == 1 ):
            state1 = (state1+1)%2

#print states_Nres

# now, get all 2 res states
states_2res = []
state1 = 0
state2 = 0
for gstate in range(2**2):
    states_2res.append([state1,state2])
    state2 = (state2+1)%2
    if ( (state2+1) % 2 == 1 ):
        state1 = (state1+1)%2

#print states_2res

# now, define the Nres variables
PN = []
for s1 in range(len(states_Nres)):
    PN.append([])
    seq1 = ''.join(str(s) for s in states_Nres[s1])
    for s2 in range(len(states_Nres)):
        seq2 = ''.join(str(s) for s in states_Nres[s2])
        PN[s1].append(Symbol('P_'+seq2+'g'+seq1))

#print PN

# now, define the 2res variables
P2 = []
for s1 in range(len(states_2res)):
    P2.append([])
    seq1 = ''.join(str(s) for s in states_2res[s1])
    for s2 in range(len(states_2res)):
        seq2 = ''.join(str(s) for s in states_2res[s2])
        P2[s1].append(Symbol('P_'+seq2+'g'+seq1))

#print P2

# we also need pair specific 2res variables
P2res = []
for res in range(Nres-1):
    P2res.append([])
    for s1 in range(len(states_2res)):
        P2res[res].append([])
        seq1 = ''.join(str(s) for s in states_2res[s1])
        for s2 in range(len(states_2res)):
            seq2 = ''.join(str(s) for s in states_2res[s2])
            P2res[res][s1].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1))

#print P2res

# now, the conditional prob of jumping for each res
P1res = []
for res in range(Nres):
    P1res.append([])
    for s1 in range(len(S2)):
        P1res[res].append([])
        seq1 = str(S2[s1])
        for s2 in range(len(S2)):
            seq2 = str(S2[s2])
            P1res[res][s1].append(Symbol('P_'+str(res)+'_'+seq2+'g'+seq1))

# finally, the static probabilties of the pairs and singlets

# we also need pair specific 2res variables
P2res_stat = []
for res in range(Nres-1):
    P2res_stat.append([])
    for s1 in range(len(states_2res)):
        seq1 = ''.join(str(s) for s in states_2res[s1])
        P2res_stat[res].append(Symbol('P_'+str(res)+str(res+1)+'_'+seq1))

P1res_stat = []
for res in range(Nres):
    P1res_stat.append([])
    for s1 in range(len(S2)):
        seq1 = str(S2[s1])
        P1res_stat[res].append(Symbol('P_'+str(res)+'_'+seq1))


# put together the equations for each global prob
eqns = []
# now, define the Nres variables
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
#        eqns.append(tmp)

# now, add the definitions of the 1res conditional and static probabilties
for res in range(Nres):
    for si in range(len(S2)):
        pxpx = [0,0]
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
                                seqp = ''.join(str(s) for s in states_2res[ind1p])
                                seq = ''.join(str(s) for s in states_2res[ind1])
                                P2res_end = Symbol('P_'+str(res-1)+str(res)+'_'+seqp+'g'+seq)
                                P2res_stat_end = Symbol('P_'+str(res-1)+str(res)+'_'+seq)
                                pxpx[sip] += P2res_end*P2res[res][ind2][ind2p]*P2res_stat_end*P2res_stat[res][ind2]
                            elif ( res == Nres-1 ):
                                seqp = ''.join(str(s) for s in states_2res[ind2p])
                                seq = ''.join(str(s) for s in states_2res[ind2])
                                P2res_end = Symbol('P_'+str(res)+str(res+1)+'_'+seqp+'g'+seq)
                                P2res_stat_end = Symbol('P_'+str(res)+str(res+1)+'_'+seq)
                                pxpx[sip] += P2res[res-1][ind1][ind1p]*P2res_end*P2res_stat[res-1][ind1]*P2res_stat_end
		            else:
                                pxpx[sip] += P2res[res-1][ind1][ind1p]*P2res[res][ind2][ind2p]*P2res_stat[res-1][ind1]*P2res_stat[res][ind2]
        # for each res and particular set of states, but input in pairs of states
        eqns.append( P1res[res][si][0]**2 - (pxpx[0] / np.sum(pxpx)) )
        eqns.append( P1res[res][si][1]**2 - (pxpx[1] / np.sum(pxpx)) )
        # also add the 1 res static definitions for each res
        eqns.append( P1res_stat[res][si] - np.sum(pxpx) )

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
                                        seqp = ''.join(str(s) for s in states_2res[ind1p])
                                        seq = ''.join(str(s) for s in states_2res[ind1])
                                        P2res_end = Symbol('P_'+str(res-1)+str(res)+'_'+seqp+'g'+seq)
                                        P2res_stat_end = Symbol('P_'+str(res-1)+str(res)+'_'+seq)
                                        tmp1 = P2res_end*P2res_stat_end
                                    else:
                                        tmp1 = P2res[res-1][ind1][ind1p]*P2res_stat[res-1][ind1]
                                    if ( res == Nres-2 ):
                                        seqp = ''.join(str(s) for s in states_2res[ind3p])
                                        seq = ''.join(str(s) for s in states_2res[ind3])
                                        P2res_end = Symbol('P_'+str(res+1)+str(res+2)+'_'+seqp+'g'+seq)
                                        P2res_stat_end = Symbol('P_'+str(res+1)+str(res+2)+'_'+seq)
                                        tmp3 = P2res_end*P2res_stat_end 
                                    else:
                                        tmp3 = P2res[res+1][ind3][ind3p]*P2res_stat[res+1][ind3]
                                    den = P1res[res][si][sip]*P1res[res+1][sip1][sip1p]*P1res_stat[res][si]*P1res_stat[res+1][sip1]
                                    tmp2 = ( P2res[res][ind2][ind2p]*P2res_stat[res][ind2] ) / den
                                    tmp += tmp1*tmp2*tmp3
                # for each pair of residues and particular set of states
                eqns.append( P2res_stat[res][si] - tmp )

# add the normalization conditions
# TN norm
#for s1 in range(len(states_Nres)):
#    eqns.append( np.sum(PN,axis=1)[s1] - 1 )
# 1 res cond norm
for res in range(Nres):
    for si in range(len(S2)):
        eqns.append( P1res[res][si][0] + P1res[res][si][1] - 1 )
# 2 res stat norm
for res in range(Nres-1):
    tmp = 0
    for si in range(len(S2)):
        for sip1 in range(len(S2)):
            state = [si,sip1]
            ind = np.where( np.all(states_2res==np.array(state),axis=1) == True )[0][0]
            tmp += P2res_stat[res][ind]
    eqns.append( tmp - 1 )
# 1 res stat norm
for res in range(Nres):
    tmp = 0
    for si in range(len(S2)):
        tmp += P1res_stat[res][si]
    eqns.append( tmp - 1 )

# give the T2res as input to simplify, same for each res
P2res_inp = [[ 0.6421973 ,  0.11902942,  0.16855856,  0.07021472],[ 0.06320449,  0.59824318,  0.13032524,  0.2082271 ],[ 0.09646157,  0.14045539,  0.61951799,  0.14356505],[ 0.03158511,  0.17639972,  0.11284946,  0.67916571]]
# we also need pair specific 2res variables
for res in range(-1,Nres):
    for s1 in range(len(states_2res)):
        seq1 = ''.join(str(s) for s in states_2res[s1])
        for s2 in range(len(states_2res)):
            seq2 = ''.join(str(s) for s in states_2res[s2])
            if (res == -1):
                var = Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1)
            elif (res == Nres-1):
                var = Symbol('P_'+str(res)+str(res+1)+'_'+seq2+'g'+seq1)
            else:
                var = P2res[res][s1][s2]
#            eqns.append( var - P2res_inp[s1][s2] )

# specify the boundary conditions
# set the 1res static prob to 1/0
res = -1
state = 0
var = Symbol('P_'+str(res)+'_'+str(state))
#eqns.append( var )
state = 1
var = Symbol('P_'+str(res)+'_'+str(state))
#eqns.append( var - 1 )
res = 3
state = 0
var = Symbol('P_'+str(res)+'_'+str(state))
#eqns.append( var )
state = 1
var = Symbol('P_'+str(res)+'_'+str(state))
#eqns.append( var - 1 )
# now, relate the 2res static prop
res = -1
for s1 in range(len(states_2res)):
    seq1 = ''.join(str(s) for s in states_2res[s1])
    var = Symbol('P_'+str(res)+str(res+1)+'_'+seq1)
    eqns.append( var - P1res_stat[res+1][states_2res[s1][1]] )
res = 2
for s1 in range(len(states_2res)):
    seq1 = ''.join(str(s) for s in states_2res[s1])
    var = Symbol('P_'+str(res)+str(res+1)+'_'+seq1)
    eqns.append( var - P1res_stat[res][states_2res[s1][0]] )
# finally, the 1 res cond prob
res = -1
for si in range(len(S2)):
    for sip in range(len(S2)):
        var = Symbol('P_'+str(res)+'_'+str(si)+'g'+str(sip))
        if ( si == 1 ):
            val = 1
        else:
            val = 0
#        eqns.append( var - val )
res = 3
for si in range(len(S2)):
    for sip in range(len(S2)):
        var = Symbol('P_'+str(res)+'_'+str(si)+'g'+str(sip))
        if ( si == 1 ):
            val = 1
        else:
            val = 0
#        eqns.append( var - val )

#soln = solve( eqns, np.concatenate(PN).tolist() )
soln = solve( eqns, np.concatenate(np.concatenate(P1res)).tolist() )

import pickle
def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object('soln.pkl', soln)



