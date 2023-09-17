def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)                     & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

def picker(args, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, spt=None, sst=None):

    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
        Detection probabilities. 
        
    yh2 : 1D array
        P arrival probabilities.  
        
    yh3 : 1D array
        S arrival probabilities. 
        
    yh1_std : 1D array
        Detection standard deviations. 
        
    yh2_std : 1D array
        P arrival standard deviations.  
        
    yh3_std : 1D array
        S arrival standard deviations. 
        
    spt : {int, None}, default=None    
        P arrival time in sample.
        
    sst : {int, None}, default=None
        S arrival time in sample. 
        
   
    Returns
    --------    
    matches: dic
        Contains the information for the detected and picked event.            
        
    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
            
    pick_errors : dic                
        {detection statr-time:[ P_ground_truth - P_pick, S_ground_truth - S_pick]}
        
    yh3: 1D array             
        normalized S_probability                              
                
    """               
    
 #   yh3[yh3>0.04] = ((yh1+yh3)/2)[yh3>0.04] 
 #   yh2[yh2>0.10] = ((yh1+yh2)/2)[yh2>0.10] 
             
    detection = trigger_onset(yh1, args['detection_threshold'], args['detection_threshold'])
    pp_arr = detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = detect_peaks(yh3, mph=args['S_threshold'], mpd=1)
          
    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    if len(pp_arr) > 0:
        P_uncertainty = None  
            
        for pick in range(len(pp_arr)): 
            pauto = pp_arr[pick]
                        
            if args['estimate_uncertainty'] and pauto:
                P_uncertainty = np.round(yh2_std[int(pauto)], 3)
                    
            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3) 
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})                 

    
    if len(ss_arr) > 0:
        S_uncertainty = None  
            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]
                   
            if args['estimate_uncertainty'] and sauto:
                S_uncertainty = np.round(yh3_std[int(sauto)], 3)
                    
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})             
            
    if len(detection) > 0:
        D_uncertainty = None  
        
        for ev in range(len(detection)):                                 
            if args['estimate_uncertainty']:               
                D_uncertainty = np.mean(yh1_std[detection[ev][0]:detection[ev][1]])
                D_uncertainty = np.round(D_uncertainty, 3)
                    
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
                    
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][1]]})            
    
    # matching the detection and picks
    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []
        
        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a,x] for x in l2[b:e]])
            
        best_pair = None
        for pr in ans: 
            ds = pr[1]-pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds           
        return best_pair


    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        S_error = None
        P_error = None        
        if int(ed-bg) >= 10:
                                    
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.update({Ss : S_val}) 
             
            if len(candidate_Ss) > 1:                
# =============================================================================
#                 Sr_st = 0
#                 buffer = {}
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     if S_valCan[0] > Sr_st:
#                         buffer = {SsCan : S_valCan}
#                         Sr_st = S_valCan[0]
#                 candidate_Ss = buffer
# =============================================================================              
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}


            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                        candidate_Ps.update({Ps : P_val}) 
                else:         
                    if Ps > bg-100 and Ps < ed:
                        candidate_Ps.update({Ps : P_val}) 
                    
            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan} 
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer
                    
            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}
                    
                    
# =============================================================================
#             Ses =[]; Pes=[]
#             if len(candidate_Ss) >= 1:
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     Ses.append(SsCan) 
#                                 
#             if len(candidate_Ps) >= 1:
#                 for PsCan, P_valCan in candidate_Ps.items():
#                     Pes.append(PsCan) 
#             
#             if len(Ses) >=1 and len(Pes) >= 1:
#                 PS = pair_PS(Pes, Ses, ed-bg)
#                 if PS:
#                     candidate_Ps = {PS[0] : candidate_Ps.get(PS[0])}
#                     candidate_Ss = {PS[1] : candidate_Ss.get(PS[1])}
# =============================================================================

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:                 
                matches.update({
                                bg:[ed, 
                                    EVENTS[ev][0], 
                                    EVENTS[ev][1], 
                                
                                    list(candidate_Ps)[0],  
                                    candidate_Ps[list(candidate_Ps)[0]][0], 
                                    candidate_Ps[list(candidate_Ps)[0]][1],  
                                                
                                    list(candidate_Ss)[0],  
                                    candidate_Ss[list(candidate_Ss)[0]][0], 
                                    candidate_Ss[list(candidate_Ss)[0]][1],  
                                                ] })
                
                if sst and sst > bg and sst < EVENTS[ev][2]:
                    if list(candidate_Ss)[0]:
                        S_error = sst -list(candidate_Ss)[0] 
                    else:
                        S_error = None
                                            
                if spt and spt > bg-100 and spt < EVENTS[ev][2]:
                    if list(candidate_Ps)[0]:  
                        P_error = spt - list(candidate_Ps)[0] 
                    else:
                        P_error = None
                                          
                pick_errors.update({bg:[P_error, S_error]})
      
    return matches, pick_errors, yh3


# In[2]:



import matplotlib.pyplot as plt
import numpy as np
from obspy.signal.trigger import trigger_onset



yh2 = np.load('pred_SS_mean_all.npy', allow_pickle=True)
yh2_std = np.load('pred_SS_std_all.npy', allow_pickle=True)


spt = np.load('sall.npy', allow_pickle=True)

#epik = np.load('epick.npy', allow_pickle=True)
#print(np.shape(yh2))


# In[3]:


earthq = []
nois = []
for i in spt:
    if i==None:
        nois.append(i)
    else:
        earthq.append(i)
#print(len(nois), len(earthq), len(earthq)+len(nois))


# In[4]:


thre=0.1
P_PICKall=[]
Ppickall=[]
Pproball = []
perrorall=[]
P_uncertaintyall = []

for i in range(0,len(yh2)):
    
    yh3 = yh2[i]
    yh3_std = yh2_std[i]
    
    
    sP_arr = detect_peaks(yh3, mph=thre, mpd=1)

    P_PICKS = []
    pick_errors = []
    #print(spt)
    P_uncertainty = None
    if len(sP_arr) > 0:
        P_uncertainty = None  

        for pick in range(len(sP_arr)):        
            sauto = sP_arr[pick]

            if  sauto:
                P_uncertainty = np.round(yh3_std[int(sauto)], 3)

            if sauto: 
                P_prob = np.round(yh3[int(sauto)], 3) 
                P_PICKS.append([sauto,P_prob, P_uncertainty]) 
                
    P_uncertaintyall.append(P_uncertainty)
    
    so=[]
    si=[]
    P_PICKS = np.array(P_PICKS)
    P_PICKall.append(P_PICKS)
    for ij in P_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        perrorall.append(int(spt[i]- swave))  
        Ppickall.append(int(swave))
        Pproball.append(int(np.max(so)))
    except:
        perrorall.append(None)
        Ppickall.append(None)
        Pproball.append(None)


Ppickall = np.array(Ppickall)
perrorall = np.array(perrorall)


# In[5]:


Ppick = Ppickall

pwave = []
pwavetp=[]
pwavetn=[]
pwavefp=[]
pwavefn=[]
difts=[]
iqq=[]
cc = 0
for iq in range(0,len(spt)):
    
    if (Ppick[iq]!=None) and (spt[iq]!=None):
        pwavetp.append(spt[iq]-Ppick[iq])
    elif (Ppick[iq]==None) and (spt[iq]!=None):
        pwavefn.append(iq)
        
    elif (Ppick[iq]==None) and (spt[iq]==None):
        pwavetn.append(iq)
    
    elif (Ppick[iq]!=None) and (spt[iq]==None):
        pwavefp.append(iq)


# In[6]:


samp = 50
difts = np.array(pwavetp)
TP = len(np.where(np.abs(difts)<=samp)[0])
TN = len(pwavetn) 
FP = len(pwavefp) 
FN = len(pwavefn) + len(np.where(np.abs(difts)>samp)[0])
P = TP /(TP+FP)
R = TP / (TP+FN)
F1 = 2 * (P*R) / (P+R)

#print(TP + TN + FP + FN, TP, TN, FP, FN,len(pwavefn) , len(np.where(np.abs(difts)>samp)[0]))
print('Total number of tested events is:',len(yh2))
print('TP is:',TP)
print('FP is:',FP)
print('TN is:',TN)
print('FN is:',FN)
print('Precision is:',P)
print('Recall is :',R)
print("F1-score is:",F1)
print('Number of missing Events is:',len(pwavefn))
a0 = np.where(np.abs(difts)<=samp)[0]
diftsp = difts[a0]/100
print('std is:',np.std(diftsp))
print('MAE is:',np.mean(np.abs(diftsp)))
np.save('difss.npy',diftsp)
