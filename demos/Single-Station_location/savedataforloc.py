from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

from scipy.signal import butter, lfilter, lfilter_zi

def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y


import h5py
import numpy as np
idall = np.load('/home/omar/EQCCT_Texas/ID_Texas_Test_Final.npy')
datall = []
jj = len(idall)
snr = []
ori = []
mag = []
dep = []
lis = []
idapp = []
stlat = []
stlon = []
evlat = []
evlon = []

f = h5py.File('/home/omar/EQCCT_Texas/Texas_Test_WithNoise_Final.h5', 'r')

for kl in range(jj):

    idx = idall[kl]
    dataset = f.get(idx)
    
    if (idx.split('_')[-1]=='EV') and (idx.split('_')[1]=='PECS'):
        #print(idx)
        snr.append(np.mean(dataset.attrs['snr_db']));
        mag.append(dataset.attrs['magnitude'])
        dep = dataset.attrs['ev_depth']
        latdif = dataset.attrs['sta_latitude'] - dataset.attrs['ev_latitude']
        londif = dataset.attrs['sta_longitude'] - dataset.attrs['ev_longitude']
        Psample = dataset.attrs['p_arrival_sample']
    
        dat = np.array(dataset['data'])
        #print(Psample,np.shape(dat))
        
        if Psample>=100:
            dat = dat[Psample-100:Psample+4900,:]
    
            dat[:,0] = butter_bandpass_filter_zi(dat[:,0], 1, 45, 100, order=3)
            dat[:,1] = butter_bandpass_filter_zi(dat[:,1], 1, 45, 100, order=3)
            dat[:,2] = butter_bandpass_filter_zi(dat[:,2], 1, 45, 100, order=3)

            dat = dat/np.max(np.abs(dat))
            datall.append(dat)
            lis.append([latdif,londif,dep])
            stlat.append(dataset.attrs['sta_latitude'] )
            stlon.append(dataset.attrs['sta_longitude'])
            evlon.append(dataset.attrs['ev_longitude'])
            evlat.append(dataset.attrs['ev_latitude'])
            
            idapp.append(idx)


np.save('./loc/evlat',evlat)
np.save('./loc/evlon',evlon)
np.save('./loc/stlat',stlat)
np.save('./loc/stlon',stlon)
np.save('./loc/idPECS.npy',idapp)
np.save('./loc/datloc.npy',datall)
np.save('./loc/relativelist.npy',lis)
np.save('./loc/snrLoc.npy',snr)
np.save('./loc/MagLoc.npy',mag)
f.close()
