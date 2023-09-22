"""
Created on Aug 2023

@author: Yangkang Chen

Plot picks of TexNet catalog event using TXED (Texas earthquake dataset for AI)

This example is only runnable for TXED_0919.h5 (earlier versions do not apply)

Example:
python test_event.py texnet2018ajoc Z 1
python test_event.py texnet2020kijr
python test_event.py texnet2023qnms
python test_event.py texnet2023qnms Z 1 TXEDPATH #e.g., python test_event.py texnet2023qnms Z 1 /Users/chenyk/DATALIB/TXED
"""
import argparse
import obspy
import numpy as np
import obspy.core.utcdatetime as utc

import matplotlib
# matplotlib.use('agg')
import os
import glob
from txed import plot_traces
from txed import shift3c

from obspy import Stream
from obspy import Trace
import h5py


def getargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('eid', nargs='?', 
        default='texnet2020galz')

    parser.add_argument('ic', nargs='?', 
        default='Z') #Z,N,or E

    parser.add_argument('ifmap', nargs='?', 
        default=1) #if plot source on map

    parser.add_argument('txedpath', nargs='?', 
        default='DATALIB') #if plot source on map
         
    return parser.parse_args()

if os.path.isdir(os.getenv('HOME')+'/DATALIB/tmp/TXED/pickfig') == False:  
	os.makedirs(os.getenv('HOME')+'/DATALIB/tmp/TXED/pickfig',exist_ok=True)

def run(eid,ic,ifmap,txedpath):
	'''
	eid: 		str
	ic:			str (Z,N,E component)
	ifmap:		int if plot on map (location)
	'''
	print('EVENT ID is',eid)
	if txedpath == 'DATALIB':
		fname=os.getenv('HOME')+'/DATALIB/TXED/TXED_0919.h5'
		eventid=np.load(os.getenv('HOME')+'/DATALIB/TXED/ID_0919.npy')
	else:
		print(txedpath)
		fname=txedpath+'/TXED_0919.h5'
		eventid=np.load(txedpath+'/ID_0919.npy')
		
	print('Len of eventid is',len(eventid))

	f = h5py.File(fname, 'r')
	no=1000
	
	eids=[ii for ii in eventid if ii.split("_")[0]==eid] 
	d2=[]
	staname=[]
	pick=[]
	staloc=[]
	for jj in eids: #e.g.,  ['texnet2022ptdv_WB02_EV','texnet2022ptdv_WB03_EV']
		sta=jj.split("_")[1];
		dataset = f.get(jj)
		data = np.array(dataset['data'])
# 		tshift=1
		tshift=int((utc.UTCDateTime(dataset.attrs['p_arrival_time'])-utc.UTCDateTime(dataset.attrs['origin_time']))*100)-dataset.attrs['p_arrival_sample']
		data=shift3c(data,tshift) #shift the waveform according to the P arrival time
		
		trace1=Trace();trace1.data=data[:,0];trace1.stats.network='TX';trace1.stats.station=sta;trace1.stats.channel='Z';
		trace1.stats.starttime=dataset.attrs['origin_time'];trace1.stats.delta=0.01;
		
		trace2=Trace();trace2.data=data[:,1];trace2.stats.network='TX';trace2.stats.station=sta;trace2.stats.channel='N';
		trace2.stats.starttime=dataset.attrs['origin_time'];trace2.stats.delta=0.01;
		
		trace3=Trace();trace3.data=data[:,2];trace3.stats.network='TX';trace3.stats.station=sta;trace3.stats.channel='E';
		trace3.stats.starttime=dataset.attrs['origin_time'];trace3.stats.delta=0.01;
		
		st=Stream(traces=[trace1,trace2,trace3])
# 		print(st)
		d2.append(st)
		staname.append(sta)
		
		dic={'P':None,'S':None}
		dic['P']=utc.UTCDateTime(dataset.attrs['p_arrival_time'])
		dic['S']=utc.UTCDateTime(dataset.attrs['s_arrival_time'])
		pick.append(dic)
		
		staloc.append([dataset.attrs['sta_longitude'],dataset.attrs['sta_latitude'],0])
# 		print(dataset.attrs['sta_longitude'],dataset.attrs['sta_latitude'])
	staloc=np.array(staloc,dtype='float')	
	print(staloc.shape)
# 	print(staloc)
	print('len(d2)',len(d2))
	mag=dataset.attrs['magnitude'];

	if ic == 'Z':
		d3=[ii[0].resample(100).detrend("linear").taper(max_percentage=0.02, type="cosine",max_length=2).filter('bandpass',freqmin=1,freqmax=45) for ii in d2]
	elif ic=='N':
		d3=[ii[1].resample(100).detrend("linear").taper(max_percentage=0.02, type="cosine",max_length=2).filter('bandpass',freqmin=1,freqmax=45) for ii in d2]
	elif ic=='E':
		d3=[ii[2].resample(100).detrend("linear").taper(max_percentage=0.02, type="cosine",max_length=2).filter('bandpass',freqmin=1,freqmax=45) for ii in d2]
	else:
		d3=[ii[0].resample(100).detrend("linear").taper(max_percentage=0.02, type="cosine",max_length=2).filter('bandpass',freqmin=1,freqmax=45) for ii in d2]
	
	evloc=np.array([dataset.attrs['ev_longitude'],dataset.attrs['ev_latitude'],dataset.attrs['ev_depth']/1000.0])
	dis=[np.hypot(ii[0]-evloc[0],ii[1]-evloc[1]) for ii in staloc[:]]
	inds=np.argsort(dis) ##This can verify if the random shift is effective
	dis=[dis[ii] for ii in inds]
	d3=[d3[ii] for ii in inds]
	staname=[staname[ii] for ii in inds]
	staloc=np.array([staloc[ii,:] for ii in inds])
	pick=[pick[ii] for ii in inds] #This can verify if the random shift is effective

	if ifmap==0:
		staloc=None;
		evloc=None;
		staname=None;
	
	plot_traces(d3,axoff=0,titleoff=0,picks=pick,eid=eid,mag=mag,ifmap=ifmap,staloc=staloc,evloc=evloc,staname=staname,
			figname=os.getenv('HOME')+'/DATALIB/tmp/TXED/pickfig/%s-%s.png'%(eid,ic),bbox_inches='tight',dpi=500);
	
if __name__ == '__main__':
    
    """
    plot picks file
    """
    
    args = getargs()
    print('eid:',args.eid)
    print('ic:',args.ic)
    print('ifmap:',args.ifmap)
    print('figpath:',os.getenv('HOME')+'/DATALIB/tmp/TXED/pickfig/%s-%s.png'%(args.eid,args.ic))
    run(str(args.eid),str(args.ic),int(args.ifmap),str(args.txedpath))
    
    
    
    