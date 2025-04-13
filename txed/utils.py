import numpy as np
def shift3c(data,tshift):
	'''
	shift3c: shift a 3C numpy array according to the tshift (scalar)
	
	INPUT
	data: nsample x 3 array
	tshift: shift in samples (>0 -> right shift; <0 -> left shift)
	
	OUTPUT
	data2: shifted array
	'''
	
# 	return np.roll(data, tshift, axis=0) #not best
	data2=np.zeros(data.shape)
	if tshift>0:
		data2[tshift:,:] = data[0:-tshift,:]
	else:
		data2[0:tshift,:]=data[-tshift:,:]
	
	return data2
	
def asciiread(fname):
	'''
	fname: file name
	din:   a list of lines
	withnewline: if with the newline symbol '\n': True: with; False: without
	
	Example:
	
	from txed import asciiread
	import os
	
	lines=asciiread(os.getenv('HOME')+'/chenyk.data2/various/cyksmall/texnet_stations_2022_1019.csv');
	'''
	
	f=open(fname,'r')
	lines=f.readlines()
	lines=[ii.strip() for ii in lines]
	
	return lines


def get_net_from_sta(sta):
	'''
	get_net_from_sta: output network code according to input station code (in TXED, only station code is used for naming)
	
	INPUT
	sta: station code (e.g., 'PECS')
	
	OUTPUT
	net: network code (e.g., 'TX')
	
	EXAMPLE
	from txed import get_net_from_sta
	sta='PECS'
	net=get_net_from_sta(sta)
	print(net)
	'''
	import txed
	stafile=txed.__file__[:-11]+'/data/texnet_stations_20250413.csv'
	
	p = open(stafile).readlines()
	p.pop(0)
	stnames = [line.strip().split(',')[0]+'.'+line.strip().split(',')[1] for line in p]
	
	net=[ii.split(".")[0] for ii in stnames if ii.split(".")[-1]==sta]
	
	if len(net)>=1:
		net=net[0]
		
	return net
	
def gen_stream(dataset=None,net='4O',sta='CT01',loc="",cha='HH',trace_start_time=None):
	'''
	gen_stream: generate obspy stream
	
	INPUT
	TXED data sample, HDF5 group object 
	e.g., f = h5py.File("TXED_20231111.h5", 'r'); dataset=f.get('texnet2023ncwh_CT01_EV'); 
	
	OUTPUT
	obspy stream
	
	EXAMPLE1
	from txed import gen_stream
	import numpy as np
	st=gen_stream(np.ones([6000,3]))

	EXAMPLE2
	see demos/test_rm_response.py
	
	'''
	import obspy
	if trace_start_time is None:
		trace_start_time=obspy.UTCDateTime('2023-07-06T13:30:16.144554Z')
	
	if type(dataset) == np.ndarray:
		data = dataset
	else:
		data = np.array(dataset['data'])
	
	tr_Z = obspy.Trace(data=data[:, 0])
	tr_Z.stats.starttime = trace_start_time
	tr_Z.stats.delta = 0.01
	tr_Z.stats.channel = cha+'Z'
	if len(loc)>0:
		tr_Z.stats.location = loc
	tr_Z.stats.station = sta
	tr_Z.stats.network = get_net_from_sta(sta)

	tr_N = obspy.Trace(data=data[:, 1])
	tr_N.stats.starttime = trace_start_time
	tr_N.stats.delta = 0.01
	tr_N.stats.channel = cha+'N'
	if len(loc)>0:
		tr_N.stats.location = loc
	tr_N.stats.station = sta
	tr_N.stats.network = get_net_from_sta(sta)

	tr_E = obspy.Trace(data=data[:, 2])
	tr_E.stats.starttime = trace_start_time
	tr_E.stats.delta = 0.01
	tr_E.stats.channel = cha+'E'
	if len(loc)>0:
		tr_E.stats.location = loc
	tr_E.stats.station = sta
	tr_E.stats.network = get_net_from_sta(sta)

	stream = obspy.Stream([tr_Z, tr_N, tr_E])

	return stream
	