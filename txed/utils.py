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
