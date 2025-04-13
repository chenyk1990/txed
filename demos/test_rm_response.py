import h5py,os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7,8.2)

if os.path.isdir('./waveforms') == False:  
	os.makedirs('./waveforms',exist_ok=True)

h5fname="TXED_20231111.h5"
npyfname="ID_20231111.npy"

#open the h5file
f = h5py.File(h5fname, 'r')
eventid=np.load(npyfname)

#specificy a number of waveforms to plot
no=5
idxs=eventid[0:no]

#or specify an arbitrary TexNet eventID
# idxs=[ii for ii in eventid if ii.split('_')[0]=='texnet2023qnms']
# idxs=[ii for ii in eventid if ii.split('_')[0]=='texnet2022wmmd']
idxs=[ii for ii in eventid if ii.split('_')[0]=='texnet2023ncwh']

dataset = f.get(idxs[0])
	
	
import obspy

from txed import get_net_from_sta
from txed import gen_stream


from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
	

# sta=dataset.attrs['station']
# net=get_net_from_sta(sta)
# trace_start_time=UTCDateTime(dataset.attrs['p_arrival_time'])-dataset.attrs['p_arrival_sample']*0.01
# client = Client("TEXNET")
# inventory = client.get_stations(network=net,
#                                     station=sta,
#                                     starttime=trace_start_time,
#                                     endtime=trace_start_time + 60,
#                                     loc="*", 
#                                     channel="*",
#                                     level="response")  
# cha=inventory[0].get_contents()['channels'][0].split(".")[-1][0:2]
# 
# print('This data sample comes from the station of ',net+'.'+sta+'.'+cha+'*')
# 
# st0=gen_stream(dataset,net,sta,cha,trace_start_time)
# 
# st1=st0.copy()
# st1 = st1.remove_response(inventory=inventory, output='VEL', plot=False) 
#                                    
# 
# st2=st0.copy()
# st2 = st2.remove_response(inventory=inventory, output='DISP', plot=False) 
# 
#                         
# st3=st0.copy()
# st3 = st3.remove_response(inventory=inventory, output='ACC', plot=False) 
# 

# http://rtserve.beg.utexas.edu, http://scarchive.beg.utexas.edu

id=0
for idx in idxs:
	id=id+1
	print('Plotting: %d/%d'%(id,len(idxs)))
	dataset = f.get(idx)
	
	sta=dataset.attrs['station']
	net=get_net_from_sta(sta)
	trace_start_time=UTCDateTime(dataset.attrs['p_arrival_time'])-dataset.attrs['p_arrival_sample']*0.01
	client = Client("http://scarchive.beg.utexas.edu") #change it to http://rtserve.beg.utexas.edu if outside BEG network
	inventory = client.get_stations(network=net,
                                    station=sta,
                                    starttime=trace_start_time,
                                    endtime=trace_start_time + 60,
                                    loc="*", 
                                    channel="*",
                                    level="response")  
	loc=inventory[0].get_contents()['channels'][0].split(".")[-2]                           
	cha=inventory[0].get_contents()['channels'][0].split(".")[-1][0:2]

	staname=net+'.'+sta+'.'+loc+'.'+cha+'*'
	print('This data sample comes from the station of ',staname)

	st0=gen_stream(dataset,net,sta,loc,cha,trace_start_time)

	#in case of HH1/2 instead of HHN/E
	if '1' in [ii[-1] for ii in inventory[0].get_contents()['channels']] or '2' in [ii[-1] for ii in inventory[0].get_contents()['channels']]:
		st0[1].stats.channel=st0[1].stats.channel[0:2]+'1'
		st0[2].stats.channel=st0[1].stats.channel[0:2]+'2'

	st1=st0.copy()
	st1 = st1.remove_response(inventory=inventory, output='DISP', plot=False) 
                                   

	st2=st0.copy()
	st2 = st2.remove_response(inventory=inventory, output='VEL', plot=False) 

                        
	st3=st0.copy()
	st3 = st3.remove_response(inventory=inventory, output='ACC', plot=False) 


	data = np.array(dataset['data'])
	spt = dataset.attrs['p_arrival_time'];
	sst = dataset.attrs['s_arrival_time'];
	coda_end = int(dataset.attrs['coda_end_sample']);
	snr = dataset.attrs['snr_db'];
	t=dataset.attrs['origin_time']
	mag=dataset.attrs['magnitude']
	
	fig=plt.figure(figsize=(10, 8))
	ax0 = fig.add_subplot(411)
	plt.plot(st0[0].data, 'k',label='Raw Z')
	ymin,yma = ax0.get_ylim()
# 	plt.vlines(spt,ymin,yma,color='r',linewidth=2)
# 	plt.vlines(sst,ymin,yma,color='b',linewidth=2)
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax0.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Amplitude', fontsize=12) 
	ax0.set_xticklabels([])
	
	ax1 = fig.add_subplot(412)
	plt.plot(st1[0].data, 'k',label='Displacement Z')
	ymin,yma = ax1.get_ylim()
# 	plt.vlines(spt,ymin,yma,color='r',linewidth=2)
# 	plt.vlines(sst,ymin,yma,color='b',linewidth=2)
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax1.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
# 	plt.ylabel('Displacement', fontsize=12) 
	plt.ylabel('Meters', fontsize=12) 
	ax1.set_xticklabels([])
	
	ax1 = fig.add_subplot(413) 
	plt.plot(st2[0].times("matplotlib"),st2[0].data, 'k',label='Velocity Z')
	ymin,yma = ax1.get_ylim()
# 	plt.vlines(spt,ymin,yma,color='r',linewidth=2)
# 	plt.vlines(sst,ymin,yma,color='b',linewidth=2)
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax1.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Meters/Second', fontsize=12) 
	ax1.set_xticklabels([])
	
	ax1 = fig.add_subplot(414) 
	plt.plot(st3[0].times("matplotlib"),st3[0].data, 'k',label='Acceleration Z')
	ymin,yma = ax1.get_ylim()
# 	plt.vlines(UTCDateTime(spt),ymin,yma,color='r',linewidth=2)
# 	plt.vlines(UTCDateTime(sst),ymin,yma,color='b',linewidth=2)
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax1.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
# 	plt.ylabel('Acceleration', fontsize=12) 
	plt.ylabel('Meters/Second^2', fontsize=12) 
	ax1.xaxis_date()
# 	fig.autofmt_xdate()
# 	plt.xlabel('Sample', fontsize=12) 

	eid=idx.split("_")[0]
	stname=idx.split("_")[1]
	sttime=t
	
	xmin, xmax = ax1.get_xlim()
	plt.text(xmin,(ymin-(ymax-ymin)*0.4),"Event origin time: "+sttime+'; Station name: '+staname,fontsize=12,color='k')
	ax0.set_title('Signal waveform with responses removed: %s-%s'%(eid,stname), fontsize=14)
	plt.savefig(fname='./waveforms/signal-%s-%s-noresponse'%(eid,stname)+'.png', format="png")
	plt.show()
# 	plt.close() 
	

