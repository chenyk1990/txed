import h5py,os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7,8.2)

if os.path.isdir('./waveforms') == False:  
	os.makedirs('./waveforms',exist_ok=True)

h5fname="TXED_0913.h5"
npyfname="ID_0913.npy"

#open the h5file
f = h5py.File(h5fname, 'r')
eventid=np.load(npyfname)

#specificy a number of waveforms to plot
no=5
idxs=eventid[0:no]

#or specify an arbitrary TexNet eventID
# idxs=[ii for ii in eventid if ii.split('_')[0]=='texnet2023qnms']
# idxs=[ii for ii in eventid if ii.split('_')[0]=='texnet2022wmmd']
idxs=[ii for ii in eventid if ii.split('_')[0]=='texnet2023ncwh' and ii.split('_')[1]=='PB10']

#loop over IDs and plot
id=0
for idx in idxs:
	id=id+1
	print('Plotting: %d/%d'%(id,len(idxs)))
	
	dataset = f.get(idx)
	data = np.array(dataset['data'])
	spt = int(dataset.attrs['p_arrival_sample']);
	sst = int(dataset.attrs['s_arrival_sample']);
	coda_end = int(dataset.attrs['coda_end_sample']);
	snr = dataset.attrs['snr_db'];
	t=dataset.attrs['origin_time']
	mag=dataset.attrs['magnitude']
	
	fig=plt.figure()
	ax1 = fig.add_subplot(311)
	plt.plot(data[:,0], 'k',label='Z')
	ymin,yma = ax1.get_ylim()
	plt.vlines(spt,ymin,yma,color='r',linewidth=2)
	plt.vlines(sst,ymin,yma,color='b',linewidth=2)
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax1.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Amplitude', fontsize=12) 
	ax1.set_xticklabels([])
	ax = fig.add_subplot(312) 
	plt.plot(data[:,1], 'k',label='N')
	ymin,yma = ax1.get_ylim()
	plt.vlines(spt,ymin,yma,color='r',linewidth=2)
	plt.vlines(sst,ymin,yma,color='b',linewidth=2)
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Amplitude', fontsize=12) 
	ax.set_xticklabels([])
	ax = fig.add_subplot(313) 
	plt.plot(data[:,2], 'k',label='E')
	ymin,yma = ax1.get_ylim()
	plt.vlines(spt,ymin,yma,color='r',linewidth=2)
	plt.vlines(sst,ymin,yma,color='b',linewidth=2)
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Amplitude', fontsize=12) 
	plt.xlabel('Sample', fontsize=12) 

	eid=idx.split("_")[0]
	stname=idx.split("_")[1]
	sttime=t
	plt.text(-1200,(ymin-(ymax-ymin)*0.3),eid+' (Ml=%.2g)'%mag,fontsize=12,color='k')
	plt.text(1800,(ymin-(ymax-ymin)*0.3),stname,fontsize=12,color='k')
	plt.text(3700,(ymin-(ymax-ymin)*0.3),sttime,fontsize=12,color='k')
	ax1.set_title('Signal waveform: %s-%s'%(eid,stname), fontsize=14)
	plt.savefig(fname='./waveforms/signal-%s-%s'%(eid,stname)+'.png', format="png")
	plt.show()
	plt.close() 

f.close()

#open waveforms/signal-texnet2023ncwh-PB10.png