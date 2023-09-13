import h5py,os,random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7,8.2)

if os.path.isdir('./waveforms') == False:  
	os.makedirs('./waveforms',exist_ok=True)

h5fname="TXED_0913.h5"
npyfname="ID_0913.npy"

#open the h5file
f = h5py.File(h5fname, 'r')
allid=np.load(npyfname)

noiseid=[ii for ii in allid if ii.split("_")[-1]=='NO']
print('Length of noiseid is',len(noiseid))

#locate the noise ID
idxs=[ii for ii in noiseid if ii.split("_")[0]=='24634' and ii.split("_")[1]=='PECS']

id=0
for idx in idxs:
	id=id+1
	print('Plotting: %d/%d'%(id,len(idxs)))
	dataset = f.get(idx)
	data = np.array(dataset['data'])
	t=dataset.attrs['origin_time']
	fig=plt.figure()
	ax1 = fig.add_subplot(311)
	plt.plot(data[:,0], 'k',label='Z')
	ymin,yma = ax1.get_ylim()
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax1.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Amplitude', fontsize=12) 
	ax1.set_xticklabels([])
	ax = fig.add_subplot(312) 
	plt.plot(data[:,1], 'k',label='N')
	ymin,yma = ax1.get_ylim()
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Amplitude', fontsize=12) 
	ax.set_xticklabels([])
	ax = fig.add_subplot(313) 
	plt.plot(data[:,2], 'k',label='E')
	ymin,yma = ax1.get_ylim()
	legend_properties = {'weight':'bold'}
	ymin, ymax = ax.get_ylim()
	plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
	plt.ylabel('Amplitude', fontsize=12) 
	plt.xlabel('Sample', fontsize=12) 

	no=idx.split("_")[0]
	eid='Noise #%s'%no
	stname=idx.split("_")[1]
	sttime=t
	plt.text(-1200,(ymin-(ymax-ymin)*0.3),eid,fontsize=12,color='k')
	plt.text(1200,(ymin-(ymax-ymin)*0.3),stname,fontsize=12,color='k')
	plt.text(3300,(ymin-(ymax-ymin)*0.3),sttime,fontsize=12,color='k')
	ax1.set_title('Noise waveform: #%s-%s'%(no,stname), fontsize=14)
	plt.savefig(fname='./waveforms/noise-%s-%s'%(no,stname)+'.png', format="png")
	plt.show()
	plt.close() 
f.close()

#open waveforms/noise-24634-PECS.png