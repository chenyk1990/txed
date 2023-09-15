# TXED

## Description

**TXED** is the Texas Earthquake Dataset for AI

## Reference
    Chen, et al., 2023, TXED: the Texas Earthquake Dataset for AI, TBD.
    
BibTeX:

	@article{txed,
	  title={TXED: the Texas Earthquake Dataset for AI},
	  author={Chen et al.},
	  journal={TBD},
	  volume={TBD},
	  number={TBD},
	  issue={TBD},
	  pages={TBD},
	  year={2016},
	  publisher={TBD}
	}
 
-----------
## Copyright
    Developers of the TXED package, 2021-present
-----------

## License
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)   

-----------

## Install
Using the latest version

    git clone https://github.com/chenyk1990/txed

-----------
## Download

Google drive link: https://drive.google.com/drive/folders/1WXVB8ytNB4bOaZ97oq6OmMRyAEg95trp?usp=sharing 

	wget /address_TBD/TXED_0913.h5
	wget /address_TBD/ID_0913.npy

-----------
## Examples
Check the INFO of signal waveforms

	import numpy as np
	allid = np.load("ID_0913.npy")
	signalid=[ii for ii in allid if ii.split("_")[-1]=='EV']
	print('Length of signalid is',len(signalid))

Check the INFO of noise waveforms

    import numpy as np
    allid = np.load("ID_0913.npy")
    noiseid=[ii for ii in allid if ii.split("_")[-1]=='NO']
    print('Length of noiseid is',len(noiseid))

Print attributes in TXED

	import h5py
	import numpy as np
	f = h5py.File("TXED_0913.h5", 'r')
	eventid=np.load("ID_0913.npy")
	idx=eventid[0]
	dataset = f.get(idx)
	print('TXED attributes are:',dataset.attrs.keys())
	
Plot signal waveforms

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
	idxs=[ii for ii in eventid if ii.split('_')[0]=='texnet2023ncwh']
	
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

Plot noise waveforms

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

	#specificy a number of waveforms to plot
	no=100
	
	#random shuffling the noise waveforms
	random.seed(2011)
	random.shuffle(noiseid)
	idxs=noiseid[0:no]

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

The Ipython Notebooks are examples for playing with the TXED.

    
Single-station location example


    
-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## NOTES:

-----------
## Gallery
The gallery figures of the txed package can be found at
    https://github.com/chenyk1990/gallery/tree/main/txed

These gallery figures are also presented below. 


A sample signal waveform Generated by [test_signal.py](https://github.com/chenyk1990/txed/tree/main/demos/test_signal.py)
<img src='https://github.com/chenyk1990/gallery/blob/main/txed/signal-texnet2023ncwh-PB10.png' alt='Slicing' width=960/>

A sample noise waveform Generated by [test_noise.py](https://github.com/chenyk1990/txed/tree/main/demos/test_noise.py)
<img src='https://github.com/chenyk1990/gallery/blob/main/txed/noise-24634-PECS.png' alt='Slicing' width=960/>


