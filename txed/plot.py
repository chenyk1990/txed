import matplotlib.pyplot as plt
import numpy as np

def plot_traces(st,axoff=1,ayoff=1,titleoff=1,picks=None,eid=None,mag=None,ifmap=0,staloc=None,evloc=None,staname=None,ptime=None,stime=None,figname=None,showf=True,**kwargs):
	"""
	plot_traces: plot all traces in a stream quickly side-by-side from top to bottom
	
	INPUT
	st: Stream
	axoff: axes X off
	ayoff: axes Y off
	title: title off
	picks: P/S picks object
		in this subroutine, picks is a list of two-entries Dict 
		initialized by {'P':None,'S':None}
		(compared with obspy pick object obspy.core.event.origin.Pick)
	showf: if show figure interatively
	
	EXAMPLES
	from txed import plot_traces
	plot_traces(st);
	
	from txed import plot_traces
	plot_traces(st,axoff=0,ayoff=0,titleoff=0);
	
	from txed plot_traces; 
	plot_traces(obspy.read("newwaveforms/texnet2022ugad.mseed")[0:10]);
	
	demos/test_event.py
	"""
	ntr=len(st)
	if ifmap == 0:
		fig = plt.figure(figsize=(6, 8))
	else:
		fig = plt.figure(figsize=(12, 8))
	
	for ii in range(ntr):
		if ifmap==0:
			ax = plt.subplot(ntr,1,ii+1)
		else:
			ax = plt.subplot(ntr,2,2*ii+1)
		nt=len(st[ii].data);
		twin=(nt-1)*1.0/st[ii].stats.sampling_rate;
# 		print('twin=',twin,'nt=',nt,'dt',1.0/st[ii].stats.sampling_rate)
# 		staname=st[ii].stats.network+'.'+st[ii].stats.station
		t=np.linspace(0,twin,nt)
		plt.plot(t,st[ii].data,color='k',label = st[ii].stats.station, linewidth = 1, markersize=1)
		
		if ii==0 and titleoff != 1:
			if eid is None and mag is None:
				plt.title(st[ii].stats.starttime,fontsize='large', fontweight='normal')
			else:
				plt.title(eid+' (M=%.3g)'%(mag)+' '+str(st[ii].stats.starttime),fontsize=10, fontweight='normal')

		ax.legend(loc='lower right', fontsize = 10/(ntr/4))
		if ii==ntr-1:
			if axoff == 1:
				plt.setp(ax.get_xticklabels(), visible=False)
			else:
				plt.setp(ax.get_xticklabels(), visible=True)
				ax.set_xlabel("Time (s)",fontsize='large', fontweight='normal')
		else:
			plt.setp(ax.get_xticklabels(), visible=False)
		ax.set_xlim(xmin=0)
		ax.set_xlim(xmax=t[-1])
		ymin, ymax = ax.get_ylim()
		
		if ayoff:
			plt.setp(ax.get_yticklabels(), visible=False)
			
		if picks is not None:
			if picks[ii]['P'] is not None:
				tp=picks[ii]['P']-st[ii].stats.starttime
				plt.vlines(tp, ymin, ymax, color = 'r', linewidth = 1) #for P
			
			if picks[ii]['S'] is not None:
				ts=picks[ii]['S']-st[ii].stats.starttime
				plt.vlines(ts, ymin, ymax, color = 'g', linewidth = 1) #for S
# 		plt.text(2.5,(ymin+ymax)/2,staname)
				
		if ptime is not None:
			tp=ptime[ii]-st[ii].stats.starttime
			plt.vlines(tp, ymin, ymax, color = 'r', linewidth = 1) #for P
			
		if stime is not None:
			ts=stime[ii]-st[ii].stats.starttime
			plt.vlines(ts, ymin, ymax, color = 'g', linewidth = 1) #for P
			
	if ifmap==1:	
		ax = plt.subplot(2,2,2)
		plt.plot(staloc[:,0],staloc[:,1],'v',color='b')
		plt.plot(evloc[0],evloc[1],'*',color='r')
		addcounty_tx();
		
		plt.ylabel('Latitude (deg)')
		plt.gca().set_xlim(evloc[0]-3,evloc[0]+3)
		plt.gca().set_ylim(evloc[1]-3,evloc[1]+3)
		[x1,x2]=plt.gca().get_xlim()
		
		if len(staname) == staloc.shape[0]:
			for ii in range(len(staname)):
				plt.text(staloc[ii,0],staloc[ii,1],staname[ii][:],color='k',fontsize=6)
		
		ax = plt.subplot(2,2,4)
		plt.plot(evloc[0],evloc[2],'*',color='r',label='TexNet event')
			
		plt.gca().set_ylim(-2, 25);plt.gca().set_xlim(x1, x2)
		plt.gca().invert_yaxis()
		plt.xlabel('Longitude (deg)');plt.ylabel('Depth (km)')
		
		plt.plot(staloc[:,0],-staloc[:,2]/1000.0,'v',color='b',label='Station')
		plt.plot(0, 0, color = 'k', linestyle='solid', linewidth = 1, label="Waveform") #for S
		
		## add picks legend
		plt.vlines(0, 100, 120, color = 'r', linestyle='solid', linewidth = 1, label="TexNet P") #for P
		plt.vlines(0, 100, 120, color = 'g', linestyle='solid', linewidth = 1, label="TexNet S") #for S
		
		ax.legend(loc='lower right', fontsize = 8)
		
		
	if figname is not None:
		plt.savefig(figname,**kwargs)
	
	if showf:
		plt.show()
	else:
		plt.close() #or plt.clear() ?
		

def addcounty_tx():
	'''
	addcounty_tx: add state-scale county lines in Texas
	
	from txed.plot import addcounty_tx
	import matplotlib.pyplot as plt
	plt.figure()
	addcounty_tx();
	plt.show()
	
	'''
	from txed import asciiread
	##red county lines
	lines=asciiread('../txed/data/Texas_County_XY.txt')
	lines=[ii.rstrip().split(' ') for ii in lines]
	##add county lines
	tmp=[]
	for ii in range(len(lines)):
		if lines[ii] != ['']:
			tmp.append(lines[ii])
		else:
			tmp=np.array(tmp,dtype='float32')
			plt.plot(tmp[:,0],tmp[:,1],'-',color='#929591')
			ii=ii+1;
			tmp=[];