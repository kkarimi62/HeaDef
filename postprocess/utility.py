import json
from json import JSONEncoder
import pdb
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import traceback
import os
import scipy.interpolate as scp_int
import warnings
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import patches
import sys
#import sklearn
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_validate
#import patsy
#from sklearn import linear_model, mixture
#import sklearn.mixture as skm
from scipy import optimize
import scipy
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import re
from functools import reduce
import time
import LammpsPostProcess2nd as lp
#
warnings.filterwarnings('ignore')

class Symbols:
    def __init__(self):
        self.colors = ['black','red','green','blue','cyan','brown','grey','magenta','orange','yellow']
        self.fillstyles=['white',None,'white',None,'white',None,'white',None,'white',None,'white',None,'white',None,'white',None]
        self.markers=['o','s','D','^','<','>','v']
        self.markersizes=[10,10,10,12,12,12,10]
        self.nmax=7
        
    def GetAttrs(self,count=0,label='',nevery=1,fmt='.-',zorder=1,**kwargs):
        if count > self.nmax:
            print('index out of list bound!')
            return 
        attrs={ 'color':self.colors[count],
            'markersize':self.markersizes[count],
            'marker':self.markers[count],
            'markerfacecolor':self.colors[count],
            'markeredgecolor':'white', #'black' if not self.fillstyles[count] else None,
            'label':label,
           'markevery':nevery,
           'errorevery':nevery,
           'markeredgewidth':1.75,
            'linewidth':1, 
           'barsabove':None,
           'capsize':5 if not 'capsize' in kwargs else kwargs['capsize'],
           'capthick':1,
           'elinewidth':1,
           'fmt':fmt,
             'zorder':zorder,
         }
        return attrs
    
    def GetAttrs2nd(self,count=0,label='',nevery=1,fmt='.-',zorder=1):
        if count > self.nmax:
            print('index out of list bound!')
            return 
        attrs={ 'color':self.colors[count],
            'markersize':self.markersizes[count],
            'marker':self.markers[count],
            'markerfacecolor':'white',
#            'markeredgecolor':'black' if not self.fillstyles[count] else None,
            'label':label,
           'markevery':nevery,
           'errorevery':nevery,
           'markeredgewidth':1.75,
            'linewidth':1, 
           'barsabove':None,
           'capsize':5,
           'capthick':1,
           'elinewidth':1,
           'fmt':fmt,
            'zorder':zorder,
          }
        return attrs

class Legends:
    def __init__(self
                ):
        pass
    def Set(self,fontsize=20,
                 labelspacing=0,
                 **kwargs
#                 bbox_to_anchor=(0.5,0.48,0.5,0.5),
           ):
        self.attrs = {'frameon':False,'fontsize':fontsize,
                   'labelspacing':labelspacing,
                      'handletextpad':.2,
                   'handlelength':1,
                    **kwargs,
                     }
    def Get(self):
        return self.attrs

def PlotPaperVersion(pathh_indx,
                     file0_indx,
                     runs = [0],
                     times = range(0,200+1,2),
                     **kwargs):
    
    verbose = True if 'verbose' in kwargs and kwargs['verbose'] == True else False
    timeseries = True if not 'timeseries' in kwargs else kwargs['timeseries']
    drawstyle = 'default' if not 'drawstyle' in kwargs else kwargs['drawstyle']
    xstr = '' if not 'xlabel' in kwargs else kwargs['xlabel']
    ystr = '' if not 'ylabel' in kwargs else kwargs['ylabel']
    nevery = kwargs['nevery'] if 'nevery' in kwargs else 1
    #--- setup symbols
    colors = ['C0','red','green','blue','cyan','brown','grey','magenta','orange','yellow']
    fillstyles=['white',None,'white',None,'white',None,'white',None,'white',None,'white',None,'white',None,'white',None]
    markers=['o','s','D','^','<','>','v']
    markersizes=[10,10,10,12,12,12,10]
    #--- plot
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    #

    
    for mg, marker, color, fillstyle, markersize in list(zip( [ 
                                         'FeNi',
                                           'CoNiFe',
                                           'CoNiCrFe',
                                            'CoCrFeMn',
                                            'CoNiCrFeMn',
                                            'Co5Cr2Fe40Mn27Ni26',
                                            'Co5Cr2Fe40Mn27Ni262nd',
#                                             'CuZr3'
                                        ],markers, colors, fillstyles, markersizes )):
        if 'glass' in kwargs and kwargs['glass'] != mg:
            continue
        print(mg)
        Xdata = []
        Ydata = []
        #--- loop over realizations 
        for irun in runs:
            xdata_timeseries = []
            ydata_timeseries = []
            erry_timeseries = []
            errx_timeseries = []
            
            #--- loop over time
            for itimee, count in\
            list(zip(times, range(len(times)))): 

                pathh = { 
                          0:'%s/PairCrltnT300/%s/Run%s'%(os.getcwd(),mg,irun),
                          1:'%s/VorAnlT300/%s/Run%s'%(os.getcwd(),mg,irun),
                          2:'%s/D2minAnalysisT300/%s/Run%s'%(os.getcwd(),mg,irun),
                          3:'%s/ElasticityT300/%s/eps2/itime%s/Run%s'%(os.getcwd(),mg,itimee,irun),
                          4:'%s/ElasticityT300/%s/eps2/itime%s/Run%s/ModuAnl'%(os.getcwd(),mg,itimee,irun),
                          5:'%s/Exponents/%s'%(os.getcwd(),mg),
                        }[pathh_indx]
                file0 = {
                          0:'%s/gr.txt'%pathh,
                          1:'%s/icoPercentageWithStrain.txt'%pathh,
                          2:'%s/d2min_gamma.txt'%pathh,
                          3:'%s/rc_d2min.txt'%pathh,
                          4:'%s/crsD2minRhoSro.txt'%pathh,
                          5:'%s/NegativeModulus.txt'%pathh, 
                          6:'%s/YieldDelta.txt'%pathh, 
                          7:'%s/gr_peak_gamma.txt'%pathh, 
                          8:'%s/muClustersize_gamma.txt'%pathh, 
                          9:'%s/pdfMu.txt'%pathh, 
                          10:'%s/mu_mean_std.txt'%pathh, 
                          11:'%s/mu_mean_std.txt'%pathh, 
                          12:'%s/pdfClsSize.txt'%pathh,
                          13:'%s/muClustersize_gamma.txt'%pathh, 
                          14:'%s/muClustersize_gamma.txt'%pathh, 
                          15:'%s/muClustersize_gamma.txt'%pathh, 
                          16:'%s/ps.txt'%pathh, 
                          17:'%s/muClustersize_gamma.txt'%pathh, 
                          18:'%s/ps.txt'%pathh, 
                          19:'%s/pinf.txt'%pathh, 
                          20:'%s/pinf.txt'%pathh, 
                          21:'%s/pinf.txt'%pathh, 
                          22:'%s/crltnl_gamma.txt'%pathh, 
                          23:'%s/crltnl_gamma.txt'%pathh, 
                          24:'%s/crltnl_gamma.txt'%pathh, 
                          25:'%s/hmin_gamma_exp.txt'%pathh, 
                          26:'%s/hmin_nu_exp.txt'%pathh, 
                          27:'%s/hmin_beta_exp.txt'%pathh, 
                          28:'%s/s_rg.txt'%pathh, 
                        }[file0_indx]
                
                #--- read data
                xdata, ydata, erry, errx = ReadDataa(file0,file0_indx)
                Rescale(file0_indx,xdata, ydata, erry)

                    
                if not timeseries:   #--- plot each time  
                    if verbose:
                        print('xdata=',xdata,'\nydata=',ydata,'\nerry=',erry,'\nerrx=',errx)                        
                    attrs={ 'color':colors[count],
                            'markersize':markersizes[count],
                            'marker':markers[count],
                            'markerfacecolor':colors[count],
                            'markeredgecolor':'white', #'black' if not fillstyles[count] else None,
                            'label':'%s/irun%s/itime%s'%(mg,irun,itimee),
                           'markevery':nevery,
                           'errorevery':nevery,
                           'markeredgewidth':kwargs['markeredgewidth'] if 'markeredgewidth' in kwargs else 1.75,
                            'linewidth':1, 
                           'barsabove':None,
                           'capsize':5,
                           'capthick':1,
                           'elinewidth':1,
                           'fmt':kwargs['fmt'] if 'fmt' in kwargs else '.',
                           'drawstyle':drawstyle,
                          }
                    #--- rescale
                    if 'scaley' in kwargs:
                        ydata /= kwargs['scaley'][count]
                        erry /= kwargs['scaley'][count]
                    
                    if verbose:
                        print('plot itime=%s,irun=%s,len(ax.lines)=%s'%(itimee,irun,len(ax.lines)))
#                     if erry == [np.nan]:
#                         erry = None
                    PltErr(xdata,ydata, 
                       yerr=erry,      
                       xerr=errx, #None if not 'xerr' in kwargs or not kwargs['xerr'] else  errx,      
                       ax = ax,
                       xstr = xstr,
                       ystr = ystr,
                       attrs = attrs,
                       Plot = False,
                       **kwargs,
                      )

                else:
                    assert timeseries #--- concat different times
                    xdata_timeseries += list(xdata)
                    ydata_timeseries += list(ydata)
                    erry_timeseries  += list(erry)
                    errx_timeseries  += list(errx)

                    
            if timeseries: #--- plot timeseries
                xdata_timeseries = np.array(xdata_timeseries)
                ydata_timeseries = np.array(ydata_timeseries)
                erry_timeseries = np.array(erry_timeseries)
                errx_timeseries = np.array(errx_timeseries)
            #--------------------
            #--- rescale data
            #--------------------
                if file0_indx in [ 15, 24 ]:
                    if np.any(~np.isnan(ydata_timeseries)):
                        pmax = np.max(xdata_timeseries)
                        #
                        tmp = np.array(xdata_timeseries)
                        tmp.sort()
                        q=0.95
                        ns=len(tmp)
                        pmax = tmp[int(q*ns)]
                        #
                        if 'pmax' in kwargs:
                            pmax = kwargs['pmax']

                        if verbose:
                            print('pmax=',pmax)
                        xdata_timeseries = 1.0 - np.array( xdata_timeseries ) / pmax 
                elif file0_indx in [21]:
                    if np.any(~np.isnan(ydata_timeseries)):
#                        pc = xdata_timeseries[ydata_timeseries>0.0][0]
                        ind = np.arange(len(ydata_timeseries))[ydata_timeseries>0.0][0]
                        pc = xdata_timeseries[ind-1]
                        if 'pc' in kwargs:
                            pc = kwargs['pc']
                        if verbose:
                            print('pc=',pc)
                        xdata_timeseries = np.array( xdata_timeseries ) / pc - 1.0

        
                #--- concat different realizations
                try:
                    Xdata = np.concatenate((Xdata,xdata_timeseries),axis=0) #--- concat different realizations
                    Ydata = np.concatenate((Ydata,ydata_timeseries),axis=0) 
                except:
                    traceback.print_exc()
                    Xdata = xdata_timeseries.copy()
                    Ydata = ydata_timeseries.copy()
                

                #--- plot
                if 'PlotEvery' in kwargs and kwargs['PlotEvery']:
                #--- graph-related attribute
                    attrs={ 'color':color,
                            'markersize':markersize,
                            'marker':marker,
#                            'markerfacecolor':fillstyle,
                            'markeredgecolor':'white', #'black' if not fillstyle else None,
                            'label':'%s/irun%s'%(mg,irun),
                           'markevery':nevery,
                           'errorevery':nevery,
                           'markeredgewidth':1.75,
                            'linewidth':1, 
                           'barsabove':None,
                           'capsize':5,
                           'capthick':1,
                           'elinewidth':1,
                           'fmt':kwargs['fmt'] if 'fmt' in kwargs else '.',
                           'zorder':1,
                          }                    
                    #--- plot ydata
                    if verbose:
                        print('ydata=',ydata_timeseries)
                        print('plot irun=%s,len(ax.lines)=%s'%(irun,len(ax.lines)))
                    PltErr(xdata_timeseries,ydata_timeseries, 
                           yerr=erry_timeseries,      
                           xerr=errx_timeseries,      
                           ax = ax,
                           xstr = xstr,
                           ystr = ystr,
                           attrs = attrs,
                           Plot = False,
                           **kwargs,
    #                           xerr=yerr0,
                          )

        #
        if 'PlotMean' in kwargs and kwargs['PlotMean']:
            Xdata = Xdata[~np.isnan(Ydata)]
            Ydata = Ydata[~np.isnan(Ydata)]
            #---
#             if file0_indx in [21]: #--- include positive vals
#                 Xdata = Xdata[Ydata>0]
#                 Ydata = Ydata[Ydata>0]
            if verbose:
                print('ydata=',Ydata)
                print('plot len(ax.lines)=%s'%(len(ax.lines)))
            try:
                #--- take average
                nbins=1024 if not 'nbins' in kwargs else kwargs['nbins']
                if file0_indx == 8:
                    nbins=1
                Xbin, Ybin, Yerr = BinData(Xdata,Ydata,nbins=nbins,scale='linear' if not 'scale' in kwargs else kwargs['scale'])
                attrs={ 'color':color,
                        'markersize':markersize,
                        'marker':marker,
                        'markerfacecolor':fillstyle,
                        'markeredgecolor':'black' if not fillstyle else None,
                        'label':'%s'%(mg),
                       'markevery':nevery,
                       'errorevery':nevery,
                       'markeredgewidth':0.7,
                        'linewidth':1, 
                       'barsabove':None,
                       'capsize':5,
                       'capthick':1,
                       'elinewidth':1,
                       'zorder':2,
                      }

                PltErr(Xbin,Ybin, 
                       yerr=Yerr,      
                       ax = ax,
                       attrs = attrs,
                       xstr = xstr,
                       ystr = ystr,
                       Plot = False,
                       **kwargs,
                #      xerr=yerr0,
                      )
            except:
#                    traceback.print_exc()
                pass


    return ax


def PltErr( xdata, ydata, 
            yerr = None,
            xstr = '',
            ystr = '',
            Plot = True,
            **kwargs
            ):
    fontsize=kwargs['fontsize'] if 'fontsize' in kwargs else 20
    if not 'ax' in kwargs:
        fig = plt.figure( figsize = (4,4) if 'figsize' not in kwargs else kwargs['figsize'] )
        ax = fig.add_subplot(111)
#        ax.count = 0
        ax.markerss=['o','s','D','^','<','>','v']

#        ax.set_prop_cycle(marker=['o', '+', 'x', '*', '.', 'X'])
    else:
        ax = kwargs['ax']
#        ax.count += 1

        if 'twinx' in kwargs and kwargs['twinx']:
                ax = kwargs['ax'].twinx()
    #--- setting   
    ax.set_xlabel(xstr,fontsize=fontsize)
    ax.set_ylabel(ystr,fontsize=fontsize)
    ax.tick_params(labelsize=fontsize,which='both',axis='both', top=True, right=True)
    #
    xerr = kwargs['xerr'] if 'xerr' in kwargs else None 
#
    if 'attrs' in kwargs:
        ax.errorbar( xdata, ydata,yerr = yerr, xerr = xerr, **kwargs['attrs'])
        if 'fill_between' in kwargs and kwargs['fill_between']:
            ax.fill_between(xdata, ydata-yerr, ydata+yerr)
    else:
        ax.errorbar( xdata, ydata,yerr = yerr, xerr = xerr,
                    fmt=kwargs['fmt'] if 'fmt' in kwargs else '-o',
                    label=kwargs['label'] if 'label' in kwargs else '',
                    markevery=kwargs['markevery'] if 'markevery' in kwargs else 1,
                    markersize=kwargs['markersize'] if 'markersize' in kwargs else 10,
                    marker=kwargs['marker'] if 'marker' in kwargs else 'o', #ax.markerss[(ax.count)%7],
                   )

    #--- plot
    #
#    ax.plot(ax.axis()[:2],[0.0,0.0],'-.',color='black')
    #
    if 'ylim' in kwargs:
        ylim = kwargs['ylim'] 
        ax.set_ylim(ylim)
    if 'xlim' in kwargs:
        xlim = kwargs['xlim'] 
        ax.set_xlim(xlim)
    #
    if 'xscale' in kwargs: 
        ax.set_xscale(kwargs['xscale'])
    if 'yscale' in kwargs: 
        ax.set_yscale(kwargs['yscale'])
    #
    if 'xticks' in kwargs:
        ax.set_xticks(list(map(float,kwargs['xticks'][1])))
#        ax.set_xticklabels(list(map(lambda x:'$%s$'%x,kwargs['xticks'][0])))
        ax.set_xticklabels(kwargs['xticks'][0])
    #
    if 'yticks' in kwargs:
        ax.set_yticks(list(map(float,kwargs['yticks'][1])))
        ax.set_yticklabels(list(map(lambda x:'$%s$'%x,kwargs['yticks'][0])))
        
    #
    LOGY = True if ('yscale' in kwargs and kwargs['yscale'] == 'log') else False
    LOGX = True if ('xscale' in kwargs and kwargs['xscale'] == 'log') else False
    ndecade_x = kwargs['ndecade_x'] if 'ndecade_x' in kwargs else 1
    ndecade_y = kwargs['ndecade_y'] if 'ndecade_y' in kwargs else 1
    PutMinorTicks(ax, LOGX=LOGX,LOGY=LOGY,nevery_x=ndecade_x,nevery_y=ndecade_y)
    #
    if 'DrawFrame' in kwargs: 
        DrawFrame(ax, *kwargs['DrawFrame'],LOG_Y=LOGY,LOG_X=LOGX)
    #
    if 'legend' in kwargs:
        plt.legend(**kwargs['legend'])
	#
    if 'halfopen' in kwargs and kwargs['halfopen']:
        ax.spines['right'].set_visible(False) #--- half open
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    if 'title' in kwargs: #Plot:
        plt.savefig(kwargs['title'],dpi=300 if not 'dpi' in kwargs else kwargs['dpi'],bbox_inches='tight', 
                    pad_inches=0.0)
    if Plot:
        plt.show()
    #
    
    
#    if not 'ax' in kwargs:
    return ax


def Zscore(val):
    x=val.copy()
    x -= np.mean(x)
    x /= np.std(x)
    return x

def SetTickLabels(ax, **kwargs):
    fmt='%3.1f'
    if 'xaxis' in kwargs:
        tickLabels = kwargs['xaxis']
        ax.xaxis.set_ticklabels(['$%s$'%i for i in tickLabels])
        ax.xaxis.set_ticks(tickLabels)
    if 'yaxis' in kwargs:
        tickLabels = kwargs['yaxis']
        ax.yaxis.set_ticklabels(['$%s$'%i for i in tickLabels])
        ax.yaxis.set_ticks(tickLabels)
        
def PutMinorTicks(ax, LOGY=None,LOGX=None, nevery_x=1,nevery_y=1):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if LOGY:
        #--- add major yticks
        ymin=np.ceil(np.log10(ax.axis()[2]))
        ymax=np.floor(np.log10(ax.axis()[3]))
        nbin = ymax - ymin
        ax.set_yticks(10**np.arange(ymin,ymax+nevery_y,nevery_y))
#        ax.set_yticks(np.logspace(ymin,ymax,int(nbin)+1))
        #--- put minor bins y
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),numticks=12)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if LOGX:
        #--- add major yticks
        ymin=np.ceil(np.log10(ax.axis()[0]))
        ymax=np.floor(np.log10(ax.axis()[1]))
        nbin = ymax - ymin
        ax.set_xticks(10**np.arange(ymin,ymax+nevery_x,nevery_x))
#        ax.set_xticks(np.logspace(ymin,ymax,int(nbin)+1))
#        print(10**np.arange(ymin,ymax,nevery_x))
#        ax.set_xticks(np.logspace(ymin,ymax,int(nbin/nevery_x)+1))
        #--- put minor bins y
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        
def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    
def gaussian_mixture( values, 
                     times,
                     LABELS = False, 
                     PLOT = True):
    thresh = {}
    ratio = {}
#    pdb.set_trace()
    gsFitTotal = 1.0
    for itime in sorted( times ):
    #--- convert data frema to array
        nij = values[itime]
        X=np.log10(nij)
        X=np.array(X).reshape(len(X),1)

        try:
            gm_obj = skm.BayesianGaussianMixture(n_components=2, tol=1e-8, max_iter=10000,
                                                verbose=0, 
    #                                            random_state=32,
    #                                            init_params='random',
                                               )
            gm_obj.fit(X)

            if not gm_obj.converged_:
                continue

            #--- min(\mu0, \mu1) corresponds to trig. mode
            mean0 = gm_obj.means_[0][0]
            mean1 = gm_obj.means_[1][0]
            d     = { 'trigrd':min([mean0,0],[mean1,1])[1], 
                      'backgrd':max([mean0,0],[mean1,1])[1]}
            mean  = { 'trigrd' : gm_obj.means_[d['trigrd']][0],
                      'backgrd': gm_obj.means_[d['backgrd']][0] }
            sigma = { 'trigrd' : gm_obj.covariances_[d['trigrd']][0]**0.5,
                      'backgrd': gm_obj.covariances_[d['backgrd']][0]**0.5 }
            delta = { 'trigrd' : gm_obj.weights_[d['trigrd']],
                      'backgrd': gm_obj.weights_[d['backgrd']] }
    #        print(sigma['backgrd']/mean['backgrd'])
            #--- plot scatter
            nij_red = nij[gm_obj.predict(X)==0]
            nij_blue = nij[gm_obj.predict(X)==1]
            #ratio[itime] =  sigma['backgrd']/mean['backgrd'] #
            ratio[itime] = 1.0 * len(nij_blue) / (len(nij_red)+len(nij_blue))
    #        print(len(nij_red)+len(nij_blue))
    #        prob = np.random.rand(len(X))
    #         list_of_red = prob  <   gm_obj.predict_proba(X)[:, d['trigrd']] #--- assign modes based on the responsibilities \gamma
    #         list_of_blue = prob >= gm_obj.predict_proba(X)[:, d['trigrd']]
    #         nij_red = nij[list_of_red]
    #         nij_blue = nij[list_of_blue]

            #--- plot distributions
            edge_act, hist_act = DistNij(nij,normed=None, nbins_per_decade = 16)

            if PLOT:
                print('itime=%s, ratio=%s'%(itime,ratio[itime]))
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(111)
    #            ax.set_yscale('log')
    #            ax.set_ylim(0.9,len(nij_red))#1e3)
                #
                ax.plot(edge_act,hist_act,'o',color='black',label='Total')
                #
                xv = edge_act
                ax.plot( xv, 
                        len(X)*(delta['trigrd']*gaussian(xv, mean['trigrd'], sigma['trigrd'])+
                                delta['backgrd']*gaussian(xv, mean['backgrd'], sigma['backgrd']))*(xv[1]-xv[0]), 
                        color='black')
                ax.plot( xv, len(X)*delta['trigrd']*gaussian(xv, mean['trigrd'], sigma['trigrd'])*(xv[1]-xv[0]),color='red')
                ax.plot( xv, len(X)*delta['backgrd']*gaussian(xv, mean['backgrd'], sigma['backgrd'])*(xv[1]-xv[0]),color='C0')
                if LABELS:
                    ax.set_xlabel(r'$log n_{ij}$')
                    ax.set_ylabel(r'$P(log n_{ij})$')
                ax.set_xlim(np.floor(np.min(edge_act)),np.ceil(np.max(edge_act)))
                #
                gsFitTotal = np.c_[ax.get_lines()[1].get_xdata(),ax.get_lines()[1].get_ydata()] #--- return data
            #


            #--- find decision boundary
            mu0,mu1,sigma0,sigma1,delta0,delta1=mean['trigrd'],mean['backgrd'],sigma['trigrd'],sigma['backgrd'],delta['trigrd'],delta['backgrd']
            def f(x): #,:
                return delta0*np.exp(-0.5*((x-mu0)/sigma0)**2)/sigma0 -\
                        delta1*np.exp(-0.5*((x-mu1)/sigma1)**2)/sigma1
            n_th = 0
            try:
                n_th = optimize.bisect(f, mu0-1*sigma0, mu1+1*sigma1)
                if PLOT:
                    ax.plot([n_th,n_th],ax.axis()[2:],'-.r') #--- vertical line
                thresh[itime]=10**n_th
            except:
                traceback.print_exc()
                pass
            if PLOT:
                fig.savefig('distnijGM.png',bbox_inches='tight',dpi=2*75)
                plt.show()

        except:
            traceback.print_exc()
            continue
    #     print gm_obj.predict_proba([[n_th]])
    return thresh, ratio, gsFitTotal

def DistNij(nij,normed=True, nbins_per_decade = 4, **kwargs):
#--- histogram
    nn=np.log10(nij)
    nmin=kwargs['nmin'] if 'nmin' in kwargs else nn.min()
    nmax=kwargs['nmax'] if 'nmax' in kwargs else nn.max()
    bins=np.linspace(nmin,nmax,int(nmax-nmin)*nbins_per_decade)
    hist, edge = np.histogram(nn,bins=bins,normed=normed)
    
    #--- accumulated histogram
    slist=np.array(nn)
    slist.sort()
    N = len(slist)
    d = histogramACCUMLTD( slist.tolist() )
    keys=list(d.keys())
    keys.sort()
    
    xx=[];yy=[]
    for ikey in keys:
        xx.append(d[ikey][0])
        yy.append(d[ikey][2])
    
    # ax2 = ax.twinx()
    # ax2.plot(xx,yy,
    #         linestyle='-', drawstyle='steps-post',color='red',
    #         linewidth=1.0) #--- accumulated
    # #ax2.set_xlim(-7,1)
    # #ax2.set_ylim(0,1200)
    
    # ax2.tick_params(axis='y',colors='red')
    # ax2.set_ylabel('$N(<n_{ij})$',color='red')
    
    return edge[:-1],hist

def valuesDict(d,keys):
    return list(map(d.get,sorted(keys)))


def histogramACCUMLTD( slist ):
    assert type( slist ) == type( [] ), 'arg must be a list. a %s is given!' %( type( slist ) )
    d = {}
    for item in slist:
        try:
            d[ item ] += 1
        except:
            d[ item ] = 1
    keys = list(d.keys())
    keys.sort()

    cdf = 0.0
    xi = min( slist ) - 1.0e-6
    xf = max( slist ) + 1.0e-6
    npoin = len( slist )
    adict = {}
    for ikey, index in zip( keys, range( sys.maxsize ) ):
        adict[ index ] = [ xi, ikey, cdf ]
        cdf += 1.0 * d[ ikey ] # / npoin
        xi = ikey
    adict[ index + 1 ] = [ xi, xf, cdf ]
    return adict

def GetStrain(lmpData, times, time0 ):
    ebulk = {}
    box0 = lp.Box( BoxBounds = lmpData.BoxBounds[time0] )
    box0.BasisVectors( AddMissing = np.array([0.0,0.0,0.0] ))
    for itime in times:

        box = lp.Box( BoxBounds = lmpData.BoxBounds[itime] )
        box.BasisVectors( AddMissing = np.array([0.0,0.0,0.0] ))
        #
        #--- bulk strain
        dx=box.CellVector[0,1]-box0.CellVector[0,1]
        l1=box.CellVector[1,1]
        ebulk[itime] = dx/l1
    return ebulk
        
    
def to_dict( df ):
    sdict = {}
    skeys = df.keys().to_list()
    for ikey in skeys:
        sdict[ikey] = df[ikey].to_list()

    return sdict

def isSane( AddColumns, columnList ):
    #--- avoid duplicates
    assert len( set( AddColumns ) ) == len( AddColumns ), 'duplicates in the list!'
    #--- assert column list is not already included
    n = len( AddColumns )
    AddColumns = list( set(AddColumns) - set( columnList ) )
    if len(AddColumns) != n:
        print('column already included!')
        return False
    return True

        
def PrintOvito( cordc, sfile, footer, attr_list=['x','y','z'] ):
    smat = cordc[attr_list]
    np.savetxt(sfile,[len(smat)],fmt='%s', footer='%s, %s'%(footer,str(attr_list)))
    np.savetxt(sfile,smat,fmt='%s')
#    sfile.close()
    
def PltBitmap( value,
              xlabel = 'x', ylabel = 'y',
              xlim = (-0.5,0.5), ylim = (-0.5,0.5),
#              frac = 1.0, #--- plot a patch
              zscore = True,
              title = 'cxy.png',
              colorbar=False,
 #           Plot = True,
              **kwargs
             ):
        
    val = value.copy()
#    pdb.set_trace()
    #--- z-score
    if zscore:
        val -= np.mean(val)
        val /= np.std(val)
        val[val>1.0]=1
        val[val<-1.0]=-1
    if 'vminmax' in kwargs:
        (vmin,vmax) = kwargs['vminmax']
    else:
        (vmin,vmax) = (np.min(val[~np.isnan(val)]), np.max(val[~np.isnan(val)]))
    #--- plot
    (mgrid,ngrid) = val.shape
    center = (ngrid/2,mgrid/2)
    #
#    if 'ax' in kwargs:
#        ax = kwargs['ax']
#    else:
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
	#
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
	#
    ax.tick_params(labelsize=20,which='both',axis='both', top=True, right=True)

#    fig = plt.figure(figsize=(4,4))
#    ax = fig.add_subplot(111)
    fontsize=20 if not 'fontsize' in kwargs else kwargs['fontsize']
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.tick_params(labelsize=fontsize,which='both',axis='both', top=True, right=True)
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
    #
    origin = kwargs['origin'] if 'origin' in kwargs else 'lower'
    pos = ax.imshow(val.real,cmap='bwr' if not 'cmap' in kwargs else kwargs['cmap'],
                     extent=(xlim[0],xlim[1],ylim[0],ylim[1]),origin=origin ,
                    vmin=vmin, vmax=vmax,
                   interpolation=None if not 'interpolation' in kwargs else kwargs['interpolation']
                   )
    if 'mask' in kwargs:
#        print('mask called')
        assert kwargs['mask'].shape == val.shape
        ax.contourf(kwargs['mask'], 1, hatches=['','....'], linewidths=1.0,colors='red',alpha=0.0,
                     extent=(xlim[0],xlim[1],ylim[0],ylim[1]),origin=origin)
    if 'frac' in kwargs:
        frac = kwargs['frac']
        ax.set_xlim(xlim[0]*frac,xlim[1]*frac)
        ax.set_ylim(ylim[0]*frac,ylim[1]*frac)
    if 'fracx' in kwargs:
        fracx = kwargs['fracx']
        ax.set_xlim(fracx)
    if 'fracy' in kwargs:
        fracy = kwargs['fracy']
        ax.set_ylim(fracy)

    if colorbar:
        fig.colorbar( pos, pad=0.05 if not 'pad' in kwargs else kwargs['pad'], 
					shrink=0.5,fraction = 0.04, orientation='vertical' if not 'orientation' in kwargs else kwargs['orientation'] )
    if 'DrawFrame' in kwargs: 
        DrawFrame(ax, *kwargs['DrawFrame'])
    if 'set_title' in kwargs:
        ax.set_title(kwargs['set_title'],fontsize=fontsize)
    #
    LOGY = True if ('yscale' in kwargs and kwargs['yscale'] == 'log') else False
    LOGX = True if ('xscale' in kwargs and kwargs['xscale'] == 'log') else False
    PutMinorTicks(ax, LOGX=LOGX,LOGY=LOGY)
    #
    if 'xticks' in kwargs:
        ax.set_xticks(list(map(float,kwargs['xticks'][1])))
#        ax.set_xticklabels(list(map(lambda x:'$%s$'%x,kwargs['xticks'][0])))
        ax.set_xticklabels(kwargs['xticks'][0])
    #
    if 'yticks' in kwargs:
        ax.set_yticks(list(map(float,kwargs['yticks'][1])))
        ax.set_yticklabels(list(map(lambda x:'$%s$'%x,kwargs['yticks'][0])))

	
#    if 'title' in kwargs: #Plot:
#        plt.savefig(kwargs['title'],dpi=300 if not 'dpi' in kwargs else kwargs['dpi'],bbox_inches='tight', 
#                    pad_inches=0.0)
#    if Plot:
#     plt.show()

#    plt.savefig(title,dpi=2*75,bbox_inches='tight',pad_inches=0.0)
#    plt.show()
#    return ax


    plt.savefig(title,dpi=2*75,bbox_inches='tight',pad_inches=0.0)
    plt.show()
    
    
def PltCrltn( value,
              xlabel = 'x', ylabel = 'y',
              xlim = (-0.5,0.5), ylim = (-0.5,0.5),
              frac = 1.0, #--- plot a patch
              zscore = True,
              fileName = 'cxy.png',
              dpi=75,
            ):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    #
    val = value.copy()
    #--- zscore
    if zscore:
        val -= np.mean(val)
        val /= np.std(val)
        val[val>2.0]=1.0
        val[val<-2.0]=-1.0
    #
    (mgrid,ngrid) = val.shape
    center = (ngrid/2,mgrid/2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.axes.xaxis.set_visible(False) #--- remove labels
    ax.axes.yaxis.set_visible(False)

    pos = ax.imshow((CenterMatrix( val ).real),cmap='bwr',
                     extent=(xlim[0],xlim[1],ylim[0],ylim[1]),
                     #,vmin=-.01, vmax=.01
                    )
    ax.set_xlim(xlim[0]*frac,xlim[1]*frac)
    ax.set_ylim(ylim[0]*frac,ylim[1]*frac)

#    plt.colorbar( pos, fraction = 0.04)
    plt.savefig(fileName,dpi=dpi,bbox_inches='tight')
    plt.show()
    


def GetAutoCorrelation( val ):
    value  = val.copy()
    value -= np.mean( value )
    value /= np.std( value )

    ( nx, ny, nz ) =  value.shape
    n = nx * ny * nz
    vq = np.fft.fftn(value) #, axes=(0,1,2))
    vq_sq = np.abs(vq)**2

    v_real = np.fft.ifftn( vq_sq) / n #,  axes=(0,1,2) )
    return v_real

def CenterMatrix(a):
    ( mgrid, ngrid ) = a.shape
    return np.array([[a[i,j] for j in range(-int(ngrid/2),int(ngrid/2)+ngrid%2)] 
                              for i in range(-int(mgrid/2),int(mgrid/2)+mgrid%2)])

def Get_rc( xdata, ydata, cTOL ):
    try:
#        xc1 = xdata[np.abs(ydata)<cTOL][0] 
        xc1 = xdata[ydata<cTOL][0] 
    except:
        xc1 = np.nan
    try:    
        xc2 = xdata[ydata<0.0][0] #--- correlation length
    except:
        xc2 = np.nan
    try:
        xc = np.array([xc1,xc2])
        xc = xc[~np.isnan(xc)].min()
    except:
        xc = np.nan
    return xc

def PltCrltnFunc( crltn, 
                 xv,yv,
                 cTOL = 1.0e-2,
                 PLOT = True,
                 fileName='cxy.png',
                 title = '',
                 dpi = 60,
                ):
    (ny,nx,nz) = crltn.shape
    if PLOT:
        fig = plt.figure( figsize = (4,4))
        ax = fig.add_subplot(111)
        xstr = r'$r$'
        ystr = r'$c(r)$'
#         ax.set_xlabel(xstr,fontsize=16)
#         ax.set_ylabel(ystr,fontsize=16)
        ax.tick_params(labelsize=16)
        PutMinorTicks(ax)
        ax.tick_params(labelsize=20,which='both',axis='both', top=True, right=True)
    #
    val = crltn[:,:,0].copy() #--- only xy plane
    (m,n)=val.shape
    #--- along x 
    xv2 = xv[:,:,0].copy()
    dx = xv2[0,1] - xv2[0,0]
    #
    xdata0 = np.arange(0,(int(n/2)+n%2)) * dx
    ydata0 = val[0,0:(int(n/2)+n%2)]
    #
    xc = Get_rc( xdata0, ydata0, cTOL )
    #
    if PLOT:
        ax.plot( xdata0, ydata0,
                '-o',label=r'$x$',
                markersize=10,
                color='black',
                markerfacecolor='white',
                markeredgecolor=None,
                markevery=int(len(xdata0)/10),
               )       
    #--- along y 
    yv2 = yv[:,:,0].copy()
    dy = yv2[1,0] - yv2[0,0]
    #
    xdata = np.arange(0,(int(m/2)+m%2)) * dy
    ydata = val[0:(int(m/2)+m%2),0]
    #
    yc = Get_rc( xdata, ydata, cTOL )
    #
    if PLOT:
        ax.plot( xdata, ydata,
                '-s', 
                color = 'red',
                label=r'$y$',
                markersize=10,
                markerfacecolor=None,
                markeredgecolor='black',
                markevery=int(len(xdata)/10),
               )
    #--- plot
    if PLOT:
        ax.legend(frameon=False, fontsize=20,handletextpad=.4,handlelength=1)
        ax.set_title( title )
        #ax.set_yscale('log')
        #
        ax.plot(ax.axis()[:2],[0.0,0.0],'-.',color='red')
#        ax.plot([dx,dx],ax.axis()[2:],'-.',color='black')
#        ax.plot((xc,xc),ax.axis()[2:],'-.r')
        #
        DrawFrame(ax, 0.2,0.09,0.15,0.06,0.04)
        #
        plt.savefig(fileName,dpi=2*75,bbox_inches='tight',pad_inches=0.0)
        plt.show()
    #
    return (xc, yc), (xdata0,ydata0), (xdata,ydata) 


def GetSlice2d( hist, err,
         xv, yv, zv,
         xtol = 2.5,
         z = 0.0):
############################################################
####### Get a 2D slice out off a 3D matrix
############################################################    
    dx=xtol #--- discretization length
    (ny,nx,nz) = hist.shape
    indices = np.all([np.abs(zv-z)>0.0,np.abs(zv-z)<dx],axis=0) #--- filtering based on the given range
    #--- binning in xy
    flist = hist[indices]
    rvect = np.c_[yv[indices],xv[indices]]
    rx,    bin_edges = np.histogramdd( rvect, bins = (ny, nx), weights = xv[indices] ) #--- \sum r_i
    ry,    bin_edges = np.histogramdd( rvect, bins = (ny, nx), weights = yv[indices]) #--- \sum r_i
    error,    bin_edges = np.histogramdd( rvect, bins = (ny, nx), weights = err[indices] ) #--- \sum r_i
    fmean, bin_edges = np.histogramdd( rvect, bins = (ny, nx), weights = flist ) #--- 3d histogram
    count, bin_edges = np.histogramdd( rvect, bins = (ny, nx) ) #--- n_i

    count[count==0] = 1
    rx /= count 
    ry /= count 
    fmean /= count
    error /= count
    return rx, ry, fmean, error

def GetSlice1d( hist, err,
         xv, yv,
         xtol = 2.5,
         **kwargs):
############################################################
####### Get a 1D slice out off a 2D matrix
############################################################    
    dx=xtol #--- discretization length
    (ny,nx) = hist.shape
    if 'y' in kwargs:
        y = kwargs['y']
        indices = np.all([np.abs(yv-y)>0.0,np.abs(yv-y)<dx],axis=0) #--- filtering based on the given range
        flist = hist[indices]
        rvect = xv[indices]
        rdist,    bin_edges = np.histogram( rvect, bins = nx, weights = xv[indices] ) #--- \sum r_i
        error,    bin_edges = np.histogram( rvect, bins = nx, weights = err[indices] ) #--- \sum r_i
        count, bin_edges = np.histogram( rvect, bins = nx ) #--- n_i
        fmean, bin_edges = np.histogram( rvect, bins = nx, weights = flist ) #--- 3d histogram
    elif 'x' in kwargs:
        x = kwargs['x']
        indices = np.all([np.abs(xv-x)>0.0,np.abs(xv-x)<dx],axis=0) #--- filtering based on the given range
        flist = hist[indices]
        rvect = yv[indices]
        rdist,    bin_edges = np.histogram( rvect, bins = ny, weights = yv[indices] ) #--- \sum r_i
        error,    bin_edges = np.histogram( rvect, bins = ny, weights = err[indices] ) #--- \sum r_i
        count, bin_edges = np.histogram( rvect, bins = ny ) #--- n_i
        fmean, bin_edges = np.histogram( rvect, bins = ny, weights = flist ) #--- 3d histogram
        
    #--- binning in xy

    count[count==0] = 1
    rdist /= count 
    fmean /= count
    error /= count
    return rdist, fmean, error


def PltCrltnFunc1d( crltn, err,
                 xv,
                 cTOL = 1.0e-2,
                 PLOT = True,
                 fileName='cxy.png',
                 title = '',
                 dpi = 60,
                 ylim=(-1.0,+1.0),
                ):
    if PLOT:
        fig = plt.figure( figsize = (4,4))
        ax = fig.add_subplot(111)
        xstr = r'$r$'
        ystr = r'$c(r)$'
        ax.set_xlabel(xstr,fontsize=16)
        ax.set_ylabel(ystr,fontsize=16)
        ax.tick_params(labelsize=16)
    #
    val = crltn.copy() #--- only xy plane
#    (m,n)=val.shape
    #--- along x 
    xv2 = xv.copy()
    dx = xv2[1] - xv2[0]
    #
    xdata = xv2 #np.arange(0,(n/2+n%2)) * dx
    ydata = val #[0,0:(n/2+n%2)]
    #
    xc = Get_rc( xdata, ydata, cTOL )
    #
    if PLOT:
        ax.errorbar( xdata, ydata,yerr = err, fmt='-o',label=r'$x$')       
    #--- plot
    if PLOT:
        ax.legend(frameon=False)
        ax.set_title( title )
        #ax.set_yscale('log')
        #
        ax.plot(ax.axis()[:2],[0.0,0.0],'-.',color='black')
#        ax.plot([dx,dx],ax.axis()[2:],'-.',color='black')
        ax.plot([xc,xc],ax.axis()[2:],'-.',color='black')
        #
#        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        #
        plt.savefig(fileName,dpi=dpi,bbox_inches='tight')
        plt.show()
    #
    return xc


def DrawFrame(ax, alpha_xl,alpha_xr,alpha_yb,alpha_yt,linewidth,LOG_X=None,LOG_Y=None):
    [xlo,xhi,ylo,yhi] = ax.axis()
    if LOG_X:
        [xlo,xhi,junk,junk] = np.log10(ax.axis())
    if LOG_Y:
        [junk,junk,ylo,yhi] = np.log10(ax.axis())
    lx = xhi - xlo
    ly = yhi - ylo
    xy = [xlo - alpha_xl * lx, ylo - alpha_yb * ly]
    height = ly*(1+alpha_yb+alpha_yt)
    width = lx*(1+alpha_xl+alpha_xr)
    xy_end=[xy[0]+width,xy[1]+height]
    if LOG_X:
        xy[0] = 10 ** xy[0]
        xy_end[0] = 10 ** xy_end[0]
    if LOG_Y:
        xy[1] = 10 ** xy[1]
        xy_end[1] = 10 ** xy_end[1]
    ax.add_patch( patches.Rectangle(xy=xy, width=xy_end[0]-xy[0], 
                                    height=xy_end[1]-xy[1], linewidth=linewidth,
                                    clip_on=False,facecolor=None,edgecolor='black',fill=None) ) 
    
def MultipleFrames( path='', title='', irun = 0, nmax = 10000 ):
    i=0
    append = False
    while i < nmax:
        try:
            sarr0 = np.c_[np.loadtxt('%s%i/Run%s/%s'%(path,i,irun,title))].T
            #        print i,sarr0
            if not append:
                sarr = sarr0.copy()
                append = True
            else:
                sarr = np.concatenate((sarr,sarr0),axis=0)
        except:
            i+=1
    #        traceback.print_exc()
            continue
        i+=1
    return sarr

def MultipleFrames2nd( path='', title='', nrun = 0, ncols=3 ):
    i=0
    append = False
#    pdb.set_trace()
    while i < nrun:
        sarr0 = (np.ones(ncols)*np.nan).reshape((1, ncols))
#        print(sarr0.shape)
        try:
            sarr0 = np.c_[np.loadtxt('%s/Run%s/%s'%(path,i,title))].T
#            print(sarr0.shape)
        except:
#            traceback.print_exc()
            pass
        if not append:
            sarr = sarr0.copy()
            append = True
#            print(i,sarr0)
        else:
#            print(i,sarr0)
            sarr = np.concatenate((sarr,sarr0),axis=0)
        i+=1
    return sarr

def AvalancheSize(strain, Virial):
    d={'t_end':[],'duration':[],'ds':[]}
    kount = 0
    duration = 0
    ds = 0.0
#    counter = 0
    try:
        dtt = strain[1]-strain[0] #TimeSeries[0]['Time'].iloc[1]-TimeSeries[0]['Time'].iloc[0]
        for items,sbulk in list(zip(strain, Virial)): #TimeSeries[isamp].itertuples():
#            sbulk = items[2]
            t = items #items[1]
            #--- ens. average
            if kount == 0:
                a = sbulk
                ta = t
            elif kount == 1:
                b = sbulk
                tb = t
            else:
                c = sbulk
                tc = t
            if kount < 2:
                kount += 1
                init0 = kount + 1
                continue
            sdr = 0.5 * ( c - a ); #--- derivative
#                print(t, sdr)
#            if sdr < 0.0 and counter == 0:
#                continue
            if sdr > 0.0: #--- elastic loading
                init0 = kount + 1; #--- initialize init
            else: #--- avalanche starts!
#                    print(t, sdr)
                ds += sdr #--- increment avalanche size by the stress derivative
                duration += 1 #--- increment duration
            if init0 - kount == 1 and duration != 0: #--- avalanche ends!
                print(duration,ds) #tc-duration*(tb-ta),tc,duration
    #			ax.plot([tc-duration*(tb-ta),tc], [0.0,0.0]
    #                    ,'o')
                d['t_end'].append(tc) 
                d['duration'].append(duration*dtt) 
                d['ds'].append(-ds)
                ds = 0.0 #--- initialize 
                duration = 0
#                counter+=1
    #		if counter == 1:
    #			break
            a = b #--- new assignments
            b = c;
            ta = tb #--- new assignments
            tb = tc;
            kount += 1
    except:
#        traceback.print_exc()
        pass
    if duration != 0: #--- one single avalanche!
        d['t_end'].append(tc) 
        d['duration'].append(duration*dtt) 
        d['ds'].append(-ds)
#    print(duration,ds)
#fig.show()
    df=pd.DataFrame(d)
    df=df[df['ds']!=0.0]
    
    return df

def GetPDF(slist, n_per_decade=4, ACCUM = None, linscale = None, density=True,**kwargs):
    if 'bins' in kwargs:
        bins = kwargs['bins'] 
    elif not linscale:
        xlo = np.floor(np.log10(np.min(slist)))
        xhi = np.ceil(np.log10(np.max(slist)))
        bins = np.logspace(xlo,xhi,int(xhi-xlo)*n_per_decade)    
    else:
        assert not 'bins' in kwargs and linscale
        xlo = np.min(slist)
        xhi = np.max(slist)
        bins = np.linspace(xlo,xhi,n_per_decade)

    bins = kwargs['bins'] if 'bins' in kwargs else bins
        
    hist, edges = np.histogram(slist,bins=bins,density=density)
    count, edges = np.histogram(slist,bins=bins)
    
    
    if ACCUM:
        return np.cumsum((edges[1:]-edges[:-1])*hist), edges
    
    if density:
    	nth=1   
    	hist = hist[count>nth]
    	edges = edges[:-1][count>nth]
    	count = count[count>nth]
    else:
        count=1

    return  hist, edges, hist / count**0.5


# create a definition for the short hyphen
#matplotlib.rcParams["text.latex.preamble"]+= r'\mathchardef\mhyphen="2D'#.append(r'\mathchardef\mhyphen="2D')

class MyLogFormatter(matplotlib.ticker.LogFormatterMathtext):
    def __call__(self, x, pos=None):
        pass
#         # call the original LogFormatter
#         rv = matplotlib.ticker.LogFormatterMathtext.__call__(self, x, pos)

#         # check if we really use TeX
#         if matplotlib.rcParams["text.usetex"]:
#             # if we have the string ^{- there is a negative exponent
#             # where the minus sign is replaced by the short hyphen
#             rv = re.sub(r'\^\{-', r'^{\mhyphen', rv)

#         return rv
    
def makeTickLabels(ax, xdata, ydata, **kargs):
    ylo = kargs['ylo'] if 'ylo' in kargs else 10**np.floor(np.log10(np.min(ydata)))
    yhi = kargs['yhi'] if 'yhi' in kargs else 10**np.ceil(np.log10(np.max(ydata)))
    xlo = kargs['xlo'] if 'xlo' in kargs else 10**np.floor(np.log10(np.min(xdata)))
    xhi = kargs['xhi'] if 'xhi' in kargs else 10**np.ceil(np.log10(np.max(xdata)))
    center = kargs['center'] if 'center' in kargs else True
    MINUS = kargs['MINUS'] if 'MINUS' in kargs else True
    xc = 0.5*np.log10(xhi*xlo)
    yc = 0.5*np.log10(yhi*ylo)
    
    dx = np.log10(xhi/xlo)*0.5
    dy = np.log10(yhi/ylo)*0.5

    if center:
        dx = dy = np.max([dx,dy])
    
    ax.axis(10**np.array([xc-dx,xc+dx,yc-dy,yc+dy]))
    ax.loglog()
    
    ##--- add major xticks
    xmin=np.ceil(np.log10(ax.axis()[0]))
    xmax=np.floor(np.log10(ax.axis()[1]))
    nbin = xmax - xmin
    ax.set_xticks(np.logspace(xmin,xmax,int(nbin)+1))
    
    #--- add major yticks
    ymin=np.ceil(np.log10(ax.axis()[2]))
    ymax=np.floor(np.log10(ax.axis()[3]))
    nbin = ymax - ymin
    ax.set_yticks(np.logspace(ymin,ymax,int(nbin)+1))
    
    #--- put minor bins
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if MINUS:
        ax.xaxis.set_major_formatter(MyLogFormatter()) #--- minus sign too long

    #--- put minor bins y
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if MINUS:
        ax.yaxis.set_major_formatter(MyLogFormatter()) #--- minus sign too long

        
    ax.tick_params(axis='y',left=True, right=True,which='both')
    ax.tick_params(axis='x',bottom=True, top=True,which='both')
    
def GetBinnedAverage( a, y, **kwargs ):
    n=len(a)
    if 'nbins_per_decade' in kwargs:
        nbins = kwargs['nbins_per_decade'] * int( ( np.ceil(np.max(a))-np.floor(np.min(a)) ) )
    if 'nbins' in kwargs:
        nbins = kwargs['nbins']
    
    ysum = np.histogram(a, bins=nbins, weights=y)[0]
    xsum = np.histogram(a, bins=nbins, weights=a)[0]
    xcount = np.histogram(a, bins=nbins)[0]

#    print xsum
#    print xcount
#    assert not np.any(xcount==0)
    #--- remove zero
    xsum = xsum[xcount!=0]
    ysum = ysum[xcount!=0] 
    xcount = xcount[xcount!=0]

    xmean = xsum/xcount
    ymean = ysum/xcount
    return xmean, ymean, xmean / xcount ** 0.5, ymean / xcount ** 0.5

def FilterDataFrame(df,key='id',val=[1,2,3]): #,out='C66'):
    tmp0 = df.set_index(key,drop=True,append=False).loc[val]
    return tmp0.reset_index() #.reindex(range(len(tmp0)))

def Get2dSlice( value, zlin, zc, nzll=[0] ):
        #--- get xy plane
#    zc=0.5*(zlin[0]+zlin[-1])
    dz = zlin[-1]-zlin[-2]
    lz = zlin[-1]-zlin[0]
    nz = len(zlin)
    #
    zz = zc #zlin[-1] #zc #zlin[-1] #--- arbitrary plane
    nzz=int(nz*(zz-zlin[0])/lz)
#    print(nzz,nz)
    if nzz == nz: nzz -= 1
    val = value[:,:,nzz].copy()
    #
    nzll[0] = nzz
    return val

def Intrp( d2min, box0, attr, Plot = None, title = 'test.png',**kwargs ):
    #--- mean dist between atoms 
    natoms = len( d2min.x ) 
    CellVectorOrtho, VectorNorm = lp.GetOrthogonalBasis( box0.CellVector )
    volume = np.linalg.det( CellVectorOrtho )
    dmean = 0.5*( volume / natoms ) ** (1.0/3.0) if not 'dx' in kwargs else kwargs['dx']


    #--- grid tiling mapped box with original size
    #--- values are interpolated onto this grid
    (xlin, ylin, zlin), (xv, yv, zv) = lp.GetCubicGrid( box0.CellOrigin, 
                                                     box0.CellVector, 
                                                     dmean,
                                                     margin = 0.0 * dmean )
    xi = np.array(list(zip(xv.flatten(), yv.flatten(), zv.flatten())))

    #--- expand the original box
        #--- map to square box
    mapp = lp.Map( d2min, box0 ) 
    mapp.ChangeBasis()
    mapp.Set( d2min ) #--- atoms: add mapped xyz

    cptmp = lp.Copy(d2min, box0) #--- important: must be reference frame!!
    cptmp.Expand( epsilon = 0.2, mode = 'isotropic' )
    d2exp = cptmp.Get()

    points = np.c_[d2exp.xm,d2exp.ym,d2exp.zm] #--- unstructured points
    values = np.array(d2exp[attr]) #(np.array(d2exp.C66)+np.array(d2exp.C55)+np.array(d2exp.C44))/3.0 #np.c_[-(np.array(d2exp.sxx)+np.array(d2exp.syy)+np.array(d2exp.szz))/3.0/np.array(d2exp.AtomicVolume)] #--- corresponding values
#    pdb.set_trace()
    method = kwargs['method'] if 'method' in kwargs else 'linear'
    grid_z = scp_int.griddata(points, values, xi, method=method)
    assert not np.any(np.isnan(grid_z.flatten())), 'increase ev!'

    #--- make an object
    d2intrp = lp.Atoms(**pd.DataFrame(np.c_[xi,grid_z],columns=['x','y','z',attr]).to_dict(orient='series'))

    #--- reshape value
    nx,ny,nz = len(xlin), len(ylin),len(zlin) 
    value = np.c_[d2intrp[attr]].reshape(((ny,nx,nz)))

    CellVectorOrtho, VectorNorm = lp.GetOrthogonalBasis( box0.CellVector ) #--- box length

    #--- xy plane
    #--- 2d slice
    nzl=[0]
    val = Get2dSlice( value, zlin, 
                        zlin[-1], nzll=nzl  )
    nzz=nzl[0]
        #
    #--- square bitmap
    lx=np.min([xlin[-1]-xlin[0],ylin[-1]-ylin[0]])
    xc = 0.5*(xlin[-1]+xlin[0])
    yc = 0.5*(ylin[-1]+ylin[0])

    if Plot:
        PltBitmap(val, 
#                  xlim=VectorNorm[0]*np.array([-0.5,0.5]),ylim=VectorNorm[1]*np.array([-0.5,0.5]),
                  xlim=np.array([xc-0.5*lx,xc+0.5*lx]),ylim=np.array([yc-0.5*lx,yc+0.5*lx]),
                  frac = 1.0, #--- plot a patch
                  title = title,
                  **kwargs
                )

#    return (xlin, ylin, zlin), (xv[:,:,nzz], yv[:,:,nzz], zv[:,:,nzz]), d2intrp
    return (xlin, ylin, zlin), (xv, yv, zv), d2intrp

def PltBinary(xlin,ylin,zlin, 
              val,
              box0,
              thresh = 0.0,
              **kwargs
             ):
    #--- reshape value
    (nx,ny,nz) = len(xlin), len(ylin),len(zlin) 
    value = np.c_[val].reshape(((ny,nx,nz)))

    #--- mean & variance

    #--- xy plane
    #--- 2d slice
    nzl=[0]
    value2d = Get2dSlice( value, zlin, 
                        zlin[-1], nzll=nzl  )

    #--- square bitmap
    lx=np.min([xlin[-1]-xlin[0],ylin[-1]-ylin[0]])
    xc = 0.5*(xlin[-1]+xlin[0])
    yc = 0.5*(ylin[-1]+ylin[0])                  
    
    #--- plot
    CellVectorOrtho, VectorNorm = lp.GetOrthogonalBasis( box0.CellVector ) #--- box length
    Plot = True if not 'Plot' in kwargs else kwargs['Plot']
    if Plot:
        PltBitmap(value2d<thresh, 
#              xlim=VectorNorm[0]*np.array([-0.5,0.5]),ylim=VectorNorm[1]*np.array([-0.5,0.5]),
              xlim=np.array([xc-0.5*lx,xc+0.5*lx]),ylim=np.array([yc-0.5*lx,yc+0.5*lx]),
              frac = 1.0, #--- plot a patch
              **kwargs
            )
    return value

class Stats:
    #--------------------------
    #--- cluster statistics
    #--------------------------
    def __init__( self, mask, xlin, ylin, zlin,
              verbose = False ):
        self.mask = mask
        self.xlin = xlin
        self.ylin = ylin
        self.zlin = zlin
#        if verbose:
#            print('p=%s\npinf=%s\nsmean=%s\nsi_sq=%s'%(p,pinf,smean,crltnl_sq))
        
    def GetProbInf(self):
#        percCount = self.stats[self.stats['percTrue']==True].shape[0]
#        self.pinf0 = 1.0*percCount/self.stats.shape[0]
        
        self.pinf = (self.stats['percTrue'] * self.stats['size']).sum()/self.stats['size'].sum()
#        print(self.pinf0,self.pinf)
    #--- p
    def GetProb(self):
        (ny,nx,nz) = self.mask.shape
        nsize = nx*ny*nz
        self.p = 1.0*self.mask.sum()/nsize #--- occupation prob.

    #--- <s^2>/<s>
    def GetSmean(self):
        self.smean = (self.stats['size']*self.stats['size']).sum()/self.stats['size'].sum()
    #--- correlation length
    def GetCrltnLenSq(self):
        self.si_sq = 2*(self.stats['rg_sq']*self.stats['size'] * self.stats['size']).sum()/\
                  (self.stats['size'] * self.stats['size']).sum()       

    def isPercolating(self,sliceX,sliceY,sliceZ,size):
        (ny,nx,nz)=size
        #
        xlo = sliceX.start
        xhi = sliceX.stop
        assert xhi - xlo <= nx
        #    
        ylo = sliceY.start
        yhi = sliceY.stop
        assert yhi - ylo <= ny
        #    
        zlo = sliceZ.start
        zhi = sliceZ.stop
        assert zhi - zlo <= nz
        #
        return xhi - xlo == nx or yhi - ylo == ny or zhi - zlo == nz

    def covarianceMat(self):
        '''
        returns co-variance matrix corresponding to each cluster
        '''
        (ny,nx,nz) = self.mask.shape
        self.xv,self.yv,self.zv=np.meshgrid(range(nx),range(ny),range(nz))


        labels = np.arange(1, self.nb_labels+1)
        covar_mat = np.c_[list(map(lambda x:self.GetcoVar(x),labels))]
        self.covar_mat = pd.DataFrame(np.c_[labels,covar_mat],columns='label xx xy xz yy yz zz'.split())
    #        xc=ndimage.mean(xv, label_im, np.arange(1, nb_labels+1))
    #        yc=ndimage.mean(yv, label_im, np.arange(1, nb_labels+1))
    #        zc=ndimage.mean(zv, label_im, np.arange(1, nb_labels+1))

    def GetcoVar(self,cls_label):
    #		cls_label = 1
        filtr = self.label_im == cls_label

        count = np.sum(filtr)

        xmean = np.mean(self.xv[filtr])
        ymean = np.mean(self.yv[filtr])
        zmean = np.mean(self.zv[filtr])

        varxy = np.sum((self.xv[filtr] - xmean)*(self.yv[filtr] - ymean)) / count
        varxz = np.sum((self.xv[filtr] - xmean)*(self.zv[filtr] - zmean)) / count
        varyz = np.sum((self.yv[filtr] - ymean)*(self.zv[filtr] - zmean)) / count

        varxx = np.var(self.xv[filtr].flatten())
        varyy = np.var(self.yv[filtr].flatten())
        varzz = np.var(self.zv[filtr].flatten())

        return np.array([varxx,varxy,varxz,varyy,varyz,varzz])

    def Orientation(self):
        '''
        returns cluster orientation
        '''
        ans = np.c_[list(map(lambda x:Stats.Diagonalize(self.covar_mat.iloc[x]),range(self.nb_labels)))]
        labels = np.c_[list(map(lambda x:self.covar_mat.iloc[x]['label'],range(self.nb_labels)))] 
        self.orientation = pd.DataFrame(np.c_[labels,ans],columns='label nx ny nz'.split())
		

    @staticmethod
    def Diagonalize(a):
        amat = np.array([[a.xx,a.xy,a.xz],[a.xy,a.yy,a.yz],[a.xz,a.yz,a.zz]])
        w, v = np.linalg.eigh(amat)
        assert w[0] <= w[1] and w[0] <= w[2], 'not sorted!'
        return v[:, 0]

    def GetSize(self):
        #--- clusters
        label_im, nb_labels = ndimage.label(self.mask)
#        assert nb_labels > 1, 'nb_labels == 0!'
        self.label_im = label_im
        self.nb_labels = nb_labels
        #--- cluster bounds
        sliced=ndimage.find_objects(label_im,max_label=0)
    #    sliceX = sliced[0][1]
    #    sliceY = sliced[0][0]
    #    sliceZ = sliced[0][2]
    #    isPercolating(sliceX,sliceY,sliceZ,mask.shape)
    #
        #--- percolation
        percTrue = list(map(lambda x:self.isPercolating(x[1],x[0],x[2],self.mask.shape),sliced))    
        assert len( percTrue ) == nb_labels

        #--- geometry
        xc = ndimage.measurements.center_of_mass(self.mask, label_im,np.arange(1, nb_labels+1)) #--- (yc,xc,zc)

        (ny,nx,nz) = self.mask.shape
        xv,yv,zv=np.meshgrid(range(nx),range(ny),range(nz))
        xc=ndimage.mean(xv, label_im, np.arange(1, nb_labels+1))
        yc=ndimage.mean(yv, label_im, np.arange(1, nb_labels+1))
        zc=ndimage.mean(zv, label_im, np.arange(1, nb_labels+1))
        varx=ndimage.variance(xv, label_im, np.arange(1, nb_labels+1))
        vary=ndimage.variance(yv, label_im, np.arange(1, nb_labels+1))
        varz=ndimage.variance(zv, label_im, np.arange(1, nb_labels+1))
        #---
        dx = self.xlin[1]-self.xlin[0]
        dy = self.ylin[1]-self.ylin[0]
        dz = self.zlin[1]-self.zlin[0]
        radg_sq = varx * dx * dx + vary * dy * dy + varz * dz * dz
        #
        ones = np.ones(nx*ny*nz).reshape(ny,nx,nz)
        size=ndimage.sum(ones, label_im, np.arange(1, nb_labels+1)) * dx * dy * dz
		#--- orientation
        self.covarianceMat()
        self.Orientation()


        #--- postprocess
#        df=pd.DataFrame(np.c_[label_im.flatten()],columns=['id'])
#        sdict=df.groupby(by='id').groups

    #    pdb.set_trace()
#        df_cls = pd.DataFrame(np.c_[list(sdict.keys()),
    #                                list(map(lambda x:len(sdict[x]),sdict.keys()))
 #                                  ],columns=['cls_id'])

        #--- sort based on id
#        df_cls.sort_values('cls_id',ascending=True,inplace=True)
#        df_cls = df_cls.iloc[1:] #--- remove label 0

        #--- append percTrue
#        df_cls=pd.DataFrame(np.concatenate((np.c_[df_cls],np.c_[size],np.c_[radg_sq],np.c_[percTrue]),axis=1, dtype=np.object), columns=['cls_id','size','rg_sq','percTrue'])
#        df_cls=pd.DataFrame(np.concatenate((np.c_[df_cls],np.c_[size],np.c_[radg_sq],np.c_[percTrue]),axis=1), columns=['cls_id','size','rg_sq','percTrue'])
        df_cls=pd.DataFrame(np.c_[np.arange(1, nb_labels+1),size,radg_sq,percTrue,self.orientation[['nx','ny','nz']]], columns='cls_id size rg_sq percTrue nx ny nz'.split())
#        pdb.set_trace()
        #---
        #--- sort based on size
        df_cls.sort_values('size',ascending=False,inplace=True)


        self.stats = df_cls

    def Print(self,xpath,filee,x, header):
            os.system('mkdir -p %s'%xpath)



def SetDataFrameByValue(df, key='id', key_vals=[1,2,3], value='StructureType', value_vals=[1,1,1]):
    df_0 = utl.FilterDataFrame(df, key=key, val=key_vals)
    df_0[value] = value_vals
#    pdb.set_trace()
    other_vals = set(df.id) - set(key_vals)
    df_1 = utl.FilterDataFrame(df, key=key, val=list(other_vals))
    df_merge = np.concatenate([df_0,df_1])
    return pd.DataFrame(np.c_[df_merge],columns=list(df_0.keys()))

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

    
class ReadWriteJson:
    def __init__(self):
        pass
        
    
    def Write( self, data, fout, **kwargs ):
        self.data = data
        assert type(self.data) == type([]), 'data must be a list of dicts.'
        for x in self.data:
            assert type(x) == type({}), 'elements of data must be dictionaries!'
        
        
        with open(fout, "w") as fp:
            keys = kwargs.keys()
            for x in keys:
                assert type(kwargs[x]) == type([]), '%s must be a list!'%x
                assert len(kwargs[x]) == len(self.data), 'len(%s) must be equal to data'%x
            #
            for item,indx in zip(self.data,range(len(self.data))):
                values = list(map(lambda x:kwargs[x][indx],keys))
                dictionary={**item, **dict(zip(keys,values))}
                json.dump( dictionary, fp, cls=NumpyArrayEncoder )
                fp.write('\n')

    def Read(self, finp):
        with open(finp, 'r') as inpfile:
            return list(map(lambda x:json.loads(x),inpfile))
