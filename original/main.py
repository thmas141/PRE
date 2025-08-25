import sys
import os
sys.path.insert(0, 'C:/Users/thdelvau/OneDrive - NTNU/Documents/Release')

import matplotlib.pyplot as plt

from NetSocks import NSClient ,customize_plots,Shape  # type: ignore
import numpy as np
from pathlib import Path
import scipy as sp
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import copy
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from scipy.fft import rfft, rfftfreq
import random


#print(0.1)
ns = NSClient(); 
#print(0)
ns.configure(True, False)
ns.cuda(2)
ns.reset(); ns.clearelectrodes()
#print(1)
def init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap,defect=False) :
    ns.reset()
    meshdims = (Lx, Ly, Lz)
    # Parameters
    # ps
    # Set up the antiferromagnet
    AFM = ns.AntiFerromagnet(np.array(meshdims)*1e-9, [cellsize*1e-9])
    AFM.modules(modules)
    temp = str(T) + 'K'
    ns.temperature(temp)
    if (T==0):
        ns.setode('LLG', 'RK4') 
    else :
        ns.setode('sLLG', 'RK4') # Stochastic LLG for temperature effects
    ns.setdt(10e-15)#ns.setdt(10e-15)#ns.setdt(50e-15)#ns.setdt(1e-15)
    if (material=='hematite'):
    # Set parameters   
        AFM.param.grel_AFM = 1
        AFM.param.damping_AFM =  2e-4
        AFM.param.Ms_AFM = 2.1e3
        AFM.param.Nxy = 0
        AFM.param.A_AFM = 76e-15 # J/m
        AFM.param.Ah = -460e3 # J/m^3
        AFM.param.Anh = 0.0
        AFM.param.J1 = 0
        AFM.param.J2 = 0
        AFM.param.K1_AFM = 21# J/m^3
        AFM.param.K2_AFM = 0
        AFM.param.K3_AFM =0
        AFM.param.cHa = 1
        AFM.param.ea1 = (1,0,0)
        ns.setktens('-0.001x2', '1z2')
    elif (material=='FeBO3'):
        AFM.param.grel_AFM = 1
        AFM.param.damping_AFM =1e-4#1e-6 for disp#5e-4for random simu  #5.5e-3 max
        AFM.param.Ms_AFM = 3.5e3
        AFM.param.Nxy = 0
        AFM.param.A_AFM = 24.3e-15#1.64e-24 # J/m 
        AFM.param.Ah = -228e3#-228e3 # J/m^3 
        AFM.param.Anh = 0.0
        AFM.param.J1 = 0
        AFM.param.J2 = 0
        AFM.param.K1_AFM =1#297.5#3.29e5 #2.74#3.29e5# J/m^3 ??
        AFM.param.K2_AFM = 0
        AFM.param.K3_AFM =0#0.939
        AFM.param.cHa = 1
        AFM.param.ea1 = (1,0,0)
        AFM.param.Dh=1.08e4#1.08e4#2.28e4
        AFM.param.dh_dir = [0,0,1]
        AFM.param.Ddir = (0,1,0)
        AFM.param.D_AFM=0
        ns.setktens('-0.0455x2', '297.5z2' )
    else :
        print('unknown material') 
    AFM.param.Ms_AFM.clearparamsvar()
    #if defect :
        #AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,1500*1e-9,110])
        # disk=Shape.disk([10*cellsize*1e-9,10*cellsize*1e-9,cellsize*1e-9])
        # AFM.shape_set(disk.move([0.5*Lx*1e-9,0.5*Ly*1e-9,0.5*Lz*1e-9]))
        # #AFM.shape_setangle(disk,[180,0])
        # Ms= AFM.param.Ms_AFM.setparam()
        # AFM.param.Ms_AFM.shape_setparam(disk,1)
        #AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,1500*1e-9,110])#1defect = AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,1500*1e-9,23]), 2defects= 1000
    # Add increased damping at edges along x-axis to prevent reflections. For smaller meshes this needs to be shortened
    AFM.param.damping_AFM.setparamvar('abl_tanh', [damping_x/meshdims[0], damping_x/meshdims[0], 0, 0, 0, 0, 1, 10, damping_x])
    
    
    if (Hap!=0) :
        AFM.setfield(Hap,90,90)
        ns.Relax(['time', 50*1e-12])
        ns.reset()
    else :
        ns.Relax(['time', 5*1e-14])#50
        ns.reset() 
    #ns.random()
    return AFM

def magnon_transport_SOT (material,Lx,Ly,Lz,cellsize,T,modules,damping_x,V,Hap,t,output_file):
    AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap)
    Mu0=(4*3.141*1e-7) #to convert external field in Tesla to A/M wich is used in boris
    # Set spesific params for torque
    AFM.param.SHA = 1
    AFM.param.flST = 1

    # Current along y-direction
    ns.addelectrode(np.array([(Lx/2 - 100), 0, (Lz-cellsize), (Lx/2 + 100), 0, Lz])* 1e-9)
    ns.addelectrode(np.array([(Lx/2 - 100), Ly, (Lz-cellsize), (Lx/2 + 100), Ly, Lz]) * 1e-9)
    ns.designateground('1')

    # Add step function so that torque only acts on region in the injector
    if Ly==5:
        width = 20
    elif Ly == 100:
        width = 40
    func = '(step(x-' + str(Lx/2 - width/2) + 'e-9)-step(x-' + str(Lx/2 + width/2) + 'e-9)) * (step(z-' + str(Lz-cellsize) + 'e-9)-step(z-' + str(Lz) + 'e-9))'
    AFM.param.SHA.setparamvar('equation', func)
    AFM.param.flST.setparamvar('equation',func)
    #data
    Datasave=[['time'],['Ha', AFM], ['<M>', AFM, np.array([Lx,Ly,Lz])*1e-9]]
    for i in range(200) :
        #print('data',i)
        Datasave.append(['<mxdmdt>', AFM, [10*i*1e-9,0,0,10*(i+1)*1e-9,Ly*1e-9,Lz*1e-9]])

    for i in range(100) :
        Datasave.append(['<Jc>', AFM, [20*i*1e-9,0,0,20*(i+1)*1e-9,Ly*1e-9,Lz*1e-9]])

    ns.setsavedata(output_file,*Datasave )
    #AFM.setangle(90,180)
    ns.V([V, 'time', t*1e-12,'time',1e-12])

def disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file,dir,DMI,same_plot=False,defect=False):
    file=Path(output_file)
    modules=['exchange', 'anitens', 'Zeeman']
    if DMI :
        modules.append('DMexchange')
    AFM=init(material,Lx,Ly,Lz,cellsize,0,modules,damping_x,Hap,defect)
    ####don't know if the field is reset when pressing ns.reset()  
    #time_calm=10*1e-12
    #AFM.setfield(Hap,90,90)
    #ns.Relax(['time', time_calm])
    ####don't know if the field is reset when pressing ns.reset() 
    total_time = 2 *t*1e-12
    AFM.pbc('x', 10)   
    
    if (material=='hematite') :
        He =1e11#1e10 #1e11#4e1 
        N = 600 #600
        L = Lx*1e-9
        kc = 2*np.pi*N/(2*L)
        fc =1e12 #5e12
        time_step = 1e-13#1e-13#0.1e-12
    if (material=='FeBO3') :
        He=3e10#1e10 #1e11#4e1 
        N = 500
         #600
        L = Lx*1e-9
        kc = 2*np.pi*N/(2*L)
        fc =10e12#3e12#2800e6 #5e12
        time_step = 1e-13#1e-13#0.1e-12

    H0 = He
    # if same_plot==False :
    #     if dir==2 :
    #         He=0
    #     else :
    #         H0=0
    ns.equationconstants('H0', H0)
    ns.equationconstants('Hap', Hap)
    ns.equationconstants('He', He)
    ns.equationconstants('k', kc)
    ns.equationconstants('f', fc)
    ns.equationconstants('u', total_time/2)
    ns.equationconstants('L', Lx*1e-9)
    ns.equationconstants('M', Ly*1e-9)
    #ns.equationconstants('ø',1)
    ns.setstage('Hequation')
    if (material=='hematite') :
        ns.editstagevalue(0, '0, He * sinc(k*(x-(L/2)))*sinc(k*(y-(M/2)))*sinc(2*PI*f*(t-u)), H0 * sinc(k*(x-L/2))*sinc(k*(y-M/2))*sinc(2*PI*f*(t-u))')#que par rapport a y la
    if (material=='FeBO3') :
        ns.editstagevalue(0, '0,He * sinc(k*(x-L/2))*sinc(k*(y-M/2))*sinc(2*PI*f*(t-u)), H0 * sinc(k*(x-L/2))*sinc(k*(y-M/2))*sinc(2*PI*f*(t-u))')#que par rapport a y la   2*PI*
    #setup data extraction in command buffer : need to extract a normalized magnetization profile at each time_step and append it to output file
    ns.setdata('commbuf')
    ns.editstagestop(0, 'time', total_time)
    ns.editdatasave(0, 'time', time_step)
    ns.clearcommbuffer()
    #get magnetisation profile along length through center
    ns.dp_getexactprofile((np.array([cellsize/2, Ly/2, Lz-cellsize])*1e-9), 
                        (np.array([Lx - cellsize/2, Ly/2, Lz])*1e-9), 
                            cellsize*1e-9, 0,bufferCommand = True)
    if material=='hematite':
        Ms=2.1e3
    if material=='FeBO3':
       Ms=3.5e3
    #save only the dir component of magnetisation at time_ste
    if same_plot :
        ns.dp_div(2, Ms, bufferCommand=True)
        ns.dp_saveappendasrow(file, 2, bufferCommand = True)#should be true
        ns.dp_div(3, Ms, bufferCommand=True)
        ns.dp_saveappendasrow(file, 3, bufferCommand = True)#should be true
    else :
        ns.dp_div(dir, Ms, bufferCommand=True)
        ns.dp_saveappendasrow(file, dir, bufferCommand = True)#should be true

    ns.Run()

def disp_SOT(material,c,Hap1,Hap2,t,output_files,dir,T,V):
    Haps=[Hap1,Hap2]
    for i,output_file in enumerate(output_files) :
        Hap=Haps[i]
        modules=['exchange', 'anitens', 'SOTfield', 'transport','Zeeman']
        AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap)
        Mu0=(4*3.141*1e-7) #to convert external field in Tesla to A/M wich is used in boris
        # Set spesific params for torque
        AFM.param.SHA = 1
        AFM.param.flST = 1

        # Current along y-direction
        ns.addelectrode(np.array([(Lx/2 - 100), 0, (Lz-cellsize), (Lx/2 + 100), 0, Lz])* 1e-9)
        ns.addelectrode(np.array([(Lx/2 - 100), Ly, (Lz-cellsize), (Lx/2 + 100), Ly, Lz]) * 1e-9)
        ns.designateground('1')

        # Add step function so that torque only acts on region in the injector
        if Ly==5:
             width = 20
        elif Ly == 100:
            width = 40
        func = '(step(x-' + str(Lx/2 - width/2) + 'e-9)-step(x-' + str(Lx/2 + width/2) + 'e-9)) * (step(z-' + str(Lz-cellsize) + 'e-9)-step(z-' + str(Lz) + 'e-9))'
        AFM.param.SHA.setparamvar('equation', func)
        AFM.param.flST.setparamvar('equation',func)
        #data
        time=0.0
        time_step=100
        time_relax=100
        y_vals = [Ly/2]
        if material=='hematite':
            Ms=2.1e3
        if material=='FeBO3':
            Ms=3.5e3
        while time < t:
            ns.V([V, 'time', (time+time_step)*1e-12,'time',1e-12])
            ns.dp_getexactprofile((np.array([cellsize/2, y_vals[0], Lz-cellsize])*1e-9), (np.array([Lx - cellsize/2, y_vals[0], Lz])*1e-9), cellsize*1e-9, 0)
            ns.dp_div(dir, Ms)
            ns.dp_saveappendasrow(output_file, dir)
            ns.Relax(['time', (time + time_step+time_relax)*1e-12])
            time+=time_step+time_relax
    plot_magnon_dispersion_with_zoom(output_files,theorie=True)

def stat_interf (Lx,Ly,Lz,cellsize,T,modules,damping_x,t) :
    He=1e6
    H0=0
    x1=Lx/3
    x2=2*Lx/3
    S=2*0.2e-6
    phi=0
    f=3e9
    time_step=0.1e-12
    AFM=init('FeBO3',Lx,Ly,Lz,cellsize,T,modules,damping_x,0)
    ns.equationconstants('He', He)
    ns.equationconstants('H0', H0)
    ns.equationconstants('x1', x1*1e-12)
    ns.equationconstants('x2', x2*1e-12)
    ns.equationconstants('S', S)
    ns.equationconstants('P', phi)
    ns.equationconstants('f', f)
    ns.setstage('Hequation')
    ns.editstagevalue(0, 'H0, H0, He * (exp(-(x-x1)*(x-x1)*y*y/(2*S))+exp(-(x-x2)*(x-x2)*y*y/(2*S))*cos(P))*cos(2*PI*f*t)')
    ns.editstagestop(0, 'time', t)
    #ns.editdatasave(0, 'time', time_step)
    ns.Run()

def compute_frequency(time, signal):
    """
    Calcule la fréquence d'oscillation (en Hz) d'un signal temporel.
    
    Paramètres :
        time   : liste ou array des temps (en secondes)
        signal : liste ou array du signal (par ex. Mx, My...)
    
    Retour :
        fréquence moyenne (Hz), période moyenne (s), nb pics détectés
    """
    # Détection des pics (maxima)
    time = np.array(time)
    peaks, _ = find_peaks(signal)
    if len(peaks) < 2:
        return 0.0, None, 0  # Pas assez de pics pour calculer une fréquence
    # Temps des pics
    peak_times = time[peaks]
    # Différences entre pics successifs → périodes
    periods = np.diff(peak_times)
    # Moyenne de la période
    mean_period = np.mean(periods)
    frequency = 1.0 / mean_period

    print(f"Résultat : {frequency:.2e}")
    return frequency, mean_period, len(peaks)

def plot_SA(output_file,place) :
    j=0#0
    if place=='behind':
        j=3
    X=ns.Get_Data_Columns(output_file,0)
    Y1=ns.Get_Data_Columns(output_file,1+j)
    Y2=ns.Get_Data_Columns(output_file,2+j)
    Y3=ns.Get_Data_Columns(output_file,3+j)#be carefull : behind or up is not the same
    Y1p=[i/(3500*1.01) for i in Y1]#normalisation
    Y2p=[i/(3500*1.01) for i in Y2]
    Y3p=[i/(3500*1.01) for i in Y3]

    # ##if NEEL :
    # Y1=ns.Get_Data_Columns(output_file,1+3)
    # Y2=ns.Get_Data_Columns(output_file,2+3)
    # Y3=ns.Get_Data_Columns(output_file,3+3)#be carefull : behind or up is not the same
    # Y1p=[Y1p[i]-Y1[i]/(3500*1.01) for i in range(len(Y1))]#normalisation
    # Y2p=[Y2p[i]-Y2[i]/(3500*1.01) for i in range(len(Y2))]
    # Y3p=[Y3p[i]-Y3[i]/(3500*1.01) for i in range(len(Y3))]


    # ####tests
    # # for i in range(len(Y1p)):
    # #    print(Y1p[i]**2+Y2p[i]**2+Y3p[i]**2)
    # # print('mx frequencie :')
    # # compute_frequency(X[(len(X) * 2 )// 3:], Y1p[(len(X) * 2 ) // 3:])
    # # print('my frequencie :')
    # # compute_frequency(X[len(X) * 2 // 3:], Y2p[len(X) * 2 // 3:])
    # # print('mz frequencie :')
    # # compute_frequency(X[len(X) * 2 // 3:], Y3p[len(X) * 2 // 3:])


    #mx/my/mz
    plt.figure()
    plt.plot(X,Y1p,color='blue',label='<Mx>')
    plt.plot(X,Y2p,color='green',label='<My>')
    plt.plot(X,Y3p,color='red',label='<Mz>')
    plt.legend()
    plt.show()

    #plotting distance
    # My=[]
    # # #Mx=[]
    # # #Mz=[]
    # for i in range(100) :
    # #     #mx=ns.Get_Data_Columns(output_file,1+3*i)
    # #     #Mx.append(mx[len(mx)//2+100]/3500)
    #     my=ns.Get_Data_Columns(output_file,1+i)
    #     My.append(np.mean(my[len(my)//2:])/3500)
    # #     #mz=ns.Get_Data_Columns(output_file,3+3*i)
    # #     #Mz.append(np.mean(my[len(mz)//2:])/3500)
    # d=[12*i*1e-9 for i in range(100)]      
    # plt.figure()
    # # #plt.plot(d,Mx,label='<mx>')
    # plt.plot(d,My,label='<my>')
    # # #plt.plot(d,Mz,label='<mz>')
    # plt.legend()
    # plt.show()

    #plotting tranche

def plot_disp_sinc(output_files,Hap): ###y-z on the same plot
    clim_max = 1000
    time_step = 1e-13
    ylabel = 'f (THz)'
    divisor = 1e12
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(7)
    fig1.set_figwidth(16)
    titles='yz'
    for output_file in output_files:
        pos_time = np.loadtxt(output_file)
        fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

        freq_len = len(fourier_data)
        k_len = len(fourier_data[0])
        freq = np.fft.fftfreq(freq_len, time_step)
        kvector = np.fft.fftfreq(k_len, 5e-9)

        k_max = 2 * np.pi * kvector[int(0.5 * len(kvector))] * 5e-9

        f_min = np.abs(freq[0])
        f_max = np.abs(freq[int(0.5 * len(freq))]) / divisor
        final_result = [fourier_data[j] for j in range(int(0.5 * freq_len), freq_len)]
       
        label = 'q'
        ax1.imshow(final_result, origin='lower', interpolation='bilinear',
                      extent=[-k_max, k_max, f_min, f_max], aspect="auto",
                      clim=(0, clim_max))
        ax1.set_xlabel(label)
        ax1.set_ylabel(ylabel)

        # Inset zoom
        x1, x2, y1, y2 = -0.15, 0.15, 0, 0.3
        axins = ax1.inset_axes([0.5, 0.5, 0.47, 0.47], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[])
        axins.imshow(final_result, extent=[-k_max, k_max, f_min, f_max], origin='lower', clim=(0, clim_max))
        ax1.indicate_inset_zoom(axins, edgecolor='black')
        ax1.title.set_text(titles)
    plot_analytical(ax1,Hap,'y')
    plt.tight_layout()  
    plt.show()

def plot_magnon_dispersion_with_zoom(output_files,H1,H2, theorie=True):
    clim_max = 1000
    time_step = 1e-13
    ylabel = 'f (THz)'
    divisor = 1e12

    fig1, ax1 = plt.subplots(1, 2)
    fig1.set_figheight(7)
    fig1.set_figwidth(16)
    FIELDS = [H1, H2]
    titles=['y','z']
    for i, output_file in enumerate(output_files):
        pos_time = np.loadtxt(output_file)
        fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

        freq_len = len(fourier_data)
        k_len = len(fourier_data[0])
        freq = np.fft.fftfreq(freq_len, time_step)
        kvector = np.fft.fftfreq(k_len, 5e-9)

        k_max = 2 * np.pi * kvector[int(0.5 * len(kvector))] * 5e-9

        f_min = np.abs(freq[0])
        f_max = np.abs(freq[int(0.5 * len(freq))]) / divisor
        final_result = [fourier_data[j] for j in range(int(0.5 * freq_len), freq_len)]
       
        label = 'q'
        ax1[i].imshow(final_result, origin='lower', interpolation='bilinear',
                      extent=[-k_max, k_max, f_min, f_max], aspect="auto",
                      clim=(0, clim_max))
        ax1[i].set_xlabel(label)
        ax1[i].set_ylabel(ylabel)

        # Inset zoom
        x1, x2, y1, y2 = -0.15, 0.15, 0, 0.3
        axins = ax1[i].inset_axes([0.5, 0.5, 0.47, 0.47], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[])
        axins.imshow(final_result, extent=[-k_max, k_max, f_min, f_max], origin='lower', clim=(0, clim_max))
        ax1[i].indicate_inset_zoom(axins, edgecolor='black')


        ax1[i].title.set_text(titles[i])
        if theorie:
            plot_analytical(ax1[i],FIELDS[i],titles[i])
            plot_analytical(axins, FIELDS[i], titles[i])       
    plt.tight_layout()
    #savename = 'C:/Users/thdelvau/OneDrive - NTNU/Documents/test'
    #plt.savefig(savename, dpi=800)
    plt.show()

def plot_analytical(axe,Hfield,coor) :
    Ah = 228e3
    A = 24.3e-15
    Keasy = 0.0455
    Khard = 297.5
    Ms = 3.5e3
    mu0 = 4 * np.pi * 1e-7
    gamma_e_over_2pi = 2.802e10
    C = gamma_e_over_2pi/(Ms)
    Ly=1000*1e-9
    Lz=5*1e-9
    Lx=2500*1e-9
    print('plot_analytical is Ly and Lz dependent')
    # Define the dispersion relations
    def high(q_a,n,m):
        q = q_a / 5e-9
        return 4*C*np.sqrt(A*Ah*(q**2+(2*m*np.pi/Lz)**2+(2*n*np.pi/Ly)**2) + Ah*(Keasy+Khard))/1e12#
    
    def low(q_a,n,m):
        q = q_a/5e-9
        return 4*C*np.sqrt(A*Ah*(q**2+(2*n*np.pi/Ly)**2+(2*m*np.pi/Lz)**2) + Ah*Keasy+(0.25*Ms*Hfield)**2)/1e12#
    
    x = np.linspace(-0.75, 0.75, 1000)
    if coor=='y':
        for n in range(10):##10
            for m in range(1):
                axe.plot(x, low(x,n,m), color='red', linestyle = 'dashed', linewidth=1/((n+1)*(m+1)))
                #axe.plot(x, high(x,n,m), color='green', linestyle = 'dashed', linewidth=1/((n+1)*(m+1)))##si plot sur le meme, a changer manuellement c est un peu relou, faut l enlever si x,y de chaque cote
    else :
        for n in range(4):
            for m in range(1): 
                axe.plot(x, high(x,n,m), color='green', linestyle = 'dashed', linewidth=1/((n+1)*(m+1)))
    #print(low(2*np.pi*600/(2*Lx),0,0))

def angle_DMI(file):
    x=ns.Get_Data_Columns(file,0)[-1]
    y=ns.Get_Data_Columns(file,1)[-1]
    z=ns.Get_Data_Columns(file,2)[-1]
    Ma=[x,y,z]
    x=ns.Get_Data_Columns(file,3)[-1]
    y=ns.Get_Data_Columns(file,4)[-1]
    z=ns.Get_Data_Columns(file,5)[-1]
    Mb=[x,y,z]
    u=np.array(Ma)
    v=np.array(Mb)
    #print(u)
    #print(v)
    ctheta=np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
    print('angle_DMI = ',180-np.degrees(np.arccos(np.clip(ctheta,-1,1))))
    return 180-np.degrees(np.arccos(np.clip(ctheta,-1,1)))

def thermal_disp(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file,dir,DMI,T) :
    time_step=1e-13
    file=Path(output_file)
    modules=['exchange', 'anitens', 'Zeeman']
    if DMI :
        modules.append('DMexchange')
    AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap)
    ns.setdata('commbuf')
    ns.editdatasave(0, 'time', time_step)
    ns.clearcommbuffer()
    ns.dp_getexactprofile((np.array([cellsize/2, Ly/2, Lz-cellsize])*1e-9), 
                        (np.array([Lx - cellsize/2, Ly/2, Lz])*1e-9), 
                            cellsize*1e-9, 0,bufferCommand = True)
    Ms=3.5e3
    ns.dp_div(dir, Ms, bufferCommand=True)
    ns.dp_saveappendasrow(file, dir, bufferCommand = True)#should be true
    ns.Relax(['time', 50*1e-12,'time', 1*1e-13])

def setup_defect(AFM,Lx,Ly,Lz,a_x,a_y,a_z,nb=1):
    ns.setdt(1e-15)
    function=''
    def one_def(x0,y0,z0):
        func = '1.01 - ((step(x - ' + str((x0 - a_x/2)*1e-9) + ') - step(x - ' + str((x0 + a_x/2)*1e-9) + ')) * ' + \
           '(step(y - ' + str((y0 - a_y/2)*1e-9) + ') - step(y - ' + str((y0 + a_y/2)*1e-9) + ')) * ' + \
           '(step(z - ' + str((z0 - a_z/2)*1e-9) + ') - step(z - ' + str((z0 + a_z/2)*1e-9) + ')))'
        return func 

    def placer_carres_sans_chevauchement(n, ax, ay, Lx, Ly, max_essais=10000):
        centres = []
        for _ in range(n):
            for _ in range(max_essais):
                x = random.uniform(ax / 2, Lx - ax / 2)
                y = random.uniform(ay / 2, Ly - ay / 2)
                nouveau_centre = (x, y)

                # Vérifier la distance avec les autres centres
                trop_proche = False
                for cx, cy in centres:
                    if abs(x - cx) < ax and abs(y - cy) < ay:
                        trop_proche = True
                        break

                if not trop_proche:
                    centres.append(nouveau_centre)
                    break
                else:
                    print("⚠️ Impossible de placer tous les carrés sans chevauchement.")
                    break
        return centres

    if nb==1:
        x0=Lx*0.3
        y0=Ly*0.5
        z0=Lz/2
        function=one_def(x0,y0,z0)
    elif nb==2:
        c1=(720,600)
        c2=(765,600)
        c3=(740,620)
        c4=(740,580)
        centres=[c2]
        for c in centres :
            function+=one_def(c[0],c[1],Lz/2)
    else :
        centres=placer_carres_sans_chevauchement(nb, a_x, a_y, Lx/2, Ly, max_essais=10000)
        for c in centres :
            function+=one_def(c[0],c[1],Lz/2)
    AFM.param.Ms_AFM.setparamvar('equation', function)

def setup_data(AFM,cellsize,i):
    if i==1 :
        datasave=[['time'],['<M>', AFM, [740*1e-9,600*1e-9,0,740*1e-9+cellsize*1e-9,600*1e-9+cellsize*1e-9,cellsize*1e-9]],['<M>', AFM, [600*1e-9,0.5*Ly*1e-9,0,600*1e-9+cellsize*1e-9,0.5*Ly*1e-9+cellsize*1e-9,cellsize*1e-9]]]
    if i==2:
        datasave=[['time'],['<M>', AFM, [740*1e-9,600*1e-9,0,740*1e-9+cellsize*1e-9,600*1e-9+cellsize*1e-9,cellsize*1e-9]],['<M2>', AFM, [740*1e-9,600*1e-9,0,740*1e-9+cellsize*1e-9,600*1e-9+cellsize*1e-9,cellsize*1e-9]]]
    if i==3:
        datasave=[['time']]
        for i in range (100):
            datasave.append(['<My>', AFM, [745*1e-9,(600-3*i)*1e-9,0,740*1e-9+cellsize*1e-9,(600-3*i)*1e-9+cellsize*1e-9,cellsize*1e-9]])
    if i==4:
        datasave=[['time']]
        for i in range (100):
            datasave.append(['<My>', AFM, [(1250-(i+1)*12)*1e-9,295*1e-9,0,(1250-(i)*12)*1e-9,300*1e-9,5*1e-9]])
    if i==5:
        datasave=[['time']]
        for i in range (100):
            datasave.append(['<M>', AFM, [(1250-(i+1)*12)*1e-9,500*1e-9,0,(1250-(i)*12)*1e-9,505*1e-9,5*1e-9]])
    return datasave

def antenna(t,output_file,DM,defects,dir=2):
    time_step=t*1e-12*1e-4# for gif
    Lx=2500
    Ly=1000
    Lz=5
    T=0#.3
    cellsize=5
    taille=10*(cellsize*1e-9)
    modules=['exchange', 'anitens','Zeeman']
    if DM :
        modules.append('DMexchange')
    file=Path(output_file)
    AFM=init('FeBO3',Lx,Ly,Lz,cellsize,T,modules,500,0,defects)#remember to add bool defect
    if defects : 
        #AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,8*cellsize*1e-9,8*cellsize*1e-9,1500*1e-9,100])#AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,8*cellsize*1e-9,8*cellsize*1e-9,1500*1e-9,100]) max
        #setup_defect(AFM,Lx,Ly,Lz,4*cellsize,4*cellsize,cellsize)   
        setup_defect(AFM,Lx,Ly,Lz,2*cellsize,2*cellsize,cellsize,10)     
    H0=0
    He=1e5#4e5
    ns.setstage('Hequation')
    ns.equationconstants('H0', H0)
    ns.equationconstants('He', He)
    ns.equationconstants('L', 0.5*Lx*1e-9)#
    ns.equationconstants('S', taille**2)
    ns.equationconstants('f',0.3*1e12)#ns.equationconstants('f',0.3*1e12)
    ns.editstagevalue(0, 'H0, H0, He * exp(-(x-L)*(x-L)/(2*S))*sin(f*(t))')

    ns.editstagestop(0, 'time', t*1e-12)
    ns.editdatasave(0,'time',time_step)
    Datasave = setup_data(AFM,cellsize,2)
    ns.setsavedata(file, *Datasave)#------>a remettre sinon pas de data
    ns.Run()

def antenna_plot(output_file,Lx,Ly,Lz,cellsize):
    X=ns.Get_Data_Columns(output_file,0)
    def f(m,n):
        mz=ns.Get_Data_Columns(output_file,1+30*m+n)[-1]
        return mz #np.mean(mz[len(mz)//2:])
    
    Z = np.zeros((59, 59))
    for i in range(58):
        print(i)
        for j in range(58):
            # print('i=',i)
            # print('j=',j)
            Z[j, i] = f(i, j)  # Z[row, col] = f(m, n)
    
    # Tracé 3D
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=[0, 0.75 * Lx * 1e-9,0, 1.0 * Ly * 1e-9 ], origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='<Mz>')
    plt.title('<Mz> over the sample')
    plt.xlabel('m')
    plt.ylabel('n')
    plt.show()

#'C:\Users\thdelvau\OneDrive - NTNU\Documents\main\CleanV1\test.gif'
from matplotlib import animation

def antenna_animation_with_defect(output_file, Lx, Ly, Lz, cellsize, save_as='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/testV2/antenna/test2.gif'):
    times = ns.Get_Data_Columns(output_file, 0)
    nx, ny = 59, 59
    pos_defect = [1850*1e-9, 140*1e-9, 0]
    idx_x = int(round(pos_defect[0] / (cellsize*1e-9)))
    idx_y = int(round(pos_defect[1] / (cellsize*1e-9)))
    defect_size = 2  # Mettre à 1,2,3 selon la taille souhaitée du masque
    print(1)
    def get_Mz_map(frame_idx):
        Mz_map = np.zeros((ny, nx))
        for i in range(nx):
            print(i)
            for j in range(ny):
                mz = ns.Get_Data_Columns(output_file, nx*i+j)
                Mz_map[j, i] = mz[frame_idx] if len(mz) > frame_idx else np.nan

        # Mettre du noir autour du défaut
        for dx in range(-defect_size, defect_size+1):
            for dy in range(-defect_size, defect_size+1):
                xi = idx_x + dx
                yi = idx_y + dy
                if 0 <= xi < nx and 0 <= yi < ny:
                    Mz_map[yi, xi] = np.nan  # np.nan = sera rendu noir si cmap l’indique
        return Mz_map
    print(2)
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [0, 0.75 * Lx * 1e-9, 0, 1.0 * Ly * 1e-9]
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='black')
    im = ax.imshow(np.zeros((ny, nx)), extent=extent, origin='lower', cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='<Mz>')
    plt.title('<Mz> over the sample')
    plt.xlabel('m')
    plt.ylabel('n')

    def update(frame_idx):
        Z = get_Mz_map(frame_idx)
        im.set_array(Z)
        ax.set_title(f"<Mz> (t={times[frame_idx]*1e12:.2f} ps)")
        return [im]

    nframes = min([len(ns.Get_Data_Columns(output_file, i)) for i in range(nx*ny)])
    ani = animation.FuncAnimation(fig, update, frames=nframes, blit=True, interval=50)
    ani.save(save_as, writer=PillowWriter(fps=20))
    plt.show()
    plt.close(fig)

def test_plot(file): 
    X = ns.Get_Data_Columns(file, 0)
    Y1 = [i/3500 for i in ns.Get_Data_Columns(file, 1)]
    Y2 = [i/3500 for i in ns.Get_Data_Columns(file, 2)]
    Y3 = [i/3500 for i in ns.Get_Data_Columns(file, 3)]

    fig, ax = plt.subplots()
    quiver = ax.quiver(0, 0, Y1[0], Y2[0], angles='xy', scale_units='xy', scale=1, color='red')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    
    def update(n):
        faster=100*n
        ax.clear()
        # Flèche (origine, composante Mx, My) et couleur selon Mz
        color = plt.cm.viridis((Y3[faster]+1)/2)  # Mz normalisé [-1,1] vers [0,1]
        ax.quiver(0, 0, Y1[faster], Y2[faster], angles='xy', scale_units='xy', scale=1, color=color)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(f'Time = {X[faster]:.2e}')
        ax.set_aspect('equal')


    ani = FuncAnimation(fig, update, frames=len(X)-500, interval=0.0001)
    ani.save('C:/Users/thdelvau\OneDrive - NTNU/Documents/results/3/M_above_defect/m.gif', writer='pillow', fps=20)
    plt.show()



def multi_plot(file1, file2, file3, file4):
    def load_data(file):
        X = ns.Get_Data_Columns(file, 0)
        Y1 = [i / 3500 for i in ns.Get_Data_Columns(file, 1)]
        Y2 = [i / 3500 for i in ns.Get_Data_Columns(file, 2)]
        Y3 = [i / 3500 for i in ns.Get_Data_Columns(file, 3)]
        return X, Y1, Y2, Y3

    data_list = [load_data(file) for file in [file1, file2, file3, file4]]
    print(1)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()

    for ax in axs:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

    quivers = [
        axs[i].quiver(0, 0, data_list[i][1][0], data_list[i][2][0], angles='xy', scale_units='xy', scale=1, color='red')
        for i in range(4)
    ]
    fixed_titles = ['nothing', 'DMI', 'defect', 'defect+DMI']
    def update(n):
        print(n)
        step =  10*n
        for i in range(4):
            X, Y1, Y2, Y3 = data_list[i]
            if step >= len(X): continue  # skip if index out of bounds
            axs[i].clear()
            color = cm.viridis((Y3[step] + 1) / 2)  # Normalize Mz [-1,1] → [0,1]
            axs[i].quiver(0, 0, Y1[step], Y2[step], angles='xy', scale_units='xy', scale=1, color=color)
            axs[i].set_xlim(-1, 1)
            axs[i].set_ylim(-1, 1)
            axs[i].set_aspect('equal')
            axs[i].set_title(fixed_titles[i], fontsize=10)
            axs[i].annotate(f'Time = {X[step]:.2e}', xy=(0.5, -0.12), xycoords='axes fraction', ha='center', fontsize=8)

    total_frames = min(len(d[0])//10  for d in data_list)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=0.0001)
    ani.save('C:/Users/thdelvau/OneDrive - NTNU/Documents/main/MultiCleanV1.gif', writer='pillow', fps=20)
    plt.show()



def find_1_defect(AFM,x1,x2,y1,y2,cellsize):
    #print([x1,x2,y1,y2])
    if ((abs(x1-x2)<=cellsize) or (abs(y1-y2)<=cellsize)):
        return [x1,x2,y1,y2]
    else :
        n1,n2,m1,m2=x1,x2,y1,y2
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/find-defect.txt'
        if os.path.exists(output_file):
                os.remove(output_file)
        Datasave=[['|M|mm', AFM, np.array([n1,0.5*(m1+m2),0,0.5*(n1+n2),m2,cellsize])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),0.5*(m1+m2),0,n2,m2,cellsize])*1e-9],['|M|mm', AFM, np.array([n1,m1,0,0.5*(n1+n2),0.5*(m1+m2),cellsize])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),m1,0,n2,0.5*(m1+m2),cellsize])*1e-9]]
        ns.setsavedata(output_file,*Datasave )
        ns.Relax(['time', 1e-13,'time',1e-13])
        ns.reset()
        M=[ns.Get_Data_Columns(output_file,0)[-1],ns.Get_Data_Columns(output_file,2)[-1],ns.Get_Data_Columns(output_file,4)[-1],ns.Get_Data_Columns(output_file,6)[-1]]
        #Mz=[abs(i) for i in M]
        #for m in M:
            #print(m)
        min=np.argmin(M)
        #print('argmin=',min)
        if min==0:
            return find_1_defect(AFM,n1,0.5*(n1+n2),0.5*(m1+m2),m2,cellsize)#find_1_defect(AFM,n1,0.5*(m1+m2),0.5*(n1+n2),m2,cellsize)
        if min==1:
            return find_1_defect(AFM,0.5*(n1+n2),n2,0.5*(m1+m2),m2,cellsize)#find_1_defect(AFM,0.5*(n1+n2),0.5*(m1+m2),n1,m2,cellsize)
        if min==2:
            return find_1_defect(AFM,n1,0.5*(n1+n2),m1,0.5*(m1+m2),cellsize)# find_1_defect(AFM,n1,m1,0.5*(n1+n2),0.5*(m1+m2),cellsize)
        if min==3:
            return find_1_defect(AFM,0.5*(n1+n2),n2,m1,0.5*(m1+m2),cellsize)


def number_defects(AFM,x1,x2,y1,y2,cellsize):
    defec = []
    def create_list_defect(AFM,x1,x2,y1,y2,cellsize,defec):
    #print([x1,x2,y1,y2])
        if ((abs(x1-x2)<=2*cellsize) or (abs(y1-y2)<=2*cellsize)):
            defec.append([x1,x2,y1,y2])
            return 1
        n1,n2,m1,m2=x1,x2,y1,y2
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/find-defect.txt'
        if os.path.exists(output_file):
                os.remove(output_file)
        Datasave=[['|M|mm', AFM, np.array([n1,0.5*(m1+m2),0,0.5*(n1+n2),m2,cellsize])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),0.5*(m1+m2),0,n2,m2,cellsize])*1e-9],['|M|mm', AFM, np.array([n1,m1,0,0.5*(n1+n2),0.5*(m1+m2),cellsize])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),m1,0,n2,0.5*(m1+m2),cellsize])*1e-9]]
        ns.setsavedata(output_file,*Datasave )
        ns.Relax(['time', 1e-15,'time',1e-15])
        ns.reset()
        M=[ns.Get_Data_Columns(output_file,0)[-1],ns.Get_Data_Columns(output_file,2)[-1],ns.Get_Data_Columns(output_file,4)[-1],ns.Get_Data_Columns(output_file,6)[-1]]
        arr=np.array(M)
        if np.allclose(arr,3.5e3):
            return 0
        nbr=0
        create_list_defect(AFM,n1,0.5*(n1+n2),0.5*(m1+m2),m2,cellsize,defec)#find_1_defect(AFM,n1,0.5*(m1+m2),0.5*(n1+n2),m2,cellsize)
        create_list_defect(AFM,0.5*(n1+n2),n2,0.5*(m1+m2),m2,cellsize,defec)#find_1_defect(AFM,0.5*(n1+n2),0.5*(m1+m2),n1,m2,cellsize)
        create_list_defect(AFM,n1,0.5*(n1+n2),m1,0.5*(m1+m2),cellsize,defec)# find_1_defect(AFM,n1,m1,0.5*(n1+n2),0.5*(m1+m2),cellsize)
        create_list_defect(AFM,0.5*(n1+n2),n2,m1,0.5*(m1+m2),cellsize,defec)
    def rectangles_are_close(rect1, rect2, tol):
        x_close = not (rect1[1] + tol < rect2[0] or rect2[1] + tol < rect1[0])
        y_close = not (rect1[3] + tol < rect2[2] or rect2[3] + tol < rect1[2])
        return x_close and y_close
    def count_unique_defects(defec, tol):
        n = len(defec)
        visited = [False]*n
        count = 0
        for i in range(n):
            if not visited[i]:
                stack = [i]
                while stack:
                    idx = stack.pop()
                    if not visited[idx]:
                        visited[idx] = True
                        for j in range(n):
                            if not visited[j] and rectangles_are_close(defec[idx], defec[j], tol):
                                stack.append(j)
                count += 1
        return count
    create_list_defect(AFM, x1, x2, y1, y2, cellsize, defec)
    uniq_defects = count_unique_defects(defec, tol=2*cellsize)
    return uniq_defects



def fit_sine_sum(file, column, nb_terms=2):
    """
    approximate culumn from file by sum of sin, sum of nb_terms terms
    return :time, data,function,parramters of sin
    """
    # 1. Chargement des données
    t = np.array(ns.Get_Data_Columns(file, 0))   # temps en secondes
    y = np.array(ns.Get_Data_Columns(file, column))
    t_new = t[(len(t) * 9) // 10:]/3500
    y_new = y[(len(t) * 9) // 10:]/3500
    # 2. Analyse spectrale (FFT réelle)
    N = len(t_new)
    dt = t_new[1] - t_new[0]  # supposé échantillonné uniformément
    yf = rfft(y_new)
    xf = rfftfreq(N, dt)
    # 3. Identifier les `nb_terms` plus grandes composantes fréquentielles
    magnitudes = np.abs(yf)

    ###TRIE AR FREQ
    indices = np.argsort(magnitudes)[-nb_terms:]  # top N indices
    freqs = xf[indices]
    freqs = np.sort(freqs)
    ###TRIE PAR AMP 
    # indices = np.argsort(magnitudes)[-nb_terms:][::-1]
    # freqs = xf[indices]
    # amps=magnitudes[indices]

    # 4. Fonction modèle : somme de sinus
    def sine_sum(t, *params):
        result = np.zeros_like(t)
        for i in range(nb_terms):
            A = params[3*i]
            w = params[3*i+1]
            phi = params[3*i+2]
            result += A * np.sin(2 * np.pi * w * t + phi)
        return result
    # 5. Estimation initiale des paramètres : amplitude, fréquence, phase
    p0 = []
    for f in freqs:
        p0 += [1.0, f, 0.0]  # A=1, f estimée, phi=0
    # 6. Ajustement non linéaire
    popt, _ = curve_fit(sine_sum, t_new, y_new, p0=p0,maxfev=10000)
    # 7. Fonction ajustée
    fitted = sine_sum(t_new, *popt)
    return t_new, y_new, fitted, popt

def print_interpolated_formula(popt):
    """
    Affiche la formule interpolée en tant que somme de sinusoïdes.
    popt : liste des paramètres [A0, f0, phi0, A1, f1, phi1, ...]
    """
    terms = []
    nb_terms = len(popt) // 3
    for i in range(nb_terms):
        A = popt[3*i]
        f = popt[3*i+1]
        phi = popt[3*i+2]

        # Format propre : A * sin(2π f t + φ)
        term = f"{A:.3f}·sin(2π·{f:.2e}·t + {phi:.3f})"
        terms.append(term)
    formula = " + ".join(terms)
    return formula


def plot_fit_sine_sum(file,loc, nb_terms=2):
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)  # 3 lignes, 1 colonne
    j=0
    if loc=='behind':
        j=3
    for i in range(3) :
        t, y, fitted, params=fit_sine_sum(file, i+j+1, nb_terms)
        axs[i].plot(t, y, color='r')
        axs[i].plot(t, fitted, label=print_interpolated_formula(params))
        axs[i].legend(loc='upper center', ncol=1,fontsize=8)
    axs[0].set_ylabel('m_x')
    axs[1].set_ylabel('m_y')
    axs[2].set_ylabel('m_z')
    axs[2].set_xlabel('Temps (ns)')

    plt.tight_layout()
    plt.show()


def fourier_analysis(time, signal, nb_peaks=5, plot=True):
    time = np.array(time)
    signal = np.array(signal) - np.mean(signal)  # enlever la composante DC
    N = len(signal)
    dt = time[1] - time[0]  # intervalle de temps

    # 2. FFT réelle
    yf = rfft(signal)
    xf = rfftfreq(N, dt)

    # 3. Magnitude normalisée
    magnitudes = np.abs(yf) / N

    # 4. Trouver les pics dominants
    peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes)*0.01)  # seuil = 1% du max
    sorted_peaks = sorted(peaks, key=lambda x: magnitudes[x], reverse=True)
    dominant_peaks = sorted_peaks[:nb_peaks]

    peaks_info = [(xf[i], magnitudes[i]) for i in dominant_peaks]

    # 5. Plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(xf, magnitudes, label="FFT Spectrum")
        plt.scatter([xf[i] for i in dominant_peaks],
                    [magnitudes[i] for i in dominant_peaks],
                    color='red', label="Dominant Peaks")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Fourier Analysis")
        plt.legend()
        plt.xscale('log')  # utile si spectre large
        plt.grid()
        plt.show()

    return xf, magnitudes, peaks_info


def plot_fourier_xyz(output_file, place='above', nb_peaks=5):
    
    j = 0
    if place == 'behind':
        j = 3

    # Lecture et normalisation
    X = ns.Get_Data_Columns(output_file, 0)
    Y1 = ns.Get_Data_Columns(output_file, 1 + j)
    Y2 = ns.Get_Data_Columns(output_file, 2 + j)
    Y3 = ns.Get_Data_Columns(output_file, 3 + j)

    Y1p = [i / (3500 * 1.01) for i in Y1]
    Y2p = [i / (3500 * 1.01) for i in Y2]
    Y3p = [i / (3500 * 1.01) for i in Y3]


    ######If NEEL
    Y1=ns.Get_Data_Columns(output_file,1+3)
    Y2=ns.Get_Data_Columns(output_file,2+3)
    Y3=ns.Get_Data_Columns(output_file,3+3)
    Y1p=[Y1p[i]-Y1[i]/(3500*1.01) for i in range(len(Y1))]#normalisation
    Y2p=[Y2p[i]-Y2[i]/(3500*1.01) for i in range(len(Y2))]
    Y3p=[Y3p[i]-Y3[i]/(3500*1.01) for i in range(len(Y3))]
    ###################


    # Dernier tiers
    start = (len(X) * 2) // 3
    time = np.array(X[start:])
    components = {
        'X': np.array(Y1p[start:]),
        'Y': np.array(Y2p[start:]),
        'Z': np.array(Y3p[start:])
    }

    color_map = {'X': 'r', 'Y': 'g', 'Z': 'b'}
    all_peaks = {}

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Fourier', fontsize=14)

    for i, (label, signal) in enumerate(components.items()):
        signal -= np.mean(signal)
        N = len(signal)
        dt = time[1] - time[0]

        yf = rfft(signal)
        xf = rfftfreq(N, dt)
        magnitudes = np.abs(yf) / N

        peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes) * 0.01)
        sorted_peaks = sorted(peaks, key=lambda i: magnitudes[i], reverse=True)[:nb_peaks]
        peaks_info = [(xf[i], magnitudes[i]) for i in sorted_peaks]
        all_peaks[label] = peaks_info

        axs[i].plot(xf, magnitudes, color=color_map[label], label=f"{label}-axis")#2*np.pi*xf
      #  axs[i].scatter([xf[p] for p in sorted_peaks],
                     #  [magnitudes[p] for p in sorted_peaks],
                     #  color=color_map[label], marker='x')
        axs[i].set_ylabel("Amplitude")
        axs[i].legend()
        axs[i].set_xscale("log")
        axs[i].grid(True)

    axs[-1].set_xlabel("Fréquencies (Hz)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Affichage console
    # print(f"\nTop {nb_peaks} fréquences dominantes pour '{place}' :")
    # for label, peaks in all_peaks.items():
    #     print(f"\n{label}-axis:")
    #     for f, amp in peaks:
    #         print(f"f = {f:.2e} Hz, amplitude = {amp:.3e}")


def plot_compute_field(ovf_filename, z0,defect_pos=None):
    # Read OVF2 file
    vec, nodes, rect = ns.Read_OVF2(ovf_filename)
    Nx, Ny, Nz = nodes  # number of cells in x, y, z

        # vec is (360000, 3) → reshape to (Nz, Ny, Nx, 3), then transpose to (Nx, Ny, Nz, 3)
    vec = vec.reshape((Nz, Ny, Nx, 3)).transpose(2, 1, 0, 3)

    Hx = vec[:, :, :, 0]  # shape (Nx, Ny, Nz)
    Hy = vec[:, :, :, 1]
    Hz = vec[:, :, :, 2]

    # Index of layer closest to z0
    z_min, z_max = rect[2], rect[5]
    dz = (z_max - z_min) / (Nz - 1)
    iz = int(round((z0 - z_min) / dz))
    iz = max(0, min(Nz-1, iz))

    # Extract 2D slices at z = z0, with step
    Hx_2D = Hx[:, :, iz][::1, ::1]
    Hy_2D = Hy[:, :, iz][::1, ::1]
    Hz_2D = Hz[:, :, iz][::1, ::1]

    # Define spatial bounds for display (meters)
    x_min, x_max = rect[0], rect[3]
    y_min, y_max = rect[1], rect[4]

    # Adjust extents accordingly
    x_vals = np.linspace(x_min, x_max, Nx)[::3]
    y_vals = np.linspace(y_min, y_max, Ny)[::3]
    extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]

    if defect_pos:
        x_frac, y_frac = defect_pos
        defect_size = 0#0.06 # taille du défaut (en fraction de la taille totale, ex: 5%)

        # Taille des tableaux downsamplés
        nx, ny = Hx_2D.shape

        # Position centrale du défaut
        i_center = int(x_frac * nx)
        j_center = int(y_frac * ny)

        # Taille du défaut en nombre de pixels
        half_width = max(1, int(0.5 * defect_size * nx))  # au moins 1 pixel

        # Bornes à découper (avec clamp dans les limites)
        i_start = max(0, i_center - half_width)
        i_end = min(nx, i_center + half_width + 1)
        j_start = max(0, j_center - half_width)
        j_end = min(ny, j_center + half_width + 1)

        # Mise à zéro dans les 3 composantes
        # Hx_2D[i_start:i_end, j_start:j_end] = np.nan
        # Hy_2D[i_start:i_end, j_start:j_end] = np.nan
        # Hz_2D[i_start:i_end, j_start:j_end] = np.nan




    # Plot all three components side by side in columns
    plt.figure(figsize=(15, 5))  # wider figure to fit 3 plots

    plt.subplot(1, 3, 1)
    plt.imshow(Hx_2D.T, origin='lower', extent=extent)
    plt.title(f'Hx at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (T)')

    plt.subplot(1, 3, 2)
    plt.imshow(Hy_2D.T, origin='lower', extent=extent)
    plt.title(f'Hy at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (T)')

    plt.subplot(1, 3, 3)
    plt.imshow(Hz_2D.T, origin='lower', extent=extent)
    plt.title(f'Hz at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (T)')
    plt.tight_layout()
    plt.show()
    #plt.pause(1)
    #plt.close()


def main(simu,material,DMI=False):
    defects=False
    #print(2)
    Lx = 2500
    Ly = 1000
    Lz = 5
    cellsize = 5
    damping_x=600
    modules=['exchange', 'anitens','Zeeman']
    T=0
    if DMI :
        modules.append('DMexchange')
    Mu0=(4*3.141*1e-7) #to convert external field in Tesla to A/M wich is used in boris
    if material=='hematite':
        Hfield=5.7/Mu0
    else :
        Ms = 3.5e3
        Ah = -228e3
        Khard=297.5
        Hfield=(1/(Mu0*Ms))*np.sqrt(16*abs(Ah)*Khard)
        #print('HcFeBO3=',Mu0*Hfield,'T')
    if (simu=='transport_SOT'and material=='hematite'):
        t=1000
        T = 0
        modules=['exchange', 'anitens', 'SOTfield', 'transport','Zeeman']
        V=-4e-5        
        Hap=1*Hfield
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV1/magnontransport.txt'
        magnon_transport_SOT (material,Lx,Ly,Lz,cellsize,T,modules,damping_x,V,Hap,t,output_file)
        plot_SA(output_file)
    elif (simu=='sinc_dispersion'and material=='hematite') :
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV1/sinc_disp_H0.txt'
        if os.path.exists(output_file):
            os.remove(output_file)
        t=50
        T = 0
        Hap=0*Hfield
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file,2)
        output_file_2='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV1/sinc_disp_H03.txt'
        if os.path.exists(output_file_2):
            os.remove(output_file_2)
        Hap=0.3*Hfield
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file_2,2)
        output_file_3='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV1/sinc_disp_H06.txt'
        if os.path.exists(output_file_3):
            os.remove(output_file_3)
        Hap=0.6*Hfield
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file_3,2)
        output_file_4='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV1/sinc_disp_Hc.txt'
        if os.path.exists(output_file_4):
            os.remove(output_file_4)
        Hap=1*Hfield
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file_4,2)
        output_files=[output_file,output_file_2]
        plot_magnon_dispersion_with_zoom(output_files,True)
        output_files=[output_file_3,output_file_4]
        plot_magnon_dispersion_with_zoom(output_files,False)
    elif (simu=='sinc_dispersion'and material=='FeBO3') :
        print(3)
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/sinc_disp_FeBO3_y.txt'
        if os.path.exists(output_file):
            os.remove(output_file)
        t=50
        T = 0
        Hap1=0*Hfield
        for i in range(1):
            disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap1,t,output_file,2,DMI)
        print(4)
        Hap2=0*Hfield
        output_file_2='C:/Users/thdelvau/OneDrive - NTNU/Documents/sinc_disp_FeBO3_H1.txt'
        if os.path.exists(output_file_2):
            os.remove(output_file_2)
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap2,t,output_file_2,3,DMI)#should be 3 for z
        output_files=[output_file,output_file_2]
        plot_magnon_dispersion_with_zoom(output_files,Hap1*Mu0,Hap2*Mu0)
    elif (simu=='disp_sot'):
        output_file1='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV1/sinc_disp_FeBO3_H0.txt'
        if os.path.exists(output_file1):
            os.remove(output_file1)
        output_file2='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV1/SOT_disp_FeBO3_H1.txt'
        if os.path.exists(output_file2):
            os.remove(output_file2)
        output_files=[output_file1,output_file2]
        Hap1=0.3*Hfield
        Hap2=0.5*Hfield
        t=1000
        dir=2
        T=0.3
        V=-1e-4
        disp_SOT(material,Lx,Ly,Lz,cellsize,damping_x,Hap1,Hap2,t,output_files,dir,T,V)
    elif (simu=='transport_SOT'and material=='FeBO3'):
        Lx=2000
        print(6)
        t=1000
        T = 0.3
        modules=['exchange', 'anitens', 'SOTfield', 'transport','Zeeman']
        V=-0.8e-4#crit :-1e-4#-2e-5       
        Hap=0.5*Hfield
        ##tests
        Hap=1*Hfield#0.01*5.7/Mu0
        V=-4.5e-4#-4e-4
        ##
        print('Hematite field =',5.7/Mu0)
        print('FeBO3 field =',Hfield)
        print('Applied field =',Hap)
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/magnontransport_FeBO3.txt'
        #plot_SA(output_file)#test
        magnon_transport_SOT (material,Lx,Ly,Lz,cellsize,T,modules,damping_x,V,Hap,t,output_file)
        plot_SA(output_file)
        print(7)
    elif (simu=='interf'):
        Lx=3000
        Ly=2000
        Lz=5
        cellsize=5
        T=0
        t=150
        modules =['exchange', 'anitens','Zeeman']
        stat_interf (Lx,Ly,Lz,cellsize,T,modules,damping_x,t)
    #magnon_transport_SOT (material,Lx,Ly,Lz,cellsize,0,['exchange', 'anitens'],damping_x,0,0,1000,' ')
    elif(simu=='test defects'):
        t=1000
        T = 0.3
        modules=['exchange', 'anitens', 'SOTfield', 'transport','Zeeman']
        V=0       
        Hap=0
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/defects.txt'
        AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap,defect=True)
        ns.Relax(['time', 500*1e-12])
    elif (simu=='Ground_state') :
        Angle=[]
        AFM=init(material,Lx,Ly,Lz,cellsize,0,modules,damping_x,0)
        for i in range (1,2) :
            cube=(i**8)*cellsize
            Datasave=[['time'],['<M>', AFM, [0.5*Lx*1e-9,0.5*Ly*1e-9,0,(0.5*Lx+cube)*1e-9,(0.5*Ly+cube)*1e-9,cellsize*1e-9]],['<M2>', AFM, [0.5*Lx*1e-9,0.5*Ly*1e-9,0,(0.5*Lx+cube)*1e-9,(0.5*Ly+cube)*1e-9,cellsize*1e-9]]]
            output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/test.txt'
            ns.setsavedata(output_file,*Datasave )
            ns.reset()
            ns.Relax(['time', 50*1e-12,'time', 25*1e-12])
            Angle.append(angle_DMI(output_file))
        print('average angle_DMI = ',np.mean(Angle))
    elif (simu=='disp_sameplot') :
        output_file1='C:/Users/thdelvau/OneDrive - NTNU/Documents/sinc_disp_FeBO3_sameplot1.txt'
        output_file2='C:/Users/thdelvau/OneDrive - NTNU/Documents/sinc_disp_FeBO3_sameplot2.txt'
        output_files=[output_file1,output_file2]
        for output_file in output_files :
            if os.path.exists(output_file):
                os.remove(output_file)
        t=50
        T = 0
        Hap1=0*Hfield
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap1,t,output_file1,2,DMI,defects)
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap1,t,output_file2,3,DMI,defects)
        plot_magnon_dispersion_with_zoom(output_files,0,0, theorie=True)
        #plot_disp_sinc(output_files,Hap1)
    elif (simu=='thermal_disp') :
        output_file1='C:/Users/thdelvau/OneDrive - NTNU/Documents/thermal_disp_FeBO3_y.txt'
        output_file2='C:/Users/thdelvau/OneDrive - NTNU/Documents/thermal_disp_FeBO3_z.txt'
        output_files=[output_file1,output_file2]
        for output_file in output_files :
            if os.path.exists(output_file):
                os.remove(output_file)
        t=100
        T = 5
        Hap1=0*Hfield
        thermal_disp(material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file1,2,DMI,T)
        thermal_disp(material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file2,3,DMI,T)
        output_files=[output_file1,output_file2]
        #plot_magnon_dispersion_with_zoom(output_files,0,0, theorie=True)
        plot_disp_sinc(output_files,Hap1)
    elif (simu=='antenna'):
        # DM=False
        t=1000
        #output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/3/Smaller_defect/above/def.txt'
        output_file="C:/Users/thdelvau/OneDrive - NTNU/Documents/results/5/10def/data.txt"
        # if os.path.exists(output_file):
        #     os.remove(output_file)
        # antenna(t,output_file,True,True)
        plot_SA(output_file,'above')
        plot_fourier_xyz(output_file,'above')


        #antenna_plot(output_file,Lx,Ly,Lz,cellsize)
        #antenna_animation_with_defect(output_file, Lx, Ly, Lz, cellsize)
        #test_plot(output_file)

        #gif
        # path='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/3/Smaller_defect/Behind/'
        # file1,file2, file3, file4=path+'nothing.txt',path+'nothing+DMI.txt',path+'def.txt',path+'def+DMI.txt'
        # multi_plot(file1, file2, file3, file4)

        ##interpol
        # print(print_interpolated_formula(params))
        # t, y, fitted, params = fit_sine_sum(output_file,1)
        # plt.plot(t, y, label="mx")
        # plt.plot(t, fitted, label='-0.89+0.1*sin(2*PI*3,34e14*t+23.8)')
        # plt.legend(loc='upper center', ncol=1,fontsize=8)
        # plt.show()
        #plot_fit_sine_sum(output_file,'above',2)



        # ###disp
        # t=100
        # output_file1='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/4/dispy.txt'
        # output_file2='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/4/dispz.txt'
        # output_files=[output_file1,output_file1]
        # for i,output_file in enumerate(output_files) :
        #     if os.path.exists(output_file):
        #         os.remove(output_file)
        #         antenna(t,output_file1,DMI,False,i)
        # plot_magnon_dispersion_with_zoom(output_files,0,0, False)

        # basepath='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/4/test_around'
        # ovf_filename = os.path.join(basepath, "test.ovf")
        # plot_compute_field(ovf_filename, 50e-9,defect_pos=None)


    elif (simu=='find_defect'):
        AFM=init(material,Lx,Ly,Lz,cellsize,0,modules,damping_x,0,defect=False)
        AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,1500*1e-9,110])
        print('defects :',number_defects(AFM,cellsize,Lx-cellsize,cellsize,Ly-cellsize,cellsize))
        D=find_1_defect(AFM,cellsize,Lx-cellsize,cellsize,Ly-cellsize,cellsize)
        print('defects is in the rectangle :(',D[0],',',D[2],'),(',D[1],',',D[3],')')
    elif (simu=='nbr_defects'):
        AFM=init(material,Lx,Ly,Lz,cellsize,0,modules,damping_x,0,defect=False)
        AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,440*1e-9,110])
        print('defects :',number_defects(AFM,cellsize,Lx-cellsize,cellsize,Ly-cellsize,cellsize))



main('antenna','FeBO3',False)

