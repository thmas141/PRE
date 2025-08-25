import sys
import os
sys.path.insert(0, 'C:/Users/thdelvau/OneDrive - NTNU/Documents/Release')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from NetSocks import NSClient ,customize_plots,Shape  # type: ignore
import numpy as np
from pathlib import Path
import scipy as sp
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import copy
print(0.1)
ns = NSClient(); 
print(0)
ns.configure(True, False)
ns.cuda(2)
ns.reset(); ns.clearelectrodes()
print(1)
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
    ns.setdt(10e-15)
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
        AFM.param.damping_AFM =1e-8#1e-8 for disp#5e-4for random simu  #5.5e-3 max
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
        # disk=Shape.disk([10*cellsize*1e-9,10*cellsize*1e-9,cellsize*1e-9])
        # AFM.shape_set(disk.move([0.5*Lx*1e-9,0.5*Ly*1e-9,0.5*Lz*1e-9]))
        # #AFM.shape_setangle(disk,[180,0])
        # Ms= AFM.param.Ms_AFM.setparam()
        # AFM.param.Ms_AFM.shape_setparam(disk,1)
    #AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,1500*1e-9,110])
    # Add increased damping at edges along x-axis to prevent reflections. For smaller meshes this needs to be shortened
    AFM.param.damping_AFM.setparamvar('abl_tanh', [damping_x/meshdims[0], damping_x/meshdims[0], 0, 0, 0, 0, 1, 10, damping_x])
    
    
    if (Hap!=0) :
        AFM.setfield(Hap,90,90)
        ns.Relax(['time', 5*1e-12])
        ns.reset()
    else :
        ns.Relax(['time', 50*1e-12])#50
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

def disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file,dir,DMI,same_plot=False):
    file=Path(output_file)
    modules=['exchange', 'anitens', 'Zeeman']
    if DMI :
        modules.append('DMexchange')
    AFM=init(material,Lx,Ly,Lz,cellsize,0,modules,damping_x,Hap)
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
        He=3e8#1e10 #1e11#4e1 
        N = 300
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
        if same_plot :
            ns.editstagevalue(0, '0,Hap+He * sinc(k*(x-L/2))*sinc(2*PI*f*(t-u)), H0 * sinc(k*(x-L/2))*sinc(2*PI*f*(t-u))')
        else :
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

def disp_sin_sans_init(AFM,material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file,dir,DMI,same_plot=False,He=3e8):
    file=Path(output_file)
    modules=['exchange', 'anitens', 'Zeeman']
    if DMI :
        modules.append('DMexchange')
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
        He=3e8#1e10 #1e11#4e1 
        N = 300
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
        if same_plot :
            ns.editstagevalue(0, '0,He * sinc(k*(x-L/2))*sinc(2*PI*f*(t-u)), H0 * sinc(k*(x-L/2))*sinc(2*PI*f*(t-u))')
        else :
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

def plot_SA(output_file) :
    X=ns.Get_Data_Columns(output_file,0)
    Y1=ns.Get_Data_Columns(output_file,4)
    Y2=ns.Get_Data_Columns(output_file,5)
    Y3=ns.Get_Data_Columns(output_file,6)
    plt.figure()
    plt.plot(X,Y1,color='blue',label='<Mx>')
    plt.plot(X,Y2,color='green',label='<My>')
    plt.plot(X,Y3,color='red',label='<Mz>')
    plt.legend()
    plt.show()
    #plotting qccumulation
    Mux=[]
    Muy=[]
    Muz=[]
    Jcy=[]
    for i in range(200) :
        mux=ns.Get_Data_Columns(output_file,7+3*i)
        muy=ns.Get_Data_Columns(output_file,8+3*i)
        muz=ns.Get_Data_Columns(output_file,9+3*i)
        #print('plot',i)
        Mux.append(np.mean(mux[len(mux)//2:]))
        Muy.append(np.mean(muy[len(muy)//2:]))
        Muz.append(np.mean(muz[len(muz)//2:]))
    for i in range(100) :    
        jcy=ns.Get_Data_Columns(output_file,608+3*i)
        Jcy.append(-50*np.mean(jcy[len(jcy)//2:]))
        Jcy.append(-50*np.mean(jcy[len(jcy)//2:]))
    d=[10*i for i in range(-100,100)]      
    plt.figure()
    plt.plot(d,Mux,label='<mux>')
    plt.plot(d,Muy,label='<muy>')
    plt.plot(d,Muz,label='<muz>')
    plt.plot(d,Jcy,label='<Jcy>')
    plt.legend()
    plt.show()

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
    plot_analytical(ax1,Hap,'z')
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
        theorie=True
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
        for n in range(1):##10
            for m in range(1):
                axe.plot(x, low(x,n,m), color='red', linestyle = 'dashed', linewidth=1/((n+1)*(m+1)))
                #axe.plot(x, high(x,n,m), color='green', linestyle = 'dashed', linewidth=1/((n+1)*(m+1)))##si plot sur le meme, a changer manuellement c est un peu relou, faut l enlever si x,y de chaque cote
    else :
        for n in range(1):
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

def thermal_disp(material,Lx,Ly,Lz,cellsize,damping_x,Hap,t,output_file,dir,DMI,T,defect=False) :
    time_step=1e-13
    file=Path(output_file)
    modules=['exchange', 'anitens', 'Zeeman']
    if DMI :
        modules.append('DMexchange')
    AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap)
    if defect :
        AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,5*cellsize*1e-9,5*cellsize*1e-9,1500*1e-9,100])
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

def setup_defect(AFM,Lx,Ly,Lz,a_x,a_y,a_z):
    ns.setdt(1e-15)
    x0=Lx*0.3
    y0=Ly*0.5
    z0=Lz/2
    function='1.01'
    def one_def(x0,y0,z0):
        func = ' - ((step(x - ' + str((x0 - a_x/2)*1e-9) + ') - step(x - ' + str((x0 + a_x/2)*1e-9) + ')) * ' + \
           '(step(y - ' + str((y0 - a_y/2)*1e-9) + ') - step(y - ' + str((y0 + a_y/2)*1e-9) + ')) * ' + \
           '(step(z - ' + str((z0 - a_z/2)*1e-9) + ') - step(z - ' + str((z0 + a_z/2)*1e-9) + ')))'
        return func 
    function+=one_def(x0,y0,z0)
    # function+=one_def(Lx*0.4,Ly*0.8,Lz/2)
    # function+=one_def(Lx*0.75,Ly*0.6,Lz/2)
    # function+=one_def(Lx*0.9,Ly*0.32,Lz/2)
    # function+=one_def(Lx*0.8,Ly*0.2,Lz/2)
    AFM.param.Ms_AFM.setparamvar('equation', function)


def disp_antenna(t,output_file,dir,DM,defects,sameplot=False):
    time_step = 1e-14
    time_simu=t*1e-12
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
    AFM=init('FeBO3',Lx,Ly,Lz,cellsize,T,modules,0,0,defects)#remember to add bool defect
    if defects : 
        #AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,8*cellsize*1e-9,8*cellsize*1e-9,1500*1e-9,100])#AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,8*cellsize*1e-9,8*cellsize*1e-9,1500*1e-9,100]) max
        setup_defect(AFM,Lx,Ly,Lz,2*cellsize,2*cellsize,cellsize)        
    H0=0
    He=1e3#4e5
    ns.setstage('Hequation')
    #ns.equationconstants('u', t)
    ns.equationconstants('H0', H0)
    ns.equationconstants('He', He)
    ns.equationconstants('L', 0.5*Lx*1e-9)
    ns.equationconstants('S', taille**2)
    ns.equationconstants('f',0.3e12)#ns.equationconstants('f',0.3*1e12)
    ns.editstagevalue(0, 'H0,H0,  He * exp(-(x-L)*(x-L)/(2*S))*sin(f*(t))')

    ns.setdata('commbuf')
    ns.editstagestop(0, 'time', time_simu)
    ns.editdatasave(0, 'time', time_step)
    ns.clearcommbuffer()
    #get magnetisation profile along length through center
    ns.dp_getexactprofile((np.array([cellsize/2, Ly/2, Lz-cellsize])*1e-9), 
                        (np.array([Lx - cellsize/2, Ly/2, Lz])*1e-9), 
                            cellsize*1e-9, 0,bufferCommand = True)
    if sameplot:
        ns.dp_div(2, 3500, bufferCommand=True)
        ns.dp_saveappendasrow(file, 2, bufferCommand = True)#should be true
        ns.dp_div(3, 3500, bufferCommand=True)
        ns.dp_saveappendasrow(file, 3, bufferCommand = True)#should be true
    else :
        ns.dp_div(dir, 3500, bufferCommand=True)
        ns.dp_saveappendasrow(file, dir, bufferCommand = True)#should be true
    ns.Run()


def disp_bulk (t,output_file,dir,DM,defects,sameplot=False):
    time_simu=t*1e-12
    time_step=time_simu*1e-2# for gif
    Lx=1500
    Ly=600
    Lz=100
    T=0#.3
    cellsize=5
    taille=10*(cellsize*1e-9)
    modules=['exchange', 'anitens','Zeeman']
    if DM :
        modules.append('DMexchange')
    file=Path(output_file)
    AFM=init('FeBO3',Lx,Ly,Lz,cellsize,T,modules,0,0,defects)#remember to add bool defect
    ns.setdt(50e-15)
    if defects : 
        #AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,8*cellsize*1e-9,8*cellsize*1e-9,1500*1e-9,100])#AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,8*cellsize*1e-9,8*cellsize*1e-9,1500*1e-9,100]) max
        setup_defect(AFM,Lx,Ly,Lz,5*cellsize,5*cellsize,0.5*Lz)        
    H0=0
    He=8e5 
    ns.setstage('Hequation')
    #ns.equationconstants('u', t)
    ns.equationconstants('H0', H0)
    ns.equationconstants('He', He)
    ns.equationconstants('L', 0.5*Lx*1e-9)
    ns.equationconstants('S', taille**2)
    ns.equationconstants('f',0.3*1e12)
    ns.equationconstants('H',Lz*1e-9) 
    ns.equationconstants('P',0.2*Lz*1e-9)#distance of the antenna to the sample (variable)
    #ns.equationconstants('s',10*cellsize)
    ns.editstagevalue(0, 'H0, H0, He * (P/(H+P-z))*exp(-(x-L)*(x-L)/(2*S))*sin(f*(t))')

    ns.setdata('commbuf')
    ns.editstagestop(0, 'time', time_simu)
    ns.editdatasave(0, 'time', time_step)
    ns.clearcommbuffer()
    #get magnetisation profile along length through center
    ns.dp_getexactprofile((np.array([cellsize/2, Ly/2, Lz-cellsize])*1e-9), 
                        (np.array([Lx - cellsize/2, Ly/2, Lz])*1e-9), 
                            cellsize*1e-9, 0,bufferCommand = True)
    if sameplot:
        ns.dp_div(2, 3500, bufferCommand=True)
        ns.dp_saveappendasrow(file, 2, bufferCommand = True)#should be true
        ns.dp_div(3, 3500, bufferCommand=True)
        ns.dp_saveappendasrow(file, 3, bufferCommand = True)#should be true
    else :
        ns.dp_div(dir, 3500, bufferCommand=True)
        ns.dp_saveappendasrow(file, dir, bufferCommand = True)#should be true
    ns.Run()


def antenna_plot(output_file,Lx,Ly,Lz,cellsize):
    X=ns.Get_Data_Columns(output_file,0)
    def f(m,n):
        mz=ns.Get_Data_Columns(output_file,10*m+n)
        return np.mean(mz[len(mz)//2:])
    
    Z = np.zeros((39, 39))
    for i in range(39):
        print(i)
        for j in range(39):
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


def disp_defects (material,Lx,Ly,Lz,cellsize,T,modules,damping_x,t,output_files,DMI,defe):
    nb_def=[]
    for i,output_file in enumerate(output_files) :
        if os.path.exists(output_file):
            os.remove(output_file)
        AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,0)
        d=defe[i]
        AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,d*1e-9,100])
        nb_def.append(number_defects(AFM,cellsize,Lx-cellsize,cellsize,Ly-cellsize,cellsize))
        disp_sin_sans_init(AFM,material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file,2,DMI,True,9e9)
    def plot(output_files):
        clim_max   = 1000
        time_step  = 1e-13
        ylabel     = 'f (THz)'
        divisor    = 1e12

        # Création des 5×2 subplots
        fig, axes_grid = plt.subplots(nrows=5, ncols=2, figsize=(20, 20))
        axes = axes_grid.flatten()   # tableau 1D de 10 AxesSubplot

        titles = [f'number of defects : {i}' for i in nb_def]

        for i, output_file in enumerate(output_files):
            pos_time     = np.loadtxt(output_file)
            fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

            freq_len = fourier_data.shape[0]
            k_len    = fourier_data.shape[1]
            freq     = np.fft.fftfreq(freq_len, time_step)
            kvector  = np.fft.fftfreq(k_len, 5e-9)
            k_max    = 2 * np.pi * kvector[k_len//2] * 5e-9

            f_min = abs(freq[0])
            f_max = abs(freq[freq_len//2]) / divisor
            final_result = fourier_data[freq_len//2:]

            ax_i = axes[i]

            ax_i.imshow(final_result, origin='lower', interpolation='bilinear',
                extent=[-k_max, k_max, 2*f_min, 2*f_max],
                aspect="auto", clim=(0, clim_max))
            ax_i.set_ylim(f_min, 1.0) 
            ax_i.set_xlabel('q')
            ax_i.set_ylabel(ylabel)

            # inset zoom  
            # x1, x2, y1, y2 = -0.15, 0.15, 0, 0.3
            # axins = ax_i.inset_axes([0.5, 0.5, 0.47, 0.47],
            #                 xlim=(x1, x2), ylim=(y1, y2), xticklabels=[])
            # axins.imshow(final_result, extent=[-k_max, k_max, f_min, f_max],
            #      origin='lower', clim=(0, clim_max))
            # ax_i.indicate_inset_zoom(axins, edgecolor='black')

            ax_i.set_title(titles[i])

        plt.tight_layout()
        plt.savefig('C:/Users/thdelvau/OneDrive - NTNU/Documents/default/dispersion_plots2.png', dpi=300, bbox_inches='tight')
        plt.show()

    plot(output_files)



def plot_both_test(output_file,Hfield=0):
    # Paramètres
    clim_max   = 1000
    time_step  = 1e-14
    ylabel     = 'f (THz)'
    divisor    = 1e12

    
    pos_time   = np.loadtxt(output_file)            # (Nt, )
    fourier2d  = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))
    Nt, Nk     = fourier2d.shape
    freq       =np.fft.fftfreq(Nt, time_step)
    kvector    = np.fft.fftfreq(Nk, 5e-9)
    k_max      = 2*np.pi * kvector[Nk//2] * 5e-9
    f_min      = np.abs(freq[0])
    f_max      = np.abs(freq[Nt//2]) / divisor
    disp_map   = fourier2d[Nt//2:, :]               # fréquence positive

    
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(
        disp_map,
        origin='lower',
        interpolation='bilinear',
        extent=[-k_max, k_max, 2*f_min, 2*f_max],
        aspect='auto',
        clim=(0, clim_max)
    )
    ax.set_xlabel('q ')
    ax.set_ylabel(ylabel)

    
    # Génère un vecteur q adapté au même domaine
    q_nm = np.linspace(-0.75, 0.75, 1000)  # en nm⁻¹
    # fonctions dispersion analytique 
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
    def low(q_a, n=0, m=0):
        q = q_a/5e-9
        return 4*C*np.sqrt(A*Ah*(q**2+(2*n*np.pi/Ly)**2+(2*m*np.pi/Lz)**2) + Ah*Keasy+(0.25*Ms*Hfield)**2)/1e12#

    def high(q_a, n=0, m=0):
        q = q_a / 5e-9
        return 4*C*np.sqrt(A*Ah*(q**2+(2*m*np.pi/Lz)**2+(2*n*np.pi/Ly)**2) + Ah*(Keasy+Khard))/1e12#

    # Trace la branche "low" (composante y)…
    #ax.plot(q_nm, low(q_nm),  color='red',   linestyle='--', label='y')
    # …et la branche "high" (composante z)
    #ax.plot(q_nm, high(q_nm), color='green', linestyle='-.', label='z')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()




def main(simu,material,DMI=False):
    print(2)
    Lx = 2500
    Ly = 1000
    Lz = 5
    cellsize = 5
    damping_x=100
    modules=['exchange', 'anitens','Zeeman']
    T=0
    N_defaults=[5000,1500,1000,950,800,700,650,600,500,440]
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
        print('HcFeBO3=',Mu0*Hfield,'T')
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
        output_file1='C:/Users/thdelvau/OneDrive - NTNU/Documents/sinc_disp_FeBO3_sameplot_Hc.txt'
        output_file2='C:/Users/thdelvau/OneDrive - NTNU/Documents/sinc_disp_FeBO3_sameplot2.txt'
        output_files=[output_file1,output_file2]
        for output_file in output_files :
            if os.path.exists(output_file):
                os.remove(output_file)
        t=50
        T = 0
        Hap1=0*Hfield
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap1,t,output_file1,2,DMI)#,True)
        disp_sin(material,Lx,Ly,Lz,cellsize,damping_x,Hap1,t,output_file2,3,DMI)
        #output_files=[output_file1]
        plot_magnon_dispersion_with_zoom(output_files,0,0, theorie=True)
        #plot_disp_sinc(output_files,Hap1)
        #plot_both_test(output_file1,Hap1)
    elif (simu=='thermal_disp') :
        output_file1='C:/Users/thdelvau/OneDrive - NTNU/Documents/thermal_disp_FeBO3_y.txt'
        output_file2='C:/Users/thdelvau/OneDrive - NTNU/Documents/thermal_disp_FeBO3_z.txt'
        output_files=[output_file1,output_file2]
        for output_file in output_files :
            if os.path.exists(output_file):
                os.remove(output_file)
        t=100
        T = 50
        Hap1=0*Hfield
        thermal_disp(material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file1,2,DMI,T)
        thermal_disp(material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file2,3,DMI,T)
        output_files=[output_file1,output_file2]
        #plot_magnon_dispersion_with_zoom(output_files,0,0, theorie=True)
        plot_disp_sinc(output_files,Hap1)
    elif (simu=='antenna'):
        # DM=False
        # defects=False
        # t=200
        # output_file1='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV2/antenna_disp_y.txt'
        # output_file2='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV2/antenna_disp_z.txt'
        # output_files=[output_file1,output_file2]
        # for i, output_file in enumerate(output_files):
        #     if os.path.exists(output_file):
        #         os.remove(output_file)
        #     disp_antenna(t,output_file,i+2,DM,defects)
        # plot_magnon_dispersion_with_zoom(output_files,0,0, theorie=True)
        # #plot_both_test(output_file,Hfield=0)

        DM=False
        defects=True
        t=200
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/main/CleanV2/test2.txt'
        if os.path.exists(output_file):
                os.remove(output_file)
        #disp_bulk (t,output_file,2,DM,defects,True)
        disp_antenna(t,output_file,2,DM,defects,True)
        plot_both_test(output_file,Hfield=0)

    elif (simu=='disp_defects'):
        T=0
        t=100
        ###multiple defects
        # output_files=['C:/Users/thdelvau/OneDrive - NTNU/Documents/default/'+str(i)+'.txt' for i in range(10)]
        # disp_defects (material,Lx,Ly,Lz,cellsize,T,modules,damping_x,t,output_files,DMI,N_defaults)
        # output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/default/random/6_def.txt'
        # if os.path.exists(output_file):
        #         os.remove(output_file)
        # Hap1=0*Hfield
        # AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,0)
        # d=600
        # AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,d*1e-9,100])
        # disp_sin_sans_init(AFM,material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file,2,DMI,True,8e8)
        # plot_both_test(output_file,0)



        ##lenght of defects
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/default/lenght_of_defect/10cellsize.txt'
        if os.path.exists(output_file):
            os.remove(output_file)
        AFM=init(material,Lx,Ly,Lz,cellsize,T,modules,damping_x,0)
        AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,10*cellsize*1e-9,10*cellsize*1e-9,1500*1e-9,100])
        disp_sin_sans_init(AFM,material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file,2,DMI,True)
        plot_both_test(output_file,0)

        # #thermal_disp
        # output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/default/random/thermal_disp.txt'
        # if os.path.exists(output_file):
        #         os.remove(output_file)
        # T = 0.5
        # Hap1=0*Hfield
        # thermal_disp(material,Lx,Ly,Lz,cellsize,damping_x,0,t,output_file,2,DMI,T,True)
        # plot_both_test(output_file,0)


# Ah = 228e3
# A = 24.3e-15
# Keasy = 0.0455
# Khard = 297.5
# Ms = 3.5e3
# mu0 = 4 * np.pi * 1e-7
# gamma_e_over_2pi = 2.802e10
# C = gamma_e_over_2pi/(Ms)
# Ly=1000*1e-9
# Lz=5*1e-9
# Lx=2500*1e-9
# q,m,n=0,0,0
# print(4*C*np.sqrt(A*Ah*(q**2+(2*m*np.pi/Lz)**2+(2*n*np.pi/Ly)**2) + Ah*(Keasy+Khard))/1e12)#



main('antenna','FeBO3',False)

