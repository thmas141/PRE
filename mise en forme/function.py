import sys
import os


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy as sp # type: ignore
from scipy.optimize import curve_fit # type: ignore
from scipy.fft import fft, fftfreq # type: ignore
from scipy.signal import find_peaks # type: ignore
import copy
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from scipy.fft import rfft, rfftfreq # type: ignore
import random


sys.path.insert(0, 'C:/Users/thoma/Documents/ensta/2A/pre/boris/exercice')
from NetSocks import NSClient ,customize_plots,Shape  # type: ignore

#init
def init(ns,Lx,Ly,Lz,cellsize,T,modules,damping_x) :
    ns.reset()
    meshdims = (Lx, Ly, Lz)
    AFM = ns.AntiFerromagnet(np.array(meshdims)*1e-9, [cellsize*1e-9])
    AFM.modules(modules)
    temp = str(T) + 'K'
    ns.temperature(temp)
    if (T==0):
        ns.setode('LLG', 'RK4') 
    else :
        ns.setode('sLLG', 'RK4') # Stochastic LLG for temperature effects
    ns.setdt(25e-15)
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
    AFM.param.damping_AFM.setparamvar('abl_tanh', [damping_x/meshdims[0], damping_x/meshdims[0], 0, 0, 0, 0, 1, 10, damping_x])
    ns.Relax(['time', 5*1e-14])#50
    ns.reset() 
    return AFM


###dispersion relations
def disp_sin(ns,AFM,Lx,Ly,Lz,cellsize,Hap,t,output_file,dir,same_plot=False):#no init inside the functions
    file=Path(output_file)
    total_time = 2 *t*1e-12
    AFM.pbc('x', 10)   
    He=3e8#1e10 #1e11#4e1 
    N = 300#600
    L = Lx*1e-9
    kc = 2*np.pi*N/(2*L)
    fc =10e12#3e12#2800e6 #5e12
    time_step = 1e-13#1e-13#0.1e-12

    Hy = He 
    Hz=He

    ns.equationconstants('Hz', Hz)
    ns.equationconstants('Hap', Hap)
    ns.equationconstants('Hy', Hy)
    ns.equationconstants('k', kc)
    ns.equationconstants('f', fc)
    ns.equationconstants('u', total_time/2)
    ns.equationconstants('L', Lx*1e-9)
    ns.equationconstants('M', Ly*1e-9)
    #ns.equationconstants('ø',1)
    ns.setstage('Hequation')
    if same_plot :
        ns.editstagevalue(0, '0,Hy * sinc(k*(x-L/2))*sinc(2*PI*f*(t-u)), Hz * sinc(k*(x-L/2))*sinc(2*PI*f*(t-u))')
    else :
        ns.editstagevalue(0, '0,Hy * sinc(k*(x-L/2))*sinc(k*(y-M/2))*sinc(2*PI*f*(t-u)), Hz * sinc(k*(x-L/2))*sinc(k*(y-M/2))*sinc(2*PI*f*(t-u))')
        
    #setup data extraction in command buffer : need to extract a normalized magnetization profile at each time_step and append it to output file
    ns.setdata('commbuf')
    ns.editstagestop(0, 'time', total_time)
    ns.editdatasave(0, 'time', time_step)
    ns.clearcommbuffer()
    #get magnetisation profile along length through center
    ns.dp_getexactprofile((np.array([cellsize/2, Ly/2, Lz-cellsize])*1e-9), 
                        (np.array([Lx - cellsize/2, Ly/2, Lz])*1e-9), 
                            cellsize*1e-9, 0,bufferCommand = True)
    Ms=3.5e3
    #save only the dir component of magnetisation at time_ste
    if same_plot :
        ns.dp_div(2, Ms, bufferCommand=True)
        ns.dp_saveappendasrow(file, 2, bufferCommand = True)
        ns.dp_div(3, Ms, bufferCommand=True)
        ns.dp_saveappendasrow(file, 3, bufferCommand = True)
    else :
        ns.dp_div(dir, Ms, bufferCommand=True)
        ns.dp_saveappendasrow(file, dir, bufferCommand = True)

    ns.Run()

def number_defects(ns,AFM,x1,x2,y1,y2,cellsize,Lz): 
    "counting the number of defects"
    defec = []
    def create_list_defect(AFM,x1,x2,y1,y2,cellsize,defec,Lz):
    #print([x1,x2,y1,y2])
        if ((abs(x1-x2)<=2*cellsize) or (abs(y1-y2)<=2*cellsize)):
            defec.append([x1,x2,y1,y2])
            return 1
        n1,n2,m1,m2=x1,x2,y1,y2
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/find-defect.txt'
        if os.path.exists(output_file):
                os.remove(output_file)
        Datasave=[['|M|mm', AFM, np.array([n1,0.5*(m1+m2),0,0.5*(n1+n2),m2,Lz])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),0.5*(m1+m2),0,n2,m2,Lz])*1e-9],['|M|mm', AFM, np.array([n1,m1,0,0.5*(n1+n2),0.5*(m1+m2),Lz])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),m1,0,n2,0.5*(m1+m2),Lz])*1e-9]]
        ns.setsavedata(output_file,*Datasave )
        ns.Relax(['time', 1e-15,'time',1e-15])
        ns.reset()
        M=[ns.Get_Data_Columns(output_file,0)[-1],ns.Get_Data_Columns(output_file,2)[-1],ns.Get_Data_Columns(output_file,4)[-1],ns.Get_Data_Columns(output_file,6)[-1]]
        arr=np.array(M)
        if np.allclose(arr,3.5e3):
            return 0
        nbr=0
        create_list_defect(AFM,n1,0.5*(n1+n2),0.5*(m1+m2),m2,cellsize,defec,Lz)
        create_list_defect(AFM,0.5*(n1+n2),n2,0.5*(m1+m2),m2,cellsize,defec,Lz)
        create_list_defect(AFM,n1,0.5*(n1+n2),m1,0.5*(m1+m2),cellsize,defec,Lz)
        create_list_defect(AFM,0.5*(n1+n2),n2,m1,0.5*(m1+m2),cellsize,defec,Lz)
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
    create_list_defect(AFM, x1, x2, y1, y2, cellsize, defec,Lz)
    uniq_defects = count_unique_defects(defec, tol=2*cellsize)
    return uniq_defects

def disp_defects (ns,material,Lx,Ly,Lz,cellsize,T,modules,damping_x,t,output_files,DMI,defe): ####plotting disp relations on the same plot with increasing number of defects
    nb_def=[]
    for i,output_file in enumerate(output_files) :
        if os.path.exists(output_file):
            os.remove(output_file)
        AFM=init(ns,Lx,Ly,Lz,cellsize,T,modules,damping_x)
        d=defe[i]
        AFM.param.Ms_AFM.setparamvar(['defects',0,1e-2,cellsize*1e-9,cellsize*1e-9,d*1e-9,100])
        nb_def.append(number_defects(ns,AFM,cellsize,Lx-cellsize,cellsize,Ly-cellsize,cellsize,Lz))
        disp_sin(ns,AFM,Lx,Ly,Lz,cellsize,0,t,output_file,dir,same_plot=False)
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


####### setup

def setup_data(AFM,cellsize,i):
    if i==1 :
        datasave=[['time'],['<M>', AFM, [750*1e-9,500*1e-9,60,750*1e-9-cellsize*1e-9,500*1e-9-cellsize*1e-9,60-cellsize*1e-9]],['<M>', AFM, [600*1e-9,0.5*Ly*1e-9,0,600*1e-9+cellsize*1e-9,0.5*Ly*1e-9+cellsize*1e-9,cellsize*1e-9]]]
    if i==2:
        datasave=[['time'],['<M>', AFM, [(460-cellsize)*1e-9,(310-cellsize)*1e-9,(0)*1e-9,460*1e-9,310*1e-9,cellsize*1e-9]],['<M2>', AFM, [(460-cellsize)*1e-9,(310-cellsize)*1e-9,(0)*1e-9,460*1e-9,310*1e-9,cellsize*1e-9]]]
        #print(datasave)
    if i==4:
        datasave=[['time']]
        for i in range (100):
            datasave.append(['<My>', AFM, [(1250+i*12)*1e-9,0,0,(1250+(i+1)*12)*1e-9,1000*1e-9,5*1e-9]])
    if i==5:
        datasave=[['time']]
        for i in range (100):
            datasave.append(['<M>', AFM, [(1250+i*12)*1e-9,500*1e-9,0,(1250+(i+1)*12)*1e-9,505*1e-9,5*1e-9]])
    return datasave

def setup_defect(ns,AFM,Lx,Ly,Lz,a_x,a_y,a_z,nb):
    function='1.01'
    def one_def(x0,y0,z0):
        func = '- ((step(x - ' + str((x0 - a_x/2)*1e-9) + ') - step(x - ' + str((x0 + a_x/2)*1e-9) + ')) * ' + \
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
    if nb==0 : #choose the position of multiple the defect
        c1=(720,600,Lz/2)
        c2=(765,600,Lz/2)
        c3=(740,620,Lz/2)
        c4=(740,580,Lz/2)
        centres=[c1,c2,c3,c4]
        for c in centres :
            function+=one_def(c[0],c[1],c[2])
    elif nb==1: #1 defect
        x0=Lx*0.3
        y0=Ly*0.5
        z0=Lz/2
        function+=one_def(x0,y0,z0)
    else : #random nb defect
        centres=placer_carres_sans_chevauchement(nb, a_x, a_y, Lx, Ly, max_essais=10000)
        for c in centres :
            z = random.uniform(Lz/3, Lz - a_z / 2)
            function+=one_def(c[0],c[1],z)
    AFM.param.Ms_AFM.setparamvar('equation', function)
    ns.setdt(2e-15)


#antenna
def antenna_2D(ns,AFM,Lx,Ly,Lz,cellsize,t,output_file,DM,defects,dir=2):
    time_step=t*1e-12*1e-4# for gif

    taille=10*(cellsize*1e-9)


    #important parameters
    if defects :  
        setup_defect(AFM,Lx,Ly,Lz,2*cellsize,2*cellsize,cellsize,10) 
    Datasave = setup_data(AFM,cellsize,2)
    He=1e5#4e5


    H0=0
    ns.setstage('Hequation')
    ns.equationconstants('H0', H0)
    ns.equationconstants('He', He)
    ns.equationconstants('L', 0.5*Lx*1e-9)#
    ns.equationconstants('S', taille**2)
    ns.equationconstants('f',0.3*1e12)
    ns.editstagevalue(0, 'H0, H0, He * exp(-(x-L)*(x-L)/(2*S))*sin(f*(t))')
    ns.editstagestop(0, 'time', t*1e-12)
    ns.editdatasave(0,'time',time_step)
    
    ns.setsavedata(file, *Datasave)
    ns.Run()

def antenna_bulk(ns,AFM,Lx,Ly,Lz,cellsize,t,output_file,DM,defects):
    time_step=t*1e-12*1e-4# for gif
    taille=10*(cellsize*1e-9)

    #important data
    if defects : 
        setup_defect(ns,AFM,Lx,Ly,Lz,3*cellsize,3*cellsize,2*cellsize,10) 
    He=3e5
    Datasave = setup_data(AFM,cellsize,2)

    H0=0 
    ns.setstage('Hequation')
    ns.equationconstants('H0', H0)
    ns.equationconstants('He', He)
    ns.equationconstants('L', 0.5*Lx*1e-9)
    ns.equationconstants('S', taille**2)
    ns.equationconstants('f',0.3*1e12)
    ns.equationconstants('H',Lz*1e-9) 
    ns.equationconstants('P',0.2*Lz*1e-9)#distance of the antenna to the sample (variable)
    #ns.equationconstants('s',10*cellsize)
    ns.editstagevalue(0, 'H0, H0, He * (P/(H+P-z))*exp(-(x-L)*(x-L)/(2*S))*sin(f*(t))')

    ns.editstagestop(0, 'time', t*1e-12)
    
    ns.setsavedata(output_file, *Datasave)
    ns.editdatasave(0,'time',time_step)
    ns.Run()


def bulk_analysis_stray(AFM,Lx,Ly,Lz,cellsize,ns,t,ovf_filename,output_file,DM,defects): #antenna : reel data + strayfields data
    time_step=t*1e-12*1e-4
    ns.setdt(10e-15)
###############test taille supermesh
    margin_x = 0  # marge en nm
    margin_y = 0  # marge en nm  
    margin_z = -50   # marge en nm 

    if defects : 
        setup_defect(ns,AFM,Lx,Ly,Lz,2*cellsize,2*cellsize,2*cellsize,2)        
    He=6e5
    Datasave = setup_data(AFM,cellsize,2)

    
    taille=10*cellsize*1e-9
    H0=0
    ns.setstage('Hequation')
    ns.equationconstants('H0', H0)
    ns.equationconstants('He', He)
    ns.equationconstants('L', 0.5*Lx*1e-9)
    ns.equationconstants('S', taille**2)
    ns.equationconstants('f',0.3*1e12)
    ns.equationconstants('H',Lz*1e-9) 
    ns.equationconstants('P',0.2*Lz*1e-9)#distance of the antenna to the sample (variable)
    ns.editstagevalue(0, 'H0, H0, He * (P/(H+P-z))*exp(-(x-L)*(x-L)/(2*S))*sin(f*(t))')
    ns.editstagestop(0, 'time', t*1e-12)
    
    ns.setsavedata(output_file, *Datasave)
    ns.editdatasave(0,'time',time_step)
    ns.Run()

    ###strayfields
    Vis = ns.Ferromagnet([(-margin_x)*1e-9, (-margin_y)*1e-9, (margin_z)*1e-9,(Lx + margin_x)*1e-9, (Ly + margin_y)*1e-9, (0)*1e-9], [cellsize*1e-9])
    #Vis = ns.Ferromagnet([(-margin_x-10)*1e-9, (-margin_y-10)*1e-9, (margin_z)*1e-9,(Lx + margin_x+10)*1e-9, (Ly + margin_y+10)*1e-9, (0)*1e-9], [cellsize*1e-9])
    ns.display('Vis', 'Nothing')
    ns.delrect(Vis)
    ns.addmodule('supermesh', 'sdemag')
    ns.displaymodule(Vis,'demag')
    ns.display(Vis,'Heff')

    ns.computefields()
    ns.saveovf2(Vis, 'Heff', ovf_filename)



