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
from scipy.fft import rfft, rfftfreq
import itertools
import time
import random


ns = NSClient(); 
ns.configure(True)#,False)
ns.cuda(2)
ns.reset(); ns.clearelectrodes()
def init(Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap,defect=False) :
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
    ns.setdt(1e-15)
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
    AFM.param.Ms_AFM.clearparamsvar()
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


def init_FM(Lx,Ly,Lz,cellsize,T,modules,damping_x,Hap,defect=False) :
    ns.reset()
    meshdims = (Lx, Ly, Lz)
    FM = ns.Ferromagnet(np.array(meshdims)*1e-9, [cellsize*1e-9])
    FM.modules(modules)
    temp = str(T) + 'K'
    ns.temperature(temp)
    if (T==0):
        ns.setode('LLG', 'RK4') 
    else :
        ns.setode('sLLG', 'RK4') # Stochastic LLG for temperature effects
    ns.setdt(1e-15)
    ns.setktens('-0.0455x2', '297.5z2' )
    return FM


def setup_defect(AFM,Lx,Ly,Lz,a_x,a_y,a_z,nb=1):
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
    if nb==0 :
        c1=(720,600)
        c2=(765,600)
        c3=(740,620)
        c4=(740,580)
        centres=[c1,c2,c3,c4]
        for c in centres :
            function+=one_def(c[0],c[1],Lz/2)
    elif nb==1:
        x0=Lx*0.3
        y0=Ly*0.5
        z0=Lz/2
        function+=one_def(x0,y0,z0)
    elif nb==2:
        c1=(720,600)
        c2=(765,600)
        c3=(740,620)
        c4=(740,580)
        centres=[c1,c2,c3,c4]
        for c in centres :
            z = random.uniform(Lz/3, Lz - a_z / 2)
            function+=one_def(c[0],c[1],z)
    else :
        centres=placer_carres_sans_chevauchement(nb, a_x, a_y, Lx, Ly, max_essais=10000)
        for c in centres :
            z = random.uniform(Lz/3, Lz - a_z / 2)
            function+=one_def(c[0],c[1],z)
    AFM.param.Ms_AFM.setparamvar('equation', function)
    ns.setdt(2e-15)


def number_defects(AFM,x1,x2,y1,y2,cellsize,Lz):
    defec = []
    def create_list_defect(AFM,x1,x2,y1,y2,defec,Lz):
    #print([x1,x2,y1,y2])
        if ((abs(x1-x2)<=2*cellsize) or (abs(y1-y2)<=2*cellsize)):
            defec.append([x1,x2,y1,y2])
            return 1
        n1,n2,m1,m2=x1,x2,y1,y2
        output_file='C:/Users/thdelvau/OneDrive - NTNU/Documents/find-defect.txt'
        if os.path.exists(output_file):
                os.remove(output_file)
        Datasave=[['|M|mm', AFM, np.array([n1,0.5*(m1+m2),0,0.5*(n1+n2),m2,1.0001*Lz])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),0.5*(m1+m2),0,n2,m2,Lz])*1e-9],['|M|mm', AFM, np.array([n1,m1,0,0.5*(n1+n2),0.5*(m1+m2),Lz])*1e-9],['|M|mm', AFM, np.array([0.5*(n1+n2),m1,0,n2,0.5*(m1+m2),Lz])*1e-9]]
        ns.setsavedata(output_file,*Datasave )
        ns.Relax(['time', 20e-15,'time',20e-15])
        ns.reset()
        M=[ns.Get_Data_Columns(output_file,0)[-1],ns.Get_Data_Columns(output_file,2)[-1],ns.Get_Data_Columns(output_file,4)[-1],ns.Get_Data_Columns(output_file,6)[-1]]
        arr=np.array(M)
        if np.allclose(arr,3.5e3):
            return 0
        nbr=0
        create_list_defect(AFM,n1,0.5*(n1+n2),0.5*(m1+m2),m2,cellsize,defec,Lz)#find_1_defect(AFM,n1,0.5*(m1+m2),0.5*(n1+n2),m2,cellsize)
        create_list_defect(AFM,0.5*(n1+n2),n2,0.5*(m1+m2),m2,cellsize,defec,Lz)#find_1_defect(AFM,0.5*(n1+n2),0.5*(m1+m2),n1,m2,cellsize)
        create_list_defect(AFM,n1,0.5*(n1+n2),m1,0.5*(m1+m2),cellsize,defec,Lz)# find_1_defect(AFM,n1,m1,0.5*(n1+n2),0.5*(m1+m2),cellsize)
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
    create_list_defect(AFM, x1, x2, y1, y2, cellsize, defec)
    uniq_defects = count_unique_defects(defec, tol=2*cellsize)
    return uniq_defects



def strayfields_FM(t,ovf_filename,DM,defects):
    height=10*1e-9
    Lx=1500
    Ly=600
    Lz=100
    T=0#.3
    cellsize=5
    taille=10*(cellsize*1e-9)
    modules=['exchange', 'anitens','Zeeman','mstrayfield']
    if DM :
        modules.append('DMexchange')
    AFM=init_FM(Lx,Ly,Lz,cellsize,T,modules,0,0,defects)#remember to add bool defect
    ns.setdt(50e-15)
###############test taille supermesh
    margin_x = 15  # marge en nm
    margin_y = 15  # marge en nm  
    margin_z = 50   # marge en nm 
    Vis = ns.Ferromagnet([(-margin_x)*1e-9, (-margin_y)*1e-9, (-margin_z)*1e-9,(Lx + margin_x)*1e-9, (Ly + margin_y)*1e-9, (Lz + margin_z)*1e-9], [cellsize*1e-9])
    ns.display('Vis', 'Nothing')
    ns.delrect(Vis)
    mag=Shape.rect(np.array([Lx,Ly,Lz])*1e-9)
    ns.addmodule('supermesh', 'sdemag')
    ns.displaymodule(Vis,'demag')
    ns.display(Vis,'Heff')
    ns.display(mag,'Nothing')
    if defects : 
        setup_defect(AFM,Lx,Ly,Lz,5*cellsize,5*cellsize,0.5*Lz)        


    #ns.addmodule('supermesh', 'strayfield')  # Module strayfield sur supermesh
    #ns.Relax(['mxh', 1e-5]) 
    #ns.setangle(90,180)
    #ns.Relax(['time', 10e-12]) 
    ns.computefields()     # Calcule tous les champs modules activés, dont strayfield
    #ns.display('supermesh', 'Hstray')  # Affiche le champ strayfield dans GUI
    # Sauver le champ strayfield pour post-traitement en OVF2
    #ns.saveovf2('supermesh', 'Hstray', ovf_filename)
    ns.saveovf2(Vis, 'Heff', ovf_filename)
    #plot_field(ovf_filename,-5*1e-9)


def strayfields_bulk(t,ovf_filename,DM,defects):
    height=10*1e-9
    Lx=1500
    Ly=600
    Lz=100
    T=0#.3
    cellsize=5
    taille=10*(cellsize*1e-9)
    modules=['exchange', 'anitens','Zeeman','mstrayfield']
    if DM :
        modules.append('DMexchange')
    AFM=init(Lx,Ly,Lz,cellsize,T,modules,0,0,defects)#remember to add bool defect
    ns.setdt(50e-15)
###############test taille supermesh
    margin_x = 0  # marge en nm
    margin_y = 0  # marge en nm  
    margin_z = 50   # marge en nm 
    Vis = ns.Ferromagnet([(-margin_x)*1e-9, (-margin_y)*1e-9, (-margin_z)*1e-9,(Lx + margin_x)*1e-9, (Ly + margin_y)*1e-9, (0)*1e-9], [cellsize*1e-9])
    ns.display('Vis', 'Nothing')
    ns.delrect(Vis)
    ns.addmodule('supermesh', 'sdemag')
    ns.displaymodule(Vis,'demag')
    ns.display(Vis,'Heff')
    if defects : 
        setup_defect(AFM,Lx,Ly,Lz,5*cellsize,5*cellsize,0.5*Lz)        


    #ns.addmodule('supermesh', 'strayfield')  # Module strayfield sur supermesh
    #ns.Relax(['mxh', 1e-5]) 
    #ns.setangle(90,180)
    ns.Relax(['time', t*1e-12]) 
    ns.computefields()     # Calcule tous les champs modules activés, dont strayfield
    #ns.display('supermesh', 'Hstray')  # Affiche le champ strayfield dans GUI
    # Sauver le champ strayfield pour post-traitement en OVF2
    #ns.saveovf2('supermesh', 'Hstray', ovf_filename)
    ns.saveovf2(Vis, 'Heff', ovf_filename)
    #plot_field(ovf_filename,-5*1e-9)





def plot_field(ovf_filename, z0,defect_pos=None):
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
    plt.colorbar(label='H (A/m)')

    plt.subplot(1, 3, 2)
    plt.imshow(Hy_2D.T, origin='lower', extent=extent)
    plt.title(f'Hy at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')

    plt.subplot(1, 3, 3)
    plt.imshow(Hz_2D.T, origin='lower', extent=extent)
    plt.title(f'Hz at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')
    plt.tight_layout()
    plt.show()


def save_field(ovf_filename, z0,path,i):
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





    # Plot all three components side by side in columns
    plt.figure(figsize=(15, 5))  # wider figure to fit 3 plots

    plt.subplot(1, 3, 1)
    plt.imshow(Hx_2D.T, origin='lower', extent=extent)
    plt.title(f'Hx at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')

    plt.subplot(1, 3, 2)
    plt.imshow(Hy_2D.T, origin='lower', extent=extent)
    plt.title(f'Hy at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')

    plt.subplot(1, 3, 3)
    plt.imshow(Hz_2D.T, origin='lower', extent=extent)
    plt.title(f'Hz at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')
    plt.tight_layout()

    save_name=path+str(i)+'.png'
    plt.savefig(save_name)

def antenna_bulk(t,ovf_filename,DM,defects):
    height=10*1e-9
    Lx=1500
    Ly=600
    Lz=100#100
    T=0#.3
    cellsize=5
    taille=10*(cellsize*1e-9)
    modules=['exchange', 'anitens','Zeeman','mstrayfield']
    if DM :
        modules.append('DMexchange')
    AFM=init(Lx,Ly,Lz,cellsize,T,modules,0,0,defects)#remember to add bool defect
    ns.setdt(10e-15)
###############test taille supermesh
    margin_x = 0  # marge en nm
    margin_y = 0  # marge en nm  
    margin_z = -50   # marge en nm 

    if defects : 
        #setup_defect(AFM,Lx,Ly,Lz,5*cellsize,5*cellsize,0.5*Lz)  
        setup_defect(AFM,Lx,Ly,Lz,2*cellsize,2*cellsize,2*cellsize,1)        
    He=3e5#3e5
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
    #print('defects=',number_defects(AFM,0,Lx,0,Ly,cellsize,Lz))
    ns.Run()

    #Vis = ns.Ferromagnet([(-margin_x)*1e-9, (-margin_y)*1e-9, (margin_z)*1e-9,(Lx + margin_x)*1e-9, (Ly + margin_y)*1e-9, (0)*1e-9], [cellsize*1e-9])
    Vis = ns.Ferromagnet([(-margin_x-10)*1e-9, (-margin_y-10)*1e-9, (margin_z)*1e-9,(Lx + margin_x+10)*1e-9, (Ly + margin_y+10)*1e-9, (0)*1e-9], [cellsize*1e-9])
    ns.display('Vis', 'Nothing')
    ns.delrect(Vis)
    ns.addmodule('supermesh', 'sdemag')
    ns.displaymodule(Vis,'demag')
    ns.display(Vis,'Heff')
    ns.computefields()
    ns.saveovf2(Vis, 'Heff', ovf_filename)

def plot_multiple_file(ovf_filename1,ovf_filename2,ovf_filename3, z0,comp):
    files=[ovf_filename1,ovf_filename2,ovf_filename3]
    H=[]
    for i in range (3):
    # Read OVF2 file
        vec, nodes, rect = ns.Read_OVF2(files[i])
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
        if comp==0:
            H.append(Hx_2D)
        elif comp==1:
            H.append(Hy_2D)
        elif comp==2:
            H.append(Hz_2D)

    # Plot all three components side by side in columns
    plt.figure(figsize=(15, 5))  # wider figure to fit 3 plots
    plt.subplot(1, 3, 1)
    plt.imshow(H[0].T, origin='lower', extent=extent)
    plt.title(f'f=5MHz at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')

    plt.subplot(1, 3, 2)
    plt.imshow(H[1].T, origin='lower', extent=extent)
    plt.title(f'f=25MHz  at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')

    plt.subplot(1, 3, 3)
    plt.imshow(H[2].T, origin='lower', extent=extent)
    plt.title(f'f=50MHz  at z = {z0*1e9} nm')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='H (A/m)')
    plt.tight_layout()
    plt.show()




# basepath='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/5/smaller_freq/'
# ovf_filename = os.path.join(basepath, "test.ovf")
# DM=False
# defects=True
# t=650
# # #strayfields_FM(10,ovf_filename,False,False)
# antenna_bulk(t,ovf_filename,DM,defects)
# # #strayfields_bulk(t,ovf_filename,DM,defects)
# # for i in range (50) : 
# #     basepath='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/5/1def_time/data'
# #     ovf_filename = os.path.join(basepath, str(i)+'.ovf')
# #     path='C:/Users/thdelvau/OneDrive - NTNU/Documents/results/5/1def_time/data/plot_time_sf'
# #     save_field(ovf_filename,(-5)*1e-9,path,i)

# # ovf_filename="C:/Users/thdelvau/OneDrive - NTNU/Documents/results/4/strayfields/smaller/bottom/def.ovf"

# plot_field(ovf_filename,(-5)*1e-9,(0.29, 0.5) )
# # plot_field(ovf_filename,(-20)*1e-9,(0.29, 0.5) )
# # plot_field(ovf_filename,(-50)*1e-9,(0.29, 0.5) )



ovf_filename1="C:/Users/thdelvau/OneDrive - NTNU/Documents/results/5/smaller_freq/freq5M_3D_1def_mid_small/def.ovf"
ovf_filename2="C:/Users/thdelvau\OneDrive - NTNU/Documents/results/5/smaller_freq/freq25M_3D_1def_mid_small/def.ovf"
ovf_filename3="C:/Users/thdelvau/OneDrive - NTNU/Documents/results/5\smaller_freq/freq50M_3D_1def_mid_small/def.ovf"
plot_multiple_file(ovf_filename1,ovf_filename2,ovf_filename3, -5e-9,0)  
plot_multiple_file(ovf_filename1,ovf_filename2,ovf_filename3, -5e-9,1)  
plot_multiple_file(ovf_filename1,ovf_filename2,ovf_filename3, -5e-9,2)  


