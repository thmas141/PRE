import sys
import os
sys.path.insert(0, 'C:/Users/thdelvau/OneDrive - NTNU/Documents/Release')

import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from NetSocks import NSClient ,customize_plots,Shape  # type: ignore
import numpy as np
from pathlib import Path


ns = NSClient(); ns.configure(True) ;ns.cuda(2)
ns.reset(); #ns.default()
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


def Compute_Oe_field(Lx,Ly,Lz,cellsize,taille,distance_sample,f):
    
    basepath='C:/Users/thdelvau/Documents/Oefields/0.3'
    #print(ovf_filename)
    Fil = ns.Conductor([0.5*(Lx-taille)*1e-9, 0, (Lz+distance_sample)*1e-9,0.5*(Lx+taille)*1e-9, Ly*1e-9,  ((Lz+distance_sample+taille))*1e-9],[cellsize*1e-9,cellsize*1e-9,cellsize*1e-9])  
    Fil.modules(['transport'])
    ns.addmodule('supermesh', 'Oersted')
    ns.escellsize([5e-9, 1e-9, 5e-9])
   
    AFM=init(Lx,Ly,Lz,cellsize,0,['transport,exchange'],0,0,False)


    ns.setode('LLG', 'RK4')
    ns.setdt(10e-15)

    ns.clearelectrodes()
    #ns.setdefaultelectrodes()
    ns.addelectrode(np.array([0.5*(Lx-taille),0,(Lz+distance_sample),0.5*(Lx+taille),0,(Lz+distance_sample+taille)])* 1e-9)
    ns.addelectrode(np.array([0.5*(Lx-taille),Ly,(Lz+distance_sample),0.5*(Lx+taille),Ly,(Lz+distance_sample+taille)])* 1e-9)
    ns.designateground('1')
    ns.display(Fil,'V')

    for i in range (50,51):
        print(i)
        t=(i/100)*1/f
        V0=10e-3*np.sin(2*3.14*f*t)
        ovf_filename = os.path.join(basepath, f"{int(i)}.ovf")
        ns.setpotential(V0)   
        ns.computefields()
        ns.saveovf2('supermesh','HOe', ovf_filename)
        #ns.saveovf2('AFM','Heff', ovf_filename)
    # # Current along y-direction
    # ns.addelectrode(np.array([(meshdims[0]/2 - 100), 0, (meshdims[2]-cellsize), (meshdims[0]/2 + 100), 0, meshdims[2]])* 1e-9)
    # ns.addelectrode(np.array([(meshdims[0]/2 - 100), meshdims[1], (meshdims[2]-cellsize), (meshdims[0]/2 + 100), meshdims[1], meshdims[2]]) * 1e-9)
    # ns.designateground('1')

    #ns.V([V0, 'time', 100*1e-12])
    # ns.equationconstants('V0', V0)
    # ns.equationconstants('f', f)
    # ns.setstage('Vequation')
    # ns.editstagevalue(0, '1e10*sin(1e9*t)')
    # ns.editstagestop(0, 'time', 10e-12/f)
    #ns.setpotential(1e-2)
    

    #ns.Run()
    # Calcul des champs Oersted
    #ns.computefields()
    #ns.display('supermesh','HOe')
    # Sauvegarde du champ
    #ns.saveovf2('AFM','Heff', ovf_filename)
    print('Champ Oersted sauvegardé avec excitation AC')



def plot_field(ovf_filename, z0):
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

    # Downsampled 2D slices
    Hx_2D = Hx[:, :, iz][::3, ::3]
    Hy_2D = Hy[:, :, iz][::3, ::3]
    Hz_2D = Hz[:, :, iz][::3, ::3]
    # Resolution in z direction from bounding box
    z_min, z_max = rect[2], rect[5]
    dz = (z_max - z_min) / (Nz - 1)

    # Index of layer closest to z0
    iz = int(round((z0 - z_min) / dz))
    iz = max(0, min(Nz-1, iz))  # clamp to allowed range

    # Extract 2D slices at z = z0, with step
    Hx_2D = Hx[:, :, iz][::3, ::3]
    Hy_2D = Hy[:, :, iz][::2, ::2]
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

def plot_Hz_line_vs_z(ovf_filename, x_ratio=0.5, y_ratio=0.5):
    # Lecture du fichier OVF
    vec, nodes, rect = ns.Read_OVF2(ovf_filename)
    Nx, Ny, Nz = nodes

    # Reshape du champ (Nx, Ny, Nz, 3)
    vec = vec.reshape((Nz, Ny, Nx, 3)).transpose(2, 1, 0, 3)
    Hz = vec[:, :, :, 2]

    # Convertir ratios [0,1] en indices
    ix = int(round(x_ratio * (Nx - 1)))
    iy = int(round(y_ratio * (Ny - 1)))

    # Clamp dans la grille au cas où
    ix = max(0, min(Nx - 1, ix))
    iy = max(0, min(Ny - 1, iy))

    # Extraire Hz en fonction de z à ce point
    Hz_line = Hz[ix, iy, :]  # (Nz,)

    # Calcul des coordonnées physiques en z
    z_min, z_max = rect[2], rect[5]
    z_vals = np.linspace(z_min, z_max, Nz)

    # Affichage
    plt.figure(figsize=(6, 4))
    plt.plot(z_vals * 1e9, Hz_line)
    plt.xlabel('z (nm)')
    plt.ylabel(f'H_z (T) at x_ratio={x_ratio}, y_ratio={y_ratio}')
    plt.title('Composante H_z en fonction de z à un point donné')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



basepath='C:/Users/thdelvau/Documents/Oefields/0.3'
ovf_filename = os.path.join(basepath, f"{int(1)}.ovf")

Lx=1500
Ly=600
Lz=100
cellsize=5
taille=10*(cellsize)
distance_sample=0.2*Lz
f=0.3

Compute_Oe_field(Lx,Ly,Lz,cellsize,taille,distance_sample,0.3)
#plot_Hz_line_vs_z(ovf_filename,0.45,0.5)
# plot_field(ovf_filename, 5*1e-9)
# plot_field(ovf_filename, 50*1e-9)
# plot_field(ovf_filename, 100*1e-9)
