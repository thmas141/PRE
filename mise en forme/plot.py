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


#ground state plot
def angle_DMI(ns,file): #plots canting angle
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
    ctheta=np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
    print('angle_DMI = ',180-np.degrees(np.arccos(np.clip(ctheta,-1,1))))
    return 180-np.degrees(np.arccos(np.clip(ctheta,-1,1)))

#reel analysis plot
def m_trought_time(ns,output_file,j=0) :
    X=ns.Get_Data_Columns(output_file,0)#j is if multiple point are recorded
    Y1=ns.Get_Data_Columns(output_file,1+j)
    Y2=ns.Get_Data_Columns(output_file,2+j)
    Y3=ns.Get_Data_Columns(output_file,3+j)
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

    #mx/my/mz plot
    plt.figure()
    plt.plot(X,Y1p,color='blue',label='<Mx>')
    plt.plot(X,Y2p,color='green',label='<My>')
    plt.plot(X,Y3p,color='red',label='<Mz>')
    plt.legend()
    plt.show()

    #plotting distance : if data are recorded along a line
    # My=[]
    # for i in range(100) :
    #     my=ns.Get_Data_Columns(output_file,1+i)
    #     My.append(np.mean(my[len(my)//2:])/3500)
    # d=[12*i*1e-9 for i in range(100)]
          
    # plt.figure()
    # plt.plot(d,My,label='<my>')
    # plt.legend()
    # plt.show()

 
def plot_fourier_xyz(ns,output_file,j=0, nb_peaks=5):

    # Lecture et normalisation
    X = ns.Get_Data_Columns(output_file, 0)
    Y1 = ns.Get_Data_Columns(output_file, 1 + j)
    Y2 = ns.Get_Data_Columns(output_file, 2 + j)
    Y3 = ns.Get_Data_Columns(output_file, 3 + j)

    Y1p = [i / (3500 * 1.01) for i in Y1]
    Y2p = [i / (3500 * 1.01) for i in Y2]
    Y3p = [i / (3500 * 1.01) for i in Y3]


    ######If NEEL
    # Y1=ns.Get_Data_Columns(output_file,1+3)
    # Y2=ns.Get_Data_Columns(output_file,2+3)
    # Y3=ns.Get_Data_Columns(output_file,3+3)
    # Y1p=[Y1p[i]-Y1[i]/(3500*1.01) for i in range(len(Y1))]#normalisation
    # Y2p=[Y2p[i]-Y2[i]/(3500*1.01) for i in range(len(Y2))]
    # Y3p=[Y3p[i]-Y3[i]/(3500*1.01) for i in range(len(Y3))]
    ###################


    # last third
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
        axs[i].scatter([xf[p] for p in sorted_peaks],
                       [magnitudes[p] for p in sorted_peaks],
                       color=color_map[label], marker='x')
        axs[i].set_ylabel("Amplitude")
        axs[i].legend()
        axs[i].set_xscale("log")
        axs[i].grid(True)

    axs[-1].set_xlabel("Fréquencies (Hz)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    ## if one want to print peak frequencies in the cmd 
    # print(f"\nTop {nb_peaks} fréquences dominantes pour '{place}' :")
    # for label, peaks in all_peaks.items():
    #     print(f"\n{label}-axis:")
    #     for f, amp in peaks:
    #         print(f"f = {f:.2e} Hz, amplitude = {amp:.3e}")


def plot_stray_field(ns,ovf_filename, z0,defect_pos=None):
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


############plot disp, multiple ways of doing it with y and z on the same plot
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

def plot_both(output_file,Hfield=0):
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
##