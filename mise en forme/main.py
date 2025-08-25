import sys
import os

import plot # type: ignore
import function # type: ignore



sys.path.insert(0, 'C:/Users/thoma/Documents/ensta/2A/pre/boris/exercice')
from NetSocks import NSClient ,customize_plots,Shape  # type: ignore


ns = NSClient(); 
ns.configure(True, False)
ns.cuda(2)
ns.reset(); ns.clearelectrodes()

def main(DMI,defect):
    Lx=2500
    Ly=1000
    Lz=10
    cellsize=5
    damping_x=50
    modules=['exchange', 'anitens','Zeeman','mstrayfield']#carefull to add mstrayfield if strayfields needed
    T=0
    if DMI :
        modules.append('DMexchange')
    AFM=function.init(ns,Lx,Ly,Lz,cellsize,T,modules,damping_x)

    t=1
    ovf_filename="C:/Users/thoma/Documents/ensta/2A/pre/boris/exercice/test.ovf"
    output_file="C:/Users/thoma/Documents/ensta/2A/pre/boris/exercice/test.txt"
    function.bulk_analysis_stray(AFM,Lx,Ly,Lz,cellsize,ns,t,ovf_filename,output_file,DMI,defect)
    plot.m_trought_time(ns,output_file)
    plot.plot_fourier_xyz(ns,output_file)
    plot.plot_stray_field(ns,ovf_filename,-5e-9)

DMI,defect=True,True
main(DMI,defect)