import numpy as np
import unyt as u

#%matplotlib inline
import matplotlib.pyplot as plt

#import sys
#if '/cosma/home/dp004/dc-syke1/dev/my_python/' not in sys.path:
#    sys.path.append('/cosma/home/dp004/dc-syke1/dev/my_python/')

import my_python.uvb as uvb
import my_python.ions as ions

zarr = np.arange(2, 5.01, 1, dtype=np.int_)
hm01 = {z: uvb.UVB('HM01', z) for z in zarr}
hm12 = {z: uvb.UVB('HM12', z) for z in zarr}
mh15 = {z: uvb.UVB('MH15', z) for z in zarr}

#z = 2
#plt.figure()
#for uvb_dict in [hm01, hm12, mh15]:
#    uvb_z = uvb_dict[z]
#    plt.loglog(uvb_z.tab_nu, uvb_z.tab_spectrum, label=uvb_z.name)
#plt.legend()
#plt.show()

ion = ions.Ion('H I')

for z in zarr:
    gamma_01 = hm01[z].photoionisation_rate(ion)
    gamma_12 = hm12[z].photoionisation_rate(ion)
    gamma_15 = mh15[z].photoionisation_rate(ion)

    print(hm12[z].photoheating_rate(ion))

    print(f"z={z}")
    print('HM01:', gamma_01, 'HM12:', gamma_12, 'MH15:', gamma_15)
    print('HM01/HM12:', gamma_01 / gamma_12)
    print()
