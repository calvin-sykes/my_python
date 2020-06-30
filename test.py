import numpy as np
import unyt as u

#%matplotlib inline
import matplotlib.pyplot as plt

#import sys
#if '/cosma/home/dp004/dc-syke1/dev/my_python/' not in sys.path:
#    sys.path.append('/cosma/home/dp004/dc-syke1/dev/my_python/')

import my_python.uvb as uvb
import my_python.ions as ions

hm01 = uvb.UVB('HM01', 3.0)
hm12 = uvb.UVB('HM12', 3.0)
mh15 = uvb.UVB('MH15', 3.0)

h1 = ions.Ion('H I')

gamma_01 = hm01.photoionisation_rate()
gamma_12 = hm12.photoionisation_rate()
gamma_15 = mh15.photoionisation_rate()

#plt.figure()
#for uvb in [hm01, hm12, mh15]:
#    plt.loglog(uvb.tab_nu, uvb.tab_spectrum)
#plt.show()
#
print(gamma_01, gamma_12, gamma_15)
print(gamma_12 / gamma_01)

print(hm01.photon_flux())
print(hm01.photon_weighted_cross_section())
