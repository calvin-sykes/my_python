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

gamma_01 = hm01.HI_photoionisation_rate()
gamma_12 = hm12.HI_photoionisation_rate()
gamma_15 = mh15.HI_photoionisation_rate()

plt.figure()
for uvb in [hm01, hm12, mh15]:
    plt.loglog(uvb.tab_nu, uvb.tab_spectrum)
plt.show()

print(gamma_01, gamma_12, gamma_15)
print(gamma_12 / gamma_15)
