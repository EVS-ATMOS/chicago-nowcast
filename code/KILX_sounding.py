import numpy as np
import matplotlib.pyplot as plt
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.plots import SkewT
from datetime import datetime, timedelta
from metpy.units import units
import metpy.calc as metcalc


dataset = WyomingUpperAir.request_data(datetime(2018,6,27,0), 'ILX')


#get_ipython().run_line_magic('matplotlib', 'inline')

p = dataset['pressure'].values * units(dataset.units['pressure'])
#ip100 = np.where(p.magnitude==100.)[0][0]+1
#p = p[:ip100]
T = dataset['temperature'].values * units(dataset.units['temperature'])
#T = T[:ip100]
Td = dataset['dewpoint'].values * units(dataset.units['dewpoint'])
#Td = Td[:ip100]
u = dataset['u_wind'].values * units(dataset.units['u_wind'])
#u = u[:ip100]
v = dataset['v_wind'].values * units(dataset.units['v_wind'])
#v = v[:ip100]

fig = plt.figure(figsize=(9,9))
skew = SkewT(fig)
skew.plot(p,T,'r')
skew.plot(p,Td,'g')
skew.plot_barbs(p[:-1:2], u[:-1:2], v[:-1:2])
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000,100)
skew.ax.set_xlim(-40,60)
prof = metcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k', linewidth=2)
plt.title('KILX ROAB Obs 00 UTC 27 June 2018', loc='center')
plt.savefig('/home/jhemedinger/suli_projects/chicago-nowcast/images/ilx_sounding_00UTC.png', dpi=300)
plt.show()
