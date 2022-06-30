# Classical_MD-Tutorial
The aim of this tutorial is to provide a practical guide to using CP2K for classical molecular dynamics with a focus on systems of interest to electrochemistry namely, metal/water interfaces. 
---

# Tutorial adapted to CP2K
In this tutorial we will run classical molecular dynamics on CP2K for a system composed of two parallel metallic electrodes with water between. 
The tutorial is structured into 3 different sections:
- **Section 1:** Setup of the electrodes using ASE 
- **Section 2:** Using packmol to add water 
- **Section 3:** Forcefield selection
- **Section 4:** Running classical MD using the NVT ensemble

<br/><br/>

---

## Section 1: Setup of the electrodes using ASE 
When running classical MD, one of the necessary steps to take is define topology information for the system being studied. Unlike quantum mechanical calculations, it is incorrect to initiate a simulation without specifying/generating topology information such as connectivity between atoms when simulating systems with molecules. With that said, any connectivity entails bonds, which in a classical sense can be envisaged as 'springs'. The property of the bonds need to be also defined, which we will do later. For modelling systems of interest to electrochemistry, it is simpler to let CP2K generate topological information, namely connectivity between atoms, subject to distance criteria. CP2K has a robust framework to generate connectivity in a custom and complex way if deemed necessary. Since we will be using distance criteria later on for generating connectivity, our system will be created in a way to make things easier in this regard. In plain language, we will make sure atoms we do not want to be connected, are separated sufficiently without complicating solvation.

ASE is a powerful tool for generating solid state structures and will thus be used to construct the electrodes. A general code can be seen below, which is also attached as a jupyter notebook to this repository. This code allows you to create a system of two electrodes, where each electrode is akin to a slab. The {111} surface is assumed here for both electrodes, however this can be changed to facets such as {100}. You have the flexibility of specify even/odd number of layers (the sum of layers of both electrodes which is what we see when individual units are stacked during PBC application), the lateral size of the electrodes and the size of the water box between them.

### ASE structure code
Here, the code has already been filled to generate gold electrodes of lateral dimensions of 6x6 and 3 layers individually (6 total). 

```python
#!/usr/bin/python

from ase.build import fcc111
from ase import Atoms
from ase.io import write,read
from ase.visualize import view
import numpy as np
import math

# alpha, d & Nx, Ny are the lattice parameter (bulk), interplanar spacing and later dimensions (Nx x Ny xlayers)
alpha=4.1983
d = alpha/(3**0.5)
Nx = 6  # use either even or odd 
Ny = 6  # only use even values because we will setup orthorombic cells

# junction layers considering PBC (sum of bottom and upper)
layers= 6

#The following loop enables usage of odd number of layers e.g., 5 layers = 3 for bottom and 2 for top electrode
if (layers % 2) == 0:
    Nz = layers
else:
    Nz = layers + 1
       
# Defining the lower base electrode orthogonally
elec = fcc111('Au', size=(Nx,Ny,Nz), vacuum=None, a=alpha, orthogonal=True)

# extracting cell parameters useful for later 
unit = elec.get_cell()

# cuts the upper layers of the electrode; (x-1)*d means x layers are kept; done this way for symmetry reasons
Z = ((Nz/2)-1)*d
del elec[elec.positions[:, 2] > Z]

N_z = int(Nz/2)
elec2 = fcc111('Au', size=(Nx,Ny,N_z), vacuum=None, a=alpha, orthogonal=True)

# translate the 2nd electrode in the z-direction by an amount = size of water box you want later; 20 Å used here
water_Zsize = 20
if (layers % 2) == 0:
    elec2.translate([0, 0, (water_Zsize+2*d)])
else:
    del elec2[elec2.positions[:, 2] < d/2 ]
    elec2.translate([0, 0, (water_Zsize+2*d)])

# This combines both electrodes giving your system  
system = elec + elec2

# generate proper unit cell parameters for system, ref denotes Z-position of last atom; used to get C unit cell vector 
ref = elec2.get_positions()[(len(elec2.get_positions()) - 1), :]
system.set_cell([[unit[0, 0], unit[0, 1], unit[0, 2]], [unit[1, 0], unit[1, 1], unit[1, 2]], [0.0, 0.0, (d+ref[2])]])

# get proper indices
system.set_tags(range(len(system)))

# to check if correct uncomment below
#view(system)

# extract structure file
write('electrodes.xyz', system)

# now we generate packmol script to solvate our system
# we define our water box dimensions as follows: inside box xmin ymin zmin xmax ymax zmax
xmin = 0
ymin = 0
R = (alpha/(2*math.sqrt(2)))  #radius of metal atom
zmin = ((Nz/2) - 1)*d + R + 2.2

xmax = unit[0, 0] - 2
ymax = unit[1, 1] - 2
zmax = ((Nz/2) - 1)*d + water_Zsize - R - 2.2

rho = 1*10**(-24) # g/Å3
NA = 6.022*10**23 # avogadro constant
Mr = 18 # molecular mass of water

# for volume, we subtract 1/2 spheres from box volume because water molecules may seep in between spheres
V = unit[0, 0]*unit[1, 1]*(water_Zsize) - 2*((N**2)*(1/2)*(4/3)*math.pi*(R**3))

#number of water molecules
No_W = round((rho*V*NA)/Mr)

packmol = open('packmol.inp', 'w')
print("output solvated_system.xyz",\
      "\nfiletype xyz",\
      "\nstructure electrodes.xyz",\
      "\n   fixed 0 0 0 0 0 0",\
      "\nend structure",\
      "\nstructure water.xyz",\
      "\n   number", No_W,\
      "\n   tolerance 2.0",\
      "\n   inside box", xmin, ymin, zmin, xmax, ymax, zmax,\
      "\nend structure", file= packmol)
packmol.close()
```
## Section 2: Using packmol to add water
The python code above, auotmatically generates a packmol input script for the system defined. After running successfully, you should be able to see 'packmol.inp' generated correctly, with reasonable box sizes and number of water molecules. There are two important sides to the packmol script that might need clarification; the box dimensions and the number of water molecules.
- The box is defined by specifying the minimum and maximum coordinates. As our box is confined between two electrodes, the zmin cannot be zero, but rather it should have a value greater than the z-position of the surface layer of the bottom electrode. In the above code, zmin is defined as 2.2 Å above the surface spheres. It is done this way to avoid issues with connectivity generation between the metal and water. Zmax is likewise defined as 2.2 Å below the surface spheres of the upper electrode. The position of the upper electrode itself is set by the size of the water box we want as seen earlier.
- The number of water molecules is defined quite simply from the following equation:

$$ No_w = {\rho *V*N_A \over M_R} $$
