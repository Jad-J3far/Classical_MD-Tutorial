# Classical_MD-Tutorial
The aim of this tutorial is to provide a practical guide to using CP2K for classical molecular dynamics with a focus on systems of interest to electrochemistry namely, metal/water interfaces. We begin with an example where the electrodes are charge neutral i.e., no applied biases.
---

# Tutorial adapted to CP2K
In this tutorial we will run classical molecular dynamics on CP2K for a system composed of two parallel metallic electrodes with water between. 
The tutorial is structured into 4 different sections:
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
V = unit[0, 0]*unit[1, 1]*(water_Zsize) - 2*((Nx*Ny)*(1/2)*(4/3)*math.pi*(R**3))

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
- The number of water molecules is defined quite simply by the following equation:

$$ No_w = {\rho *V*N_A \over M_R} $$

The volume, V, is calculated by calculating the volume of the box bounded by the position of atoms in the surface layers for the height, and the A & B parameters as the length and widths, and subtracting from this box volume, the volume of $(2 \times N_x \times N_y)$ half spheres since this volume is not available to water molecules.

$$ V = {A \times B \times Z_w} -  {2(N_x \times N_y \times (1/2) \times (4/3) \times \pi \times R^3)} $$

With this described, packmol.inp should be run in the same directory (with packmol installed) as the files electrodes.xyz and water.xyz. The latter is attached in the repository and simply consists of one water molecule which packmol will use to build a box of the desired number of water molecules, all separated by 2 Å as defined in the input using the keyword 'tolerance'. The command for running packmol is simply:

```
packmol < packmol.inp

```
Or, for mac OS users:

```
./packmol < packmol.inp

```

## Section 3: Forcefield Selection
Another necessary part of setting up a classical MD simulation, is defining forcefields, which dictate nonbonded interactions between atoms or molecules. These forces are then computed using the classical newtonian equations of motion to determine velocities and trajectories. In our system, we need to define a water model or forcefield for the explicit water atoms and how they interact with one another. Additionally, we need to define interaction potentials for the metals and hydrogen/oxygen. Since we will be completely fixing the metal atoms, we do not need to provide metal-metal interaction potentials. In the case the metals are not fixed however, the embedded atom model (EAM) can be used to reproduce bond interactions.

### 3.1 Water Model 
A vast number of explicit water models have been developed over the past few decades. These models undoubtedly vary in complexity and accuracy with regards to reproducing the overall properties of bulk water. In typical periodic classical MD simulations involving water, water molecules account for a large portion of atoms/molecules present and thus the choice of water model heavily influences the computational costs. In general, models can be classified by the following three points; (i) the number of interaction points called site, (ii) whether the model is rigid or flexible, (iii) whether the model includes polarization effects. The following is great paper comparing different models for different properties https://pubs.acs.org/doi/10.1021/acs.jcim.1c00794. 

It is important to note that water models are typically optimized to reproduce certain features of bulk water and in particular phases. Broadly speaking, individual water models do not reproduce all bulk water properties simultaneously. Thus, the choice of water model can be said to depend on the application. With that said, here we aim to select a water model that can reproduce the structure and density of bulk water reliably. For this, we will be using TIP3P (transferable intermolecular potential with 3 points) as described in the following paper https://doi.org/10.1063/1.1884609. In more detail, the model employed is the flexible TIP3P model which is changed in such a way that it allows for the flexibility of bonds. The model hamiltonian can be found in the paper along with parameter details. It needs to be described in CP2K under both *Bonded* and *Nonbonded* sections as will soon be seen. The former is described in the form of a harmonic constraint while the latter in the form of the 12-6 Lennard-Jones (LJ) potential.
angles This is one of the simpler models that is computationally cheap. 

### 3.2 Metal-Water Interactions
Similarly to the water models, there are different interaction potential functional forms that can be employed to describe the metal-H and metal-O interactions. Ultimately, these are *nonbonded* interactions and over the course of standard classical simulations, no chemical reactions i.e., bond breaking/formation, occur. This interaction potential should desirably reproduce the orientation and structuring of water at the interface, that we would see using AIMD, except bond formation. Different functional forms include:
- LJ 12-6 or 9-6
- Buckingham Potential
- Morse Potential

These can be parameterized in different ways including fitting against *ab intio* calculations. Other more specialized potentials have been developped over the years for certain metals such as platinum. Since the system built above uses gold, we will use the LJ 12-6 interaction potential for Au-water from the following paper https://doi.org/10.1021/acs.jctc.7b00612 which parameterized the function against AIMD simulations. You may search for a similar forcefield for your metal of choice, or as an initial attempt, use the forcefields as will be described in the CP2K input in the following section.

## Section 4: Running Classical MD using the NVT ensemble
We can now begin with setting up the CP2K input. This can be found below, and is attached as a file to this repository. 

