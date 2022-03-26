# "A first step towards controllability of partial differential equations via physics-informed neural networks"

For reproducibility purposes, this repository contains the code used in the examples described in the "Numerical experiments" section of the manuscript "A first step towards controllability of partial differential equations via physics-informed neural networks", (2022), by Carlos J. García-Cervera[^1], Mathieu Kessler[^2] and Francisco Periago[^2].

## Requirements and instructions 

1. Clone the present project in the folder of your choice:
```
git clone git@github.com:mkesslerct/deepcontrol.git
```
2. Create the conda virtual enviromment `deepcontrol` 
```
cd deepcontrol
conda env create -f deepcontrol_env.yml
```
3. Activate the deepcontrol virtual environment:
```
conda activate deepcontrol
```
4. Download and install the development version of `deepxde`, the library for scientific machine learning and physics-informed learning, see [Github repo](https://github.com/lululxvi/deepxde)
> Note: as of March 25th 2022, the development version if required for the multidimensional linear heat equation, section 4.2
```
git clone git@github.com:lululxvi/deepxde.git
cd deepxde
python3 -m pip install .
cd ..
```

5. Run any of the scripts by:
```
python3 scripts/script_name.py
``` 



[^1]: Department of Mathematics. University of California, Santa Barbara. CA 93106, USA
[^2]: Department of Applied Mathematics and Statistics. Universidad Politécnica de Cartagena, Campus Muralla del Mar, 30202, Cartagena (Murcia), Spain
