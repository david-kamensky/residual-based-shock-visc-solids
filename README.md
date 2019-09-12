# Residual-based shock capturing in solids

This repository contains tIGAr-based code examples to accompany the paper
```
@article{Bazilevs2019,
title = "Residual-based shock capturing in solids",
journal = "Computer Methods in Applied Mechanics and Engineering",
author = "Y. Bazilevs and D. Kamensky and G. Moutsanidis and S. Shende",
year = "2019",
note = "Accepted."
}
```
Usage requires [tIGAr](https://github.com/david-kamensky/tIGAr) (and its dependencies).  Installation information for tIGAr can be found in the linked repository's README file.  Some examples also require [TSFC](https://doi.org/10.1137/17M1130642), which can be installed for FEniCS as follows:
```
$ sudo pip3 install git+https://github.com/blechta/tsfc.git@2018.1.0
$ sudo pip3 install git+https://github.com/blechta/COFFEE.git@2018.1.0
$ sudo pip3 install git+https://github.com/blechta/FInAT.git@2018.1.0
$ sudo pip3 install singledispatch networkx pulp
```
