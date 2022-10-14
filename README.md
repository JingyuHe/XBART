# XBART

XBART -- Accelerated Bayesian Additive Regression Trees is an optimized machine learning library that provides efficient and accurate predictions. It implements a tree ensemble model inspired by the Bayesian Additive Regression Trees. XBART discards the slow random walk Metropolis-Hastings algorithms implemented by BART, rather fit the trees recursively and maintains the regularization from BART model. It can solve many data analytics problem efficiently and accurately. Furthermore it works for both regression and classification tasks, provides package in R and python, and can take advantage of parallel computing.

## Warm-start BART

The BART Markov chain Monte Carlo algorithm can initialize at XBART fitted trees (rather than the default root nodes) to speed up convergence. See the demo examples under /tests folder for more details.

Currently, the warm-start BART relies on a customerized version of BART package [Github Link](https://github.com/jingyuhe/BART). We are working with the developers of BART pacakage to bring this feature to the original package.

## Reference

He, Jingyu, Saar Yalov, and P. Richard Hahn. "XBART: Accelerated Bayesian additive regression trees." *The 22nd International Conference on Artificial Intelligence and Statistics*. PMLR, 2019. [Link](http://jingyuhe.com/files/xbart.pdf)

He, Jingyu, and P. Richard Hahn. "Stochastic tree ensembles for regularized nonlinear regression." *Journal of the American Statistical Association* (2021): 1-20. [Link](http://jingyuhe.com/files/scalabletrees.pdf)



## Extension

The XBCF package implements the Bayesian causal forest under XBART framework. Details can be found [here](https://github.com/socket778/XBCF).



## Contributors

Jingyu He, Saar Yalov, Meijia Wang, Nikolay Krantsevich, Lee Reeves and P. Richard Hahn



## Install Instruction

It can be installed from GitHub directly using devtools package in R. The CRAN version will be submitted soon.

```R
library(devtools)
install_github("JingyuHe/XBART")
```

You may also install the customerized BART package to try the warm-start feature.
```R
library(devtools)
install_github("JingyuHe/BART")
```



#### Trouble shooting

##### Windows

If you have any compiler error, please install the latest version of Rtools [Link](https://cran.r-project.org/bin/windows/Rtools/rtools42/rtools.html), it will install all necessary compilers and dependencies.

##### Mac

You might need to install the Xcode command line tools for compilers. (Not necessary to install the entire large Xcode software.)

Open a terminal, run 

```R
xcode-select --install
```

##### Linux

Linux is already shipped with all necessary compilers. Since you are using Linux, you must be an expert ;-)

##### GSL

If you can't in stall it on Mac becase of an error message says 'gsl/gsl_sf_bessel.h' not found. Try following steps.
1, Open a terminal, run ```brew install gsl```.
2, Check if gsl is installed in the following directory: /opt/homebrew/Cellar/gsl/2.7.1 (if not, it should be somewhere similar, try searching for gsl).
3, In terminal, type 
```
cd ~/.R
open Makevars
```
(If you don’t have the Makevars file, create one by ```touch Makevars```)
4, in the file, paste in:
```
LDFLAGS+=-L/opt/homebrew/Cellar/gsl/2.7.1/lib
CPPFLAGS+=-I/opt/homebrew/Cellar/gsl/2.7.1/include
```
or with equivalent directory where your gsl library is installled.
