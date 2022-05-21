# XBART

XBART -- Accelerated Bayesian Additive Regression Trees is an optimized machine learning library that provides efficient and accurate predictions. It implements a tree ensemble model inspired by the Bayesian Additive Regression Trees. XBART discards the slow random walk Metropolis-Hastings algorithms implemented by BART, rather fit the trees recursively and maintains the regularization from BART model. It can solve many data analytics problem efficiently and accurately. Furthermore it works for both regression and classification tasks, provides package in R and python, and can take advantage of parallel computing.



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



#### Trouble shooting

##### Windows

If you have any compiler error, please install the latest version of Rtools [Link](https://cran.r-project.org/bin/windows/Rtools/rtools42/rtools.html), it will install all necessary compilers and dependencies.

##### Mac


For mac users seeing error clang: error: unsupported option '-fopenmp'

It is because the default C++ compiler on Mac does not support openmp.

To solve it

1. Install necessary packages via homebrew 
   
   brew cask install gfortran

   brew install llvm boost libomp

   brew info llvm
   See what the LDFLAGS is, mine is like: 
   LDFLAGS=“-L/opt/homebrew/opt/llvm/lib -Wl,-rpath,/opt/homebrew/opt/llvm/lib”

2. Run the following code in the terminal, under your user directory (/Users/your_user_name)
   
   mkdir -p ~/.R

   cd .R

   touch Makevars

   open -e Makevars

3. Then the text editor pops out, copy and paste the following lines to the file
   
   CC=/opt/homebrew/opt/llvm/lclang
   CXX=/opt/homebrew/opt/llvm/clang++
   CXX11=/opt/homebrew/opt/llvm/clang++
   CXX14=/opt/homebrew/opt/llvm/clang++
   CXX17=/opt/homebrew/opt/llvm/clang++
   CXX1X=/opt/homebrew/opt/llvm/clang++
   LDFLAGS=-L/opt/homebrew/opt/llvm/lib

4. Save, exit and reboot your Mac

##### Linux

Since you are using Linux, you must be an expert ;-)
