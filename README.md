# Jingyu He, Saar Yalov and P. Richrad Hahn (2019). XBART: Accelerated Bayesian Additive Regression Trees. The 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)

# For mac users seeing error clang: error: unsupported option '-fopenmp'
It is because the default C++ compiler on Mac does not support openmp.

To solve it
1. Install necessary packages via homebrew 
   brew cask install gfortran
   brew install llvm boost libomp

2. Run the following code in the terminal, under your user directory (/Users/your_user_name)
   
   mkdir -p ~/.R
   cd .R
   touch Makevars
   open -e Makevars

3. Then the text editor pops out, copy and paste the following lines to the file
   
   CC=/usr/local/opt/llvm/bin/clang
   CXX=/usr/local/opt/llvm/bin/clang++
   CXX11=/usr/local/opt/llvm/bin/clang++
   CXX14=/usr/local/opt/llvm/bin/clang++
   CXX17=/usr/local/opt/llvm/bin/clang++
   CXX1X=/usr/local/opt/llvm/bin/clang++
   LDFLAGS=-L/usr/local/opt/llvm/lib

4. Save, exit and reboot your Mac