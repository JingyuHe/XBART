To install in R
library(devtools)
install_github("JingyuHe/XBART")




For mac users seeing error clang: error: unsupported option '-fopenmp'

It is because the default C++ compiler on Mac does not support openmp.

To solve it

1. Install necessary packages via homebrew 
   
   brew cask install gfortran

   brew install llvm boost

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
