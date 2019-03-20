CC = g++
#CFLAGS = -c  -Wall
CXX_STD = -std=gnu++11
CFLAGS  = -c -fPIC -g 
#-pedantic -Wall -Wextra -ggdb3
# LDFLAGS      = -shared

all: main.so

main.so: common.o sample_int_crank.o tree.o treefuns.o utility.o fit_std_main_loop.o forest.o XBART.o
	$(CC) $(CXX_STD) -fPIC -g  -shared  -o common.o sample_int_crank.o tree.o treefuns.o utility.o fit_std_main_loop.o forest.o XBART.o  

common.o: common.cpp
	$(CC) $(CFLAGS) $(CXX_STD)  common.cpp

xbart.o: xbart.cpp
	$(CC) $(CFLAGS) $(CXX_STD)  XBART.cpp

sample_int_crank.o: sample_int_crank.cpp
	$(CC) $(CFLAGS) $(CXX_STD)  sample_int_crank.cpp

tree.o: tree.cpp
	$(CC) $(CFLAGS) $(CXX_STD) tree.cpp

treefuns.o: treefuns.cpp
	$(CC) $(CFLAGS) $(CXX_STD)  treefuns.cpp

utility.o: utility.cpp
	$(CC) $(CFLAGS) $(CXX_STD)  utility.cpp

fit_std_main_loop.o: fit_std_main_loop.cpp
	$(CC) $(CFLAGS) $(CXX_STD)  fit_std_main_loop.cpp

forest.o: forest.cpp
	$(CC) $(CFLAGS) $(CXX_STD)  forest.cpp
