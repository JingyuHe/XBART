/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#ifndef GUARD_common_h
#define GUARD_common_h

#ifdef MATHLIB_STANDALONE
#define NoRcpp
#else
#define RNG_Rcpp
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <vector>

#include <cassert>
//#include <RcppParallel.h>
#include <map>
#include <climits>

//using namespace RcppParallel;

using std::endl;

#ifdef BEGIN_RCPP   // May need to change this
#include <RcppArmadillo.h>
#define COUT Rcpp::Rcout
#else
#define COUT std::cout
#endif



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// #include <Rcpp.h>

// #define printf Rprintf

// #define err Rcpp::Rerr
// #define assert Rcpp::Rassert

// log(2*pi)
#define LTPI 1.83787706640934536

// #include "rn.h" // Removed

std::ostream &operator<<(std::ostream &out, const std::vector<double> &v);
std::ostream &operator<<(std::ostream &out, const std::vector<size_t> &v);
std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<double>> &v);
std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<size_t>> &v);

typedef std::vector<double> vec_d;     //double vector
typedef std::vector<vec_d> xinfo;      //vector of vectors, will be split rules
typedef std::vector<size_t> vec_sizet; // unsigned int vector
typedef std::vector<vec_sizet> xinfo_sizet;

template <typename T>
using matrix = std::vector<std::vector<T>>;

#endif
