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

// #ifndef GUARD_rn_h
// #define GUARD_rn_h

//pure virtual base class for random numbers
class rn
{
  public:
    rn() {}
    virtual double normal() = 0;        //standard normal
    virtual double uniform() = 0;       //uniform(0,1)
    virtual double chi_square() = 0;    //chi-square
    virtual double exp() = 0;           //exponential
    virtual void set_df(size_t df) = 0; //set df for chi-square
    virtual ~rn() {}
};

#ifdef RNG_random

#include <Rmath.h>
#include <random>

//abstract random number generator based on C++ <random>
class arn : public rn
{
    //typedefs
    typedef std::default_random_engine genD;
    typedef std::normal_distribution<double> norD;
    typedef std::uniform_real_distribution<double> uniD;
    typedef std::chi_squared_distribution<double> chiD;

  public:
    //constructor
    arn() {}
    arn(unsigned size_t n1, unsigned size_t n2)
    {
        this->n1 = n1;
        this->n2 = n2;
    }
    //virtual
    virtual ~arn() {}
    virtual double normal() { return (nor)(gen); }
    virtual double uniform() { return (uni)(gen); }
    virtual double chi_square() { return (chi)(gen); }
    virtual double exp() { return -log(uniform()); }
    virtual void set_df(size_t df) { this->df = df; }
    size_t get_df() { return df; }

  private:
    size_t df;
    unsigned size_t n1, n2;
    genD gen;
    norD nor;
    uniD uni;
    chiD chi;
};

#elif defined(RNG_Rmath)

#include <Rmath.h>

//abstract random number generator based on Rmath
class arn : public rn
{
  public:
    //constructor
    arn() : df(1) {}
    arn(unsigned size_t n1, unsigned size_t n2) : df(1) { ::set_seed(n1, n2); }
    //virtual
    virtual ~arn() {}
    virtual double normal() { return ::norm_rand(); }
    virtual double uniform() { return ::unif_rand(); }
    virtual double chi_square() { return ::rchisq((double)df); }
    virtual double exp() { return ::exp_rand(); }
    virtual void set_df(size_t df) { this->df = df; }
    size_t get_df() { return df; }
    void set_seed(unsigned size_t n1, unsigned size_t n2)
    {
        ::set_seed(n1, n2);
    }
    void get_seed(unsigned size_t *n1, unsigned size_t *n2)
    {
        ::get_seed(n1, n2);
    }

  private:
    size_t df;
};

#else // YesRcpp

//abstract random number generator based on C++ <random>
// class arn : public rn
// {
//     //typedefs
//     typedef std::default_random_engine genD;
//     typedef std::normal_distribution<double> norD;
//     typedef std::uniform_real_distribution<double> uniD;
//     typedef std::chi_squared_distribution<double> chiD;

//   public:
//     //constructor
//     arn() {}
//     arn(unsigned size_t n1, unsigned size_t n2)
//     {
//         this->n1 = n1;
//         this->n2 = n2;
//     }
//     //virtual
//     virtual ~arn() {}
//     virtual double normal() { return (nor)(gen); }
//     virtual double uniform() { return (uni)(gen); }
//     virtual double chi_square() { return (chi)(gen); }
//     virtual double exp() { return -log(uniform()); }
//     virtual void set_df(size_t df) { this->df = df; }
//     size_t get_df() { return df; }

//   private:
//     size_t df;
//     unsigned size_t n1, n2;
//     genD gen;
//     norD nor;
//     uniD uni;
//     chiD chi;
// };

// // Added:
// #include <RcppArmadillo.h>
// #include <RcppParallel.h>

// //abstract random number generator based on R/Rcpp
// class arn : public rn
// {
//   public:
//     //constructor
//     arn() : df(1) {}
//     //virtual
//     virtual ~arn() {}
//     virtual double normal() { return R::norm_rand(); }
//     virtual double uniform() { return R::unif_rand(); }
//     virtual double chi_square() { return R::rchisq((double)df); }
//     virtual double exp() { return R::exp_rand(); }
//     virtual void set_df(size_t df) { this->df = df; }
//     size_t get_df() { return df; }

//   private:
//     size_t df;
//     Rcpp::RNGScope RNGstate;
// };

//  #endif

#endif
