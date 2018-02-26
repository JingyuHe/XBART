/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert Gramacy
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

#include "rand_draws.h"

int NS;
rk_state **states;

/* 
 * newRNGstates:
 * 
 * seeding (potentiall multiple) RNGs
 */

void newRNGstates(void)
{
  unsigned long s;
  int i;

#ifdef _OPENMP
  NS = omp_get_max_threads();
#else
  NS = 1;
#endif
  states = (rk_state**) malloc(sizeof(rk_state*) * NS);

  for(i=0; i<NS; i++) {
    states[i] = (rk_state*) malloc(sizeof(rk_state));
    s = 10000 * unif_rand();
    rk_seed(s, states[i]);
  }
}


/* 
 * deleteRNGstates:
 *
 * freeing (potentially multiple) RNGs
 */

void deleteRNGstates(void)
{
  int i; //unsigned int i;
  for(i=0; i<NS; i++) {
    free(states[i]);
  }
  free(states); 
  states = NULL;
  NS = 0;
}


/* 
 * runi:
 * 
 * one from a uniform(0,1)
 * from jenise 
 */

double runi(rk_state *state)
{
    unsigned long rv;
    assert(state);
    rv = rk_random((rk_state*) state);
    return ((double) rv) / RK_MAX;
}



/*
 * rnor:
 * 
 * one draw from a from a univariate normal
 * modified from jenise's code
 */

void rnor(double *x, rk_state *state)
{
  double e,v1,v2,w;

  do {
    v1 = 2*runi(state)-1.;
    v2 = 2*runi(state)-1.;
    w = v1*v1+v2*v2;
  } while(w>1.);

  e = sqrt((-2.*log(w))/w);
  x[0] = v2*e;
  x[1] = v1*e;
}


/*
 * rexpo:
 *
 * exponential deviates; modified from R's with a 
 * custom state variable 
 */

double rexpo(double scale, rk_state* state)
{
    assert(scale > 0);
    return scale * expo_rand(state); 
}


/*
 * expo_rand:
 *
 * exponeitial deviates with scale=1; modified from R's with a 
 * custom state variable */

double expo_rand(rk_state *state)
{
    /* q[k-1] = sum(log(2)^k / k!)  k=1,..,n, */
    /* The highest n (here 16) is determined by q[n-1] = 1.0 */
    /* within standard precision */
    const static double q[] =
    {
        0.6931471805599453,
        0.9333736875190459,
        0.9888777961838675,
        0.9984959252914960,
        0.9998292811061389,
        0.9999833164100727,
        0.9999985691438767,
        0.9999998906925558,
        0.9999999924734159,
        0.9999999995283275,
        0.9999999999728814,
        0.9999999999985598,
        0.9999999999999289,
        0.9999999999999968,
        0.9999999999999999,
        1.0000000000000000
    };

    double a = 0.;
    double u = runi(state); 
    while(u <= 0. || u >= 1.) u = runi(state);
    for (;;) {
        u += u;
        if (u > 1.)
            break;
        a += q[0];
    }
    u -= 1.;

    if (u <= q[0])
        return a + u;

    int i = 0;
    double ustar = runi(state), umin = ustar;
    do {
        ustar = runi(state);
        if (umin > ustar)
            umin = ustar;
        i++;
    } while (u > q[i]);
    return a + umin * q[0];
}


/*
 * sq:
 * 
 * calculate the square of x
 */

double sq(double x)
{
  return x*x;
}


/*
 * rinvgauss:
 *
 * Michael/Schucany/Haas Method for generating Inverse Gaussian
 * random variable, as given in Gentle's book on page 193; not
 * thread safe -- doesn't use RNG state
 */

double rinvgauss(const double mu, const double lambda)
{
  double u, y, x1, mu2, l2;

  y = sq(norm_rand());
  mu2 = sq(mu);
  l2 = 2*lambda;
  x1 = mu + mu2*y/l2 - (mu/l2)* sqrt(4*mu*lambda*y + mu2*sq(y));

  u = unif_rand();
  if(u <= mu/(mu + x1)) return x1;
  else return mu2/x1;
}


/*
 * rtnorm_reject:
 *
 * dummy function in place of the Robert (1995) algorithm 
 * based on proposals from the exponential, should have that 
 * mean < tau 
 */

double rtnorm_reject(double mean, double tau, double sd, rk_state* state)
{
  double x, z, lambda;
  //int cnt;

  /* Christian Robert's way */
  assert(mean < tau);
  tau = (tau - mean)/sd;

  /* optimal exponential rate parameter */
  lambda = 0.5*(tau + sqrt(sq(tau) + 4.0));

  /* do the rejection sampling */
  //cnt = 0;
  do {
    z = rexpo(1.0/lambda, state) + tau;
  } while (runi(state) > exp(0.0-0.5*sq(z - lambda)));

  /* put x back on the right scale */
  x = z*sd + mean;

  assert(x > 0);
  return(x);

}
