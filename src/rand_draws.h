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

#ifndef __RAND_DRAWS_H__
#define __RAND_DRAWS_H__

#include "common.h"
#include "randomkit.h"
#include <stdlib.h>
#include <assert.h>

void newRNGstates(void);
void deleteRNGstates(void);
double runi(rk_state *state);
void rnor(double *x, rk_state *state);
double rexpo(double lambda, rk_state *state);
double sq(double x);
double rinvgauss(const double mu, const double lambda);
double rtnorm_reject(double mean, double tau, double sd, rk_state* state);
double rexpo(double scale, rk_state* state);
double expo_rand(rk_state *state);

#endif
