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

#include "rtnorm.h"

double rtnorm(double mean, double tau, double sd, rn &gen)
{
    double x, z, lambda;

    /* Christian Robert's way */
    //assert(mean < tau); //assert unnecessary: Rodney's way
    tau = (tau - mean) / sd;

    /* originally, the function did not draw this case */
    /* added case: Rodney's way */
    if (tau <= 0.)
    {
        /* draw until we get one in the right half-plane */
        do
        {
            z = gen.normal();
        } while (z < tau);
    }
    else
    {
        /* optimal exponential rate parameter */
        lambda = 0.5 * (tau + sqrt(tau * tau + 4.0));

        /* do the rejection sampling */
        do
        {
            z = lambda * gen.exp() + tau;
        } while (gen.uniform() > exp(-0.5 * pow(z - lambda, 2.)));
    }

    /* put x back on the right scale */
    x = z * sd + mean;

    //assert(x > 0); //assert unnecessary: Rodney's way
    return (x);
}
