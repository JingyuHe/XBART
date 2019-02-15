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

#include "treefuns.h"
 
using namespace std;
//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo &xi)
{
    cout << "xinfo: \n";
    for (size_t v = 0; v != xi.size(); v++)
    {
        cout << "v: " << v << std::endl;
        for (size_t j = 0; j != xi[v].size(); j++)
            cout << "j,xi[v][j]: " << j << ", " << xi[v][j] << std::endl;
    }
    cout << "\n\n";
}



void fit_new_std(tree &tree, const double *X_std, size_t N, size_t p, std::vector<double> &output)
{
    tree::tree_p bn;
    for (size_t i = 0; i < N; i++)
    {
        bn = tree.search_bottom_std(X_std, i, p, N);
        //output[i] = bn->gettheta();
        output[i] = bn->gettheta_vector()[0];
    }
    return;
}




void fit_new_theta_noise_std(tree &tree, const double *X, size_t p, size_t N, std::vector<double> &output)
{
    tree::tree_p bn;
    for (size_t i = 0; i < N; i++)
    {
        bn = tree.search_bottom_std(X, i, p, N);
        // Add a loop?
        //output[i] = bn->gettheta_noise();
        output[i] = bn->gettheta_vector()[0];
    }
    return;
}

