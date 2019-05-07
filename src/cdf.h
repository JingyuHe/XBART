// distribution and special functions declarations
//
// CDF(x) is Prob(X <= x)
// CDFInv(x) gives the quantile corresponding to x
// zCDF(x) is x converted into an equivalent gaussian
//
// In principle, for any distribution *, zCDF*(x) should be normCDFInv(CDF*(x))
//   but if * always takes positive values, we want to map to positive normals
//   also, a clever implementation may be needed to avoid overflow

#ifndef __CDF_H
#define __CDF_H

//const double PI = 3.1415926535;
//const double SQRT2PI = 2.5066282746;
//const double SQRT2 = 1.414214;


#include"common.h"

double normCDF(const double x);
double normCDFInv(const double x);

double tCDF(const double x, const double n);
double ztCDF(const double x, const double n);
double tCDFInv(const double x, const double n); // not yet defined

double fCDF(const double x, const double n1, const double n2);
double zfCDF(const double x, const double n1, const double n2);
double fCDFInv(const double x, const double n1, const double n2); // not yet defined

// digamma and trigamma are slow, very approximate and only work for x,y>0
// should fix this some day
double digamma(const double x);
double trigamma(const double x);
double trigammainv(const double y); // error up to 0.25


#endif