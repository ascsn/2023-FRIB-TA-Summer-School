//     tnorm.cpp: Truncated Normal helper functions for probit model.
//     Copyright (C) 2012-2019 Matthew T. Pratola
//
//     This file is part of OpenBT.
//
//     OpenBT is free software: you can redistribute it and/or modify
//     it under the terms of the GNU Affero General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     OpenBT is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU Affero General Public License for more details.
//
//     You should have received a copy of the GNU Affero General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//     Author contact information
//     Matthew T. Pratola: mpratola@gmail.com


//# include <cstdlib>
//# include <iostream>
//# include <iomanip>
#include <cmath>
//# include <ctime>
//# include <cstring>

#include "crn.h"
#include "tnorm.h"
using namespace std;


// Draw from the truncated Normal distribution with left,right truncation points
// given by (lwr,upr).
// Based on https://people.sc.fsu.edu/~jburkardt/cpp_src/truncated_normal/truncated_normal.html
// which in turn borrows much from Abrowitz & Stegun.
double gen_trunc_normal(double mu,double sig,double lwr, double upr,rn &gen)
{
	double alpha_cdf;
	double beta_cdf;
	double x;
	double xi;
	double xi_cdf;

	alpha_cdf=normal_01_cdf((lwr-mu)/sig);
	beta_cdf=normal_01_cdf((upr-mu)/sig);

	xi_cdf=alpha_cdf+gen.uniform()*(beta_cdf-alpha_cdf);
	xi=normal_01_cdf_inv(xi_cdf);

	x=mu+sig*xi;

	return x;
}

double gen_left_trunc_normal(double mu,double sig,double upr,rn &gen)
{
	double beta_cdf;
	double x;
	double xi;
	double xi_cdf;

	beta_cdf=normal_01_cdf((upr-mu)/sig);

	xi_cdf=gen.uniform()*(beta_cdf);
	xi=normal_01_cdf_inv(xi_cdf);

	x=mu+sig*xi;

	return x;
}

double gen_right_trunc_normal(double mu,double sig,double lwr,rn &gen)
{
	double alpha_cdf;
	double x;
	double xi;
	double xi_cdf;

	alpha_cdf=normal_01_cdf((lwr-mu)/sig);

	xi_cdf=alpha_cdf+gen.uniform()*(1.0-alpha_cdf);
	xi=normal_01_cdf_inv(xi_cdf);

	x=mu+sig*xi;

	return x;
}

double normal_01_cdf(double x)
{
  double a1 = 0.398942280444;
  double a2 = 0.399903438504;
  double a3 = 5.75885480458;
  double a4 = 29.8213557808;
  double a5 = 2.62433121679;
  double a6 = 48.6959930692;
  double a7 = 5.92885724438;
  double b0 = 0.398942280385;
  double b1 = 3.8052E-08;
  double b2 = 1.00000615302;
  double b3 = 3.98064794E-04;
  double b4 = 1.98615381364;
  double b5 = 0.151679116635;
  double b6 = 5.29330324926;
  double b7 = 4.8385912808;
  double b8 = 15.1508972451;
  double b9 = 0.742380924027;
  double b10 = 30.789933034;
  double b11 = 3.99019417011;
  double cdf;
  double q;
  double y;

  if (fabs(x) <= 1.28)
  {
    y = 0.5 * x * x;
    q = 0.5 - fabs ( x ) * ( a1 - a2 * y / ( y + a3 - a4 / ( y + a5
      + a6 / ( y + a7 ) ) ) );
  }
  else if (fabs (x) <= 12.7)
  {
    y = 0.5 * x * x;

    q = exp ( - y ) * b0 / ( fabs ( x ) - b1
      + b2  / ( fabs ( x ) + b3
      + b4  / ( fabs ( x ) - b5
      + b6  / ( fabs ( x ) + b7
      - b8  / ( fabs ( x ) + b9
      + b10 / ( fabs ( x ) + b11 ) ) ) ) ) );
  }
  else
  {
    q = 0.0;
  }
//
//  Take account of negative X.
//
  if (x < 0.0)
  {
    cdf = q;
  }
  else
  {
    cdf = 1.0 - q;
  }

  return cdf;
}

double normal_01_cdf_inv (double p)
{
  double a[8] = {
    3.3871328727963666080,     1.3314166789178437745E+2,
    1.9715909503065514427E+3,  1.3731693765509461125E+4,
    4.5921953931549871457E+4,  6.7265770927008700853E+4,
    3.3430575583588128105E+4,  2.5090809287301226727E+3 };
  double b[8] = {
    1.0,                       4.2313330701600911252E+1,
    6.8718700749205790830E+2,  5.3941960214247511077E+3,
    2.1213794301586595867E+4,  3.9307895800092710610E+4,
    2.8729085735721942674E+4,  5.2264952788528545610E+3 };
  double c[8] = {
    1.42343711074968357734,     4.63033784615654529590,
    5.76949722146069140550,     3.64784832476320460504,
    1.27045825245236838258,     2.41780725177450611770E-1,
    2.27238449892691845833E-2,  7.74545014278341407640E-4 };
  double const1 = 0.180625;
  double const2 = 1.6;
  double d[8] = {
    1.0,                        2.05319162663775882187,
    1.67638483018380384940,     6.89767334985100004550E-1,
    1.48103976427480074590E-1,  1.51986665636164571966E-2,
    5.47593808499534494600E-4,  1.05075007164441684324E-9 };
  double e[8] = {
    6.65790464350110377720,     5.46378491116411436990,
    1.78482653991729133580,     2.96560571828504891230E-1,
    2.65321895265761230930E-2,  1.24266094738807843860E-3,
    2.71155556874348757815E-5,  2.01033439929228813265E-7 };
  double f[8] = {
    1.0,                        5.99832206555887937690E-1,
    1.36929880922735805310E-1,  1.48753612908506148525E-2,
    7.86869131145613259100E-4,  1.84631831751005468180E-5,
    1.42151175831644588870E-7,  2.04426310338993978564E-15 };
  double q;
  double r;
  double split1 = 0.425;
  double split2 = 5.0;
  double value;

  if (p <= 0.0)
  {
    value = -r8_huge ( );
    return value;
  }

  if (1.0 <= p)
  {
    value = r8_huge ( );
    return value;
  }

  q = p - 0.5;

  if( fabs (q) <= split1)
  {
    r = const1 - q * q;
    value = q * r8poly_value_horner ( 7, a, r ) 
              / r8poly_value_horner ( 7, b, r );
  }
  else
  {
    if (q < 0.0)
    {
      r = p;
    }
    else
    {
      r = 1.0 - p;
    }

    if (r <= 0.0)
    {
      value = r8_huge ( );
    }
    else
    {
      r = sqrt ( - log ( r ) );

      if (r <= split2)
      {
        r = r - const2;
        value = r8poly_value_horner ( 7, c, r ) 
              / r8poly_value_horner ( 7, d, r );
       }
       else
       {
         r = r - split2;
         value = r8poly_value_horner ( 7, e, r ) 
               / r8poly_value_horner ( 7, f, r );
      }
    }

    if (q < 0.0)
    {
      value = - value;
    }

  }

  return value;
}


double r8_huge(void)
{
  return HUGE_VAL; //usually defined in math.h or stdlib.h
}


// Evaluate polynomial using Horner's method.
double r8poly_value_horner (int m,double c[],double x)
{
  int i;
  double value;

  value=c[m];

  for(i=m-1;0<=i;i--)
  {
    value=value*x+c[i];
  }

  return value;
}