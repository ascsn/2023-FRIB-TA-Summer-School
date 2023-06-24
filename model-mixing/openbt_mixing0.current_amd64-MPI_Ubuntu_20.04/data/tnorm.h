//     tnorm.h: Truncated Normal helper functions for probit model.
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



// Draw from the truncated Normal distribution with left,right truncation points
// given by (lwr,upr).
// Based on https://people.sc.fsu.edu/~jburkardt/cpp_src/truncated_normal/truncated_normal.html
double gen_trunc_normal(double mu,double sig,double lwr, double upr,rn &gen);

// lwr is -Inf
double gen_left_trunc_normal(double mu,double sig,double upr,rn &gen);

// upr is +Inf
double gen_right_trunc_normal(double mu,double sig,double lwr,rn &gen);

// CDF of N(0,1) at x.
double normal_01_cdf(double x);

// inverse of N(0,1) CDF at p.
double normal_01_cdf_inv (double p);

// helper function
double r8_huge(void);

// helper function -- evaluate polynomial using Horner's method.
double r8poly_value_horner (int m,double c[],double x);