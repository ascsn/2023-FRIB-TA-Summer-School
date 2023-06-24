//     brtfuns.h: Base BT model class help functions header file.
//     Copyright (C) 2012-2016 Matthew T. Pratola, Robert E. McCulloch and Hugh A. Chipman
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
//     Robert E. McCulloch: robert.e.mculloch@gmail.com
//     Hugh A. Chipman: hughchipman@gmail.com


#ifndef GUARD_brtfuns_h
#define GUARD_brtfuns_h

#include <iostream>
#include <list>
#include "tree.h"
#include "treefuns.h"
#include "brt.h"

using std::cout;
using std::endl;

//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc);
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, double pipb, tree::npv& goodbots);
//--------------------------------------------------
//bprop: function to generate birth proposal
void bprop(tree& x, xinfo& xi, brt::tprior& tp, double pb, tree::npv& goodbots, double& PBx, tree::tree_p& nx, size_t& v, size_t& c, double& pr, rn& gen);
//--------------------------------------------------
// death proposal
void dprop(tree& x, xinfo& xi, brt::tprior& tp, double pb, tree::npv& goodbots, double& PBx, tree::tree_p& nx, double& pr, rn& gen);
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo& xi, brt::tprior& tp);
//--------------------------------------------------
//calculate beginning and end points of data vector to be accessed in parallel computations
void calcbegend(int n, int my_rank, int thread_count, int* beg, int* end);

//--------------------------------------------------
// Functions to support change-of-variable proposal
//--------------------------------------------------
// update the correlation matrix for chgv move taking into account that not all
// variables may be eligible at pertnode.
void updatecormat(tree::tree_p pertnode, xinfo& xi, std::vector<std::vector<double> >& chgv);
//--------------------------------------------------
// renormalize the correlation matrix so that the probability of row sums to 1.
void normchgvrow(size_t row, std::vector<std::vector<double> >& chgv);
//--------------------------------------------------
// MPI version of the above 2 combined into 1 call.
void mpi_update_norm_cormat(size_t rank, size_t tc, tree::tree_p pertnode, xinfo& xi, std::vector<double>& chgvrow, int* chv_lwr, int* chv_upr);

//--------------------------------------------------
// randomly choose a new variable to transition to from oldv
size_t getchgv(size_t oldv, std::vector<std::vector<double> >& chgv, rn& gen);
size_t getchgvfromrow(size_t oldv, std::vector<double>& chgvrow, rn& gen);


//--------------------------------------------------
// Functions to support rotate proposal
//--------------------------------------------------
//setup the initial right rotation
void rotright(tree::tree_p n);
//--------------------------------------------------
//setup the initial left rotation
void rotleft(tree::tree_p n);
//--------------------------------------------------
//eliminate immediate dead ends from the rotate
void reduceleft(tree::tree_p n, size_t v, size_t c);
//--------------------------------------------------
//eliminate immediate dead ends from the rotate
void reduceright(tree::tree_p n, size_t v, size_t c);
//--------------------------------------------------
//split tree along variable v at cutpoint c retaining only 
//part of the tree that is ``left'' of this v,c rule
void splitleft(tree::tree_p t, size_t v, size_t c);
//--------------------------------------------------
//split tree along variable v at cutpoint c retaining only 
//part of the tree that is ``right'' of this v,c rule
void splitright(tree::tree_p t, size_t v, size_t c);
//--------------------------------------------------
//does an actual merge (randomly chosen) 
bool merge(tree::tree_p tl, tree::tree_p tr, tree::tree_p t, size_t v, size_t c, rn& gen);
//--------------------------------------------------
// only to get nways, not to actually do the merge.
bool mergecount(tree::tree_p tl, tree::tree_p tr, size_t v, size_t c, int* nways);
//--------------------------------------------------
// End of functions to support rotate proposal
//--------------------------------------------------

//--------------------------------------------------
// Functions to support collapsing BART ensemble into single supertree
//--------------------------------------------------
void collapsetree(tree& st, tree::tree_p t, tree::tree_p tprime);
void splitall(tree::tree_p t, tree::npv& tlefts, tree::npv& trights);

//--------------------------------------------------
// Functions to support calculation of Sobol indices for BART
// Based on Hiroguchi, Pratola and Santner (2020).
//--------------------------------------------------
double probxnoti_termk(size_t i, size_t k, std::vector<std::vector<double> >& a, std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx);
double probxi_termk(size_t i, size_t k, std::vector<std::vector<double> >& a, std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx);
double probxij_termk(size_t i, size_t j, size_t k, std::vector<std::vector<double> >& a, std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx);
double probxnoti_termkl(size_t i, size_t k, size_t l, std::vector<std::vector<double> >& a, std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx);
double probxi_termkl(size_t i, size_t k, size_t l, std::vector<std::vector<double> >& a, std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx);
double probxij_termkl(size_t i, size_t j, size_t k, size_t l, std::vector<std::vector<double> >& a, std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx);
double probxall_termkl(size_t k, size_t l, std::vector<std::vector<double> >& a, std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx);

//--------------------------------------------------
// This function only used for determining Pareto front/set.
// Based on Hiroguchi, Santner, Sun and Pratola (2020).
//--------------------------------------------------
double probxall_termkl_rect(size_t k, size_t l, std::vector<std::vector<double> >& a0, 
   std::vector<std::vector<double> >& b0, std::vector<std::vector<double> >& a1, 
   std::vector<std::vector<double> >& b1, std::vector<double>& minx, std::vector<double>& maxx, std::vector<double>& aout, std::vector<double>& bout);

std::vector<size_t> find_pareto_front(size_t start, size_t end, std::list<std::vector<double> > theta);
bool not_dominated(size_t index, std::vector<size_t> R, std::list<std::vector<double> > theta);

//--------------------------------------------------
// Functions to Model Mixing with BART and/or Vector Parameters
//--------------------------------------------------
void collapsetree_vec(tree& st, tree::tree_p t, tree::tree_p tprime); //collapse tree for vector parameter theta
void makefinfo(size_t k, int n, double *f, finfo &fi); 
void matrix_to_array(Eigen::MatrixXd &M, double *b);
void vector_to_array(Eigen::VectorXd &V, double *b);
void array_to_matrix(Eigen::MatrixXd &M, double *b);
void array_to_vector(Eigen::VectorXd &V, double *b);

//--------------------------------------------------
//Helper Functions for tree models with vector parameters & model mixing 
//--------------------------------------------------
//Compute spectral decomposition of covariance matrix -- update later 

#endif
