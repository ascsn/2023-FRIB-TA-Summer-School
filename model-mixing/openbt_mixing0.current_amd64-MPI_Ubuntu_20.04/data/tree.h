//     tree.h: BT tree class definition.
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


#ifndef GUARD_tree_h
#define GUARD_tree_h

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <vector>
#include "Eigen/Dense"

#include "rn.h"

//--------------------------------------------------
//xinfo xi, then xi[v][c] is the c^{th} cutpoint for variable v.
//left if x[v] < xi[v][c]
typedef std::vector<double> vec_d; //double vector
typedef std::vector<vec_d> xinfo; //vector of vectors, will be split rules

//--------------------------------------------------
//Defined for model mixing
typedef Eigen::VectorXd vxd; //used for eigen vectors with unfixed dimension
typedef Eigen::MatrixXd mxd; //used for eigen matricies with unfixed dimension 
typedef mxd finfo; //defined to be consistent with xinfo naming convention

//--------------------------------------------------
//info contained in a node, used by input operator
struct node_info {
   std::size_t id; //node id
   std::size_t v;  //variable
   std::size_t c;  //cut point
   double theta;   //univariate theta
   Eigen::VectorXd thetavec;   //multivariate theta
};

//--------------------------------------------------
class tree {
public:
   //friends--------------------
   friend std::istream& operator>>(std::istream&, tree&);
   //typedefs--------------------
   typedef tree* tree_p;
   typedef const tree* tree_cp;
   typedef std::vector<tree_p> npv; 
   typedef std::vector<tree_cp> cnpv;
   
   //prior
   //contructors,destructors--------------------
   //---include initialization of thetavec
   tree(): theta(0.0),k(2),thetavec(vxd::Zero(2)),v(0),c(0),p(0),l(0),r(0){}
   tree(const tree& n): theta(0.0),k(2),thetavec(vxd::Zero(2)),v(0),c(0),p(0),l(0),r(0) {cp(this,&n);}
   tree(double itheta): theta(itheta),k(2),thetavec(vxd::Zero(2)),v(0),c(0),p(0),l(0),r(0) {}
   tree(vxd itheta): theta(0.0),k(itheta.rows()),thetavec(itheta),v(0),c(0),p(0),l(0),r(0) {} //constructor for multivariate parameter

   void tonull(); //like a "clear", null tree has just one node
   ~tree() {tonull();}
   
   //operators----------
   tree& operator=(const tree&);
   
   //interface--------------------
   // Vectorized nput/output methods to support saving/loading to R.
   void treetovec(int* id, int* v, int* c, double* theta);
   void vectotree(size_t inn, int* id, int* v, int* c, double* theta);
   //set
   void settheta(double theta) {this->theta=theta;}
   void setv(size_t v) {this->v = v;}
   void setc(size_t c) {this->c = c;}
   //get
   double gettheta() const {return theta;}
   size_t getv() const {return v;}
   size_t getc() const {return c;}
   tree_p getp() {return p;}  
   tree_p getl() {return l;}
   tree_p getr() {return r;}
   //tree functions--------------------
   tree_p getptr(size_t nid); //get node pointer from node id, 0 if not there
   void pr(bool pc=true); //to screen, pc is "print children"
   size_t treesize(); //number of nodes in tree
   size_t nnogs();    //number of nog nodes (no grandchildren nodes)
   size_t nbots();    //number of bottom nodes
   bool birth(size_t nid, size_t v, size_t c, double thetal, double thetar);
   bool death(size_t nid, double theta);
   void birthp(tree_p np,size_t v, size_t c, double thetal, double thetar);
   void deathp(tree_p nb, double theta);
   void getbots(npv& bv);         //get bottom nodes
   void getnogs(npv& nv);         //get nog nodes (no granchildren)
   void getintnodes(npv& v);      //get interior nodes i.e. aren't bottom nodes
   void getrotnodes(npv& v);      //get rot nodes
   void getnodes(npv& v);         //get vector of all nodes
   void getnodes(cnpv& v) const;  //get vector of all nodes (const)
   void getnodesonv(npv& v, size_t var);  //get all nodes that split on variable v
   void getnodesonvc(npv& v, size_t var, size_t cut); //get all nodes that split on variable v at cutpoint c
   void getpathtoroot(npv& n);     //get all nodes from this node back to root of tree.
   void getpathtorootlr(npv& nl, npv& nr); //get all "left" and "right" nodes from this node back to root of tree EXCLUDING this.
   bool xonpath(npv& path, size_t nodedx, double *x, xinfo& xi);  //true if x follows path down tree, false otherwise
   void swaplr();                  //swap the left and right branches of this node in a tree
   tree_p bn(double *x,xinfo& xi); //find Bottom Node
   size_t nuse(size_t v);          // Number of nodes splitting on var v
   void rl(size_t v, int *L);      // find lower region
   void ru(size_t v, int *U);      // find upper region
   void rg(size_t v, int* L, int* U); //recursively find region [L,U] for var v
   void rgi(size_t v, int* L, int* U);  //recursively rind the inteval [L,U] for var v, used in Sobol indices calculations.
   //node functions--------------------
   size_t nid() const; //nid of a node
   size_t depth();  //depth of a node
   char ntype(); //node type t:top, b:bot, n:no grandchildren i:interior (t can be b)
   bool isnog();
   bool isleft() const;
   bool isright() const;
   //these are in public right now so brt::rot compiles
   double theta; //univariate double parameter
   size_t k; //Dimension of thetavec == number of models to be fixed
   vxd thetavec; //multivariate double parameter -- using Eigen VectorXd
   
   size_t v;
   size_t c;
   //tree structure
   tree_p p; //parent
   tree_p l; //left child
   tree_p r; //right child

   //Tree functions when using vector parameters-------------------
   // Vectorized nput/output methods to support saving/loading to R.
   void treetovec(int* id, int* v, int* c, double* thetavec, int k); //needs an input value of k
   void vectotree(size_t inn, int* id, int* iv, int* ic, double* ithetavec, int ik); //needs an input value of k
   //set and get
   void setthetavec(vxd thetavec) {this->thetavec=thetavec;}
   vxd getthetavec() const {return thetavec;}
   //Tree functions--birth and death (these are not renamed right now, the override with different datatype should work here)
   bool birth(size_t nid, size_t v, size_t c, vxd thetavecl, vxd thetavecr);
   bool death(size_t nid, vxd thetavec);
   void birthp(tree_p np,size_t v, size_t c, vxd thetavecl, vxd thetavecr);
   void deathp(tree_p nb, vxd thetavec);
   //print functions
   void pr_vec(bool pc=true); //to screen, pc is "print children"

private:
   //double theta; //univariate double parameter
   //rule: left if x[v] < xinfo[v][c]
   // size_t v;
   // size_t c;
   // //tree structure
   // tree_p p; //parent
   // tree_p l; //left child
   // tree_p r; //right child
   //utiity functions
   void cp(tree_p n,  tree_cp o); //copy tree
   void cpvec(tree_p n,  tree_cp o); //copy tree with vector parameters
};
std::istream& operator>>(std::istream&, tree&);
std::ostream& operator<<(std::ostream&, const tree&);

#endif
