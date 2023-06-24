//     tree.cpp: BT tree class methods.
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


#include <string>
#include <map>
#include "tree.h"
#include "Eigen/Dense"

//--------------------
// node id
size_t tree::nid() const 
{
   if(!p) return 1; //if you don't have a parent, you are the top
   if(this==p->l) return 2*(p->nid()); //if you are a left child
   else return 2*(p->nid())+1; //else you are a right child
}
//--------------------
tree::tree_p tree::getptr(size_t nid)
{
   if(this->nid() == nid) return this; //found it
   if(l==0) return 0; //no children, did not find it
   tree_p lp = l->getptr(nid);
   if(lp) return lp; //found on left
   tree_p rp = r->getptr(nid);
   if(rp) return rp; //found on right
   return 0; //never found it
}
// Vectorized nput/output methods to support saving/loading to R.
// we require that id,v,c,theta are already of dim numnodesx1 vectors.
void tree::treetovec(int* oid, int* ov, int* oc, double* otheta)
{
   tree::cnpv nds;
   this->getnodes(nds);
   for(size_t i=0;i<nds.size();i++) {
      oid[i]=(int)nds[i]->nid();
      ov[i]=(int)nds[i]->getv();
      oc[i]=(int)nds[i]->getc();
      otheta[i]=nds[i]->gettheta();
   }
}
// nn is number of nodes
void tree::vectotree(size_t inn, int* iid, int* iv, int* ic, double* itheta)
{
   size_t itid,ipid;                     //itid: id of current node, ipid: parent's id
   std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id

   this->tonull(); // obliterate old tree (if there)

   //first node has to be the top one
   pts[1] = this; //careful! this is not the first pts, it is pointer of id 1.
   this->setv((size_t)iv[0]); this->setc((size_t)ic[0]); this->settheta(itheta[0]);
   this->p=0;

   //now loop through the rest of the nodes knowing parent is already there.
   for(size_t i=1;i!=inn;i++) {
      tree::tree_p np = new tree;
      np->v = (size_t)iv[i]; np->c=(size_t)ic[i]; np->theta=itheta[i];
      itid = (size_t)iid[i];
      pts[itid] = np;
      ipid = itid/2;
      // set pointers
      if(itid % 2 == 0) { //left child has even id
         pts[ipid]->l = np;
      } else {
         pts[ipid]->r = np;
      }
      np->p = pts[ipid];
   }
}
//--------------------
//add children to  bot node nid
bool tree::birth(size_t nid,size_t v, size_t c, double thetal, double thetar)
{
   tree_p np = getptr(nid);
   if(np==0) {
      std::cout << "error in birth: bottom node not found\n";
      return false; //did not find note with that nid
   }
   if(np->l!=0) {
      std::cout << "error in birth: found node has children\n";
      return false; //node is not a bottom node
   }

   //add children to bottom node np
   tree_p l = new tree;
   l->theta=thetal;
   tree_p r = new tree;
   r->theta=thetar;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;

   return true;
}
//--------------------
//depth of node
size_t tree::depth()
{
   if(!p) return 0; //no parents
   else return (1+p->depth());
}
//--------------------
//tree size
size_t tree::treesize()
{
   if(l==0) return 1;  //if bottom node, tree size is 1
   else return (1+l->treesize()+r->treesize());
}
//--------------------
//node type
char tree::ntype()
{
   //t:top, b:bottom, n:no grandchildren, i:internal
   if(!p) return 't';
   if(!l) return 'b';
   if(!(l->l) && !(r->l)) return 'n';
   return 'i';
}
//--------------------
//print out tree(pc=true) or node(pc=false) information
void tree::pr(bool pc) 
{
   size_t d = depth();
   size_t id = nid();

   size_t pid;
   if(!p) pid=0; //parent of top node
   else pid = p->nid();

   std::string pad(2*d,' ');
   std::string sp(", ");
   if(pc && (ntype()=='t')) {
      std::cout << "tree size: " << treesize() << std::endl;
   }
   std::cout << pad << "(id,parent): " << id << sp << pid;
   std::cout << sp << "(v,c): " << v << sp << c;
   std::cout << sp << "theta: " << theta;
   std::cout << sp << "type: " << ntype();
   std::cout << sp << "depth: " << depth();
   std::cout << sp << "pointer: " << this << std::endl;

   if(pc) {
      if(l) {
         l->pr(pc);
         r->pr(pc);
      }
   }
}
//--------------------
//kill children of  nog node nid
bool tree::death(size_t nid, double theta)
{
   tree_p nb = getptr(nid);
   if(nb==0) {
      std::cout << "error in death, nid invalid\n";
      return false;
   }
   if(nb->isnog()) {
      delete nb->l;
      delete nb->r;
      nb->l=0;
      nb->r=0;
      nb->v=0;
      nb->c=0;
      nb->theta=theta;
      return true;
   } else {
      std::cout << "error in death, node is not a nog node\n";
      return false;
   }
}
//--------------------
//is the node a nog node
bool tree::isnog() 
{
   bool isnog=true;
   if(l) {
      if(l->l || r->l) isnog=false; //one of the children has children.
   } else {
      isnog=false; //no children
   }
   return isnog;
}
bool tree::isleft() const
{
   bool isleft=false;
   if(p && p->l==this)
      isleft=true;

   return isleft;
}

bool tree::isright() const
{
   bool isright=false;
   if(p && p->r==this)
      isright=true;

   return isright;
}
//--------------------
size_t tree::nnogs() 
{
   if(!l) return 0; //bottom node
   if(l->l || r->l) { //not a nog
      return (l->nnogs() + r->nnogs());
   } else { //is a nog
      return 1;
   }
}
//--------------------
size_t tree::nbots() 
{
   if(l==0) { //if a bottom node
      return 1;
   } else {
      return l->nbots() + r->nbots();
   }
}
//--------------------
//get bottom nodes
void tree::getbots(npv& bv)
{
   if(l) { //have children
      l->getbots(bv);
      r->getbots(bv);
   } else {
      bv.push_back(this);
   }
}
//--------------------
//get nog nodes
void tree::getnogs(npv& nv)
{
   if(l) { //have children
      if((l->l) || (r->l)) {  //have grandchildren
         if(l->l) l->getnogs(nv);
         if(r->l) r->getnogs(nv);
      } else {
         nv.push_back(this);
      }
   }
}
//--------------------
// Get nodes that are not bottom nodes
void tree::getintnodes(npv& v)
{
   if(this->l) //left node has children
   {
      v.push_back(this);
      this->l->getintnodes(v);
      if(this->r->l)
         this->r->getintnodes(v);
   }
}
//--------------------
// Get nodes of tree minus root and leafs
void tree::getrotnodes(npv& v)
{
   if(!this->p && this->l)  //this is the root node and it has children, so lets get the rot nodes
   {
      this->l->getintnodes(v);
      this->r->getintnodes(v);
   }
}
//--------------------
//get all nodes
void tree::getnodes(npv& v)
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
void tree::getnodes(cnpv& v)  const
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
//--------------------
//get all nodes that split on variable v
void tree::getnodesonv(npv& v, size_t var)
{
   if(this->v==var)
      v.push_back(this);
   if(l) {
      l->getnodesonv(v,var);
      r->getnodesonv(v,var);
   }
}
//--------------------
//get all nodes that split on variable v at cutpoint c
void tree::getnodesonvc(npv& v, size_t var, size_t cut)
{
   if(this->v==var && this->c==cut)
      v.push_back(this);
   if(l) {
      l->getnodesonvc(v,var,cut);
      r->getnodesonvc(v,var,cut);
   }
}
//--------------------------------------------------
// Get path from proposal node back to root node.
void tree::getpathtoroot(npv& n)
{
   n.push_back(this);  //add current node to list
   if(this->p) //has a parent?  then not at root yet..
      this->p->getpathtoroot(n);
}
//--------------------------------------------------
// Get path from this node back to root node.
// Note that if this node is the root, it is not added to either nl or nr.
void tree::getpathtorootlr(npv& nl, npv& nr)
{
   if(this->p==0) { //already at root
      return;
   }
   if(this == p->l) { //this is a left child
      nl.push_back(this->p);  //add left split to list
   }
   if(this == p->r) { //this is a right child
      nr.push_back(this->p); //add right split to list
   }
   if(this->p) //has a parent?  then not at root yet..
      this->p->getpathtorootlr(nl,nr);
}

//--------------------------------------------------
// does x map along the path?
// for a path of size m having elements n_(m-1),n_(m-2),...,n_1,root, we
// initialize nodedx to path.size()-1 (here it will be nodedx=m-1) and determine
// if x follows the path by calling root.xonpath(path,nodedx,x,xi);
// If true, then we can get the remaining map by calling n_(m-1).bn(x,xi);
bool tree::xonpath(npv& path, size_t nodedx, double *x, xinfo& xi)
{
   tree_p next;

   if(nodedx==0) return true;

   if(x[v] < xi[v][c])
      next = this->l;
   else
      next = this->r;

   if(next == path[nodedx-1])
      return next->xonpath(path,nodedx-1,x,xi);
   else
      return false;
}
//--------------------------------------------------
// swap the left and right branches of this node in a tree
void tree::swaplr()
{
   tree::tree_p temp;
   temp=this->r;
   this->r=this->l;
   this->l=temp;
}
//--------------------
tree::tree_p tree::bn(double *x,xinfo& xi)
{
   if(l==0) return this; //no children
   if(x[v] < xi[v][c]) {
      return l->bn(x,xi);
   } else {
      return r->bn(x,xi);
   }
}
//--------------------
// Number of nodes splitting on var v
size_t tree::nuse(size_t v)
{
   npv nds;
   this->getnodes(nds);
   size_t nu=0; //return value
   for(size_t i=0;i!=nds.size();i++) {
      if(nds[i]->l && nds[i]->v==v) nu+=1;
   }
   return nu;
}
//--------------------
// find lower region
void tree::rl(size_t v, int *L)
{
   if(l==0) { //no children
      return;
   }
   if(this->v==v && (int)(this->c) >= (*L)) {
      *L=(int)(this->c)+1;
      r->rl(v,L);
   }
   else {
      l->rl(v,L);
      r->rl(v,L); 
   }   
}
//--------------------
// find upper region
void tree::ru(size_t v, int *U)
{
   if(l==0) { //no children
      return;
   }
   if(this->v==v && (int)(this->c) <= (*U)) {
      *U=(int)(this->c)-1;
      l->ru(v,U);
   }
   else {
      l->ru(v,U);
      r->ru(v,U);
   }
}
//--------------------
//find region for a given variable
void tree::rg(size_t v, int* L, int* U)
{
   if(this->p==0)  {
      return;
   }
   if((this->p)->v == v) { //does my parent use v?
      if(this == p->l) { //am I left or right child
         if((int)(p->c) <= (*U)) *U = (p->c)-1;
         p->rg(v,L,U);
      } else {
         if((int)(p->c) >= *L) *L = (p->c)+1;
         p->rg(v,L,U);
      }
   } else {
      p->rg(v,L,U);
   }
}
//--------------------
//find interval for a given variable
//This is very different from rg().  The rg() function returns still 
//eligible split points whereas rgi returns the raw interval splits
//that have been used.
//So far, this is only used in computing the Sobol indicies.
void tree::rgi(size_t v, int* L, int* U)
{
   if(this->p==0)  {
      return;
   }
   if((this->p)->v == v) { //does my parent use v?
      if(this == p->l) { //am I left or right child
         if((int)(p->c) <= (*U)) *U = (p->c);
         p->rgi(v,L,U);
      } else {
         if((int)(p->c) >= *L) *L = (p->c);
         p->rgi(v,L,U);
      }
   } else {
      p->rgi(v,L,U);
   }
}
//--------------------
//cut back to one node
void tree::tonull()
{
   size_t ts = treesize();
   //loop invariant: ts>=1
   while(ts>1) { //if false ts=1
      npv nv;
      getnogs(nv);
      for(size_t i=0;i<nv.size();i++) {
         delete nv[i]->l;
         delete nv[i]->r;
         nv[i]->l=0;
         nv[i]->r=0;
      }
      ts = treesize(); //make invariant true
   }
   k = 1;
   theta=0.0; thetavec = vxd::Zero(k);
   v=0;c=0;
   p=0;l=0;r=0;
}
//--------------------
//copy tree tree o to tree n
void tree::cp(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
{
   if(n->l) {
      std::cout << "cp:error node has children\n";
      return;
   }

   n->theta = o->theta;
   n->thetavec = o->thetavec;
   n->v = o->v;
   n->c = o->c;

   if(o->l) { //if o has children
      n->l = new tree;
      (n->l)->p = n;
      cp(n->l,o->l);
      n->r = new tree;
      (n->r)->p = n;
      cp(n->r,o->r);
   }
}
//--------------------------------------------------
//operators
tree& tree::operator=(const tree& rhs)
{
   if(&rhs != this) {
      tonull(); //kill left hand side (this)
      cp(this,&rhs); //copy right hand side to left hand side
   }
   return *this;
}
//--------------------------------------------------
//functions
std::ostream& operator<<(std::ostream& os, const tree& t)
{
   tree::cnpv nds;
   t.getnodes(nds);
   os << nds.size() << std::endl;
   for(size_t i=0;i<nds.size();i++) {
      os << nds[i]->nid() << " ";
      os << nds[i]->getv() << " ";
      os << nds[i]->getc() << " ";
      os << nds[i]->gettheta() << std::endl;
   }
   return os;
}
std::istream& operator>>(std::istream& is, tree& t)
{
   size_t tid,pid; //tid: id of current node, pid: parent's id
   std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id
   size_t nn; //number of nodes

   t.tonull(); // obliterate old tree (if there)

   //read number of nodes----------
   is >> nn;
   if(!is) {
      //cout << ">> error: unable to read number of nodes" << endl;
      return is;
   }

   //read in vector of node information----------
   std::vector<node_info> nv(nn);
   for(size_t i=0;i!=nn;i++) {
      is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta;
      if(!is) {
         //cout << ">> error: unable to read node info, on node  " << i+1 << endl;
         return is;
      }
   }
   //first node has to be the top one
   pts[1] = &t; //careful! this is not the first pts, it is pointer of id 1.
   t.setv(nv[0].v); t.setc(nv[0].c); t.settheta(nv[0].theta);
   t.p=0;

   //now loop through the rest of the nodes knowing parent is already there.
   for(size_t i=1;i!=nv.size();i++) {
      tree::tree_p np = new tree;
      np->v = nv[i].v; np->c=nv[i].c; np->theta=nv[i].theta;
      tid = nv[i].id;
      pts[tid] = np;
      pid = tid/2;
      // set pointers
      if(tid % 2 == 0) { //left child has even id
         pts[pid]->l = np;
      } else {
         pts[pid]->r = np;
      }
      np->p = pts[pid];
   }
   return is;
}
//--------------------
//add children to bot node *np
void tree::birthp(tree_p np,size_t v, size_t c, double thetal, double thetar)
{
   tree_p l = new tree;
   l->theta=thetal;
   tree_p r = new tree;
   r->theta=thetar;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;
}
//--------------------
//kill children of  nog node *nb
void tree::deathp(tree_p nb, double theta)
{
   delete nb->l;
   delete nb->r;
   nb->l=0;
   nb->r=0;
   nb->v=0;
   nb->c=0;
   nb->theta=theta;
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------
//Functions to accomodate for vector parameters
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//treetovec_vec: Tree-to-vector for vector parameters -- override scalar parameter function by including the value of k
void tree::treetovec(int* oid, int* ov, int* oc, double* othetavec, int k)
{
   tree::cnpv nds;
   vxd thetavec_temp(k);
   this->getnodes(nds);
   for(size_t i=0;i<nds.size();i++) {
      oid[i]=(int)nds[i]->nid();
      ov[i]=(int)nds[i]->getv();
      oc[i]=(int)nds[i]->getc();
      thetavec_temp = nds[i]->getthetavec();

      //Temporary fix for annoyance caused in rotation step -- check to see if any of the internal theta's are a zero vector of dimension 2
      //Right now, k = 2 by default, so when assigning a new theta to a rotated internal node, it gets a zero vec of dim 2. If we have more than 2 models then this is an issue
      if(thetavec_temp.size() != k){
         if(thetavec_temp == vxd::Zero(2)){
            thetavec_temp = vxd::Zero(k);
         }else{
            std::cout << "You have an error with rotate that is not fixed" << std::endl;
         }
      }

      for(int j = 0; j<k; j++){
         othetavec[i*k+j]=thetavec_temp(j); 
      }
   }
}

//---------------------------------------------------------------------
//Vector-to-tree for vector parameters -- ithetavec is a n*k vector if the others are pointers to n vectors 
// -- override scalar parameter function by including the value of k 
void tree::vectotree(size_t inn, int* iid, int* iv, int* ic, double* ithetavec, int k){
   size_t itid,ipid;                     //itid: id of current node, ipid: parent's id
   std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id
   vxd thetavec_temp(k);
   this->tonull(); // obliterate old tree (if there)

   //Populate the first theta vector 
   for(int j = 0; j<k; j++){
      thetavec_temp(j) = (double)ithetavec[j];
   }
   //first node has to be the top one
   pts[1] = this; //careful! this is not the first pts, it is pointer of id 1.
   this->setv((size_t)iv[0]); this->setc((size_t)ic[0]); this->setthetavec(thetavec_temp);
   this->p=0;

   //now loop through the rest of the nodes knowing parent is already there.
   for(size_t i=1;i!=inn;i++) {
      tree::tree_p np = new tree;

      //Populate the temp theta vector 
      for(int j = 0; j<k; j++){
         thetavec_temp(j) = (double)ithetavec[i*k + j];
      }
      
      np->v = (size_t)iv[i]; np->c=(size_t)ic[i]; np->thetavec=thetavec_temp;
      itid = (size_t)iid[i];
      pts[itid] = np;
      ipid = itid/2;
      // set pointers
      if(itid % 2 == 0) { //left child has even id
         pts[ipid]->l = np;
      } else {
         pts[ipid]->r = np;
      }
      np->p = pts[ipid];
   }
}

//---------------------------------------------------------------------
//Birth: add children to  bot node nid -- Another override
bool tree::birth(size_t nid,size_t v, size_t c, vxd thetavecl, vxd thetavecr)
{
   tree_p np = getptr(nid);
   if(np==0) {
      std::cout << "error in birth: bottom node not found\n";
      return false; //did not find note with that nid
   }
   if(np->l!=0) {
      std::cout << "error in birth: found node has children\n";
      return false; //node is not a bottom node
   }

   //add children to bottom node np
   tree_p l = new tree;
   l->thetavec=thetavecl;
   tree_p r = new tree;
   r->thetavec=thetavecr;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;

   return true;
}

//---------------------------------------------------------------------
//Death: kill children of  nog node nid -- another override
bool tree::death(size_t nid, vxd thetavec)
{
   tree_p nb = getptr(nid);
   if(nb==0) {
      std::cout << "error in death, nid invalid\n";
      return false;
   }
   if(nb->isnog()) {
      delete nb->l;
      delete nb->r;
      nb->l=0;
      nb->r=0;
      nb->v=0;
      nb->c=0;
      nb->thetavec=thetavec;
      return true;
   } else {
      std::cout << "error in death, node is not a nog node\n";
      return false;
   }
}

//---------------------------------------------------------------------
//Copy tree with vector parameters: new function defined for tree with vecotor parameters copy tree tree o to tree n 
void tree::cpvec(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
{
   if(n->l) {
      std::cout << "cp:error node has children\n";
      return;
   }

   n->thetavec = o->thetavec;
   n->v = o->v;
   n->c = o->c;

   if(o->l) { //if o has children
      n->l = new tree;
      (n->l)->p = n;
      cpvec(n->l,o->l);
      n->r = new tree;
      (n->r)->p = n;
      cpvec(n->r,o->r);
   }
}

//---------------------------------------------------------------------
//Birthp: add children to bot node *np -- another override 
void tree::birthp(tree_p np,size_t v, size_t c, vxd thetavecl, vxd thetavecr)
{
   tree_p l = new tree;
   l->thetavec=thetavecl;
   tree_p r = new tree;
   r->thetavec=thetavecr;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;
}

//---------------------------------------------------------------------
//deathp: kill children of  nog node *nb -- another override
void tree::deathp(tree_p nb, vxd thetavec)
{
   delete nb->l;
   delete nb->r;
   nb->l=0;
   nb->r=0;
   nb->v=0;
   nb->c=0;
   nb->thetavec=thetavec;
}

//---------------------------------------------------------------------
//print tree with vector parameters
//print out tree(pc=true) or node(pc=false) information
void tree::pr_vec(bool pc) 
{
   size_t d = depth();
   size_t id = nid();

   size_t pid;
   if(!p) pid=0; //parent of top node
   else pid = p->nid();

   std::string pad(2*d,' ');
   std::string sp(", ");
   if(pc && (ntype()=='t')) {
      std::cout << "tree size: " << treesize() << std::endl;
   }
   std::cout << pad << "(id,parent): " << id << sp << pid;
   std::cout << sp << "(v,c): " << v << sp << c;
   std::cout << sp << "thetavec: " << thetavec.transpose();
   std::cout << sp << "type: " << ntype();
   std::cout << sp << "depth: " << depth();
   std::cout << sp << "pointer: " << this << std::endl;

   if(pc) {
      if(l) {
         l->pr_vec(pc);
         r->pr_vec(pc);
      }
   }
}