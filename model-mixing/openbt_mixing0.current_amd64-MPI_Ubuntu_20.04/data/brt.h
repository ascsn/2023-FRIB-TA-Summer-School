//     brt.h: Base BT model class definition.
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


#ifndef GUARD_brt_h
#define GUARD_brt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"

#ifdef _OPENMP
#   include <omp.h>
#endif

#ifdef _OPENMPI
#   include <mpi.h>
#   include <crn.h>
#   define SIZE_UINT1 16  // sizeof(unsigned int)
#   define SIZE_UINT2 32  // sizeof(unsigned int)*2
#   define SIZE_UINT3 48  // sizeof(unsigned int)*3
#   define SIZE_UINT4 64  // sizeof(unsigned int)*4
#   define SIZE_UINT5 80  // sizeof(unsigned int)*5
#   define SIZE_UINT6 96  // sizeof(unsigned int)*6
#   define MPI_TAG_BD_BIRTH_VC 10
#   define MPI_TAG_BD_BIRTH_VC_ACCEPT 11
#   define MPI_TAG_BD_BIRTH_VC_REJECT 12
#   define MPI_TAG_BD_DEATH_LR 20
#   define MPI_TAG_BD_DEATH_LR_ACCEPT 21
#   define MPI_TAG_BD_DEATH_LR_REJECT 22
#   define MPI_TAG_RESET_RNG 30
#   define MPI_TAG_PERTCV 40
#   define MPI_TAG_PERTCV_ACCEPT 41
#   define MPI_TAG_PERTCV_REJECT 42
#   define MPI_TAG_PERTCHGV 50
#   define MPI_TAG_PERTCHGV_ACCEPT 51
#   define MPI_TAG_PERTCHGV_REJECT 52
#   define MPI_TAG_PERTCHGV_MATRIX_UPDATE 53
#   define MPI_TAG_ROTATE 60
#   define MPI_TAG_ROTATE_ACCEPT 61
#   define MPI_TAG_ROTATE_REJECT 62
#endif

class sinfo { //sufficient statistics (will depend on end node model)
public:
   sinfo(): n(0) {}
   sinfo(const sinfo& is):n(is.n) {}
   virtual ~sinfo() {}  //need this so memory is properly freed in derived classes.

   size_t n;
   // compound addition operator needed when adding suff stats
   virtual sinfo& operator+=(const sinfo& rhs) {
      n=n+rhs.n;
      return *this;
   }
   // assignment operator for suff stats
   virtual sinfo& operator=(const sinfo& rhs)
   {
      if(&rhs != this) {
         this->n = rhs.n;
      }
      return *this; 
   }
   // addition opertor is defined in terms of compound addition
   const sinfo operator+(const sinfo& other) const {
      sinfo result = *this; //copy of myself.
      result += other;
      return result;
   }
};

class brt {
public:
   //--------------------
   //classes
   class tprior { //prior on the tree structure
   public:
      tprior(): alpha(.95),beta(1.0) {}
      //prob(split) = alpha/(1+d)^beta, where d is node depth.
      double alpha;
      double beta;
   };
   class mcmcinfo { //algorithm parameters (eg move probabilities)
   public:
      mcmcinfo(): pbd(1.0),pb(.5),minperbot(5),dopert(true),pertalpha(0.1),pertproposal(1),
                  pertaccept(0),rotproposal(0),rotaccept(1),bproposal(0),baccept(1),dproposal(0),
                  daccept(1),pchgv(0.2),chgvproposal(1),chgvaccept(0),corv(0),
                  dostats(false),varcount(0),tavgd(0.0),tmaxd(0) {} 
      double pbd;
      double pb;
      size_t minperbot;
      bool dopert;
      double pertalpha;
      size_t pertproposal;  //number of perturb proposals
      size_t pertaccept;    //number of accepted perturb proposals
      size_t rotproposal;   //number of rotate proposals
      size_t rotaccept;     //number of accepted rotate proposals
      size_t bproposal;     //number of birth proposals
      size_t baccept;       //number of birth accepts
      size_t dproposal;     //number of death proposals
      size_t daccept;       //number of death accepts
      double pchgv;         //probability of change of variable proposal.  Probability of perturb proposal is 1-pchgv
      size_t chgvproposal;  //number of change of variable proposals
      size_t chgvaccept;    //number of accepted chnage of varialbe proposals
      std::vector<std::vector<double> >* corv; //initial proposal distribution for changing variables in pert
      //statistics
      bool dostats;    //keep track of statistics yes or no?
      unsigned int* varcount;//count number of splits in tree on each variable
      double tavgd;         //average tree depth
      unsigned int tmaxd;   //maximum tree depth
      unsigned int tmind;   //minimum tree depth
   };
   class cinfo { //parameters for end node model prior
   };
   //--------------------
   //constructors/destructors
   brt():t(0.0),tp(),xi(0),ci(),di(0),mi(),tc(1),rank(0) {}
   //brt(size_t ik):t(vxd::Zero(ik)),tp(),xi(0),ci(),di(0),mi(),tc(1),rank(0) {}
   virtual ~brt() { if(mi.varcount) delete[] mi.varcount; }
   //--------------------
   //methods
   void settc(int tc) {this->tc = tc;}  // this is numslaves for MPI, or numthreads for OPEN_MP
   void setmpirank(int rank) {this->rank = rank;}  //only needed for MPI
   void setmpicvrange(int* lwr, int* upr) {this->chv_lwr=lwr; this->chv_upr=upr;} //only needed for MPI
   void setxi(xinfo *xi) {this->xi=xi; this->ncp1=2.0;
                          for(size_t i=0;i<(*xi).size();i++) 
                            if(this->ncp1<(double)((*xi)[i].size()+1.0))
                              this->ncp1=(double)((*xi)[i].size()+1.0);
                         }
   void setdata(dinfo *di) {this->di=di;resid.resize(di->n);yhat.resize(di->n);setf();setr();}
   void pr();
   void settp(double alpha, double beta) {tp.alpha=alpha;tp.beta=beta;}
   void setmi(double pbd, double pb, size_t minperbot, bool dopert, double pertalpha, double pchgv, std::vector<std::vector<double> >* chgv)
             {mi.pbd=pbd; mi.pb=pb; mi.minperbot=minperbot; mi.dopert=dopert;
              mi.pertalpha=pertalpha; mi.pchgv=pchgv; mi.corv=chgv; }
   void setstats(bool dostats) { mi.dostats=dostats; if(dostats) mi.varcount=new unsigned int[xi->size()]; }
   void getstats(unsigned int* vc, double* tad, unsigned int* tmd, unsigned int* tid) { *tad=mi.tavgd; *tmd=mi.tmaxd; *tid=mi.tmind; for(size_t i=0;i<xi->size();i++) vc[i]=mi.varcount[i]; }
   void addstats(unsigned int* vc, double* tad, unsigned int* tmd, unsigned int* tid) { *tad+=mi.tavgd; *tmd=std::max(*tmd,mi.tmaxd); *tid=std::min(*tid,mi.tmind); for(size_t i=0;i<xi->size();i++) vc[i]+=mi.varcount[i]; }
   void resetstats() { mi.tavgd=0.0; mi.tmaxd=0; mi.tmind=0; for(size_t i=0;i<xi->size();i++) mi.varcount[i]=0; }
   void setci() {}
   void draw(rn& gen);
   void draw_mpislave(rn& gen);
   void mpislave_bd(rn& gen);
   virtual sinfo* newsinfo() { return new sinfo; }
   virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
   virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new sinfo); return *si; }
   void setf();  //set the vector of predicted values
   void setr();  //set the vector of residuals
   double f(size_t i) { return yhat[i]; }  //get the i'th predicted value
   double r(size_t i) { return resid[i]; }  //get the i'th residual
   std::vector<double>* getf() { return &yhat; }
   std::vector<double>* getr() { return &resid; }
   void predict(dinfo* dipred); // predict y at the (npred x p) settings *di.x
//   void savetree(int* id, int* v, int* c, double* theta);  //save tree to vector output format
   void savetree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
//   void loadtree(size_t nn, int* id, int* v, int* c, double* theta);  //load tree from vector input format
   void loadtree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   //--------------------
   //data
   tree t;
   //--------------------------------------------------
   //stuff that maybe should be protected
   void bd(rn& gen);      //uses getsuff
   void pertcv(rn& gen);  //uses getpertsuff which in turn uses subsuff
   void drawtheta(rn& gen);
   void allsuff(tree::npv& bnv,std::vector<sinfo*>& siv);  //assumes brt.t is the root node
   void subsuff(tree::tree_p nx, tree::npv& bnv, std::vector<sinfo*>& siv); //does NOT assume brt.t is the root node.
                                                                           //Instead, uses the path from nx that it constructs.
   bool rot(tree::tree_p tnew, tree& x, rn& gen);  //uses subsuff
   void adapt();

   //--------------------------------------------------
   //Methods for vector parameters and model mixing
   //--------------------------------------------------
   //draw theta vector -- used for vector parameter
   void drawvec(rn& gen);
   void drawvec_mpislave(rn& gen);
   void drawthetavec(rn& gen);

   //Birth and Death for vector parameters
   void bd_vec(rn& gen);

   //Set the data, the vector of predicted values, and residuals
   void setdata_mix(dinfo *di) {this->di=di; resid.resize(di->n); yhat.resize(di->n); setf_mix(); setr_mix();}
   void setfi(finfo *fi, size_t k){this->fi = fi; this->k = k; this->t.thetavec.resize(k); this->t.thetavec=vxd::Zero(k);this->nsprior = false;} //sets the pointer for the f matrix and k as members of brt 
   void setk(size_t k){this->k = k; this->t.thetavec.resize(k); this->t.thetavec=vxd::Zero(k);} //sets the number of models for mixing--used in programs that do not need to read in function data (ex: mixingwts.cpp))
   void setfsd(finfo *fsd){this->fisd = fsd; this->nsprior = true;} //sets the function discrepancies 
   void setf_mix();
   void setr_mix(); 
   void predict_mix(dinfo* dipred, finfo* fipred); // predict y at the (npred x p) settings *di.x 
   void predict_mix_fd(dinfo* dipred, finfo* fipred, finfo* fpdmean, finfo* fpdsd, rn& gen); // predict y at the (npred x p) settings *di.x with functional discrepancy mean & sd
   void get_mix_wts(dinfo* dipred, mxd* wts);
   void get_mix_theta(dinfo* dipred, mxd* wts);
   void get_fi(){std::cout << "fi = \n" << *fi << std::endl;}
    
   //Print brt object with vector parameters
   void pr_vec(); 

   //Save and load the tree
   void savetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   void loadtree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta); 

protected:
   //--------------------
   //model information
   tprior tp; //prior on tree (alpha and beta)
   xinfo *xi; //cutpoints for each x.
   double ncp1; //number of cutpoints for each x plus 1.
   cinfo ci; //conditioning info (e.g. other parameters and prior and end node models)
   //--------------------
   //data
   dinfo *di; //n,p,x,y
   std::vector<double> yhat; //the predicted vector
   std::vector<double> resid; //the actual residual vector
   //--------------------
   //mcmc info
   mcmcinfo mi;
   //thread count
   int tc;
   //slave rank for MPI
   int rank;
   //vectors of length #slave nodes (tc) describing which variables each node handles when
   //updating mi.corv during an MPI change-of-variable proposal.
   int* chv_lwr;
   int* chv_upr;

   //number of trees -- is always 1 in the base class.
//   size_t m;
   //--------------------
   //methods
   virtual void add_observation_to_suff(diterator& diter, sinfo& si); //add in observation i (from di) into si (possibly using ci)
   void getsuff(tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir);  //assumes brt.t is the root node
   void getsuff(tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir);       //assumes brt.t is the root node
   void getchgvsuff(tree::tree_p pertnode, tree::npv& bnv, size_t oldc, size_t oldv, bool didswap, 
                  std::vector<sinfo*>& sivold, std::vector<sinfo*>& sivnew);     //uses subsuff
   void getpertsuff(tree::tree_p pertnode, tree::npv& bnv, size_t oldc,        //uses subsuff
                  std::vector<sinfo*>& sivold, std::vector<sinfo*>& sivnew);
   void local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir); 
   void local_getsuff(diterator& diter, tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir);
   virtual double lm(sinfo& si); //uses pi. 
   virtual double drawnodetheta(sinfo& si, rn& gen);
   void local_allsuff(diterator& diter, tree::npv& bnv,std::vector<sinfo*>& siv);
   void local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv);
   virtual void local_setf(diterator& diter);
   virtual void local_setr(diterator& diter);
   virtual void local_predict(diterator& diter);
//#  ifdef _OPENMP
   void local_ompgetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir);
   void local_ompgetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir);
   void local_ompallsuff(dinfo di, tree::npv bnv,std::vector<sinfo*>& siv);
   void local_ompsubsuff(dinfo di, tree::tree_p nx, tree::npv& path, tree::npv bnv,std::vector<sinfo*>& siv);
   void local_ompsetf(dinfo di);
   void local_ompsetr(dinfo di);
   void local_omppredict(dinfo dipred);

   void local_ompsavetree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   virtual void local_savetree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   void local_omploadtree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   virtual void local_loadtree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
//#  endif
//# ifdef _OPENMPI
   void local_mpigetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir);
   void local_mpigetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir);
   void local_mpiallsuff(diterator& diter, tree::npv& bnv,std::vector<sinfo*>& siv);
   virtual void local_mpi_reduce_allsuff(std::vector<sinfo*>& siv);
   virtual void local_mpi_sr_suffs(sinfo& sil, sinfo& sir);
   void mpi_resetrn(rn& gen);
   void local_mpisubsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv);


   //-------------------------------------------
   //Protected Model Mixing functions and data
   //-------------------------------------------
   finfo *fi; //pointer to the f matrix
   finfo *fisd; //pointer to the std matrix for fhat  
   bool nsprior; //True/False -- use stationary or nonstationy prior
   size_t k;
   
   virtual Eigen::VectorXd drawnodethetavec(sinfo& si, rn& gen);
   virtual void local_setf_mix(diterator& diter);
   virtual void local_setr_mix(diterator& diter);
   virtual void local_predict_mix(diterator& diter, finfo& fipred);
   virtual void local_get_mix_wts(diterator& diter, mxd& wts);
   virtual void local_get_mix_theta(diterator& diter, mxd& wts);
   
   // #ifdef _OPENMP
   //void local_ompgetsuff_mix(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir);
   //void local_ompgetsuff_mix(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir);
   //void local_ompallsuff_mix(dinfo di, tree::npv bnv,std::vector<sinfo*>& siv);
   //void local_ompsubsuff_mix(dinfo di, tree::tree_p nx, tree::npv& path, tree::npv bnv,std::vector<sinfo*>& siv);
   void local_ompsetf_mix(dinfo di);
   void local_ompsetr_mix(dinfo di);
   void local_omppredict_mix(dinfo dipred, finfo fipred);
   void local_ompget_mix_wts(dinfo dipred, mxd wts);
   void local_ompget_mix_theta(dinfo dipred, mxd wts);

   //Save and Load tree with vector parameters
   void local_ompsavetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   virtual void local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   void local_omploadtree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
   virtual void local_loadtree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);

};

#endif

