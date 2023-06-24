#ifndef GUARD_amxbrt_h
#define GUARD_amxbrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "mxbrt.h"


class amxbrt : public mxbrt {
public:
   //--------------------
   //classes
   // tprior and mcmcinfo are same as in brt
   // cinfo same as in mxbrt
   //--------------------
   //constructors/destructors
   amxbrt(): mxbrt(),st(0),m(200),mb(m),notjmus(m),divec(m) {}
   amxbrt(size_t im): mxbrt(),st(0),m(im),mb(m),notjmus(m),divec(m) {}
   //amxbrt(size_t im, size_t ik): mxbrt(ik),st(0),m(im),mb(m),notjmus(m),divec(m) {}
   virtual ~amxbrt() {
      if(!notjmus.empty()) {
         for(size_t j=0;j<m;j++) notjmus[j].clear();
         notjmus.clear();
         for(size_t j=0;j<m;j++) delete divec[j];
      }
      st.tonull();
   }

   //--------------------
   //methods
   void drawvec(rn& gen);
   void drawvec_mpislave(rn& gen);
   void adapt();
   void setmpirank(int rank) { this->rank = rank; for(size_t j=0;j<m;j++) mb[j].setmpirank(rank); }  //only needed for MPI
   void setmpicvrange(int* lwr, int* upr) { this->chv_lwr=lwr; this->chv_upr=upr; for(size_t j=0;j<m;j++) mb[j].setmpicvrange(lwr,upr); } //only needed for MPI
   void setci(double tau, double beta0, double* sigma) { ci.tau=tau; ci.sigma=sigma; ci.beta0=beta0; for(size_t j=0;j<m;j++) mb[j].setci(tau,beta0,sigma); }
   void setci(mxd invtau2_matrix, vxd beta_vec, double* sigma) {ci.invtau2_matrix.resize(invtau2_matrix.rows(),invtau2_matrix.rows()); ci.beta_vec.resize(beta_vec.rows());
      ci.invtau2_matrix=invtau2_matrix; ci.beta_vec=beta_vec; ci.sigma=sigma,ci.diffpriors = true; for(size_t j=0;j<m;j++) mb[j].setci(invtau2_matrix,beta_vec,sigma);} //Set when using prior's that differ by function
   void settc(int tc) { this->tc = tc; for(size_t j=0;j<m;j++) mb[j].settc(tc); }
   void setxi(xinfo *xi) { this->xi=xi; for(size_t j=0;j<m;j++) mb[j].setxi(xi); }
   void setfi(finfo *fi, size_t k) {this->fi = fi; this->k = k; this-> nsprior = false ;for(size_t j=0;j<m;j++) mb[j].setfi(fi,k); }
   void setfsd(finfo *fsd) {
      this->fisd = fsd; this->nsprior= true; 
      for(size_t j=0;j<m;j++) mb[j].setfsd(fsd);}
   void setk(size_t k) {this->k = k; for(size_t j=0;j<m;j++) mb[j].setk(k); }
   void setdata_mix(dinfo *di);
   void settp(double alpha, double beta) { tp.alpha=alpha;tp.beta=beta; for(size_t j=0;j<m;j++) mb[j].settp(alpha,beta); }
   tree::tree_p gettree(size_t i) { return &mb[i].t; }
   void setmi(double pbd, double pb, size_t minperbot, bool dopert, double pertalpha, double pchgv, std::vector<std::vector<double> >* chgv)
             { mi.pbd=pbd; mi.pb=pb; mi.minperbot=minperbot; mi.dopert=dopert;
               mi.pertalpha=pertalpha; mi.pchgv=pchgv; mi.corv=chgv; 
               for(size_t j=0;j<m;j++) mb[j].setmi(pbd,pb,minperbot,dopert,pertalpha,pchgv,chgv); }
   void setstats(bool dostats) { mi.dostats=dostats; for(size_t j=0;j<m;j++) mb[j].setstats(dostats); if(dostats) mi.varcount=new unsigned int[xi->size()]; }
   void pr_vec();
   // drawnodetheta, lm, add_observation_to_suff and newsinfo/newsinfovec unused here.

   // convert BART ensemble to single supertree
   void resetst() { st.tonull(); st=mb[0].t; } //copy mb0's tree to st.
   void collapseensemble();

   //Method for sampling homoscedastic variance for paramter sigma^2 -- not sure if this works
   void setvi(double nu, double lambda) {ci.nu = nu; ci.lambda = lambda; for(size_t j=0;j<m;j++) mb[j].setvi(nu, lambda);} //Use to change the defualt parameters

    //--------------------
    //data
    //--------------------------------------------------
    //stuff that maybe should be protected
    tree st;

protected:
    //--------------------
    //model information
    size_t m;  //number of trees in sum representation
    std::vector<mxbrt> mb;  // the vector of individual mu trees for sum representation
    //--------------------
    //data
    std::vector<std::vector<double> > notjmus;
    std::vector<dinfo*> divec;
    //--------------------
    //mcmc info
    //--------------------
    //methods
    virtual void local_setf_mix(diterator& diter);  //set the vector of predicted values
    virtual void local_setr_mix(diterator& diter);  //set the vector of residuals
    virtual void local_predict_mix(diterator& diter, finfo& fipred); // predict y at the (npred x p) settings *di.x
    virtual void local_get_mix_wts(diterator& diter, mxd& wts); // extract model weights at each *di.x settings
    virtual void local_get_mix_theta(diterator& diter, mxd& wts); // extract the terminal node parameters for the first *di.x settings
    virtual void local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                    std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
    virtual void local_loadtree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                    std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);

};


#endif