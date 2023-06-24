#include "amxbrt.h"
#include "brtfuns.h"
#include <iostream>
#include <map>
#include <vector>

using std::cout;
using std::endl;

//--------------------------------------------------
//a single iteration of the MCMC for brt model
void amxbrt::drawvec(rn& gen){
    for(size_t j=0;j<m;j++) {
        //Uses operators defined in dinfo on y to compute the jth residual Rij
        *divec[j] = *di;
        *divec[j]-= *getf(); //Yi - sum_k g(xi, Tk, Mk) 
        *divec[j]+= *mb[j].getf(); //Computes the jth residual in BART model

        //Draw parameter vector in the jth tree
        mb[j].drawvec(gen);
        
        // Update the in-sample predicted vector
        setf_mix();

        // Update the in-sample residual vector
        setr_mix();
    }
    // overall statistics from the subtrees.  Need to divide by m*N to get
    // useful numbers after the MCMC is done.
    if(mi.dostats) {
    resetstats();
    for(size_t j=0;j<m;j++){
        mb[j].addstats(mi.varcount,&mi.tavgd,&mi.tmaxd,&mi.tmind);
    }
  }
}

//--------------------------------------------------
//slave controller for draw when using MPI
void amxbrt::drawvec_mpislave(rn& gen){
    for(size_t j=0;j<m;j++) {
        //Get the jth resiudal
        *divec[j]= *di;
        *divec[j]-= *getf();
        *divec[j]+= *mb[j].getf();

        // do the draw for jth component
        mb[j].drawvec_mpislave(gen);
        
        // Update the in-sample predicted vector
        setf_mix();

        // Update the in-sample residual vector
        setr_mix();
    }
}

//--------------------------------------------------
//adapt the proposal widths for perturb proposals,
//bd or rot proposals and b or d proposals.
void amxbrt::adapt(){
  for(size_t j=0;j<m;j++) {
#ifndef SILENT
    cout << "\nAdapt ambrt[" << j << "]:";
#endif
    mb[j].adapt();
  }
}

//--------------------------------------------------
//setdata for amxbrt
void amxbrt::setdata_mix(dinfo *di) {
    this->di=di;
        
    // initialize notjsigmavs.
    for(size_t j=0;j<m;j++){
        notjmus[j].resize(this->di->n,0.0);
    }
    for(size_t j=0;j<m;j++){
        for(size_t i=0;i<di->n;i++){
            //notjmus[j][i]=this->di->y[i]/((double)m);
            notjmus[j][i]=this->di->y[i];
        }
    }
    for(size_t j=0;j<m;j++){
        divec[j]=new dinfo(this->di->p,this->di->n,this->di->x,&notjmus[j][0],this->di->tc); //constructing a new dinfo with notjmus[j][0] as the y value 
    }
    /*
    diterator diter(divec[0]);
    for(;diter<diter.until();diter++){
        cout << diter.getx() << " ------- " << *diter << " ------- " << diter.gety() << endl;
    }
    */
    // each mb[j]'s data is the appropriate row in notjmus
    for(size_t j=0;j<m;j++){
        mb[j].setdata_mix(divec[j]); //setdata_mix is a method of mb[j] which is a member of mxbrt class. This is different than setdata_mix in mxbrt
    }
    resid.resize(di->n);
    yhat.resize(di->n);
    setf_mix();
    setr_mix();
}

//--------------------------------------------------
//set vector of predicted values for psbrt model
void amxbrt::local_setf_mix(diterator& diter){
   for(;diter<diter.until();diter++){
      yhat[*diter]=0.0;
      for(size_t j=0;j<m;j++)
        yhat[*diter]+=mb[j].f(*diter); //sum of trees - add the fitted value from each tree
   }
}

//--------------------------------------------------
//set vector of residuals for psbrt model
void amxbrt::local_setr_mix(diterator& diter){
   for(;diter<diter.until();diter++) {
      resid[*diter]=di->y[*diter]-f(*diter);
   }
}

//--------------------------------------------------
//predict the response at the (npred x p) input matrix *x
//Note: the result appears in *dipred.y.
void amxbrt::local_predict_mix(diterator& diter, finfo& fipred){
    tree::tree_p bn;
    double temp;
    vxd thetavec_temp(k); 
    for(;diter<diter.until();diter++) {
        //Initialize to zero 
        temp = 0;
        thetavec_temp = vxd::Zero(k);
        
        for(size_t j=0;j<m;j++) {
            bn = mb[j].t.bn(diter.getxp(),*xi);
            thetavec_temp = bn->getthetavec();
            temp = temp + fipred.row(*diter)*thetavec_temp;
        }
        //std::cout << temp << std::endl;
        diter.sety(temp);
    }
}

//--------------------------------------------------
//extract model weights
void amxbrt::local_get_mix_wts(diterator& diter, mxd& wts){
    tree::tree_p bn;
    vxd thetavec_temp(k);
    for(;diter<diter.until();diter++) {
        //Initialize to zero 
        thetavec_temp = vxd::Zero(k);
        //Get sum of trees for the model weights
        for(size_t j=0;j<m;j++) {
            bn = mb[j].t.bn(diter.getxp(),*xi);
            thetavec_temp = thetavec_temp + bn->getthetavec();
        }
        wts.col(*diter) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix. 
    }
}

//--------------------------------------------------
//extract terminal node parameters for a specific point
void amxbrt::local_get_mix_theta(diterator& diter, mxd& wts){
    tree::tree_p bn;
    vxd thetavec_temp(k);
    bool enter = true;
    for(;diter<diter.until();diter++) {
        //Initialize to zero 
        thetavec_temp = vxd::Zero(k);
        //Get sum of trees for the model weights
        if(enter){
            for(size_t j=0;j<m;j++) {
                bn = mb[j].t.bn(diter.getxp(),*xi);
                thetavec_temp = bn->getthetavec();
                wts.col(j) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix.
            }
            enter = false;
        }
        
    }
}

//--------------------------------------------------
//Local Save tree
void amxbrt::local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
    std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta){
    size_t indx=iter*m;
    for(size_t i=(indx+(size_t)beg);i<(indx+(size_t)end);i++) {
        nn[i]=mb[i-indx].t.treesize();
        id[i].resize(nn[i]);
        v[i].resize(nn[i]);
        c[i].resize(nn[i]);
        theta[i].resize(k*nn[i]);
        mb[i-indx].t.treetovec(&id[i][0],&v[i][0],&c[i][0],&theta[i][0],k);
    }
}

//--------------------------------------------------
//Local load tree
void amxbrt::local_loadtree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta){
  size_t indx=iter*m;
  for(size_t i=(indx+(size_t)beg);i<(indx+(size_t)end);i++)
    mb[i-indx].t.vectotree(nn[i],&id[i][0],&v[i][0],&c[i][0],&theta[i][0],k);
}

//--------------------------------------------------
//pr for brt
void amxbrt::pr_vec()
{
   std::cout << "***** ambrt object:\n";
   cout << "Number of trees in product representation:" << endl;
   cout << "        m:   m=" << m << endl;
   cout << "Conditioning info on each individual tree:" << endl;
   cout << "   mean:   tau=" << ci.tau << endl;
   cout << "   mean:   beta0 =" << ci.beta0 << endl;
   if(!ci.sigma)
     cout << "         sigma=[]" << endl;
   else
     cout << "         sigma=[" << ci.sigma[0] << ",...," << ci.sigma[di->n-1] << "]" << endl;
   brt::pr_vec();
   cout << "**************Trees in sum representation*************:" << endl;
   for(size_t j=0;j<m;j++) mb[j].t.pr_vec();
}

//--------------------------------------------------
//Collapse BART ensemble into one supertree
//The single supertree created will be stored in st.
void amxbrt::collapseensemble()
{
   tree::npv bots;
   resetst();

   for(size_t j=1;j<m;j++)
   {
      st.getbots(bots);
      //collapse each tree j=1..m into the supertree
      for(size_t i=0;i<bots.size();i++)
         collapsetree_vec(st,bots[i],this->gettree(j)); //mb[j]->t);
      bots.clear();
   }

}