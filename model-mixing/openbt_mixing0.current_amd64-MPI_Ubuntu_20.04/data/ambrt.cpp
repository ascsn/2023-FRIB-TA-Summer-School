//     ambrt.cpp: Additive mean BART model class methods.
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


#include "ambrt.h" 
#include "brtfuns.h"
#include <iostream>
#include <map>
#include <vector>

using std::cout;
using std::endl;


//--------------------------------------------------
//a single iteration of the MCMC for brt model
void ambrt::draw(rn& gen)
{
  for(size_t j=0;j<m;j++) {
    //update this row of notjmus
    // for(size_t i=0;i<di->n;i++) {
    //   notjmus[j][i]=di->y[i]-f(i)+mb[j].f(i);  //res_j=y-sum_{k!=j}tree_j
    // }
   *divec[j]= *di;
   *divec[j]-= *getf();
   *divec[j]+= *mb[j].getf();

    // do the draw for jth component
    mb[j].draw(gen);
   
    // Update the in-sample predicted vector
    setf();
   
    // Update the in-sample residual vector
    setr();
  }
  // overall statistics from the subtrees.  Need to divide by m*N to get
  // useful numbers after the MCMC is done.
  if(mi.dostats) {
    resetstats();
    for(size_t j=0;j<m;j++)
      mb[j].addstats(mi.varcount,&mi.tavgd,&mi.tmaxd,&mi.tmind);
  }
}
//--------------------------------------------------
//slave controller for draw when using MPI
void ambrt::draw_mpislave(rn& gen)
{
  for(size_t j=0;j<m;j++) {
    *divec[j]= *di;
    /*
    cout << "here slave 1" << endl;
    cout << "yhat.size() = "<< yhat.size() << endl;
    cout << "di->n = " << (*di).n << endl;
    cout << "di->x = " << (*di).x << endl;
    cout << "di->y = " << (*di).y << endl;
    cout << "di->p = " << (*di).p << endl;
    */
    *divec[j]-= *getf();
    //cout << "here slave 2" << endl;
    *divec[j]+= *mb[j].getf();
    //cout << "here slave 3" << endl;
    // do the draw for jth component
    mb[j].draw_mpislave(gen);
    //cout << "here slave 4" << endl;
    // Update the in-sample predicted vector
    setf();
    //cout << "here slave 5" << endl;
    // Update the in-sample residual vector
    setr();
    //cout << "here slave 6" << endl;  
  }
}
//--------------------------------------------------
//adapt the proposal widths for perturb proposals,
//bd or rot proposals and b or d proposals.
void ambrt::adapt()
{
  for(size_t j=0;j<m;j++) {
#ifndef SILENT
    cout << "\nAdapt ambrt[" << j << "]:";
#endif
    mb[j].adapt();
  }
}
//--------------------------------------------------
//setdata for ambrt
void ambrt::setdata(dinfo *di) {
  this->di=di;

  // initialize notjsigmavs.
  for(size_t j=0;j<m;j++)
      notjmus[j].resize(this->di->n,0.0);
  for(size_t j=0;j<m;j++)
    for(size_t i=0;i<di->n;i++)
      notjmus[j][i]=this->di->y[i]/((double)m);
  
  for(size_t j=0;j<m;j++)
    divec[j]=new dinfo(this->di->p,this->di->n,this->di->x,&notjmus[j][0],this->di->tc);  
  
  // each mb[j]'s data is the appropriate row in notjmus
  for(size_t j=0;j<m;j++)
    mb[j].setdata(divec[j]);
  resid.resize(di->n);
  yhat.resize(di->n);
  setf();
  setr();
  
  
}
//--------------------------------------------------
//set vector of predicted values for psbrt model
void ambrt::local_setf(diterator& diter)
{
   for(;diter<diter.until();diter++) {
      yhat[*diter]=0.0;
      for(size_t j=0;j<m;j++)
        yhat[*diter]+=mb[j].f(*diter);
   }
}
//--------------------------------------------------
//set vector of residuals for psbrt model
void ambrt::local_setr(diterator& diter)
{
   for(;diter<diter.until();diter++) {
      resid[*diter]=di->y[*diter]-f(*diter);
   }
}
//--------------------------------------------------
//predict the response at the (npred x p) input matrix *x
//Note: the result appears in *dipred.y.
void ambrt::local_predict(diterator& diter)
{
  tree::tree_p bn;
  double temp;

  for(;diter<diter.until();diter++) {
    temp=0.0;
    for(size_t j=0;j<m;j++) {
      bn = mb[j].t.bn(diter.getxp(),*xi);
      temp+=bn->gettheta();
    }
    diter.sety(temp);
  }
}
void ambrt::local_savetree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
     std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
  size_t indx=iter*m;
  for(size_t i=(indx+(size_t)beg);i<(indx+(size_t)end);i++) {
    nn[i]=mb[i-indx].t.treesize();
    id[i].resize(nn[i]);
    v[i].resize(nn[i]);
    c[i].resize(nn[i]);
    theta[i].resize(nn[i]);
    mb[i-indx].t.treetovec(&id[i][0],&v[i][0],&c[i][0],&theta[i][0]);
  }
}
void ambrt::local_loadtree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
  size_t indx=iter*m;
  for(size_t i=(indx+(size_t)beg);i<(indx+(size_t)end);i++)
    mb[i-indx].t.vectotree(nn[i],&id[i][0],&v[i][0],&c[i][0],&theta[i][0]);
}

//--------------------------------------------------
//pr for brt
void ambrt::pr()
{
   std::cout << "***** ambrt object:\n";
   cout << "Number of trees in product representation:" << endl;
   cout << "        m:   m=" << m << endl;
   cout << "Conditioning info on each individual tree:" << endl;
   cout << "   mean:   tau=" << ci.tau << endl;
   if(!ci.sigma)
     cout << "         sigma=[]" << endl;
   else
     cout << "         sigma=[" << ci.sigma[0] << ",...," << ci.sigma[di->n-1] << "]" << endl;
   brt::pr();
   cout << "**************Trees in sum representation*************:" << endl;
   for(size_t j=0;j<m;j++) mb[j].t.pr();
}

//--------------------------------------------------
//Collapse BART ensemble into one supertree
//The single supertree created will be stored in st.
void ambrt::collapseensemble()
{
   tree::npv bots;
   resetst();

   for(size_t j=1;j<m;j++)
   {
      st.getbots(bots);
      //collapse each tree j=1..m into the supertree
      for(size_t i=0;i<bots.size();i++)
         collapsetree(st,bots[i],this->gettree(j)); //mb[j]->t);
      bots.clear();
   }

}

//--------------------------------------------------
//Calculate Sobol Si, Sij and Tsi indices
//Draw is a p-dim vector to store the S_i's for all i=1,..,p variables.
//Based on Hiroguchi, Pratola and Santner (2020).
void ambrt::sobol(std::vector<double>& Si, std::vector<double>& Sij, 
                  std::vector<double>& TSi, double& V, std::vector<double>& minx,
                  std::vector<double>& maxx, size_t p)
{
  double term1,term2,term3;
  tree::npv bots;
  size_t vij=0;

  // concatentate all m tree's bottom nodes into one long vector
  for(size_t j=0;j<m;j++)
    mb[j].t.getbots(bots);

  size_t B=bots.size();

  std::vector<double> pxnoti(B);
  std::vector<std::vector<double> > a(p,std::vector<double>(B));
  std::vector<std::vector<double> > b(p,std::vector<double>(B));

  for(size_t i=0;i<p;i++)
    for(size_t k=0;k<B;k++) {
      int L,U;
      L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
      bots[k]->rgi(i,&L,&U);

      // Now we have the interval endpoints, put corresponding values in a,b matrices.
      if(L!=std::numeric_limits<int>::min()) a[i][k]=(*xi)[i][L];
      else a[i][k]=minx[i];
      if(U!=std::numeric_limits<int>::max()) b[i][k]=(*xi)[i][U];
      else b[i][k]=maxx[i];
    }

  for(size_t i=0;i<p;i++)
  {
    Si[i]=0.0;
    TSi[i]=0.0;
    for(size_t k=0;k<B;k++)
      pxnoti[k]=probxnoti_termk(i,k,a,b,minx,maxx);


    // Compute normalization constant, only need to do on the first outer loop iteration
    if(i==0) {
      V=0.0;
      for(size_t k=0;k<B;k++)
      for(size_t l=0;l<B;l++)
      {
        term1=bots[k]->gettheta()*bots[l]->gettheta();
        term2=probxall_termkl(k,l,a,b,minx,maxx);
        term3=pxnoti[k]*probxi_termk(0,k,a,b,minx,maxx)*pxnoti[l]*probxi_termk(0,l,a,b,minx,maxx);
        V+=term1*(term2-term3);
      }
    }

    // Compute Si and Tsi
    for(size_t k=0;k<B;k++)
    for(size_t l=0;l<B;l++)
    {
      term1=bots[k]->gettheta()*pxnoti[k];
      term2=bots[l]->gettheta()*pxnoti[l];
      term3=probxi_termkl(i,k,l,a,b,minx,maxx)-probxi_termk(i,k,a,b,minx,maxx)*probxi_termk(i,l,a,b,minx,maxx);
      Si[i]+=term1*term2*term3;

      term1=bots[k]->gettheta()*probxi_termk(i,k,a,b,minx,maxx);
      term2=bots[l]->gettheta()*probxi_termk(i,l,a,b,minx,maxx);
      term3=probxnoti_termkl(i,k,l,a,b,minx,maxx)-pxnoti[k]*pxnoti[l];
      TSi[i]+=term1*term2*term3;
    }
    TSi[i]=V-TSi[i];

    // Compute Sij
    double temp;
    for(size_t j=(i+1);j<p;j++)
    {
      temp=0.0;
      for(size_t k=0;k<B;k++)
      for(size_t l=0;l<B;l++)
      {
        term1=bots[k]->gettheta()*pxnoti[k]/probxi_termk(j,k,a,b,minx,maxx);
        term2=bots[l]->gettheta()*pxnoti[l]/probxi_termk(j,l,a,b,minx,maxx);
        term3=probxij_termkl(i,j,k,l,a,b,minx,maxx)-probxij_termk(i,j,k,a,b,minx,maxx)*probxij_termk(i,j,l,a,b,minx,maxx);
        temp+=term1*term2*term3;
      }
      Sij[vij]=temp;
      vij++;
    }
  }

  // subtract off the 1-way components from the Sij's to get final result.
  vij=0;
  for(size_t i=0;i<p;i++)
    for(size_t j=(i+1);j<p;j++) {
      Sij[vij] -= (Si[i]+Si[j]);
      vij++;
    }

// cout << "leaving sobol" << endl;
// for(size_t i=0;i<p;i++) cout << "S" << i+1 << "=" << Si[i]/V << endl;
// for(size_t i=0;i<p;i++) cout << "TS" << i+1 << "=" << TSi[i]/V << endl;
// for(size_t i=0;i<Sij.size();i++) cout << "Sij" << i+1 << "=" << Sij[i]/V << endl;
// cout << " done." << endl;

  bots.clear(); 
}


/* Original supertree version, broken.
void ambrt::sobol(std::vector<double>& Si, std::vector<double>& Sij, 
                  std::vector<double>& TSi, double& V, std::vector<double>& minx,
                  std::vector<double>& maxx, size_t p)
{
  double term1,term2,term3;
  tree::npv bots;

cout << "get st bots" << endl;
  st.getbots(bots);
cout << "supertree:" << endl;
st.pr();
cout << "allocate p=" << p << " x |bots|=" << bots.size() << " matrices" << endl;
  std::vector<double> pxnoti(bots.size());
  std::vector<std::vector<double> > a(p,std::vector<double>(bots.size()));
  std::vector<std::vector<double> > b(p,std::vector<double>(bots.size()));

  for(size_t i=0;i<p;i++)
    for(size_t k=0;k<bots.size();k++) {
      int L,U;
cout << "xi[" << i << "].size=" << (*xi)[i].size() << endl;
      L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
      bots[k]->rgi(i,&L,&U);

      // Slightly different than regular use of rg: we want the entire interval
      // in xi, not the eligble cutpoints that remain in the interval.
      // So we have to potentially +/- 1 fo U/L respectively.
     // L=std::max(0,L-1);
     // U=std::min((int)((*xi)[i].size()-1),U+1);
cout << "i=" << i << " k=" << k << " L=" << L << " U=" << U << endl;
      // Now we have the interval endpoints, put corresponding values in a,b matrices.
      if(L!=std::numeric_limits<int>::min()) a[i][k]=(*xi)[i][L];
      else a[i][k]=minx[i];
      if(U!=std::numeric_limits<int>::max()) b[i][k]=(*xi)[i][U];
      else b[i][k]=maxx[i];
    }
cout << "a,b matrices comlete" << endl;

  for(size_t i=0;i<p;i++)
  {
cout << "calculating indices for variable " << i << endl;
    Si[i]=0.0;
    TSi[i]=0.0;
    for(size_t k=0;k<bots.size();k++)
      pxnoti[k]=probxnoti_termk(i,k,a,b,minx,maxx);


    // Compute normalization constant, only need to do on the first outer loop iteration
    if(i==0) {
      V=0.0;
      for(size_t k=0;k<bots.size();k++)
      {
        V+=bots[k]->gettheta()*bots[k]->gettheta()*pxnoti[k]*probxi_termk(0,k,a,b,minx,maxx);
      }
cout << "V=" << V << endl;
    }

    // Compute Si and Tsi
    for(size_t k=0;k<bots.size();k++)
    for(size_t l=0;l<bots.size();l++)
    {
      term1=bots[k]->gettheta()*pxnoti[k];
      term2=bots[l]->gettheta()*pxnoti[l];
      term3=probxi_termkl(i,k,l,a,b,minx,maxx)-probxi_termk(i,k,a,b,minx,maxx)*probxi_termk(i,l,a,b,minx,maxx);
if(term3<0) cout << "Sinegative term 3=" << term3 << endl;
      Si[i]+=term1*term2*term3;

      term1=bots[k]->gettheta()*probxi_termk(i,k,a,b,minx,maxx);
      term2=bots[l]->gettheta()*probxi_termk(i,l,a,b,minx,maxx);
      term3=probxnoti_termkl(i,k,l,a,b,minx,maxx)-pxnoti[k]*pxnoti[l];
if(term3<0) cout << "Tsinegative term 3" << endl;
      TSi[i]+=term1*term2*term3;
cout << "Si=" << Si[i] << " TSi=" << TSi[i] << endl;
    }

    // Compute Sij
    for(size_t j=(i+1);j<p;j++)
    {
      double temp=0.0;
      for(size_t k=0;k<bots.size();k++)
      for(size_t l=0;l<bots.size();l++)
      {
        term1=bots[k]->gettheta()*pxnoti[k]/probxi_termk(j,k,a,b,minx,maxx);
        term2=bots[l]->gettheta()*pxnoti[l]/probxi_termk(j,l,a,b,minx,maxx);
        term3=probxij_termkl(i,j,k,l,a,b,minx,maxx)-probxij_termk(i,j,k,a,b,minx,maxx)*probxij_termk(i,j,l,a,b,minx,maxx);
if(term3<0) cout << "Sijnegative term 3" << endl;
        temp+=term1*term2*term3;
      }
      Sij.push_back(temp);
cout << "Sij=" << temp << endl;
    }

  }
cout << "leaving sobol" << endl;
for(size_t i=0;i<p;i++) cout << "S" << i+1 << "=" << Si[i]/V << endl;
for(size_t i=0;i<p;i++) cout << "TS" << i+1 << "=" << TSi[i]/V << endl;
for(size_t i=0;i<Sij.size();i++) cout << "Sij" << i+1 << "=" << Sij[i]/V << endl;
cout << " done." << endl;
}
*/



//--------------------------------------------------
//Calculate Pareto Front and Pareto Set for Multiobjective Optimization using
//fitted BART models.
//Currently we only handle the bi-objective case -- ie, 2 objective functions
//represented as BART models.
//Draw is a p-dim vector to store the S_i's for all i=1,..,p variables.
//Based on Hiroguchi, Pratola and Santner (2020).
void ambrt::ens2rects(std::vector<std::vector<double> >& asol, std::vector<std::vector<double> >& bsol, 
                  std::vector<double>& thetasol, std::vector<double>& minx,
                  std::vector<double>& maxx, size_t p)
{
  tree::npv bots0, botsnext;
  std::vector<std::vector<double> > a0,b0,anext,bnext;
  std::vector<double> theta0, thetanext;
  std::vector<double> aout(p);
  std::vector<double> bout(p);
  size_t B0,Bnext;

  // Start with the first tree in the ensemble.
  mb[0].t.getbots(bots0);
  B0=bots0.size();
  a0.resize(B0, std::vector<double>(p));//  a0.resize(B0,p);
  b0.resize(B0, std::vector<double>(p));
  theta0.resize(B0);
  // Convert bots0 nodes from first tree into hyperrectangles in 
  // the a0,b0 vectors and corresponding thetas in theta0 vector.
  for(size_t i=0;i<p;i++)
    for(size_t k=0;k<B0;k++) {
      int L,U;
      L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
      bots0[k]->rgi(i,&L,&U);

      // Now we have the interval endpoints, put corresponding values in a,b matrices.
      if(L!=std::numeric_limits<int>::min()) a0[k][i]=(*xi)[i][L];
      else a0[k][i]=minx[i];
      if(U!=std::numeric_limits<int>::max()) b0[k][i]=(*xi)[i][U];
      else b0[k][i]=maxx[i];
    }
  for(size_t k=0;k<B0;k++) theta0[k]=bots0[k]->gettheta();
  bots0.clear();

  // the solution so far is just a0,b0,theta0
  asol=a0; bsol=b0; thetasol=theta0;
  a0.clear(); b0.clear(); theta0.clear();

  // Now the main loop -- we will loop over trees 1..m, each time converting
  // tree j to the same a,b,theta vector format and then taking the pairwise
  // intersection of the current solution in a0,b0,theta0 with the next tree to add.
  for(size_t j=1;j<m;j++) {
    // get terminal nodes from tree j
    mb[j].t.getbots(botsnext);
    Bnext=botsnext.size();
    anext.resize(Bnext,std::vector<double>(p));
    bnext.resize(Bnext,std::vector<double>(p));
    thetanext.resize(Bnext);

    // convert tree j to a,b,theta vector format
    for(size_t i=0;i<p;i++)
      for(size_t k=0;k<Bnext;k++) {
        int L,U;
        L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
        botsnext[k]->rgi(i,&L,&U);

        // Now we have the interval endpoints, put corresponding values in a,b matrices.
        if(L!=std::numeric_limits<int>::min()) anext[k][i]=(*xi)[i][L];
        else anext[k][i]=minx[i];
        if(U!=std::numeric_limits<int>::max()) bnext[k][i]=(*xi)[i][U];
        else bnext[k][i]=maxx[i];
      }
    for(size_t k=0;k<Bnext;k++) thetanext[k]=botsnext[k]->gettheta();
    botsnext.clear();


    // now intersect the solution we have so far with this j'th tree and
    // update the resulting stored solution
    a0=asol; b0=bsol; theta0=thetasol;
    asol.clear(); bsol.clear(); thetasol.clear();
    B0=a0.size();

    for(size_t k=0;k<B0;k++)
      for(size_t l=0;l<Bnext;l++)
      {
        // if the rectangles defined by a0[k],b0[k] and anext[l],bnext[l] intersect
        if(probxall_termkl_rect(k,l,a0,b0,anext,bnext,minx,maxx,aout,bout)>0.0) { 
          asol.push_back(aout);
          bsol.push_back(bout);
          thetasol.push_back(theta0[k]+thetanext[l]);
        }
      }

    // reset anext,bnext,thetanext
    anext.clear(); bnext.clear(); thetanext.clear();
    // reset a0,b0,theta0
    a0.clear(); b0.clear(); theta0.clear();
  } // end of the main loop

}
