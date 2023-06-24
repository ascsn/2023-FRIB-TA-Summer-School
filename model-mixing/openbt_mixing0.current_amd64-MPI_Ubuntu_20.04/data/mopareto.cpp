//     mopareto.cpp: Implement Pareto-front multiobjective optimization using OpenBT.
//     Copyright (C) 2020 Matthew T. Pratola, Akira Horiguchi
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
//     Akira Horiguchi: horiguchi.6@osu.edu


#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

#include <fstream>
#include <vector>
#include <limits>

#include "crn.h"
#include "tree.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mbrt.h"
#include "ambrt.h"
#include "psbrt.h"

using std::cout;
using std::endl;



// Calculate Pareto Front and Pareto Set given 2 trained BART models.
int main(int argc, char* argv[])
{
   std::string folder("");
   std::string folder2("");

   if(argc>1)
   {
      //argument on the command line is path to config file.
      folder=std::string(argv[1]);
      folder=folder+"/";
   }

   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config.mopareto");

   //model name, number of saved draws and number of trees
   std::string modelname;
   std::string modelname2;
   std::string xicore;

   //model name and xi
   conf >> modelname;
   conf >> modelname2;
   conf >> xicore;

   //location of the second fitted model
   conf >> folder2;
   folder2=folder2+"/";

   //number of saved draws and number of trees
   size_t nd;
   size_t m1;
   size_t mh1;
   size_t m2;
   size_t mh2;

   conf >> nd;
   conf >> m1;
   conf >> mh1;
   conf >> m2;
   conf >> mh2;

   //number of predictors
   size_t p;
   conf >> p;

   //min and max of predictors
   std::vector<double> minx(p);
   std::vector<double> maxx(p);
   for(size_t i=0;i<p;i++)
      conf >> minx[i];
   for(size_t i=0;i<p;i++)
      conf >> maxx[i];

   //global means of each response
   double fmean1, fmean2;
   conf >> fmean1;
   conf >> fmean2;

   //thread count
   int tc;
   conf >> tc;
   conf.close();


   //MPI initialization
   int mpirank=0;
#ifdef _OPENMPI
   int mpitc;
   MPI_Init(NULL,NULL);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
   MPI_Comm_size(MPI_COMM_WORLD,&mpitc);
#ifndef SILENT
   cout << "\nMPI: node " << mpirank << " of " << mpitc << " processes." << endl;
#endif
   if(tc<=1){
      cout << "Error: tc=" << tc << endl;
      MPI_Finalize();
      return 0; //need at least 2 processes! 
   } 
   if(tc!=mpitc) {
      cout << "Error: tc does not match mpitc" << endl;
      MPI_Finalize();
      return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
   }
// #else
//    if(tc!=1) return 0; //serial mode should have no slave threads!
#endif


   //--------------------------------------------------
   // Banner
   if(mpirank==0) {
      cout << endl;
      cout << "-----------------------------------" << endl;
      cout << "OpenBT Multiobjective Optimization using Pareto Front/Set CLI" << endl;
      cout << "Loading config file at " << folder << endl;
      cout << "Loading config file at " << folder2 << endl;
   }


   //--------------------------------------------------
   //make xinfo
   xinfo xi;
   xi.resize(p);

   for(size_t i=0;i<p;i++) {
      std::vector<double> xivec;
      double xitemp;

      std::stringstream xifss;
      std::string xifs;
      xifss << folder << xicore << (i+1);
      xifs=xifss.str();
      std::ifstream xif(xifs);
      while(xif >> xitemp)
         xivec.push_back(xitemp);
      xi[i]=xivec;
   }
#ifndef SILENT
   cout << "&&& made xinfo\n";
#endif

   //summarize input variables:
#ifndef SILENT
   if(mpirank==0)
      for(size_t i=0;i<p;i++)
      {
         cout << "Variable " << i << " has numcuts=" << xi[i].size() << " : ";
         cout << xi[i][0] << " ... " << xi[i][xi[i].size()-1] << endl;
      }
#endif



   // set up ambrt objects
   ambrt ambm1(m1);
   ambm1.setxi(&xi); //set the cutpoints for this model object
   ambrt ambm2(m2);
   ambm2.setxi(&xi); //set the cutpoints for this model object

   //setup psbrt objects
   psbrt psbm1(mh1);
   psbm1.setxi(&xi); //set the cutpoints for this model object
   psbrt psbm2(mh1);
   psbm2.setxi(&xi); //set the cutpoints for this model object



   //load first fitted model from file
#ifndef SILENT
   if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
   size_t ind,im,imh;
   std::ifstream imf(folder + modelname + ".fit");
   imf >> ind;
   imf >> im;
   imf >> imh;
#ifdef _OPENMPI
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(m1!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mh1!=imh) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(m1!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mh1!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
#endif

   size_t temp=0;
   imf >> temp;
   std::vector<int> e1_ots(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_ots.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_oid(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_oid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_ovar(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_ovar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_oc(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_oc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e1_otheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e1_otheta.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_sts(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_sts.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_sid(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_sid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_svar(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_svar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_sc(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_sc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e1_stheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e1_stheta.at(i);

   imf.close();




   //load second fitted model from file
#ifndef SILENT
   if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
   imf.open(folder2 + modelname2 + ".fit");
   imf >> ind;
   imf >> im;
   imf >> imh;
#ifdef _OPENMPI
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(m2!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mh2!=imh) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(m2!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mh2!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
#endif

   temp=0;
   imf >> temp;
   std::vector<int> e2_ots(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_ots.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_oid(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_oid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_ovar(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_ovar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_oc(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_oc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e2_otheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e2_otheta.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_sts(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_sts.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_sid(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_sid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_svar(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_svar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_sc(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_sc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e2_stheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e2_stheta.at(i);

   imf.close();




   // Calculate range of posterior samples to do Pareto front/set on for MPI.
   int startnd=0,endnd=nd-1;
   size_t snd,end,rnd=nd;
#ifdef _OPENMPI
   calcbegend(nd,mpirank,tc,&startnd,&endnd);
   snd=(size_t)startnd;
   end=(size_t)endnd;
   rnd=end-snd;
#ifndef SILENT
   cout << "Node " << mpirank << " calculating Pareto front and set for posterior draws " << startnd << " to " << endnd-1 << " (range="<< rnd << ")" << endl;
#endif
#endif

   //objects where we'll store the realizations
   std::vector<std::vector<double> > asol;
   std::vector<std::vector<double> > bsol;
   std::vector<double> thetasol;

   // Temporary vectors used for loading one model realization at a time.
   std::vector<int> onn1(m1,1);
   std::vector<std::vector<int> > oid1(m1, std::vector<int>(1));
   std::vector<std::vector<int> > ov1(m1, std::vector<int>(1));
   std::vector<std::vector<int> > oc1(m1, std::vector<int>(1));
   std::vector<std::vector<double> > otheta1(m1, std::vector<double>(1));
   std::vector<int> snn1(mh1,1);
   std::vector<std::vector<int> > sid1(mh1, std::vector<int>(1));
   std::vector<std::vector<int> > sv1(mh1, std::vector<int>(1));
   std::vector<std::vector<int> > sc1(mh1, std::vector<int>(1));
   std::vector<std::vector<double> > stheta1(mh1, std::vector<double>(1));

   std::vector<int> onn2(m2,1);
   std::vector<std::vector<int> > oid2(m2, std::vector<int>(1));
   std::vector<std::vector<int> > ov2(m2, std::vector<int>(1));
   std::vector<std::vector<int> > oc2(m2, std::vector<int>(1));
   std::vector<std::vector<double> > otheta2(m2, std::vector<double>(1));
   std::vector<int> snn2(mh2,1);
   std::vector<std::vector<int> > sid2(mh2, std::vector<int>(1));
   std::vector<std::vector<int> > sv2(mh2, std::vector<int>(1));
   std::vector<std::vector<int> > sc2(mh2, std::vector<int>(1));
   std::vector<std::vector<double> > stheta2(mh2, std::vector<double>(1));

   // Draw realizations of the posterior predictive.
   size_t curdx1=0;
   size_t cumdx1=0;
   size_t curdx2=0;
   size_t cumdx2=0;
   std::vector<std::vector<double> > a1,a2,b1,b2;
   std::vector<double> theta1,theta2;
#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
#endif


   // Mean trees first
   if(mpirank==0) cout << "Calculating Pareto front and set for mean trees" << endl;

   size_t ii=0;
   std::vector<std::vector<std::vector<double> > > aset(nd, std::vector<std::vector<double> >(0, std::vector<double>(0)));
   std::vector<std::vector<std::vector<double> > > bset(nd, std::vector<std::vector<double> >(0, std::vector<double>(0)));
   std::vector<std::vector<std::vector<double> > > front(nd, std::vector<std::vector<double> >(0, std::vector<double>(0)));

   for(size_t i=0;i<nd;i++) {

      // Load a realization from model 1
      curdx1=0;
      for(size_t j=0;j<m1;j++) {
         onn1[j]=e1_ots.at(i*m1+j);
         oid1[j].resize(onn1[j]);
         ov1[j].resize(onn1[j]);
         oc1[j].resize(onn1[j]);
         otheta1[j].resize(onn1[j]);
         for(size_t k=0;k< (size_t)onn1[j];k++) {
            oid1[j][k]=e1_oid.at(cumdx1+curdx1+k);
            ov1[j][k]=e1_ovar.at(cumdx1+curdx1+k);
            oc1[j][k]=e1_oc.at(cumdx1+curdx1+k);
            otheta1[j][k]=e1_otheta.at(cumdx1+curdx1+k);
         }
         curdx1+=(size_t)onn1[j];
      }
      cumdx1+=curdx1;
      ambm1.loadtree(0,m1,onn1,oid1,ov1,oc1,otheta1);

      // Load a realization from model 2
      curdx2=0;
      for(size_t j=0;j<m2;j++) {
         onn2[j]=e2_ots.at(i*m2+j);
         oid2[j].resize(onn2[j]);
         ov2[j].resize(onn2[j]);
         oc2[j].resize(onn2[j]);
         otheta2[j].resize(onn2[j]);
         for(size_t k=0;k< (size_t)onn2[j];k++) {
            oid2[j][k]=e2_oid.at(cumdx2+curdx2+k);
            ov2[j][k]=e2_ovar.at(cumdx2+curdx2+k);
            oc2[j][k]=e2_oc.at(cumdx2+curdx2+k);
            otheta2[j][k]=e2_otheta.at(cumdx2+curdx2+k);
         }
         curdx2+=(size_t)onn2[j];
      }
      cumdx2+=curdx2;
      ambm2.loadtree(0,m2,onn2,oid2,ov2,oc2,otheta2);


      // Calculate Pareto front and set
      std::vector<std::vector<double> > asol,bsol;
      std::vector<double> aout(p),bout(p);
      std::list<std::vector<double> > thetasol;
      if(i>=snd && i<end) {
         // convert ensembles to hyperrectangle format
         ambm1.ens2rects(a1, b1, theta1, minx, maxx, p);
         ambm2.ens2rects(a2, b2, theta2, minx, maxx, p);

         // calculate Pareto front and set for this realization
         for(size_t j=0;j<theta1.size();j++)
            for(size_t k=0;k<theta2.size();k++) {
               // if the rectangles defined by a1[j],b1[j] and a2[k],b2[k] intersect
              if(probxall_termkl_rect(j,k,a1,b1,a2,b2,minx,maxx,aout,bout)>0.0) { 
                asol.push_back(aout);
                bsol.push_back(bout);
                //The asol.size() is to record the index in the unsorted list so we can back
                //out the correct VHR for the theta's on the PF later on(line 497-498).
                //It would be cleaner if we had a list of vector, size_t tuple but this works as well.
                std::vector<double> th{theta1[j]+fmean1,theta2[k]+fmean2,(double)(asol.size())};
                thetasol.push_back(th);
              }
            }

         // clear the hyperrectangles for this realization
         a1.clear();
         b1.clear();
         theta1.clear();
         a2.clear();
         b2.clear();
         theta2.clear();

         // then we can get the front,set, and then clear these vectors
         thetasol.sort();

         std::vector<size_t> frontdx;
         frontdx=find_pareto_front(1,thetasol.size(),thetasol);
         // Save the Pareto front and set
         // Note frontdx has indices in 1..sizeof(front) so we need to -1 to get correct vector entry.
         // Also note we have to remap to original index to get the correct corresponding VHRs in the Pareto Set.
         aset[ii].resize(frontdx.size());
         bset[ii].resize(frontdx.size());
         front[ii].resize(frontdx.size());
         for(size_t k=0;k<frontdx.size();k++) {
               aset[ii][k].resize(p);
               bset[ii][k].resize(p);
               std::list<std::vector<double> >::iterator it = std::next(thetasol.begin(),frontdx[k]-1);
               for(size_t j=0;j<p;j++) {
                  // aset[ii][k][j]=asol[frontdx[k]-1][j];
                  // bset[ii][k][j]=bsol[frontdx[k]-1][j];
                  aset[ii][k][j]=asol[(size_t)((*it).at(2))-1][j];
                  bset[ii][k][j]=bsol[(size_t)((*it).at(2))-1][j];
               }
               front[ii][k].resize(2);
               front[ii][k][0]=(*it).at(0);
               front[ii][k][1]=(*it).at(1);
         }

         // if(mpirank==0) cout << "thetasol=" << thetasol.size() << " frontsize=" << frontdx.size() << endl;
         asol.clear();
         bsol.clear();
         thetasol.clear();
         frontdx.clear();

         ii++;
      }
   }



/* Variances trees Pareto Front currently not implemented.

   // Variance trees second
   if(mpirank==0) cout << "Drawing sd response from posterior predictive" << endl;
   cumdx=0;
   curdx=0;
   for(size_t i=0;i<nd;i++) {
      curdx=0;
      for(size_t j=0;j<mh;j++) {
         snn[j]=e_sts.at(i*mh+j);
         sid[j].resize(snn[j]);
         sv[j].resize(snn[j]);
         sc[j].resize(snn[j]);
         stheta[j].resize(snn[j]);
         for(size_t k=0;k< (size_t)snn[j];k++) {
            sid[j][k]=e_sid.at(cumdx+curdx+k);
            sv[j][k]=e_svar.at(cumdx+curdx+k);
            sc[j][k]=e_sc.at(cumdx+curdx+k);
            stheta[j][k]=e_stheta.at(cumdx+curdx+k);
         }
         curdx+=(size_t)snn[j];
      }
      cumdx+=curdx;

      psbm.loadtree(0,mh,snn,sid,sv,sc,stheta);
      // draw realization
      psbm.predict(&dip);
      for(size_t j=0;j<np;j++) tedrawh[i][j] = fp[j];
   }

   // For probit models we'll also construct probabilities
   if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
      if(mpirank==0) cout << "Drawing posterior predictive probabilities" << endl;
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<np;j++)
            tedrawp[i][j]=normal_01_cdf(tedraw[i][j]/tedrawh[i][j]);
   }
*/

#ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Pareto front and set draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif




   // Save the draws.
   if(mpirank==0) cout << "Saving Pareto front and set...";

   std::ofstream omf(folder + modelname + ".mopareto" + std::to_string(mpirank));
   for(size_t i=0;i<rnd;i++) {
      omf << front[i].size() << " ";
      for(size_t j=0;j<front[i].size();j++)
         omf << std::scientific << front[i][j][0] << " ";
      for(size_t j=0;j<front[i].size();j++)
         omf << std::scientific << front[i][j][1] << " ";
      for(size_t j=0;j<p;j++)
         for(size_t k=0;k<front[i].size();k++)
            omf << std::scientific << aset[i][k][j] << " ";
      for(size_t j=0;j<p;j++)
         for(size_t k=0;k<front[i].size();k++)
            omf << std::scientific << bset[i][k][j] << " ";
      omf << endl;
   }
   omf.close();

   if(mpirank==0) cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   MPI_Finalize();
#endif

   return 0;
}

