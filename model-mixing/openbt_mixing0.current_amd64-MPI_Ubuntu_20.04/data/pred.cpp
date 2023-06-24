//     pred.cpp: Implement model prediction interface for OpenBT.
//     Copyright (C) 2012-2018 Matthew T. Pratola
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


#include <chrono>
#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

#include <fstream>
#include <vector>
#include <limits>

#include "Eigen/Dense"
#include <Eigen/StdVector>

#include "crn.h"
#include "tree.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mbrt.h"
#include "ambrt.h"
#include "psbrt.h"
#include "tnorm.h"
#include "mxbrt.h"
#include "amxbrt.h"

using std::cout;
using std::endl;

#define MODEL_BT 1
#define MODEL_BINOMIAL 2
#define MODEL_POISSON 3
#define MODEL_BART 4
#define MODEL_HBART 5
#define MODEL_PROBIT 6
#define MODEL_MODIFIEDPROBIT 7
#define MODEL_MIXBART 9 //Skipped 8 because MERCK is 8 in cli.cpp


// Draw predictive realizations at the prediciton points, xp.
int main(int argc, char* argv[])
{
   std::string folder("");

   if(argc>1)
   {
      //argument on the command line is path to config file.
      folder=std::string(argv[1]);
      folder=folder+"/";
   }

   //-----------------------------------------------------------
   //random number generation -- only used in model mixing with function discrepancy right now
   crn gen;
   gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count()));

   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config.pred");
   std::string modelname;
   int modeltype;
   std::string xicore;
   std::string xpcore;
   std::string fpcore;

   //model name, xi and xp
   conf >> modelname;
   conf >> modeltype;
   conf >> xicore;
   conf >> xpcore;
   conf >> fpcore;

   //number of saved draws and number of trees
   size_t nd;
   size_t m;
   size_t mh;

   conf >> nd;
   conf >> m;
   conf >> mh;

   //number of predictors
   size_t p, k;
   conf >> p;
   conf >> k;

   /*
   //model mixing fdiscrepancy -- remove
   std::string fdiscrepancy_str;
   conf >> fdiscrepancy_str;
   bool fdiscrepancy = false;
   if(fdiscrepancy_str == "TRUE"){ fdiscrepancy = true;}
   */

   //thread count
   int tc;
   conf >> tc;

   //mean offset
   double fmean;
   conf >> fmean;
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
      cout << "OpenBT prediction CLI" << endl;
      cout << "Loading config file at " << folder << endl;
   }

   //--------------------------------------------------
   //read in xp.
   std::vector<double> xp;
   double xtemp;
   size_t np;
//   std::string folder("."+modelname+"/");
   std::stringstream xfss;
   std::string xfs;
   xfss << folder << xpcore << mpirank;
   xfs=xfss.str();
   std::ifstream xf(xfs);
   while(xf >> xtemp)
      xp.push_back(xtemp);
   np = xp.size()/p;
#ifndef SILENT
   cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;
#endif
   cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;
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
   //--------------------------------------------------
   //Initialize f matrix and make finfo -- used only for model mixing 
   std::vector<double> fpd;
   double ftemp;
   finfo fi_test(np,k);
   if(modeltype==MODEL_MIXBART) {
      std::stringstream ffss;
      std::string ffs;
      ffss << folder << fpcore << mpirank; //get the file to read in -- every processor reads in a different file (dictated by mpirank and R files)
      ffs=ffss.str();
      std::ifstream ff(ffs);
      while(ff >> ftemp)
         fpd.push_back(ftemp);
      Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> fi_test_temp(fpd.data(),k,np);
      fi_test = fi_test_temp.transpose();
      //fi_test.col(i) = Eigen::Map<Eigen::VectorXd>(fpd.data(),np);  
      //cout << "-------------------------------\n" << "MPI = " << mpirank << "\n np = " << np << "\nk =" << k  << "\nfi_test = \n" << fi_test << endl;
#ifndef SILENT
   cout << "&&& made finfo for test data\n";
#endif
   }

   //--------------------------------------------------
   //Initialize f mean and std discrepancy information -- used only for model mixing when fdiscrepancy = TRUE 
   /*
   std::vector<double> fdm, fds;
   double fdmtemp, fdstemp;
   finfo fdeltamean(np,k), fdeltasd(np,k);
   if(modeltype==MODEL_MIXBART && fdiscrepancy){   
         std::stringstream fdmfss;
         std::string fdmfs;
         fdmfss << folder << fpdmcore << mpirank;
         fdmfs=fdmfss.str();
         std::ifstream fdmf(fdmfs);
         while(fdmf >> fdmtemp)
            fdm.push_back(fdmtemp);
         Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> fdm_temp(fdm.data(),k,np);
         fdeltamean = fdm_temp.transpose();
         
         std::stringstream fdsfss;
         std::string fdsfs;
         fdsfss << folder << fpdscore << mpirank;
         fdsfs=fdsfss.str();
         std::ifstream fdsf(fdsfs);
         while(fdsf >> fdstemp)
            fds.push_back(fdstemp);
         Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> fds_temp(fdm.data(),k,np);
         fdeltasd = fds_temp.transpose();
   }
   */
   
if(modeltype!=MODEL_MIXBART){
   //--------------------------------------------------
   // set up ambrt object
   ambrt ambm(m);
   ambm.setxi(&xi); //set the cutpoints for this model object

   //setup psbrt object
   psbrt psbm(mh);
   psbm.setxi(&xi); //set the cutpoints for this model object



   //load from file
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
   if(m!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mh!=imh) { cout << "Error loading posterior trees"  << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(m!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mh!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
#endif

   size_t temp=0;
   imf >> temp;
   std::vector<int> e_ots(temp);
   for(size_t i=0;i<temp;i++) imf >> e_ots.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_oid(temp);
   for(size_t i=0;i<temp;i++) imf >> e_oid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_ovar(temp);
   for(size_t i=0;i<temp;i++) imf >> e_ovar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_oc(temp);
   for(size_t i=0;i<temp;i++) imf >> e_oc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e_otheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_otheta.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sts(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sts.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sid(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_svar(temp);
   for(size_t i=0;i<temp;i++) imf >> e_svar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sc(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e_stheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_stheta.at(i);

   imf.close();



   //objects where we'll store the realizations
   std::vector<std::vector<double> > tedraw(nd,std::vector<double>(np));
   std::vector<std::vector<double> > tedrawh(nd,std::vector<double>(np));
   std::vector<std::vector<double> > tedrawp(nd,std::vector<double>(np));
   double *fp = new double[np];
   dinfo dip;
   dip.x = &xp[0]; dip.y=fp; dip.p = p; dip.n=np; dip.tc=1;

   // Temporary vectors used for loading one model realization at a time.
   std::vector<int> onn(m,1);
   std::vector<std::vector<int> > oid(m, std::vector<int>(1));
   std::vector<std::vector<int> > ov(m, std::vector<int>(1));
   std::vector<std::vector<int> > oc(m, std::vector<int>(1));
   std::vector<std::vector<double> > otheta(m, std::vector<double>(1));
   std::vector<int> snn(mh,1);
   std::vector<std::vector<int> > sid(mh, std::vector<int>(1));
   std::vector<std::vector<int> > sv(mh, std::vector<int>(1));
   std::vector<std::vector<int> > sc(mh, std::vector<int>(1));
   std::vector<std::vector<double> > stheta(mh, std::vector<double>(1));

   // Draw realizations of the posterior predictive.
   size_t curdx=0;
   size_t cumdx=0;
#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
#endif


   // Mean trees first
   if(mpirank==0) cout << "Drawing mean response from posterior predictive" << endl;
   for(size_t i=0;i<nd;i++) {
      curdx=0;
      for(size_t j=0;j<m;j++) {
         onn[j]=e_ots.at(i*m+j);
         oid[j].resize(onn[j]);
         ov[j].resize(onn[j]);
         oc[j].resize(onn[j]);
         otheta[j].resize(onn[j]);
         for(size_t k=0;k< (size_t)onn[j];k++) {
            oid[j][k]=e_oid.at(cumdx+curdx+k);
            ov[j][k]=e_ovar.at(cumdx+curdx+k);
            oc[j][k]=e_oc.at(cumdx+curdx+k);
            otheta[j][k]=e_otheta.at(cumdx+curdx+k);
         }
         curdx+=(size_t)onn[j];
      }
      cumdx+=curdx;

      ambm.loadtree(0,m,onn,oid,ov,oc,otheta);
      // draw realization
      ambm.predict(&dip);
      for(size_t j=0;j<np;j++) tedraw[i][j] = fp[j] + fmean;
   }


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

#ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Posterior predictive draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif

   // Save the draws.
   if(mpirank==0) cout << "Saving posterior predictive draws...";
   std::ofstream omf(folder + modelname + ".mdraws" + std::to_string(mpirank));
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<np;j++)
         omf << std::scientific << tedraw[i][j] << " ";
      omf << endl;
   }
   omf.close();

   std::ofstream smf(folder + modelname + ".sdraws" + std::to_string(mpirank));
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<np;j++)
         smf << std::scientific << tedrawh[i][j] << " ";
      smf << endl;
   }
   smf.close();

   if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
      std::ofstream pmf(folder + modelname + ".pdraws" + std::to_string(mpirank));
      for(size_t i=0;i<nd;i++) {
         for(size_t j=0;j<np;j++)
            pmf << std::scientific << tedrawp[i][j] << " ";
         pmf << endl;
      }
      pmf.close();
   }

   if(mpirank==0) cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   MPI_Finalize();
#endif

}else if(modeltype == MODEL_MIXBART){
   //--------------------------------------------------
   // set up amxbrt object
   amxbrt axb(m);
   axb.setxi(&xi); //set the cutpoints for this model object
   axb.setfi(&fi_test, k); //set the function output for this model object -- main use is to set k 
   //if(fdiscrepancy) {axb.setfdelta(&fdeltamean, &fdeltasd);}  //set individual function discrepacnies if provided -- main use is to set fdiscrepancy to TRUE
   
   //setup psbrt object
   psbrt psbm(mh);
   psbm.setxi(&xi); //set the cutpoints for this model object
   
   //load from file
#ifndef SILENT
   if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
   size_t ind,im,imh;
   std::ifstream imf(folder + modelname + ".fit");
   imf >> ind;
   imf >> im;
   imf >> imh;
#ifdef _OPENMPI
   if(nd!=ind) { cout << "Error loading posterior trees"<< "nd = " << nd << " -- ind = " << ind << endl; MPI_Finalize(); return 0; }
   if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; MPI_Finalize(); return 0; }
   if(mh!=imh) { cout << "Error loading posterior trees"  << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << "nd = " << nd << " -- ind = " << ind << endl; return 0; }
   if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; return 0; }
   if(mh!=imh) { cout << "Error loading posterior trees"  << endl; return 0; }
#endif

   size_t temp=0;
   imf >> temp;
   std::vector<int> e_ots(temp);
   for(size_t i=0;i<temp;i++) imf >> e_ots.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_oid(temp);
   for(size_t i=0;i<temp;i++) imf >> e_oid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_ovar(temp);
   for(size_t i=0;i<temp;i++) imf >> e_ovar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_oc(temp);
   for(size_t i=0;i<temp;i++) imf >> e_oc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e_otheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_otheta.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sts(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sts.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sid(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_svar(temp);
   for(size_t i=0;i<temp;i++) imf >> e_svar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sc(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e_stheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_stheta.at(i);

   imf.close();

   //objects where we'll store the realizations
   std::vector<std::vector<double> > tedraw(nd,std::vector<double>(np));
   std::vector<std::vector<double> > tedrawh(nd,std::vector<double>(np));
   std::vector<std::vector<double> > tedrawp(nd,std::vector<double>(np));
   double *fp = new double[np];
   dinfo dip;
   dip.x = &xp[0]; dip.y=fp; dip.p = p; dip.n=np; dip.tc=1;

   // Temporary vectors used for loading one model realization at a time.
   std::vector<int> onn(m,1);
   std::vector<std::vector<int> > oid(m, std::vector<int>(1));
   std::vector<std::vector<int> > ov(m, std::vector<int>(1));
   std::vector<std::vector<int> > oc(m, std::vector<int>(1));
   std::vector<std::vector<double> > otheta(m, std::vector<double>(1));
   std::vector<int> snn(mh,1);
   std::vector<std::vector<int> > sid(mh, std::vector<int>(1));
   std::vector<std::vector<int> > sv(mh, std::vector<int>(1));
   std::vector<std::vector<int> > sc(mh, std::vector<int>(1));
   std::vector<std::vector<double> > stheta(mh, std::vector<double>(1));


   // Draw realizations of the posterior predictive.
   size_t curdx=0;
   size_t cumdx=0;
#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
#endif

   // Mean trees first
   if(mpirank==0) cout << "Drawing mean response from posterior predictive" << endl;
   for(size_t i=0;i<nd;i++) {
      curdx=0;
      for(size_t j=0;j<m;j++) {
         onn[j]=e_ots.at(i*m+j);
         oid[j].resize(onn[j]);
         ov[j].resize(onn[j]);
         oc[j].resize(onn[j]);
         otheta[j].resize(onn[j]*k);
         for(size_t l=0;l<(size_t)onn[j];l++) {
            oid[j][l]=e_oid.at(cumdx+curdx+l);
            ov[j][l]=e_ovar.at(cumdx+curdx+l);
            oc[j][l]=e_oc.at(cumdx+curdx+l);
            for(size_t r=0;r<k;r++){
               otheta[j][l*k+r]=e_otheta.at((cumdx+curdx+l)*k+r);
               //cout << "Theta_i = " << otheta[j][l*k+r] << endl;;
            }
            
         }
         curdx+=(size_t)onn[j];
      }
      cumdx+=curdx;

      axb.loadtree_vec(0,m,onn,oid,ov,oc,otheta); 
      // draw realization
      /*
      if(fdiscrepancy){
         axb.predict_mix_fd(&dip, &fi_test, &fdeltamean, &fdeltasd, gen);
      }else{
         axb.predict_mix(&dip, &fi_test);
      }
      */
      axb.predict_mix(&dip, &fi_test);
      for(size_t j=0;j<np;j++) tedraw[i][j] = fp[j] + fmean;
   }

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

   #ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Posterior predictive draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif
   // Save the draws.
   if(mpirank==0) cout << "Saving posterior predictive draws...";
   std::ofstream omf(folder + modelname + ".mdraws" + std::to_string(mpirank));
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<np;j++)
         omf << std::scientific << tedraw[i][j] << " ";
      omf << endl;
   }
   omf.close();

   std::ofstream smf(folder + modelname + ".sdraws" + std::to_string(mpirank));
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<np;j++)
         smf << std::scientific << tedrawh[i][j] << " ";
      smf << endl;
   }
   smf.close();

   if(mpirank==0) cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   MPI_Finalize();
#endif
   }
   return 0;
}

