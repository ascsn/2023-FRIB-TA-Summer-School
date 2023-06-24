//     vartivity.cpp: Implement variable activity interface for OpenBT.
//     Copyright (C) 2012-2018 Matthew T. Pratola.
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

   //--------------------------------------------------
   // Banner
   cout << endl;
   cout << "-----------------------------------" << endl;
   cout << "OpenBT variable activity CLI" << endl;
   cout << "Loading config file at " << folder << endl;

   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config.vartivity");

   //model name, number of saved draws and number of trees
   std::string modelname;
   size_t nd;
   size_t m;
   size_t mh;

   conf >> modelname;
   conf >> nd;
   conf >> m;
   conf >> mh;
//   std::string folder("." + modelname + "/");
 
   //number of predictors
   size_t p;
   conf >> p;

   conf.close();

   //load from file
#ifndef SILENT
   cout << "Loading saved posterior tree draws" << endl;
#endif
   size_t ind,im,imh;
   std::ifstream imf(folder + modelname + ".fit");
   imf >> ind;
   imf >> im;
   imf >> imh;
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(m!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mh!=imh) { cout << "Error loading posterior trees" << endl; return 0; }

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
   std::vector<std::vector<double> > vdraw(nd,std::vector<double>(p));
   std::vector<std::vector<double> > vdrawh(nd,std::vector<double>(p));

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
   size_t cid=0;
   bool haschild=false;


   // Mean trees first
   cout << "Drawing variable activity for mean posterior predictive" << endl;
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

      for(size_t j=0;j<m;j++)
         for(size_t k=0;k< (size_t)onn[j];k++) {
            cid=2*oid[j][k];
            for(size_t l=0;l< (size_t)onn[j];l++)
               if((size_t) oid[j][l]==cid)
                  haschild=true;

            if(haschild) vdraw[i][ov[j][k]]++;
            haschild=false;
         }
   }


   // Variance trees second
   cout << "Drawing variable activity for sd posterior predictive" << endl;
   cumdx=0;
   curdx=0;
   cid=0;
   haschild=false;
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

      for(size_t j=0;j<mh;j++)
         for(size_t k=0;k< (size_t)snn[j];k++) {
            cid=2*sid[j][k];
            for(size_t l=0;l< (size_t)snn[j];l++)
               if((size_t) sid[j][l]==cid)
                  haschild=true;

            if(haschild) vdrawh[i][sv[j][k]]++;
            haschild=false;
         }

   }


   // Save the draws.
   cout << "Saving posterior variable activity draws...";
   std::ofstream omf(folder + modelname + ".vdraws");
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<p;j++)
         omf << std::scientific << vdraw[i][j] << " ";
      omf << endl;
   }
   omf.close();

   std::ofstream smf(folder + modelname + ".vdrawsh");
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<p;j++)
         smf << std::scientific << vdrawh[i][j] << " ";
      smf << endl;
   }
   smf.close();
   cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.

   return 0;
}

