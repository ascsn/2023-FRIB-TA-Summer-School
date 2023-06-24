//     test.cpp: Poisson tree model class testing/validation code.
//     Copyright (C) 2012-2019 Matthew T. Pratola
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
#include <fstream>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "singlepoisson.h"

using std::cout;
using std::endl;

int main()
{

   cout << "\n*****into test for brt\n";
   cout << "\n\n";

   crn gen;
   gen.set_seed(199);

   int tc=4; //thread count for OpenMP

   //--------------------------------------------------
   //read in y
   std::vector<double> y;
   double ytemp;

   std::ifstream yf("y.txt");
   while(yf >> ytemp)
      y.push_back(ytemp);
   size_t n = y.size();
   cout << "n from y.txt: " << n <<endl;

   //--------------------------------------------------
   //read in x
   std::vector<double> x;
   double xtemp;
   size_t p;

   std::ifstream xf("x.txt");
   while(xf >> xtemp)
      x.push_back(xtemp);
   p = x.size()/n;
   cout << "n,p: " << n << ", " << p << endl;
   dinfo di;
   di.n=n;di.p=p,di.x = &x[0];di.tc=tc;
   di.y = &y[0];


   //--------------------------------------------------
   //make xinfo
   xinfo xi;
   size_t nc=1000;
   makexinfo(p,n,&x[0],xi,nc);

   //--------------------------------------------------
   // read in the initial change of variable rank correlation matrix
   std::vector<std::vector<double>> chgv;
   std::vector<double> cvvtemp;
   double cvtemp;
   std::ifstream chgvf("../brt/chgv.txt");
   for(size_t i=0;i<di.p;i++) {
      cvvtemp.clear();
      for(size_t j=0;j<di.p;j++) {
         chgvf >> cvtemp;
         cvvtemp.push_back(cvtemp);
      }
      chgv.push_back(cvvtemp);
   }
   cout << "change of variable rank correlation matrix loaded:" << endl;
   for(size_t i=0;i<di.p;i++) {
      for(size_t j=0;j<di.p;j++)
         cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
      cout << endl;
   }

   //--------------------------------------------------
   // try instantiating a singlepoissonbrt
   singlepoissonbrt bm;

   double alpha=2.22; //4.1;
   double beta=0.16; //0.66;

   cout  << "##################################################\n";
   cout << "*****Print out first singlepoissonbrt object !!!!\n";
   bm.pr();

   singlepoissonsinfo mi;
   cout << "mi:\n";
   cout << mi.n << ", " << mi.sumy << endl;
   mi.n=10;mi.sumy=3;
   singlepoissonsinfo mi2(mi);
   cout << "mi2:\n";
   cout << mi2.n << ", " << mi2.sumy << endl;

   singlepoissonsinfo mi3(mi2);
   mi3 += mi2;
   cout << "mi3:\n";
   cout << mi3.n << ", " << mi3.sumy << endl;

   mi3=mi2;
   cout << "mi3 (after =mi2):\n";
   cout << mi3.n << ", " << mi3.sumy << endl;

   // msinfo mi4 = mi3+mi2;
   singlepoissonsinfo mi4;
   mi4=mi3+mi2;
   cout << "mi4:\n";
   cout << mi4.n << ", " << mi4.sumy << endl;

   //cutpoints
   bm.setxi(&xi);    //set the cutpoints for this model object
   //data objects
   bm.setdata(&di);  //set the data
   //thread count
   bm.settc(tc);      //set the number of threads when using OpenMP, etc.
   //tree prior
   bm.settp(0.95, //the alpha parameter in the tree depth penalty prior
         2.0     //the beta parameter in the tree depth penalty prior
         );
   //MCMC info
   bm.setmi(
         0.5,  //probability of birth/death
         0.5,  //probability of birth
         5,    //minimum number of observations in a bottom node
         true, //do perturb/change variable proposal?
         0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   bm.setci(alpha,beta);

   cout << "\n*****after init:\n";
   bm.pr();

   cout << "\n*****after 1 draw:\n";
   bm.draw(gen);
   bm.pr();



   //--------------------------------------------------
   cout << "\n#####EX2: In-sample prediction of Branin function after burn-in\n";
   size_t nadapt=1000;
   size_t adaptevery=100;
   size_t nburn=500;
   size_t nds=5000;
   std::vector<double> fitted(n);

   for(size_t i=0;i<nadapt;i++) { bm.draw(gen); if((i+1)%adaptevery==0) bm.adapt(); }
   for(size_t i=0;i<nburn;i++) bm.draw(gen);

   cout << "Collecting statistics" << endl;
   bm.setstats(true);
   for(size_t i=0;i<nds;i++) {
      if((i % 500) ==0) cout << "draw " << i << endl;
      bm.draw(gen);
      for(size_t j=0;j<n;j++) fitted[j]+=bm.f(j)/nds;
   }
   std::ofstream bmfit("insample.txt");
   for(size_t j=0;j<n;j++) bmfit << fitted[j] << "\n";
   bm.pr();
/*
   // summary statistics
   unsigned int varcount[p];
   unsigned int totvarcount=0;
   for(size_t i=0;i<p;i++) varcount[i]=0;
   unsigned int tmaxd=0;
   unsigned int tmind=0;
   double tavgd=0.0;

   mbm.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
   for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
   tavgd/=(double)(nds);

   cout << "Average tree depth: " << tavgd << endl;
   cout << "Maximum tree depth: " << tmaxd << endl;
   cout << "Minimum tree depth: " << tmind << endl;
   cout << "Variable perctg:    ";
   for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
   cout << endl;
   cout << "                    ";
   for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
   cout << endl;

*/
   return 0;
}
