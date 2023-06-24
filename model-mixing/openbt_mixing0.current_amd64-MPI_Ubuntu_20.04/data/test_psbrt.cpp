//     test.cpp: Product variance BT model class testing/validation code.
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


#include <iostream>
#include <fstream>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "sbrt.h"
#include "psbrt.h"

using std::cout;
using std::endl;

int main()
{

   cout << "\n*****into test for psbrt\n";
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
   std::ifstream chgvf("chgv.txt");
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
   // EX#1: Try instantiating a psbrt
   size_t m=20;
   psbrt psbm(m,1.728307); //1.787354);
   //double overallnu=5.0;
   double overallnu=50.0;
   double opm=1.0/((double)m);
   double nu=2.0*pow(overallnu,opm)/(pow(overallnu,opm)-pow(overallnu-2.0,opm));
   double overalllambda=1.0;
   double lambda=pow(overalllambda,opm);

//   double nu=5.0;
//   double lambda=1.0;

   cout  << "##################################################\n";
   cout << "*****Print out first psbrt object !!!!\n";
   psbm.pr();


   //cutpoints
   psbm.setxi(&xi);    //set the cutpoints for this model object
   //data objects
   psbm.setdata(&di);  //set the data
   //thread count
   psbm.settc(tc);      //set the number of threads when using OpenMP, etc.
   //tree prior
   psbm.settp(0.95, //the alpha parameter in the tree depth penalty prior
         1.0     //the beta parameter in the tree depth penalty prior
         );
   //MCMC info
   psbm.setmi(0.5,  //probability of birth/death
         0.5,  //probability of birth
         5,    //minimum number of observations in a bottom node
         true, //do perturb/change variable proposal?
         0.05,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         0.0,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   psbm.setci(nu,lambda);

   cout << "\n*****after init:\n";
   psbm.pr();

   cout << "\n*****after 1 draw:\n";
   psbm.draw(gen);
   psbm.pr();

   //--------------------------------------------------
   cout << "\n#####EX2: In-sample prediction of Branin function after burn-in\n";
   size_t nadapt=20000;
   size_t adaptevery=500;
   size_t nburn=1000;
   size_t nds=1000;
   std::vector<double> fitted(n);
   dinfo inpred;
   inpred.n=n;inpred.p=p,inpred.x = &x[0];inpred.tc=tc;inpred.y=&fitted[0];

   for(size_t i=0;i<nadapt;i++) { psbm.draw(gen); if((i+1)%adaptevery==0) psbm.adapt(); }
   for(size_t i=0;i<nburn;i++) psbm.draw(gen);

   cout << "Collecting statistics" << endl;
   psbm.setstats(true);
   for(size_t i=0;i<nds;i++) {
      if((i%50) ==0) cout << "draw " << i << endl;
      psbm.draw(gen);
//      for(size_t j=0;j<n;j++) fitted[j]+=psbm.f(j)/nds;
      inpred+= *psbm.getf();
   }
   inpred/= ((double)nds);
   std::ofstream psbmfit("insample.txt");
   for(size_t j=0;j<n;j++) psbmfit << fitted[j] << "\n";
   psbm.pr();

   // summary statistics
   unsigned int varcount[p];
   unsigned int totvarcount=0;
   for(size_t i=0;i<p;i++) varcount[i]=0;
   unsigned int tmaxd=0;
   unsigned int tmind=0;
   double tavgd=0.0;

   psbm.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
   for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
   tavgd/=(double)(nds*m);

   cout << "Average tree depth: " << tavgd << endl;
   cout << "Maximum tree depth: " << tmaxd << endl;
   cout << "Minimum tree depth: " << tmind << endl;
   cout << "Variable perctg:    ";
   for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
   cout << endl;
   cout << "                    ";
   for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
   cout << endl;

   return 0;
}
