//     test.cpp: Base BT model class test/validation code.
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

#ifdef _OPENMPI
#   include <mpi.h>
#endif


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
   //read in x
   std::vector<double> x;
   double xtemp;
   size_t n,p;
   p=2;

   std::ifstream xf("x.txt");
   while(xf >> xtemp)
      x.push_back(xtemp);
   n = x.size()/p;
   if(x.size() != n*p) {
      cout << "error: input x file has wrong number of values\n";
      return 1;
   }
   cout << "n,p: " << n << ", " << p << endl;
   dinfo di;
   di.n=n;di.p=p,di.x = &x[0];di.tc=tc;

   //--------------------------------------------------
   //make xinfo
   xinfo xi;
   size_t nc=100;
   makexinfo(p,n,&x[0],xi,nc);

   //prxi(xi);

   //--------------------------------------------------
   // read in the initial change of variable rank correlation matrix
   std::vector<std::vector<double> > chgv;
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
   cout << "\n#####EX1: make a brt object and print it out\n";
   brt bm;
   cout << "\nbefore init:\n";
   bm.pr();
   //cutpoints
   bm.setxi(&xi);    //set the cutpoints for this model object
   //data objects
   bm.setdata(&di);  //set the data
   //thread count
   bm.settc(tc);      //set the number of threads when using OpenMP, etc.
   //tree prior
   bm.settp(0.95, //the alpha parameter in the tree depth penalty prior
         1.0     //the beta parameter in the tree depth penalty prior
         );
   //MCMC info
   bm.setmi(0.7,  //probability of birth/death
         0.5,  //probability of birth
         5,    //minimum number of observations in a bottom node
         true, //do perturb/change variable proposal?
         0.2,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         0.2,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   cout << "\nafter init:\n";
   bm.pr();

 
   //--------------------------------------------------
   cout << "\n#####EX2: try some draws of brt and print it out\n";
   cout << "\n1 draw:\n";
   bm.draw(gen);
   bm.pr();
   size_t nd=1000;
   cout << "\n" << nd << " draws:\n";
   for(size_t i=0;i<nd;i++)
      bm.draw(gen);
   bm.pr();
   //5seconds bd
   //5.3seconds bd+rotate@30%
   //17.5seconds bd+rotate@30%+perturb
   //17.9seconds bd+rotate@30%+perturb+changeofvariable@20%


   if(0) { //from hugh visit fall 2014
   //--------------------------------------------------
   //read in x
   std::vector<double> x;
   double xtemp;
   size_t n,p;
   p=2;

   std::ifstream xf("x.txt");
   while(xf >> xtemp)
      x.push_back(xtemp);
   n = x.size()/p;
   if(x.size() != n*p) {
      cout << "error: input x file has wrong number of values\n";
      return 1;
   }
   cout << "n,p: " << n << ", " << p << endl;
   dinfo di;
   di.n=n;di.p=p,di.x = &x[0];

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
   //make xinfo
   xinfo xi;
   size_t nc=100;
   makexinfo(p,n,&x[0],xi,nc);
   //prxi(xi);

   //--------------------------------------------------
   tree t;
   //t.pr();

   brt bm;
   //bm.pr();
   bm.setxi(&xi);
   bm.setdata(&di);
   bm.t.birth(1,0,50,-1,1);
   //bm.pr();

 
   /* if you want to run this need to move getsuff to public
   //--------------------------------------------------
   //test getsuff
   //get bottom nodes
   tree::npv bots;
   bm.t.getbots(bots);

   size_t v=0;
   size_t c = 25;
   //brt::sinfo sil,sir,sit;
   sinfo sil,sir,sit;
   bm.getsuff(bots[0],v,c,sil,sir,sit);
   cout << "cutpoint is: " << xi[v][c] << endl;
   cout << "sil.n: " << sil.n << endl;
   cout << "sir.n: " << sir.n << endl;
   cout << "sit.n: " << sit.n << endl;
   */
/*
#ifdef _OPENMP
   int tc=4;
   bm.settc(tc);
   double start,end;
   start = omp_get_wtime();
   for(int i=0;i<20;i++) bm.bd(gen);
   end = omp_get_wtime();
#else
   int tc=1;
   double start,end;
   start = time(NULL);
   for(int i=0;i<20;i++) bm.bd(gen);
   end = time(NULL);
#endif
   cout << "++++++++time: " << end-start << endl;
   cout << "++++++++tc: " << tc << endl;
   bm.pr();
*/
   //------------------------------------------------------------
   if(0) {
   //get dist of number of bottom nodes
   std::ofstream botnf("nbots.txt");
   size_t nd = 500;  //number of times to start chain
   size_t nit = 100; //number of iterations per chain
   brt bmbot;
   bmbot.setxi(&xi);
   bmbot.setdata(&di);
   bmbot.settc(tc);
   bmbot.settp(.95,1.0);
   for(size_t i=0;i<nd;i++) {
      std::cout << "i: " << i << std::endl;
      for(size_t j=0;j<nit;j++) {
         bmbot.bd(gen);
      }
      botnf << bmbot.t.nbots() << std::endl;
      bmbot.t.tonull();
   }
   }

   //------------------------------------------------------------
   if(0) {
   //std::vector<brt::sinfo> siv(2);
   std::vector<sinfo> siv(2);
   std::cout << "testing vectors of sinfos\n";
   std::cout << siv[0].n << ", " << siv[1].n << std::endl;

   siv.clear();
   siv.resize(2);
   std::cout << siv[0].n << ", " << siv[1].n << std::endl;
   }
   //------------------------------------------------------------
   /* tested allsuff (with methods moved out of protected region)
   if(1) {
   brt ckas;  //check allsuff
   ckas.setxi(&xi);
   ckas.setdata(&di);
   ckas.t.birth(1,0,50,-1,1);
   ckas.t.pr();
   ckas.settc(4);
   //check cutpoint
   std::cout << "the cutpoint is " << xi[0][50] << std::endl;

   tree::npv bnv;
   //std::vector<brt::sinfo> siv;
   std::vector<sinfo> siv;
   std::cout << "num bots: " << ckas.t.nbots() << std::endl;
   for(int i=0;i<4000;i++) ckas.allsuff(bnv,siv);
   std::cout << "length siv: " << siv.size() << std::endl;
   std::cout << siv[0].n << ", " << siv[1].n << std::endl;
   }
   */
   //------------------------------------------------------------
   /* tested allsuff (with methods moved out of protected region)
   if(0) {
   brt ckas;  //check allsuff
   ckas.setxi(&xi);
   ckas.setdata(&di);
   ckas.t.birth(1,0,50,-1,1);
   ckas.drawtheta(gen);
   ckas.t.pr();
   ckas.settc(4);
   //check cutpoint
   std::cout << "the cutpoint is " << xi[0][50] << std::endl;

   tree::npv bnv;
   //std::vector<brt::sinfo> siv;
   std::vector<sinfo> siv;
   std::cout << "num bots: " << ckas.t.nbots() << std::endl;
   ckas.allsuff(bnv,siv);
   std::cout << "length siv: " << siv.size() << std::endl;
   std::cout << siv[0].n << ", " << siv[1].n << std::endl;
   }
   */
   //------------------------------------------------------------
   /* tested MCMC draw
   if(0) {
   brt ckdraw; //check MCMC iteration
   ckdraw.setxi(&xi);
   ckdraw.setdata(&di);
   ckdraw.settc(4);
   for(size_t i=0;i<5;i++) {
      ckdraw.draw(gen);
      ckdraw.t.pr();
   }
   }
   */
   //------------------------------------------------------------
   // test perturb manually 
   /* In R, do hist(data,breaks=0:101-0.5)
   if(1) {
   brt b;
   b.setxi(&xi);
   b.setdata(&di);
   b.t.birth(1,0,50,-1,1);
   b.settc(4);
   for(size_t i=0;i<5000;i++) {
      b.pertcv(gen);
      std::cout << b.t.getc() << std::endl;
   }
   }
   */
   //------------------------------------------------------------
   /* tested b/d and rotate
   if(1) {
   brt b;
   tree::npv bnv;

   b.setxi(&xi);
   b.setdata(&di);
   b.settc(4);
   //      pbd,pb,minperbot,dopert,pertalpha  
   b.setmi(0.8, 0.5, 5,true,0.1,chgv); 
   for(size_t i=0;i<5000;i++) {
      b.draw(gen);
      bnv.clear();
      b.t.getbots(bnv);
// we can look at distribution of n in each bottom node:
//      for(size_t j=0;j<bnv.size();j++)
//         cout << bnv[j]->gettheta() << " ";
//      cout << endl;
// or we can look at distribution of number of bottom nodes:
      cout << bnv.size() << endl;
   }
   }
   */
   //------------------------------------------------------------
   // what the overall MCMC algorithm should look like
   if(1) {
   size_t tuneevery=2500;
   size_t tune=50000;
   size_t burn=5000;
   size_t draws=20000;
   brt b;

   b.setxi(&xi);
   b.setdata(&di);
   b.settc(tc);
   //      pbd,pb,minperbot,dopert,pertalpha,pchgv,chgv
   b.setmi(0.8,0.5,5,true,0.1,0.2,&chgv);

   // tune the sampler   
   for(size_t i=0;i<tune;i++)
   {
      b.draw(gen);
      if((i+1)%tuneevery==0)
         b.adapt();
   }

   b.t.pr();
   // run some burn-in, tuning is fixed now
   for(size_t i=0;i<burn;i++)
   {
      b.draw(gen);
   }

   // draw from the posterior
 
   // After burn-in, turn on statistics if we want them:
   cout << "Collecting statistics" << endl;
   b.setstats(true);
   // then do the draws
     for(size_t i=0;i<draws;i++)
   {
      b.draw(gen);
   }


      // summary statistics
      unsigned int varcount[p];
      unsigned int totvarcount=0;
      for(size_t i=0;i<p;i++) varcount[i]=0;
      unsigned int tmaxd=0;
      unsigned int tmind=0;
      double tavgd=0.0;

      b.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
      for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
      tavgd/=(double)(draws);

      cout << "Average tree depth: " << tavgd << endl;
      cout << "Maximum tree depth: " << tmaxd << endl;
      cout << "Minimum tree depth: " << tmind << endl;
      cout << "Variable perctg:    ";
      for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
      cout << endl;
      cout << "                    ";
      for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
      cout << endl;

   }

   //------------------------------------------------------------
   /* tested subsuff (with methods moved out of protected region)
   if(1) {
      brt ckas;  //check allsuff
      ckas.setxi(&xi);
      ckas.setdata(&di);
      ckas.t.birth(1,0,50,-1,1);
      ckas.drawtheta(gen);
      ckas.t.pr();
      ckas.settc(4);
      //check cutpoint
      std::cout << "the cutpoint is " << xi[0][50] << std::endl;

      tree::npv bnv;
      //std::vector<brt::sinfo> siv;
      std::vector<sinfo> siv;
      std::cout << "num bots: " << ckas.t.nbots() << std::endl;
      ckas.allsuff(ckas.t,bnv,siv);
      std::cout << "length siv: " << siv.size() << std::endl;
      std::cout << siv[0].n << ", " << siv[1].n << std::endl;

      tree::npv subbnv;
      //std::vector<brt::sinfo> subsiv;
      std::vector<sinfo> subsiv;
      ckas.subsuff(&ckas.t,subbnv,subsiv); //lame test, since the "internal" node is just the root.
      std::cout << "length subsiv: " << subsiv.size() << std::endl;
      std::cout << subsiv[0].n << ", " << subsiv[1].n << std::endl;

      //add another node, rinse and repeat.
      ckas.t.birth(2,0,25,-1,1);
      ckas.drawtheta(gen);
      ckas.t.pr();

      tree::npv bnv2;
      //std::vector<brt::sinfo> siv2;
      std::vector<sinfo> siv2;
      std::cout << "num bots: " << ckas.t.nbots() << std::endl;
      ckas.allsuff(ckas.t,bnv2,siv2);
      std::cout << "length siv: " << siv2.size() << std::endl;
      std::cout << siv2[0].n << ", " << siv2[1].n << ", " << siv2[2].n << std::endl;

      tree::npv subbnv2;
      //std::vector<sinfo> subsiv2;
      ckas.subsuff(ckas.t.l,subbnv2,subsiv2); //lame test, since the "internal" node is just the root.
      std::cout << "length subsiv: " << subsiv2.size() << std::endl;
      std::cout << subsiv2[0].n << ", " << subsiv2[1].n << std::endl;
   }
   */
   }

   return 0;
}
