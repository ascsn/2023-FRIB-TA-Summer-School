//     brt.cpp: Base BT model class methods.
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


#include "brt.h"
#include "brtfuns.h"
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

using std::cout;
using std::endl;

//--------------------------------------------------
//a single iteration of the MCMC for brt model
void brt::draw(rn& gen)
{
   // Structural/topological proposal(s)
   if(gen.uniform()<mi.pbd)
//   if(mi.pbd>0.0)
      bd(gen);
   else
   {
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      rot(tnew,t,gen);
      delete tnew;
   }
   
   // Perturbation Proposal
   if(mi.dopert)
      pertcv(gen);

   // Gibbs Step
    drawtheta(gen);

   //update statistics
   if(mi.dostats) {
      tree::npv bnv; //all the bottom nodes
      for(size_t k=0;k< xi->size();k++) mi.varcount[k]+=t.nuse(k);
      t.getbots(bnv);
      unsigned int tempdepth[bnv.size()];
      unsigned int tempavgdepth=0;
      for(size_t i=0;i!=bnv.size();i++)
         tempdepth[i]=(unsigned int)bnv[i]->depth();
      for(size_t i=0;i!=bnv.size();i++) {
         tempavgdepth+=tempdepth[i];
         mi.tmaxd=std::max(mi.tmaxd,tempdepth[i]);
         mi.tmind=std::min(mi.tmind,tempdepth[i]);
      }
      mi.tavgd+=((double)tempavgdepth)/((double)bnv.size());
   }
}
//--------------------------------------------------
//slave controller for draw when using MPI
void brt::draw_mpislave(rn& gen)
{
   #ifdef _OPENMPI
   char buffer[SIZE_UINT3];
   int position=0;
   MPI_Status status;
   typedef tree::npv::size_type bvsz;

   // Structural/topological proposal(s)
   // MPI receive the topological proposal type and nlid and nrid if applicable.
   MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();
   if(status.MPI_TAG==MPI_TAG_BD_BIRTH_VC) {
      unsigned int nxid,v,c;
      tree::tree_p nx;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nxid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&v,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&c,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nx=t.getptr((size_t)nxid);
      getsuff(nx,(size_t)v,(size_t)c,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if(status2.MPI_TAG==MPI_TAG_BD_BIRTH_VC_ACCEPT) t.birthp(nx,(size_t)v,(size_t)c,0.0,0.0); //accept birth
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_BD_DEATH_LR) {
      unsigned int nlid,nrid;
      tree::tree_p nl,nr;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nlid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nrid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nl=t.getptr((size_t)nlid);
      nr=t.getptr((size_t)nrid);
      getsuff(nl,nr,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if(status2.MPI_TAG==MPI_TAG_BD_DEATH_LR_ACCEPT) t.deathp(nl->getp(),0.0); //accept death
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_ROTATE) {
      mpi_resetrn(gen);
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      rot(tnew,t,gen);
      delete tnew;
   }
   delete &tsil;
   delete &tsir;

   // Perturbation Proposal
   // nothing to perturb if tree is a single terminal node, so we would just skip.
   if(mi.dopert && t.treesize()>1)
   {
      tree::npv intnodes;
      tree::tree_p pertnode;
      t.getintnodes(intnodes);
      for(size_t pertdx=0;pertdx<intnodes.size();pertdx++)
      {
         std::vector<sinfo*>& sivold = newsinfovec();
         std::vector<sinfo*>& sivnew = newsinfovec();
         pertnode = intnodes[pertdx];
         MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         if(status.MPI_TAG==MPI_TAG_PERTCV)
         {
            size_t oldc = pertnode->getc();
            unsigned int propcint;
            position=0;
            MPI_Unpack(buffer,SIZE_UINT1,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            pertnode->setc(propc);
            tree::npv bnv;
            getpertsuff(pertnode,bnv,oldc,sivold,sivnew);
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCV_ACCEPT) pertnode->setc(propc); //accept new cutpoint
            //else reject, for which we do nothing.
         }
         else if(status.MPI_TAG==MPI_TAG_PERTCHGV)
         {
            size_t oldc = pertnode->getc();
            size_t oldv = pertnode->getv();
            bool didswap=false;
            unsigned int propcint;
            unsigned int propvint;
            position=0;
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[oldv],chv_lwr,chv_upr);
            MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propvint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&didswap,1,MPI_CXX_BOOL,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            size_t propv=(size_t)propvint;
            pertnode->setc(propc);
            pertnode->setv(propv);
            if(didswap)
               pertnode->swaplr();
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[propv],chv_lwr,chv_upr);
            tree::npv bnv;
            getchgvsuff(pertnode,bnv,oldc,oldv,didswap,sivold,sivnew);
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCHGV_ACCEPT) { //accept change var and pert
               pertnode->setc(propc);
               pertnode->setv(propv);
               if(didswap)
                  pertnode->swaplr();
            }
            // else reject, for which we do nothing.
         }
         // no other possibilities.
         for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
         for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
         delete &sivold;
         delete &sivnew;
      }
   }

   // Gibbs Step
   drawtheta(gen);

   #endif
}
//--------------------------------------------------
//adapt the proposal widths for perturb proposals,
//bd or rot proposals and b or d proposals.
void brt::adapt()
{
//   double pert_rate,b_rate,d_rate,bd_rate,rot_rate,m_rate,chgv_rate;
   double pert_rate,m_rate,chgv_rate;

   pert_rate=((double)mi.pertaccept)/((double)mi.pertproposal);
   chgv_rate=((double)mi.chgvaccept)/((double)mi.chgvproposal);
//   pert_rate=((double)(mi.pertaccept+mi.baccept+mi.daccept+mi.rotaccept))/((double)(mi.pertproposal+mi.dproposal+mi.bproposal+mi.rotproposal));
//   b_rate=((double)mi.baccept)/((double)mi.bproposal);
//   d_rate=((double)mi.daccept)/((double)mi.dproposal);
//   bd_rate=((double)(mi.baccept+mi.daccept))/((double)(mi.dproposal+mi.bproposal));
//   rot_rate=((double)mi.rotaccept)/((double)mi.rotproposal);
   m_rate=((double)(mi.baccept+mi.daccept+mi.rotaccept))/((double)(mi.dproposal+mi.bproposal+mi.rotproposal));

   //update pbd
   // a mixture between calibrating to m_rate (25%) and not moving too quickly away from
   // the existing probability of birth/death (75%):
//   mi.pbd=0.25*mi.pbd*m_rate/0.24+0.75*mi.pbd;
   // avoid too small or large by truncating to 0.1,0.9 range:
//   mi.pbd=std::max(std::min(0.9,mi.pbd),0.1);

   //update pb
//old   mi.pb=mi.pb*bd_rate/0.24;
//old   mi.pb=mi.pb*(b_rate+d_rate)/2.0/bd_rate;
   // a mixture between calibrating to the (bd_rate and m_rate) and existing probability of birth
   // in other words, don't move too quickly away from existing probability of birth
   // and when we do move, generally we favor targetting bd_rate (90%) but also target m_rate to
   // a small degree (10%):
//   mi.pb=0.25*(0.9*mi.pb*(b_rate+d_rate)/2.0/bd_rate + 0.1*mi.pb*m_rate/0.24)+0.75*mi.pb;
   // avoid too small or large by truncating to 0.1,0.9 range:
//   mi.pb=std::max(std::min(0.9,mi.pb),0.1);

   //update pertalpha
   mi.pertalpha=mi.pertalpha*pert_rate/0.44;
//   if(mi.pertalpha>2.0) mi.pertalpha=2.0;
//   if(mi.pertalpha>(1.0-1.0/ncp1)) mi.pertalpha=(1.0-1.0/ncp1);
   if(mi.pertalpha>2.0) mi.pertalpha=2.0;
   if(mi.pertalpha<(1.0/ncp1)) mi.pertalpha=(1.0/ncp1);

   mi.pertaccept=0; mi.baccept=0; mi.rotaccept=0; mi.daccept=0;
   mi.pertproposal=1; mi.bproposal=1; mi.rotproposal=1; mi.dproposal=1;
   //if(mi.dostats) {

#ifdef SILENT
   //Ugly hack to get rid of silly compiler warning
   if(m_rate) ;
   if(chgv_rate) ;
#endif

#ifndef SILENT
   cout << "pert_rate=" << pert_rate << " pertalpha=" << mi.pertalpha << " chgv_rate=" << chgv_rate;
   // cout << "   b_rate=" << b_rate << endl;
   // cout << "   d_rate=" << d_rate << endl;
   // cout << "   bd_rate=" << bd_rate << endl;
   // cout << " rot_rate=" << rot_rate << endl;
   cout << "   m_rate=" << m_rate;
   //   cout << "mi.pbd=" << mi.pbd << "  mi.pb=" << mi.pb<< "  mi.pertalpha=" << mi.pertalpha << endl;
   //   cout << endl;
#endif
   //}
}
//--------------------------------------------------
//draw all the bottom node theta's for the brt model
void brt::drawtheta(rn& gen)
{
   tree::npv bnv;
//   std::vector<sinfo> siv;
   std::vector<sinfo*>& siv = newsinfovec();

   allsuff(bnv,siv);
#ifdef _OPENMPI
   mpi_resetrn(gen);
#endif
   
   for(size_t i=0;i<bnv.size();i++) {
      bnv[i]->settheta(drawnodetheta(*(siv[i]),gen));
      delete siv[i]; //set it, then forget it!
   }
   delete &siv;  //and then delete the vector of pointers.
}
//--------------------------------------------------
//draw theta for a single bottom node for the brt model
double brt::drawnodetheta(sinfo& si, rn& gen)
{
//   return 1.0;
   return si.n;
}
//--------------------------------------------------
//pr for brt
void brt::pr()
{
   std::cout << "***** brt object:\n";
#ifdef _OPENMPI
   std::cout << "mpirank=" << rank << endl;
#endif
   if(xi) {
      size_t p = xi->size();
      cout  << "**xi cutpoints set:\n";
      cout << "\tnum x vars: " << p << endl;
      cout << "\tfirst x cuts, first and last " << (*xi)[0][0] << ", ... ," << 
              (*xi)[0][(*xi)[0].size()-1] << endl;
      cout << "\tlast x cuts, first and last " << (*xi)[p-1][0] << ", ... ," << 
              (*xi)[p-1][(*xi)[p-1].size()-1] << endl;
   } else {
      cout << "**xi cutpoints not set\n";
   }
   if(di) {
      cout << "**data set, n,p: " << di->n << ", " << di->p << endl;
   } else {
      cout << "**data not set\n";
   }
   std::cout << "**the tree:\n";
   t.pr();
}
//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double brt::lm(sinfo& si)
{
   return 0.0;  //just drawing from prior for now.
}
//--------------------------------------------------
//getsuff used for birth.
void brt::local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir)    
{
   double *xx;//current x
   sil.n=0; sir.n=0;

   for(;diter<diter.until();diter++)
   {
      xx = diter.getxp();
      if(nx==t.bn(diter.getxp(),*xi)) { //does the bottom node = xx's bottom node
         if(xx[v] < (*xi)[v][c]) {
               //sil.n +=1;
               add_observation_to_suff(diter,sil);
          } else {
               //sir.n +=1;
               add_observation_to_suff(diter,sir);
          }
      }
   }
}
//--------------------------------------------------
//getsuff used for death
void brt::local_getsuff(diterator& diter, tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir)
{
   sil.n=0; sir.n=0;

   for(;diter<diter.until();diter++)
   {
      tree::tree_cp bn = t.bn(diter.getxp(),*xi);
      if(bn==l) {
         //sil.n +=1;
         add_observation_to_suff(diter,sil);
      }
      if(bn==r) {
         //sir.n +=1;
         add_observation_to_suff(diter,sir);
      }
   }
}
//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void brt::add_observation_to_suff(diterator& diter, sinfo& si)
{
   si.n+=1; //in add_observation_to_suff
}
//--------------------------------------------------
//getsuff wrapper used for birth.  Calls serial or parallel code depending on how
//the code is compiled.
void brt::getsuff(tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir)
{
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompgetsuff(nx,v,c,*di,sil,sir); //faster if pass dinfo by value.
   #elif _OPENMPI
      local_mpigetsuff(nx,v,c,*di,sil,sir);
   #else
      diterator diter(di);
      local_getsuff(diter,nx,v,c,sil,sir);
   #endif
}
//--------------------------------------------------
//allsuff (1)
void brt::allsuff(tree::npv& bnv,std::vector<sinfo*>& siv)
{
   //get bots once and pass them around
   bnv.clear();
   t.getbots(bnv);

   #ifdef _OPENMP
      typedef tree::npv::size_type bvsz;
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
#     pragma omp parallel num_threads(tc)
      local_ompallsuff(*di,bnv,siv); //faster if pass di and bnv by value.
   #elif _OPENMPI
      diterator diter(di);
      local_mpiallsuff(diter,bnv,siv);
   #else
      diterator diter(di);
      local_allsuff(diter,bnv,siv); //will resize siv
   #endif
}
//--------------------------------------------------
//local_subsuff
void brt::local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observation
   size_t ni;         //the  index into vector of the current bottom node
   size_t index;      //the index into the path vector.
   double *x;
   tree::tree_p root=path[path.size()-1];

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   siv.clear();
   siv.resize(nb);

   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) { bnmap[bnv[i]]=i; siv[i]=newsinfo(); }

   for(;diter<diter.until();diter++) {
      index=path.size()-1;
      x=diter.getxp();
      if(root->xonpath(path,index,x,*xi)) { //x is on the subtree, 
         tbn = nx->bn(x,*xi);              //so get the right bn below interior node n.
         ni = bnmap[tbn];
         //siv[ni].n +=1;
         add_observation_to_suff(diter, *(siv[ni]));
      }
      //else this x doesn't map to the subtree so it's not added into suff stats.
   }
}
//-------------------------------------------------- 
//local_ompsubsuff
void brt::local_ompsubsuff(dinfo di, tree::tree_p nx, tree::npv& path, tree::npv bnv,std::vector<sinfo*>& siv)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   std::vector<sinfo*>& tsiv = newsinfovec(); //will be sized in local_subsuff
   diterator diter(&di,beg,end);
   local_subsuff(diter,nx,path,bnv,tsiv);

#  pragma omp critical
   {
      for(size_t i=0;i<siv.size();i++) *(siv[i]) += *(tsiv[i]);
   }

   for(size_t i=0;i<tsiv.size();i++) delete tsiv[i];
   delete &tsiv;
#endif
}
//--------------------------------------------------
//local_mpisubsuff
void brt::local_mpisubsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   if(rank==0) {
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      typedef tree::npv::size_type bvsz;
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
 
      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
   else
   {
      local_subsuff(diter,nx,path,bnv,siv);

      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
#endif
}

//--------------------------------------------------
//get suff stats for bots that are only below node n.
//NOTE!  subsuff is the only method for computing suff stats that does not
//       assume the root of the tree you're interested is brt.t.  Instead,
//       it takes the root of the tree to be the last entry in the path
//       vector.  In other words, for MCMC proposals that physically
//       construct a new proposed tree, t', suff stats must be computed
//       on t' using subsuff.  Using getsuff or allsuff is WRONG and will
//       result in undefined behaviour since getsuff/allsuff *assume* the 
//       the root of the tree is brt.t.
void brt::subsuff(tree::tree_p nx, tree::npv& bnv, std::vector<sinfo*>& siv)
{
   tree::npv path;

   bnv.clear();
   nx->getpathtoroot(path);  //path from n back to root
   nx->getbots(bnv);  //all bots ONLY BELOW node n!!

   #ifdef _OPENMP
      typedef tree::npv::size_type bvsz;
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
#     pragma omp parallel num_threads(tc)
      local_ompsubsuff(*di,nx,path,bnv,siv); //faster if pass di and bnv by value.
   #elif _OPENMPI
      diterator diter(di);
      local_mpisubsuff(diter,nx,path,bnv,siv);
   #else
      diterator diter(di);
      local_subsuff(diter,nx,path,bnv,siv);
   #endif
}

//--------------------------------------------------
//allsuff (2)
void brt::local_ompallsuff(dinfo di, tree::npv bnv,std::vector<sinfo*>& siv)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   std::vector<sinfo*>& tsiv = newsinfovec(); //will be sized in local_allsuff

   diterator diter(&di,beg,end);
   local_allsuff(diter,bnv,tsiv);

#  pragma omp critical
   {
      for(size_t i=0;i<siv.size();i++) *(siv[i]) += *(tsiv[i]);
   }
   
   for(size_t i=0;i<tsiv.size();i++) delete tsiv[i];
   delete &tsiv;
#endif
}

//--------------------------------------------------
// reset random number generator in MPI so it's the same on all nodes.
void brt::mpi_resetrn(rn& gen)
{
#ifdef _OPENMPI
   if(rank==0) {
      // reset the rn generator so they are the same on all nodes
      // so that we can draw random numbers in parallel on each node w/o communication.
      std::stringstream state;
      crn& tempgen=static_cast<crn&>(gen);

      state << tempgen.get_engine_state();
      unsigned long ulstate = std::stoul(state.str(),nullptr,0);

//      cout << "state is " << state.str() << " and ul is " << ulstate << endl;

      MPI_Request *request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&ulstate,1,MPI_UNSIGNED_LONG,i,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
   }
   else
   {
      unsigned long ulstate;
      MPI_Status status;

      MPI_Recv(&ulstate,1,MPI_UNSIGNED_LONG,0,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&status); 

      std::string strstate=std::to_string(ulstate);
      std::stringstream state;
      state << strstate;

//      cout << "(slave) state is " << state.str() << " and ul is " << ulstate << endl;

      crn& tempgen=static_cast<crn&>(gen);
      tempgen.set_engine_state(state);
     }
/*   if(rank==0) {
      MPI_Request *request=new MPI_Request[tc];
      // reset the rn generator so they are the same on all nodes
      // so that we can draw random numbers in parallel on each node w/o communication.
      time_t timer;
      struct tm y2k = {0};
      int seconds;
      y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
      y2k.tm_year = 118; y2k.tm_mon = 0; y2k.tm_mday = 7;

      time(&timer);  // get current time
      seconds=(int)difftime(timer,mktime(&y2k));

      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&seconds,1,MPI_INT,i,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&request[i-1]);
      }

      crn& tempgen=static_cast<crn&>(gen);
      tempgen.set_seed(seconds);
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
// cout << "0) Reset seconds: " << seconds << " gen.unif:" << gen.uniform() << " gen.unif:" << gen.uniform() << endl;
   }
   else
   {
      int seconds;
      MPI_Status status;
      MPI_Recv(&seconds,1,MPI_INT,0,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&status);
      crn& tempgen=static_cast<crn&>(gen);
      tempgen.set_seed(seconds);
// cout << "1) Reset seconds: " << seconds << " gen.unif:" << gen.uniform() << " gen.unif:" << gen.uniform() << endl;
   }*/
#endif
}
//--------------------------------------------------
//allsuff (2) -- MPI version
void brt::local_mpiallsuff(diterator& diter, tree::npv& bnv,std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   if(rank==0) {
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      typedef tree::npv::size_type bvsz;
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
 
      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
   else
   {
      local_allsuff(diter,bnv,siv);

      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
#endif
}
//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void brt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   unsigned int nvec[siv.size()];

   // cast to int
   for(size_t i=0;i<siv.size();i++)
      nvec[i]=(unsigned int)siv[i]->n;  // on root node, this should be 0 because of newsinfo().
// cout << "pre:" << siv[0]->n << " " << siv[1]->n << endl;
   // MPI sum
//   MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
   if(rank==0) {
      MPI_Status status;
      unsigned int tempvec[siv.size()];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            nvec[j]+=tempvec[j];
      }

      MPI_Request *request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
      }

      // cast back to size_t
      for(size_t i=0;i<siv.size();i++)
         siv[i]->n=(size_t)nvec[i];

      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
   }
   else {
      MPI_Request *request=new MPI_Request;
      MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
      MPI_Status status;
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;

      MPI_Recv(&nvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
      // cast back to size_t
      for(size_t i=0;i<siv.size();i++)
         siv[i]->n=(size_t)nvec[i];
   }

   // cast back to size_t
   // for(size_t i=0;i<siv.size();i++)
   //    siv[i]->n=(size_t)nvec[i];
// cout << "reduced:" << siv[0]->n << " " << siv[1]->n << endl;
#endif
}
//--------------------------------------------------
//allsuff (3)
void brt::local_allsuff(diterator& diter, tree::npv& bnv,std::vector<sinfo*>& siv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observations
   size_t ni;         //the  index into vector of the current bottom node

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   siv.clear();
   siv.resize(nb);

   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) { bnmap[bnv[i]]=i; siv[i]=newsinfo(); }

   for(;diter<diter.until();diter++) {
      tbn = t.bn(diter.getxp(),*xi);
      ni = bnmap[tbn];
      //siv[ni].n +=1; 
      add_observation_to_suff(diter, *(siv[ni]));
   }
}
/*
//--------------------------------------------------
//get suff stats for nodes related to change of variable proposal
//this is simply the allsuff for all nodes under the perturb node, not the entire tree.
void brt::getchgvsuff(tree::tree_p pertnode, tree::npv& bnv, size_t oldc, size_t oldv, bool didswap, 
                  std::vector<sinfo*>& sivold, std::vector<sinfo*>& sivnew)
{
   subsuff(pertnode,bnv,sivnew);
   if(didswap) pertnode->swaplr();  //undo the swap so we can calculate the suff stats for the original variable, cutpoint.
   pertnode->setv(oldv);
   pertnode->setc(oldc);
   subsuff(pertnode,bnv,sivold);
}

//--------------------------------------------------
//get suff stats for nodes related to perturb proposal
//this is simply the allsuff for all nodes under the perturb node, not the entire tree.
void brt::getpertsuff(tree::tree_p pertnode, tree::npv& bnv, size_t oldc, 
                  std::vector<sinfo*>& sivold, std::vector<sinfo*>& sivnew)
{
   subsuff(pertnode,bnv,sivnew);
   pertnode->setc(oldc);
   subsuff(pertnode,bnv,sivold);
}
*/
//--------------------------------------------------
//getsuff wrapper used for death.  Calls serial or parallel code depending on how
//the code is compiled.
void brt::getsuff(tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir)
{
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompgetsuff(l,r,*di,sil,sir); //faster if pass dinfo by value.
   #elif _OPENMPI
      local_mpigetsuff(l,r,*di,sil,sir);
   #else
         diterator diter(di);
         local_getsuff(diter,l,r,sil,sir);
   #endif
}

//--------------------------------------------------
//--------------------------------------------------
//#ifdef _OPENMP
//--------------------------------------------------
//openmp version of getsuff for birth
void brt::local_ompgetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();

   diterator diter(&di,beg,end);
   local_getsuff(diter,nx,v,c,tsil,tsir);

#  pragma omp critical
   {
      sil+=tsil; sir+=tsir;
   }

   delete &tsil;
   delete &tsir;
#endif
}
//--------------------------------------------------
//opemmp version of getsuff for death
void brt::local_ompgetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

//   sinfo tsil, tsir;
   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();

   diterator diter(&di,beg,end);
   local_getsuff(diter,l,r,tsil,tsir);

#  pragma omp critical
   {
      sil+=tsil; sir+=tsir;
   }

   delete &tsil;
   delete &tsir;
#endif
}
//#endif

//--------------------------------------------------
//--------------------------------------------------
//#ifdef _OPENMPI
//--------------------------------------------------
// MPI version of getsuff for birth
void brt::local_mpigetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   if(rank==0) {
      char buffer[SIZE_UINT3];
      int position=0;
      MPI_Request *request=new MPI_Request[tc];
      const int tag=MPI_TAG_BD_BIRTH_VC;
      unsigned int vv,cc,nxid;

      vv=(unsigned int)v;
      cc=(unsigned int)c;
      nxid=(unsigned int)nx->nid();

      // Pack and send info to the slaves
      MPI_Pack(&nxid,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&vv,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&cc,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);

      // MPI receive all the answers from the slaves
      local_mpi_sr_suffs(sil,sir);
      delete[] request;
   }
   else
   {
      diterator diter(&di);
      local_getsuff(diter,nx,v,c,sil,sir);

      // MPI send all the answers to root
      local_mpi_sr_suffs(sil,sir);
   }
#endif
}
//--------------------------------------------------
// MPI version of getsuff for death
void brt::local_mpigetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   if(rank==0) {
      char buffer[SIZE_UINT3];
      int position=0;  
      MPI_Request *request=new MPI_Request[tc];
      const int tag=MPI_TAG_BD_DEATH_LR;
      unsigned int nlid,nrid;

      nlid=(unsigned int)l->nid();
      nrid=(unsigned int)r->nid();

      // Pack and send info to the slaves
      MPI_Pack(&nlid,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&nrid,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {   
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);

      // MPI receive all the answers from the slaves
      local_mpi_sr_suffs(sil,sir);

      delete[] request;
   }
   else {
      diterator diter(&di);
      local_getsuff(diter,l,r,sil,sir);

      // MPI send all the answers to root
      local_mpi_sr_suffs(sil,sir);
   }
#endif
}
//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
// This is model-dependent.
void brt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   if(rank==0) { // MPI receive all the answers from the slaves
      MPI_Status status;
      sinfo& tsil = *newsinfo();
      sinfo& tsir = *newsinfo();
      char buffer[SIZE_UINT2];
      int position=0;
      unsigned int ln,rn;      
      for(size_t i=1; i<=(size_t)tc; i++) {
         position=0;
         MPI_Recv(buffer,SIZE_UINT2,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer,SIZE_UINT2,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT2,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         tsil.n=(size_t)ln;
         tsir.n=(size_t)rn;
         sil+=tsil;
         sir+=tsir;
      }
      delete &tsil;
      delete &tsir;
   }
   else // MPI send all the answers to root
   {
      char buffer[SIZE_UINT2];
      int position=0;  
      unsigned int ln,rn;
      ln=(unsigned int)sil.n;
      rn=(unsigned int)sir.n;
      MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,SIZE_UINT2,&position,MPI_COMM_WORLD);
      MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,SIZE_UINT2,&position,MPI_COMM_WORLD);
      MPI_Send(buffer,SIZE_UINT2,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }
#endif   
}

//--------------------------------------------------
//--------------------------------------------------
//set the vector of predicted values
void brt::setf() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetf(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setf(diter);
   #endif
}
void brt::local_ompsetf(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setf(diter);
#endif
}
void brt::local_setf(diterator& diter)
{
   tree::tree_p bn;

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      yhat[*diter] = bn->gettheta();
   }
}
//--------------------------------------------------
//set the vector of residual values
void brt::setr() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetr(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setr(diter);
   #endif
}
void brt::local_ompsetr(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setr(diter);
#endif
}
void brt::local_setr(diterator& diter)
{
   tree::tree_p bn;

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      resid[*diter] = 0.0 - bn->gettheta();
//      resid[*diter] = di->y[*diter] - bn->gettheta();
   }
}
//--------------------------------------------------
//predict the response at the (npred x p) input matrix *x
//Note: the result appears in *dipred.y.
void brt::predict(dinfo* dipred) {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_omppredict(*dipred); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_predict(diter);
   #endif
}
void brt::local_omppredict(dinfo dipred)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_predict(diter);
#endif
}
void brt::local_predict(diterator& diter)
{
   tree::tree_p bn;

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      diter.sety(bn->gettheta());
   }
}
//--------------------------------------------------
//save/load tree to/from vector format
//Note: for single tree models the parallelization just directs
//      back to the serial path (ie no parallel execution occurs).
//      For multi-tree models, the parallelization occurs in the
//      definition of that models class.
//void brt::savetree(int* id, int* v, int* c, double* theta)
void brt::savetree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_ompsavetree(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_savetree(iter,beg,end,nn,id,v,c,theta);
   #endif
}
//void brt::local_ompsavetree(int* id, int* v, int* c, double* theta)
void brt::local_ompsavetree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_savetree(iter,beg,end,nn,id,v,c,theta);
#endif
}
void brt::local_savetree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
     std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   nn[iter]=t.treesize();
   id[iter].resize(nn[iter]);
   v[iter].resize(nn[iter]);
   c[iter].resize(nn[iter]);
   theta[iter].resize(nn[iter]);
   t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0]);
}
void brt::loadtree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_omploadtree(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_loadtree(iter,beg,end,nn,id,v,c,theta);
   #endif
}
//void brt::local_omploadtree(size_t nn, int* id, int* v, int* c, double* theta)
void brt::local_omploadtree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_loadtree(iter,beg,end,nn,id,v,c,theta);
#endif
}
void brt::local_loadtree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   t.vectotree(nn[iter],&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0]);
}

//--------------------------------------------------
//--------------------------------------------------
//bd: birth/death
void brt::bd(rn& gen)
{
//   cout << "--------------->>into bd" << endl;
   tree::npv goodbots;  //nodes we could birth at (split on)
   double PBx = getpb(t,*xi,mi.pb,goodbots); //prob of a birth at x

   if(gen.uniform() < PBx) { //do birth or death
      mi.bproposal++;
      //--------------------------------------------------
      //draw proposal
      tree::tree_p nx; //bottom node
      size_t v,c; //variable and cutpoint
      double pr; //part of metropolis ratio from proposal and prior
      bprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,v,c,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();

      getsuff(nx,v,c,sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      bool hardreject=true;
      double lalpha=0.0;
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      if((sil.n>=mi.minperbot) && (sir.n>=mi.minperbot)) { 
         lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
         hardreject=false;
         lalpha = log(pr) + (lml+lmr-lmt);
         lalpha = std::min(0.0,lalpha);
      }
      //--------------------------------------------------
      //try metrop
      double thetal,thetar; //parameters for new bottom nodes, left and right
      double uu = gen.uniform();
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if( !hardreject && (log(uu) < lalpha) ) {
         thetal = 0.0;//drawnodetheta(sil,gen);
         thetar = 0.0;//drawnodetheta(sir,gen);
         t.birthp(nx,v,c,thetal,thetar);
         mi.baccept++;
#ifdef _OPENMPI
//        cout << "accept birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   } else {
      mi.dproposal++;
      //--------------------------------------------------
      //draw proposal
      double pr;  //part of metropolis ratio from proposal and prior
      tree::tree_p nx; //nog node to death at
      dprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      //sinfo sil,sir,sit;
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();
      getsuff(nx->getl(),nx->getr(),sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
      double lalpha = log(pr) + (lmt - lml - lmr);
      lalpha = std::min(0.0,lalpha);

      //--------------------------------------------------
      //try metrop
      double theta;
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if(log(gen.uniform()) < lalpha) {
         theta = 0.0;//drawnodetheta(sit,gen);
         t.deathp(nx,theta);
         mi.daccept++;
#ifdef _OPENMPI
//        cout << "accept death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
}
//--------------------------------------------------
//mpislave_bd: birth/death code on the slave side
void mpislave_bd(rn& gen)
{


}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//Model Mixing functions for brt.cpp
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//Draw function for vector parameters -- calls drawthetavec
void brt::drawvec(rn& gen)
{
   // Structural/topological proposal(s)
   if(gen.uniform()<mi.pbd){
//   if(mi.pbd>0.0)
      //std::cout << "bd" << std::endl;
      bd_vec(gen);
   }
   else
   {
      //std::cout << "Rotate" << std::endl; 
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      //t.pr_vec();
      rot(tnew,t,gen);
      //t.pr_vec();
      delete tnew;
   }

   // Perturbation Proposal
   if(mi.dopert)
      pertcv(gen);

   // Gibbs Step
    drawthetavec(gen);

   //update statistics
   if(mi.dostats) {
      tree::npv bnv; //all the bottom nodes
      for(size_t k=0;k< xi->size();k++) mi.varcount[k]+=t.nuse(k);
      t.getbots(bnv);
      unsigned int tempdepth[bnv.size()];
      unsigned int tempavgdepth=0;
      for(size_t i=0;i!=bnv.size();i++)
         tempdepth[i]=(unsigned int)bnv[i]->depth();
      for(size_t i=0;i!=bnv.size();i++) {
         tempavgdepth+=tempdepth[i];
         mi.tmaxd=std::max(mi.tmaxd,tempdepth[i]);
         mi.tmind=std::min(mi.tmind,tempdepth[i]);
      }
      mi.tavgd+=((double)tempavgdepth)/((double)bnv.size());
   }
}


//Draw theta vector -- samples the theta vector and assigns to tree 
void brt::drawthetavec(rn& gen)
{
   tree::npv bnv;
//   std::vector<sinfo> siv;
   std::vector<sinfo*>& siv = newsinfovec();

  allsuff(bnv,siv);
#ifdef _OPENMPI
  mpi_resetrn(gen);
#endif
   for(size_t i=0;i<bnv.size();i++) {
      bnv[i]->setthetavec(drawnodethetavec(*(siv[i]),gen));
      delete siv[i]; //set it, then forget it!
   }
   delete &siv;  //and then delete the vector of pointers.
}

//--------------------------------------------------
//draw theta for a single bottom node for the brt model
Eigen::VectorXd brt::drawnodethetavec(sinfo& si, rn& gen)
{
//   return 1.0;
   Eigen::VectorXd sin_vec(k); //cast si.n to a vector of dimension 1.
   for(size_t i = 0; i<k; i++){
      sin_vec(i) = si.n; //Input si.n into each vector component
   }
   return sin_vec;
}

//--------------------------------------------------
//slave controller for draw when using MPI
void brt::drawvec_mpislave(rn& gen)
{
   #ifdef _OPENMPI
   char buffer[SIZE_UINT3];
   int position=0;
   MPI_Status status;
   typedef tree::npv::size_type bvsz;

   // Structural/topological proposal(s)
   // MPI receive the topological proposal type and nlid and nrid if applicable.
   MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();
   vxd theta0(k);
   theta0 = vxd::Zero(k);
   if(status.MPI_TAG==MPI_TAG_BD_BIRTH_VC) {
      unsigned int nxid,v,c;
      tree::tree_p nx;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nxid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&v,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&c,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nx=t.getptr((size_t)nxid);
      getsuff(nx,(size_t)v,(size_t)c,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if(status2.MPI_TAG==MPI_TAG_BD_BIRTH_VC_ACCEPT) t.birthp(nx,(size_t)v,(size_t)c,theta0,theta0); //accept birth
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_BD_DEATH_LR) {
      unsigned int nlid,nrid;
      tree::tree_p nl,nr;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nlid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nrid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nl=t.getptr((size_t)nlid);
      nr=t.getptr((size_t)nrid);
      getsuff(nl,nr,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if(status2.MPI_TAG==MPI_TAG_BD_DEATH_LR_ACCEPT) t.deathp(nl->getp(),theta0); //accept death
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_ROTATE) {
      mpi_resetrn(gen);
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      rot(tnew,t,gen);
      delete tnew;
   }
   delete &tsil;
   delete &tsir;

   // Perturbation Proposal
   // nothing to perturb if tree is a single terminal node, so we would just skip.
   if(mi.dopert && t.treesize()>1)
   {
      tree::npv intnodes;
      tree::tree_p pertnode;
      t.getintnodes(intnodes);
      for(size_t pertdx=0;pertdx<intnodes.size();pertdx++)
      {
         std::vector<sinfo*>& sivold = newsinfovec();
         std::vector<sinfo*>& sivnew = newsinfovec();
         pertnode = intnodes[pertdx];
         MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         if(status.MPI_TAG==MPI_TAG_PERTCV)
         {
            size_t oldc = pertnode->getc();
            unsigned int propcint;
            position=0;
            MPI_Unpack(buffer,SIZE_UINT1,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            pertnode->setc(propc);
            tree::npv bnv;
            getpertsuff(pertnode,bnv,oldc,sivold,sivnew);
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCV_ACCEPT) pertnode->setc(propc); //accept new cutpoint
            //else reject, for which we do nothing.
         }
         else if(status.MPI_TAG==MPI_TAG_PERTCHGV)
         {
            size_t oldc = pertnode->getc();
            size_t oldv = pertnode->getv();
            bool didswap=false;
            unsigned int propcint;
            unsigned int propvint;
            position=0;
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[oldv],chv_lwr,chv_upr);
            MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propvint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&didswap,1,MPI_CXX_BOOL,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            size_t propv=(size_t)propvint;
            pertnode->setc(propc);
            pertnode->setv(propv);
            if(didswap)
               pertnode->swaplr();
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[propv],chv_lwr,chv_upr);
            tree::npv bnv;
            getchgvsuff(pertnode,bnv,oldc,oldv,didswap,sivold,sivnew);
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCHGV_ACCEPT) { //accept change var and pert
               pertnode->setc(propc);
               pertnode->setv(propv);
               if(didswap)
                  pertnode->swaplr();
            }
            // else reject, for which we do nothing.
         }
         // no other possibilities.
         for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
         for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
         delete &sivold;
         delete &sivnew;
      }
   }

   // Gibbs Step
   drawthetavec(gen);

   #endif
}


//--------------------------------------------------
//Model Mixing Birth and Death
//--------------------------------------------------
//bd_vec: birth/death for vector parameters
void brt::bd_vec(rn& gen)
{
//   cout << "--------------->>into bd" << endl;
   tree::npv goodbots;  //nodes we could birth at (split on)
   double PBx = getpb(t,*xi,mi.pb,goodbots); //prob of a birth at x

   if(gen.uniform() < PBx) { //do birth or death
      mi.bproposal++;
      //std::cout << "Birth" << std::endl;
      //--------------------------------------------------
      //draw proposal
      tree::tree_p nx; //bottom node
      size_t v,c; //variable and cutpoint
      double pr; //part of metropolis ratio from proposal and prior
      bprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,v,c,pr,gen);
      
      //--------------------------------------------------
      //compute sufficient statistics
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();
      
      getsuff(nx,v,c,sil,sir);
      
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      bool hardreject=true;
      double lalpha=0.0;
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      if((sil.n>=mi.minperbot) && (sir.n>=mi.minperbot)) { 
         lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
         hardreject=false;
         lalpha = log(pr) + (lml+lmr-lmt);
         //std::cout << "lml" << lml << std::endl;
         //std::cout << "lmr" << lmr << std::endl;
         //std::cout << "lmt" << lmt << std::endl;
         //std::cout << "lalpha" << lalpha << std::endl;
         lalpha = std::min(0.0,lalpha);
      }
      //--------------------------------------------------
      //try metrop
      Eigen::VectorXd thetavecl,thetavecr; //parameters for new bottom nodes, left and right
      double uu = gen.uniform();
      //std::cout << "lu" << log(uu) << std::endl;
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if( !hardreject && (log(uu) < lalpha) ) {
         thetavecl = Eigen::VectorXd:: Zero(k); 
         thetavecr = Eigen::VectorXd:: Zero(k); 
         t.birthp(nx,v,c,thetavecl,thetavecr);
         mi.baccept++;
#ifdef _OPENMPI
//        cout << "accept birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   } else {
      mi.dproposal++;
      //std::cout << "Death" << std::endl;
      //--------------------------------------------------
      //draw proposal
      double pr;  //part of metropolis ratio from proposal and prior
      tree::tree_p nx; //nog node to death at
      dprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      //sinfo sil,sir,sit;
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();
      getsuff(nx->getl(),nx->getr(),sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
      double lalpha = log(pr) + (lmt - lml - lmr);
      lalpha = std::min(0.0,lalpha);

      //--------------------------------------------------
      //try metrop
      Eigen::VectorXd thetavec(k);
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if(log(gen.uniform()) < lalpha) {
         thetavec = Eigen::VectorXd::Zero(k); 
         t.deathp(nx,thetavec);
         mi.daccept++;
#ifdef _OPENMPI
//        cout << "accept death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
}

//--------------------------------------------------
//Model Mixing - set residuals and fitted values
//--------------------------------------------------
//set the vector of predicted values
void brt::setf_mix() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetf_mix(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setf_mix(diter);
   #endif
}
void brt::local_ompsetf_mix(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setf_mix(diter);
#endif
}
void brt::local_setf_mix(diterator& diter)
{
   tree::tree_p bn;
   vxd thetavec_temp(k); //Initialize a temp vector to facilitate the fitting
   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec(); 
      yhat[*diter] = (*fi).row(*diter)*thetavec_temp;
   }
}

//--------------------------------------------------
//set the vector of residual values
void brt::setr_mix() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetr_mix(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setr_mix(diter);
   #endif
}
void brt::local_ompsetr_mix(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setr_mix(diter);
#endif
}
void brt::local_setr_mix(diterator& diter)
{
   tree::tree_p bn;
   vxd thetavec_temp(k); //Initialize a temp vector to facilitate the fitting

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec();
      resid[*diter] = di->y[*diter] - (*fi).row(*diter)*thetavec_temp;
   }
}
//--------------------------------------------------
//predict the response at the (npred x p) input matrix *x
//Note: the result appears in *dipred.y.
void brt::predict_mix(dinfo* dipred, finfo* fipred) {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_omppredict_mix(*dipred, *fipred); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_predict_mix(diter, *fipred);
   #endif
}

//Local predictions for model mixing over omp
void brt::local_omppredict_mix(dinfo dipred, finfo fipred)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_predict_mix(diter, fipred);
#endif
}

//Local preditions for model mixing
void brt::local_predict_mix(diterator& diter, finfo& fipred){
   tree::tree_p bn;
   vxd thetavec_temp(k); 
   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec();
      diter.sety(fipred.row(*diter)*thetavec_temp);
   }
}

//Mix using the discrepancy
void brt::predict_mix_fd(dinfo* dipred, finfo* fipred, finfo* fpdmean, finfo* fpdsd, rn& gen) {
   size_t np = (*fpdmean).rows();
   finfo fdpred(np,k);
   double z;
   //Update the fdpred matrix to sum the point estimates + random discrepancy: fipred + fidelta 
   for(size_t i = 0; i<np; i++){
        for(size_t j=0; j<k;j++){
           z = gen.normal();
           fdpred(i,j) = (*fipred)(i,j) + (*fpdmean)(i,j) + (*fpdsd)(i,j)*z; 
        }
    }
   //cout << fdpred << endl;
   //Run the same functions -- just now using the updated prediction matrix
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_omppredict_mix(*dipred, fdpred); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_predict_mix(diter, fdpred);
   #endif
}

//--------------------------------------------------
//Get modeling mixing weights
void brt::get_mix_wts(dinfo* dipred, mxd *wts){
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompget_mix_wts(*dipred, *wts); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_get_mix_wts(diter, *wts);
   #endif   
}

void brt::local_get_mix_wts(diterator &diter, mxd &wts){
   tree::tree_p bn;
   vxd thetavec_temp(k); 
   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec();
      wts.col(*diter) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix. 
   }
}

void brt::local_ompget_mix_wts(dinfo dipred, mxd wts){
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_get_mix_wts(diter, wts);
#endif
}

//--------------------------------------------------
//Get modeling mixing weights per tree
void brt::get_mix_theta(dinfo* dipred, mxd *wts){
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompget_mix_theta(*dipred, *wts); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_get_mix_theta(diter, *wts);
   #endif   
}

void brt::local_get_mix_theta(diterator &diter, mxd &wts){
   tree::tree_p bn;
   vxd thetavec_temp(k);
   bool enter = true; 
   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec();
      if(enter){
         wts.col(0) = thetavec_temp; //sets the thetavec to be the 1st column of the wts eigen matrix.
         enter = false;
      }
   }
}

void brt::local_ompget_mix_theta(dinfo dipred, mxd wts){
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_get_mix_theta(diter, wts);
#endif
}


//--------------------------------------------------
//Print for brt with vector parameters
void brt::pr_vec(){
   std::cout << "***** brt object:\n";
#ifdef _OPENMPI
   std::cout << "mpirank=" << rank << endl;
#endif
   if(xi) {
      size_t p = xi->size();
      cout  << "**xi cutpoints set:\n";
      cout << "\tnum x vars: " << p << endl;
      cout << "\tfirst x cuts, first and last " << (*xi)[0][0] << ", ... ," << 
              (*xi)[0][(*xi)[0].size()-1] << endl;
      cout << "\tlast x cuts, first and last " << (*xi)[p-1][0] << ", ... ," << 
              (*xi)[p-1][(*xi)[p-1].size()-1] << endl;
   } else {
      cout << "**xi cutpoints not set\n";
   }
   if(di) {
      cout << "**data set, n,p: " << di->n << ", " << di->p << endl;
   } else {
      cout << "**data not set\n";
   }
   std::cout << "**the tree:\n";
   t.pr_vec();   
}


//--------------------------------------------------
//save/load tree to/from vector format -- for these functions, each double vector is of length k*nn. 
//Save tree with vector parameters
void brt::savetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_ompsavetree_vec(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_savetree_vec(iter,beg,end,nn,id,v,c,theta);
   #endif
}

//void brt::local_ompsavetree(int* id, int* v, int* c, double* theta)
void brt::local_ompsavetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_savetree_vec(iter,beg,end,nn,id,v,c,theta);
#endif
}
void brt::local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
     std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   nn[iter]=t.treesize();
   id[iter].resize(nn[iter]);
   v[iter].resize(nn[iter]);
   c[iter].resize(nn[iter]);
   theta[iter].resize(k*nn[iter]);

   //t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0]);
   t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0], k);
}
void brt::loadtree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_omploadtree_vec(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_loadtree_vec(iter,beg,end,nn,id,v,c,theta);
   #endif
}

//void brt::local_omploadtree(size_t nn, int* id, int* v, int* c, double* theta)
void brt::local_omploadtree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_loadtree_vec(iter,beg,end,nn,id,v,c,theta);
#endif
}
void brt::local_loadtree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   t.vectotree(nn[iter],&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0],k);
}


/*
//--------------------------------------------------
//peturb proposal for internal node cut points.
void brt::pertcv(rn& gen)
{
//   cout << "--------------->>into pertcv" << endl;
   tree::tree_p pertnode;
   if(t.treesize()==1) // nothing to perturb if the tree is a single terminal node
      return;

   // Get interior nodes and propose new split value
   tree::npv intnodes;
   t.getintnodes(intnodes);
   for(size_t pertdx=0;pertdx<intnodes.size();pertdx++)
   if(gen.uniform()<mi.pchgv) {
      mi.chgvproposal++;
      pertnode = intnodes[pertdx];

      //get L,U for the old variable and save it as well as oldc
      int Lo,Uo;
      getLU(pertnode,*xi,&Lo,&Uo);
      size_t oldc = pertnode->getc();

      //update correlation matrix
      bool didswap=false;
      size_t oldv=pertnode->getv();
      size_t newv;
#ifdef _OPENMPI 
      MPI_Request *request = new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCHGV,MPI_COMM_WORLD,&request[i-1]);
      }
      std::vector<double> chgvrow;
      chgvrow=(*mi.corv)[oldv]; 
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      mpi_update_norm_cormat(rank,tc,pertnode,*xi,chgvrow,chv_lwr,chv_upr);
      newv=getchgvfromrow(oldv,chgvrow,gen);
#else
      std::vector<std::vector<double> > chgv;
      chgv= *mi.corv; //initialize it
      updatecormat(pertnode,*xi,chgv);
      normchgvrow(oldv,chgv);
      newv=getchgv(oldv,chgv,gen);
#endif

      //choose new variable randomly
      pertnode->setv(newv);
      if((*mi.corv)[oldv][newv]<0.0) {
         pertnode->swaplr();
         didswap=true;
      }

      //get L,U for the new variable and save it and set newc
      int Ln,Un;
      getLU(pertnode,*xi,&Ln,&Un);
      size_t newc = Ln + (size_t)(floor(gen.uniform()*(Un-Ln+1.0)));
      pertnode->setc(newc);

#ifdef _OPENMPI
      unsigned int propcint=(unsigned int)newc;
      unsigned int propvint=(unsigned int)newv;
      request = new MPI_Request[tc];
      char buffer[SIZE_UINT3];
      int position=0;
      MPI_Pack(&propcint,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&propvint,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&didswap,1,MPI_CXX_BOOL,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,MPI_TAG_PERTCHGV,MPI_COMM_WORLD,&request[i-1]);
      }
      std::vector<double> chgvrownew;
      chgvrownew=(*mi.corv)[newv];
 
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      mpi_update_norm_cormat(rank,tc,pertnode,*xi,chgvrownew,chv_lwr,chv_upr);
      if(chgvrownew[oldv]==0.0)
         cout << "Proposal newv cannot return to oldv!  This is not possible!" << endl;

      double alpha0=chgvrownew[oldv]/chgvrow[newv];  //proposal ratio for newv->oldv and oldv->newv
#else
      //now we also need to update the row of chgv for newv->oldv to calc MH correctly
      updatecormat(pertnode,*xi,chgv);
      normchgvrow(newv,chgv);
      //sanity check:
      if(chgv[newv][oldv]==0.0)
         cout << "Proposal newv cannot return to oldv!  This is not possible!" << endl;
      double alpha0=chgv[newv][oldv]/chgv[oldv][newv];  //proposal ratio for newv->oldv and oldv->newv
#endif

      //get sufficient statistics and calculate lm
      std::vector<sinfo*>& sivold = newsinfovec();
      std::vector<sinfo*>& sivnew = newsinfovec();
      tree::npv bnv;
      getchgvsuff(pertnode,bnv,oldc,oldv,didswap,sivold,sivnew);

      typedef tree::npv::size_type bvsz;
      double lmold,lmnew;
      bool hardreject=false;
      lmold=0.0;
      for(bvsz j=0;j!=sivold.size();j++) {
         if(sivold[j]->n < mi.minperbot)
            cout << "Error: old tree has some bottom nodes with <minperbot observations!" << endl;
         lmold += lm(*(sivold[j]));
      }

      lmnew=0.0;
      for(bvsz j=0;j!=sivnew.size();j++) {
         if(sivnew[j]->n < mi.minperbot)
            hardreject=true;
         lmnew += lm(*(sivnew[j]));
      }
      double alpha1 = ((double)(Uo-Lo+1.0))/((double)(Un-Ln+1.0)); //from prior for cutpoints
      double alpha2=alpha0*alpha1*exp(lmnew-lmold);
      double alpha = std::min(1.0,alpha2);
      if(hardreject) alpha=0.0;  //change of variable led to an bottom node with <minperbot observations in it, we reject this.
#ifdef _OPENMPI
      request = new MPI_Request[tc];
#endif

      if(gen.uniform()<alpha) {
         mi.chgvaccept++;
         if(didswap) pertnode->swaplr();  //because the call to getchgvsuff unswaped if they were swapped
         pertnode->setv(newv); //because the call to getchgvsuff changes it back to oldv to calc the old lil
         pertnode->setc(newc); //because the call to getchgvsuff changes it back to oldc to calc the old lil
#ifdef _OPENMPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCHGV_ACCEPT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCHGV_REJECT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      //else nothing, pertnode->c and pertnode->v is already reset to the old values and if a swap was done in the 
      //proposal it was already undone by getchgvsuff.
      for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
      for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
      delete &sivold;
      delete &sivnew;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
   else {
      mi.pertproposal++;
      pertnode = intnodes[pertdx];

      // Get allowable range for perturbing cv at pertnode
      int L,U;
      bool hardreject=false;
      getLU(pertnode,*xi,&L,&U);
      size_t oldc = pertnode->getc();
      int ai,bi,oldai,oldbi;
      ai=(int)(floor(oldc-mi.pertalpha*(U-L+1)/2.0));
      bi=(int)(floor(oldc+mi.pertalpha*(U-L+1)/2.0));
      ai=std::max(ai,L);
      bi=std::min(bi,U);
      size_t propc = ai + (size_t)(floor(gen.uniform()*(bi-ai+1.0)));
      pertnode->setc(propc);
#ifdef _OPENMPI
         unsigned int propcint=(unsigned int)propc;
         MPI_Request *request = new MPI_Request[tc];
         char buffer[SIZE_UINT1];
         int position=0;
         MPI_Pack(&propcint,1,MPI_UNSIGNED,buffer,SIZE_UINT1,&position,MPI_COMM_WORLD);
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(buffer,SIZE_UINT1,MPI_PACKED,i,MPI_TAG_PERTCV,MPI_COMM_WORLD,&request[i-1]);
         }
#endif
      oldai=(int)(floor(propc-mi.pertalpha*(U-L+1)/2.0));
      oldbi=(int)(floor(propc+mi.pertalpha*(U-L+1)/2.0));
      oldai=std::max(oldai,L);
      oldbi=std::min(oldbi,U);

      std::vector<sinfo*>& sivold = newsinfovec();
      std::vector<sinfo*>& sivnew = newsinfovec();

      tree::npv bnv;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
      getpertsuff(pertnode,bnv,oldc,sivold,sivnew);

      typedef tree::npv::size_type bvsz;
      double lmold,lmnew;
      lmold=0.0;
      for(bvsz j=0;j!=sivold.size();j++) {
         if(sivold[j]->n < mi.minperbot)
            cout << "Error: old tree has some bottom nodes with <minperbot observations!" << endl;
         lmold += lm(*(sivold[j]));
      }

      lmnew=0.0;
      for(bvsz j=0;j!=sivnew.size();j++) {
         if(sivnew[j]->n < mi.minperbot)
            hardreject=true;
         lmnew += lm(*(sivnew[j]));
      }
      double alpha1 = ((double)(bi-ai+1.0))/((double)(oldbi-oldai+1.0)); //anything from the prior?
      double alpha2=alpha1*exp(lmnew-lmold);
      double alpha = std::min(1.0,alpha2);
#ifdef _OPENMPI
      request = new MPI_Request[tc];
#endif
      if(hardreject) alpha=0.0;  //perturb led to an bottom node with <minperbot observations in it, we reject this.

      if(gen.uniform()<alpha) {
         mi.pertaccept++;
         pertnode->setc(propc); //because the call to getpertsuff changes it back to oldc to calc the old lil.
#ifdef _OPENMPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCV_ACCEPT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCV_REJECT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      //else nothing, pertnode->c is already reset to the old value.
      for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
      for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
      delete &sivold;
      delete &sivnew;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
}

//--------------------------------------------------
//do a rotation proposal at a randomly selected internal node.
bool brt::rot(tree::tree_p tnew, tree& x, rn& gen)
{
//   cout << "--------------->>into rot" << endl;
   #ifdef _OPENMPI
   MPI_Request *request = new MPI_Request[tc];
   if(rank==0) {
      const int tag=MPI_TAG_ROTATE;
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
   }
   #endif

   tree::tree_p rotp,temp;
   tree::tree_cp xp;
   tree::npv subtold, subtnew, nbold, nbnew;
   double Qold_to_new, Qnew_to_old;
   unsigned int rdx=0;
   bool twowaystoinvert=false;
   double prinew=1.0,priold=1.0;
   size_t rotid;
   bool hardreject=false;
   std::vector<size_t> goodvars; //variables an internal node can split on

   mi.rotproposal++;

   // Get rot nodes
   tree::npv rnodes;
   tnew->getrotnodes(rnodes);
   #ifdef _OPENMPI
   if(rank==0) {
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      mpi_resetrn(gen);
   }
   delete[] request;
   #endif
   if(rnodes.size()==0)  return false;  //no rot nodes so that's a reject.

   rdx = (unsigned int)floor(gen.uniform()*rnodes.size()); //which rotatable node will we rotate at?
   rotp = rnodes[rdx];
   rotid=rotp->nid();
   xp=x.getptr(rotid);

//   Can check the funcitonality of getpathtoroot:  
//   tree::npv path;
//   rotp->getpathtoroot(path);
//   cout << "rot id=" << rotid << endl;
//   tnew->pr();
//   for(size_t i=0;i<path.size();i++)
//      cout << "i=" << i << ", node id=" << path[i]->nid() << endl;

   int nwaysm1=0,nwaysm2=0,nwayss1=0,nwayss2=0;
   double pm1=1.0,pm2=1.0,ps1=1.0,ps2=1.0;
   if(rotp->isleft()) {
      if(rotp->v==rotp->p->v) //special case, faster to handle it direclty
      {
         rotright(rotp);
         rotp=tnew->getptr(rotid);
         delete rotp->r;
         temp=rotp->l;
         rotp->p->l=temp;
         temp->p=rotp->p;
         rotp->r=0;
         rotp->l=0;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);
         //pm1=pm2=ps1=ps2=1.0 in this case
      }
      else
      {
         rotright(rotp);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceleft(rotp,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceright(rotp->p->r,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         splitleft(rotp->r,rotp->p->v,rotp->p->c);
         splitright(rotp->p->r->r,rotp->p->v,rotp->p->c);

         mergecount(rotp->r,rotp->p->r->r,rotp->p->v,rotp->p->c,&nwayss1);
         ps1=1.0/nwayss1;

         mergecount(rotp->l,rotp->p->r->l,rotp->p->v,rotp->p->c,&nwayss2);
         ps2=1.0/nwayss2;

         tree::tree_p tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->p->r->l,rotp->p->r->r,rotp->p->r->v,rotp->p->r->c,&nwaysm1);
         pm1=1.0/nwaysm1;
         merge(rotp->p->r->l,rotp->p->r->r,tmerged,rotp->p->r->v,rotp->p->r->c,gen);
         rotp->p->r->p=0;
         delete rotp->p->r;
         rotp->p->r=tmerged;

         tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->l,rotp->r,rotp->v,rotp->c,&nwaysm2);
         pm2=1.0/nwaysm2;
         size_t v,c;
         v=rotp->v;
         c=rotp->c;
         merge(rotp->l,rotp->r,tmerged,rotp->v,rotp->c,gen);
         rotp->p->l=tmerged;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);

      //end of merge code if rotp isleft.
      //there are some "extra" isleaf's here because we don't explicitly reset v,c if node becomes leaf so we need to check.
         if( !isleaf(rotp) && !isleaf(rotp->p->r) && (rotp->v!=v && rotp->c!=c) && (rotp->p->r->v != v && rotp->p->r->c != c))
            hardreject=true;
         if( isleaf(rotp) && isleaf(rotp->p->r))
            hardreject=true;
         if(rotp->p->r->v==rotp->v && rotp->p->r->c==rotp->c && !isleaf(rotp->p->r) && !isleaf(rotp))
            twowaystoinvert=true;

      }
   }
   else { //isright
      if(rotp->v==rotp->p->v) //special case, faster to handle it directly
      {
         rotleft(rotp);
         rotp=tnew->getptr(rotid);
         delete rotp->l;
         temp=rotp->r;
         rotp->p->r=temp;
         temp->p=rotp->p;
         rotp->r=0;
         rotp->l=0;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);
         //pm1=pm2=ps1=ps2=1.0 in this case
      }
      else
      {
         rotleft(rotp);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceleft(rotp->p->l,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceright(rotp,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         splitleft(rotp->p->l->l,rotp->p->v,rotp->p->c);
         splitright(rotp->l,rotp->p->v,rotp->p->c);

         mergecount(rotp->p->l->l,rotp->l,rotp->p->v,rotp->p->c,&nwayss1);
         ps1=1.0/nwayss1;

         mergecount(rotp->p->l->r,rotp->r,rotp->p->v,rotp->p->c,&nwayss2);
         ps2=1.0/nwayss2;

         tree::tree_p tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->p->l->l,rotp->p->l->r,rotp->p->l->v,rotp->p->l->c,&nwaysm1);
         pm1=1.0/nwaysm1;
         merge(rotp->p->l->l,rotp->p->l->r,tmerged,rotp->p->l->v,rotp->p->l->c,gen);
         rotp->p->l->p=0;
         delete rotp->p->l;
         rotp->p->l=tmerged;

         tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->l,rotp->r,rotp->v,rotp->c,&nwaysm2);
         pm2=1.0/nwaysm2;
         size_t v,c;
         v=rotp->v;
         c=rotp->c;
         merge(rotp->l,rotp->r,tmerged,rotp->v,rotp->c,gen);
         rotp->p->r=tmerged;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);

      //end of merge code if rotp isright
      //there are some "extra" isleaf's here because we don't explicitly reset v,c if node becomes leaf so we need to check.
         if( !isleaf(rotp) && !isleaf(rotp->p->l) && (rotp->v!=v && rotp->c!=c) && (rotp->p->l->v != v && rotp->p->l->c != c))
            hardreject=true;
         if( isleaf(rotp) && isleaf(rotp->p->l))
            hardreject=true;
         if(rotp->p->l->v==rotp->v && rotp->p->l->c==rotp->c && !isleaf(rotp->p->l) && !isleaf(rotp))
            twowaystoinvert=true;
      }
   }

   // Calculate prior probabilities, we just need to use the subtree where the rotation occured of tnew and x.
   subtold.clear();
   subtnew.clear();
   xp->p->getnodes(subtold);
   rotp->p->getnodes(subtnew);

   for(size_t i=0;i<subtold.size();i++) {
      if(subtold[i]->l) { //interior node
         priold*=tp.alpha/pow(1.0 + subtold[i]->depth(),tp.beta);
         goodvars.clear();
         getinternalvars(subtold[i],*xi,goodvars);
         priold*=1.0/((double)goodvars.size()); //prob split on v 
         priold*=1.0/((double)getnumcuts(subtold[i],*xi,subtold[i]->v)); //prob split on v at c is 1/numcutpoints
      }
      else //terminal node
         priold*=(1.0-tp.alpha/pow(1.0 + subtold[i]->depth(),tp.beta)); 
   }
   for(size_t i=0;i<subtnew.size();i++) {
      if(subtnew[i]->l) { //interior node
         prinew*=tp.alpha/pow(1.0 + subtnew[i]->depth(),tp.beta);
         goodvars.clear();
         getinternalvars(subtnew[i],*xi,goodvars);
         prinew*=1.0/((double)goodvars.size()); //prob split on v
         prinew*=1.0/((double)getnumcuts(subtnew[i],*xi,subtnew[i]->v)); //prob split on v at c is 1/numcutpoints
         if(getnumcuts(subtnew[i],*xi,subtnew[i]->v)<1)
         {
            x.pr(true);
            tnew->pr(true);
         }
      }
      else //terminal node
         prinew*=(1.0-tp.alpha/pow(1.0 + subtnew[i]->depth(),tp.beta)); 
   }

   Qold_to_new=1.0/((double)rnodes.size()); //proposal probability of rotating from x to tnew
   
   rnodes.clear();
   tnew->getrotnodes(rnodes);  //this is very inefficient, could make it much nicer later on.
//   if(rnodes.size()==0) hardreject=true; //if we're back down to just a root node we can't transition back, so this is a hard reject.

   if(!twowaystoinvert)
      Qnew_to_old=1.0/((double)rnodes.size()); //proposal probability of rotating from tnew back to x
   else
      Qnew_to_old=2.0/((double)rnodes.size());

   // Calculate log integrated likelihoods for the subtree where the rotation occured of tnew and x.
   double lmold=0.0,lmnew=0.0;
//   std::vector<sinfo> sold,snew;
   std::vector<sinfo*>& sold = newsinfovec();
   std::vector<sinfo*>& snew = newsinfovec();
   nbold.clear();
   nbnew.clear();
   sold.clear();
   snew.clear();
   x.getbots(nbold);
   tnew->getbots(nbnew);

   //get sufficient statistics for subtree involved in rotation
   //which is just everything below rotp->p.
   //Use subsuff here, which will get the suff stats for both the
   //orignal tree and the proposed tree without needed to explicitly
   //know the root node of either tree as this is recovered when
   //finding the path to rotp->p within the subsuff method.
   rotp=x.getptr(rotid);
   subsuff(rotp->p,nbold,sold);
   rotp=tnew->getptr(rotid);
   subsuff(rotp->p,nbnew,snew);

   for(size_t i=0;i<nbold.size();i++)
         lmold += lm(*(sold[i]));

   for(size_t i=0;i<nbnew.size();i++) {
      if( (snew[i]->n) >= mi.minperbot )
         lmnew += lm(*(snew[i]));
      else 
         hardreject=true;
   }

   for(size_t i=0;i<sold.size();i++) delete sold[i];
   for(size_t i=0;i<snew.size();i++) delete snew[i];
   delete &sold;
   delete &snew;

   double alpha1;
   alpha1=prinew*Qnew_to_old/priold/Qold_to_new/pm1/pm2*ps1*ps2;
   double alpha2 = alpha1*exp(lmnew-lmold);
   double alpha = std::min(1.0,alpha2);

   if(hardreject)
      alpha=0.0;

   if(gen.uniform()<alpha) {
      mi.rotaccept++;
      x = *tnew;
      return true;
   }
   else {
      return false;
   }

   return false;  // we never actually get here.
}

*/