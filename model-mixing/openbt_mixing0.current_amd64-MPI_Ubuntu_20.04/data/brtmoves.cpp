//     brtmoves.cpp: Base BT model class advanced MH methods.
//     Copyright (C) 2013-2019 Matthew T. Pratola
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

/* Can check the funcitonality of getpathtoroot:  
   tree::npv path;
   rotp->getpathtoroot(path);
   cout << "rot id=" << rotid << endl;
   tnew->pr();
   for(size_t i=0;i<path.size();i++)
      cout << "i=" << i << ", node id=" << path[i]->nid() << endl;
*/

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

