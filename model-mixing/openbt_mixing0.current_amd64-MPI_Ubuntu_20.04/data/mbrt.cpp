//     mbrt.cpp: Mean tree BT model class methods.
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


#include "mbrt.h"
//#include "brtfuns.h"
#include <iostream>
#include <map>
#include <vector>

using std::cout;
using std::endl;


//--------------------------------------------------
//a single iteration of the MCMC for brt model
void mbrt::draw(rn& gen)
{
   //All the usual steps
   brt::draw(gen);

   // Update the in-sample predicted vector
   setf();

   // Update the in-sample residual vector
   setr();

}
//--------------------------------------------------
//slave controller for draw when using MPI
void mbrt::draw_mpislave(rn& gen)
{
   //All the usual steps
   brt::draw_mpislave(gen);
   //cout << "HERE mbrt slave 1" << endl;
   // Update the in-sample predicted vector
   setf();
   //cout << "HERE mbrt slave 2" << endl;
   // Update the in-sample residual vector
   setr();
   //cout << "HERE mbrt slave 3" << endl;
}
//--------------------------------------------------
//draw theta for a single bottom node for the brt model
double mbrt::drawnodetheta(sinfo& si, rn& gen)
{
   msinfo& msi=static_cast<msinfo&>(si);
   double muhat = msi.sumwy/msi.sumw;
   double a = 1.0/(ci.tau*ci.tau);
   return (msi.sumw*muhat)/(a+msi.sumw) + gen.normal()/sqrt(a+msi.sumw);
}
//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double mbrt::lm(sinfo& si)
{
   msinfo& msi=static_cast<msinfo&>(si);
   double t2 =ci.tau*ci.tau;
   double k = msi.sumw*t2+1;

//cout << "msi.sumw=" << msi.sumw << " msi.sumwy=" << msi.sumwy << endl;
   return -.5*log(k)+.5*msi.sumwy*msi.sumwy*t2/k;
}
//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void mbrt::add_observation_to_suff(diterator& diter, sinfo& si)
{
   msinfo& msi=static_cast<msinfo&>(si);
   double w;
   w=1.0/(ci.sigma[*diter]*ci.sigma[*diter]);
   msi.n+=1;
   msi.sumw+=w;
   msi.sumwy+=w*diter.gety();
}
//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
void mbrt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir)
{
//cout << "Here1 mpi" << endl;
#ifdef _OPENMPI
   msinfo& msil=static_cast<msinfo&>(sil);
   msinfo& msir=static_cast<msinfo&>(sir);
   if(rank==0) { // MPI receive all the answers from the slaves
      MPI_Status status;
      msinfo& tsil = (msinfo&) *newsinfo();
      msinfo& tsir = (msinfo&) *newsinfo();
      char buffer[SIZE_UINT6];
      int position=0;
      unsigned int ln,rn;
      //cout << "Here2 mpi" << endl;
      for(size_t i=1; i<=(size_t)tc; i++) {
         //cout << "tsir.sumw = " << tsir.sumw << endl;
         //cout << "tsir.sumwy = " << tsir.sumwy << endl;
         
         position=0;
         MPI_Recv(buffer,SIZE_UINT6,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&tsil.sumw,1,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&tsir.sumw,1,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&tsil.sumwy,1,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&tsir.sumwy,1,MPI_DOUBLE,MPI_COMM_WORLD);

         tsil.n=(size_t)ln;
         tsir.n=(size_t)rn;
         msil+=tsil;
         msir+=tsir;
      }
      //cout << "Here3 mpi" << endl;
      delete &tsil;
      delete &tsir;
   }
   else // MPI send all the answers to root
   {
      char buffer[SIZE_UINT6];
      int position=0;  
      unsigned int ln,rn;
      //cout << "Here4 mpi" << endl;
      ln=(unsigned int)msil.n;
      rn=(unsigned int)msir.n;
      MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&msil.sumw,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&msir.sumw,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&msil.sumwy,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&msir.sumwy,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      //cout << "Here5 mpi" << endl;
      MPI_Send(buffer,SIZE_UINT6,MPI_PACKED,0,0,MPI_COMM_WORLD);
      //cout << "msir.sumw = " << msir.sumw << endl;
      //cout << "msir.sumwy = " << msir.sumwy << endl;
      //cout << "msir.n = " << msir.n << endl;
      
   }
#endif   
}
//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void mbrt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   unsigned int nvec[siv.size()];
   double sumwvec[siv.size()];
   double sumwyvec[siv.size()];

   for(size_t i=0;i<siv.size();i++) { // on root node, these should be 0 because of newsinfo().
      msinfo* msi=static_cast<msinfo*>(siv[i]);
      nvec[i]=(unsigned int)msi->n;    // cast to int
      sumwvec[i]=msi->sumw;
      sumwyvec[i]=msi->sumwy;
   }
// cout << "pre:" << siv[0]->n << " " << siv[1]->n << endl;

   // MPI sum
   // MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE,&sumwvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE,&sumwyvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

   if(rank==0) {
      MPI_Status status;
      unsigned int tempnvec[siv.size()];
      double tempsumwvec[siv.size()];
      double tempsumwyvec[siv.size()];
      
      // receive nvec, update and send back.
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempnvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            nvec[j]+=tempnvec[j]; 
      }
      MPI_Request *request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
      }
      
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         msinfo* msi=static_cast<msinfo*>(siv[i]);
         msi->n=(size_t)nvec[i];    // cast back to size_t
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      // receive sumwvec, update and send back.
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempsumwvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            sumwvec[j]+=tempsumwvec[j];
      }
      request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&sumwvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
      }
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         msinfo* msi=static_cast<msinfo*>(siv[i]);
         msi->sumw=sumwvec[i];
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
      
      // receive sumwyvec, update and send back.
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempsumwyvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            sumwyvec[j]+=tempsumwyvec[j];
      }
      request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&sumwyvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
      }
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         msinfo* msi=static_cast<msinfo*>(siv[i]);
         msi->sumwy=sumwyvec[i];
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
   }
   else {
      MPI_Request *request=new MPI_Request;
      MPI_Status status;

      // send/recv nvec      
      MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;
      MPI_Recv(&nvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // send sumwvec, update nvec, receive sumwvec
      request=new MPI_Request;
      MPI_Isend(&sumwvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         msinfo* msi=static_cast<msinfo*>(siv[i]);
         msi->n=(size_t)nvec[i];    // cast back to size_t
      }
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;
      MPI_Recv(&sumwvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // send sumwyvec, update sumwvec, receive sumwyvec
      request=new MPI_Request;
      MPI_Isend(&sumwyvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         msinfo* msi=static_cast<msinfo*>(siv[i]);
         msi->sumw=sumwvec[i];
      }
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;
      MPI_Recv(&sumwyvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // update sumwyvec
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         msinfo* msi=static_cast<msinfo*>(siv[i]);
         msi->sumwy=sumwyvec[i];
      }
   }

   // for(size_t i=0;i<siv.size();i++) {
   //    msinfo* msi=static_cast<msinfo*>(siv[i]);
   //    msi->n=(size_t)nvec[i];    // cast back to size_t
   //    msi->sumw=sumwvec[i];
   //    msi->sumwy=sumwyvec[i];
   // }
// cout << "reduced:" << siv[0]->n << " " << siv[1]->n << endl;
#endif
}

//--------------------------------------------------
//pr for brt
void mbrt::pr()
{
   std::cout << "***** mbrt object:\n";
   cout << "Conditioning info:" << endl;
   cout << "   mean:   tau=" << ci.tau << endl;
   if(!ci.sigma)
     cout << "         sigma=[]" << endl;
   else
     cout << "         sigma=[" << ci.sigma[0] << ",...," << ci.sigma[di->n-1] << "]" << endl;
   brt::pr();
}
