//     singlepoisson.cpp: Poisson tree model methods.
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


#include "singlepoisson.h"
//#include "brtfuns.h"
#include <iostream>
#include <map>
#include <vector>

using std::cout;
using std::endl;


//--------------------------------------------------
//a single iteration of the MCMC for brt model
void singlepoissonbrt::draw(rn& gen)
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
void singlepoissonbrt::draw_mpislave(rn& gen)
{
   //All the usual steps
   brt::draw_mpislave(gen);

   // Update the in-sample predicted vector
   setf();

   // Update the in-sample residual vector
   setr();
}
//--------------------------------------------------
//draw theta for a single bottom node for the brt model
double singlepoissonbrt::drawnodetheta(sinfo& si, rn& gen)
{
   singlepoissonsinfo& msi=static_cast<singlepoissonsinfo&>(si);
   double alphahat=ci.alpha+msi.sumy;
   double betahat=ci.beta+msi.n;

// cout << "drawnodetheta:" << endl;
// cout << "ci.alpha=" << ci.alpha << endl;
// cout << "ci.beta=" << ci.beta << endl;
// cout << "msi.sumy=" << msi.sumy << endl;
// cout << "msi.n=" << msi.n << endl;
   gen.set_gam(alphahat,betahat);
   return gen.gamma();
}
//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double singlepoissonbrt::lm(sinfo& si)
{
   singlepoissonsinfo& msi=static_cast<singlepoissonsinfo&>(si);

   return logam(msi.sumy+ci.alpha)-(msi.sumy+ci.alpha)*log(msi.n+ci.beta);
          // - sum i=1..n logam(yi+1.0), but this term cancels in 
          // the accept/reject of birth/death proposal so we do not calculate it.
          // Note: should check if the cancelation is preserved for other
          //       MH moves or updates (it should, I think).

//cout << "msi.sumw=" << msi.sumw << " msi.sumwy=" << msi.sumwy << endl;
}
//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void singlepoissonbrt::add_observation_to_suff(diterator& diter, sinfo& si)
{
   singlepoissonsinfo& msi=static_cast<singlepoissonsinfo&>(si);
   msi.n+=1;
   msi.sumy+=diter.gety();
}
//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
void singlepoissonbrt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   singlepoissonsinfo& msil=static_cast<singlepoissonsinfo&>(sil);
   singlepoissonsinfo& msir=static_cast<singlepoissonsinfo&>(sir);
   if(rank==0) { // MPI receive all the answers from the slaves
      MPI_Status status;
      singlepoissonsinfo& tsil = (singlepoissonsinfo&) *newsinfo();
      singlepoissonsinfo& tsir = (singlepoissonsinfo&) *newsinfo();
      char buffer[SIZE_UINT4];
      int position=0;
      unsigned int ln,rn;
      for(size_t i=1; i<=(size_t)tc; i++) {
         position=0;
         MPI_Recv(buffer,SIZE_UINT4,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&tsil.sumy,1,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&tsir.sumy,1,MPI_DOUBLE,MPI_COMM_WORLD);

         tsil.n=(size_t)ln;
         tsir.n=(size_t)rn;
         msil+=tsil;
         msir+=tsir;
      }
      delete &tsil;
      delete &tsir;
   }
   else // MPI send all the answers to root
   {
      char buffer[SIZE_UINT4];
      int position=0;  
      unsigned int ln,rn;
      ln=(unsigned int)msil.n;
      rn=(unsigned int)msir.n;
      MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);
      MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);
      MPI_Pack(&msil.sumy,1,MPI_DOUBLE,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);
      MPI_Pack(&msir.sumy,1,MPI_DOUBLE,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);

      MPI_Send(buffer,SIZE_UINT4,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }
#endif   
}
//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void singlepoissonbrt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   unsigned int nvec[siv.size()];
   double sumyvec[siv.size()];

   for(size_t i=0;i<siv.size();i++) { // on root node, these should be 0 because of newsinfo().
      singlepoissonsinfo* msi=static_cast<singlepoissonsinfo*>(siv[i]);
      nvec[i]=(unsigned int)msi->n;    // cast to int
      sumyvec[i]=msi->sumy;
   }
// cout << "pre:" << siv[0]->n << " " << siv[1]->n << endl;

   // MPI sum
   // MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE,&sumwvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE,&sumwyvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

   if(rank==0) {
      MPI_Status status;
      unsigned int tempnvec[siv.size()];
      double tempsumyvec[siv.size()];

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
         singlepoissonsinfo* msi=static_cast<singlepoissonsinfo*>(siv[i]);
         msi->n=(size_t)nvec[i];    // cast back to size_t
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      // receive sumwvec, update and send back.
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempsumyvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            sumyvec[j]+=tempsumyvec[j];
      }
      request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&sumyvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
      }
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         singlepoissonsinfo* msi=static_cast<singlepoissonsinfo*>(siv[i]);
         msi->sumy=sumyvec[i];
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      // receive sumyvec, update and send back.
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempsumyvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            sumyvec[j]+=tempsumyvec[j];
      }
      request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&sumyvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
      }
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         singlepoissonsinfo* msi=static_cast<singlepoissonsinfo*>(siv[i]);
         msi->sumy=sumyvec[i];
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
      MPI_Isend(&sumyvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         singlepoissonsinfo* msi=static_cast<singlepoissonsinfo*>(siv[i]);
         msi->n=(size_t)nvec[i];    // cast back to size_t
      }
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;
      MPI_Recv(&sumyvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // send sumwyvec, update sumwvec, receive sumwyvec
      request=new MPI_Request;
      MPI_Isend(&sumyvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         singlepoissonsinfo* msi=static_cast<singlepoissonsinfo*>(siv[i]);
         msi->sumy=sumyvec[i];
      }
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;
      MPI_Recv(&sumyvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // update sumwyvec
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         singlepoissonsinfo* msi=static_cast<singlepoissonsinfo*>(siv[i]);
         msi->sumy=sumyvec[i];
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
void singlepoissonbrt::pr()
{
   std::cout << "***** singlepoissonbrt object:\n";
   cout << "Conditioning info:" << endl;
   cout << "  alpha:   " << ci.alpha << endl;
   cout << "   beta:   " << ci.beta << endl;
   brt::pr();
}

//--------------------------------------------------
// compute the logarithm of the Gamma function
// people.sc.fsu.edu/~jburkardt/cpp_src/toms291/toms291.html
double singlepoissonbrt::logam (double x)
{
  double f;
  double value;
  double y;
  double z;

  if ( x <= 0.0 )
  {
    value = 0.0;
    return value;
  }

  y = x;

  if ( x < 7.0 )
  {
    f = 1.0;
    z = y;

    while ( z < 7.0 )
    {
      f = f * z;
      z = z + 1.0;
    }
    y = z;
    f = - log ( f );
  }
  else
  {
    f = 0.0;
  }

  z = 1.0 / y / y;

  value = f + ( y - 0.5 ) * log ( y ) - y 
    + 0.918938533204673 + 
    ((( 
    - 0.000595238095238   * z 
    + 0.000793650793651 ) * z 
    - 0.002777777777778 ) * z 
    + 0.083333333333333 ) / y;

  return value;
}
