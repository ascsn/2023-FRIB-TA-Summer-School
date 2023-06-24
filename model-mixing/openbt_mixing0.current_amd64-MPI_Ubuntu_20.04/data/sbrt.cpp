//     sbrt.cpp: Variance tree BT model class methods.
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


#include "sbrt.h"
//#include "brtfuns.h"
#include <iostream>
#include <map>
#include <vector>

using std::cout;
using std::endl;


//--------------------------------------------------
//a single iteration of the MCMC for brt model
void sbrt::draw(rn& gen)
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
void sbrt::draw_mpislave(rn& gen)
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
double sbrt::drawnodetheta(sinfo& si, rn& gen)
{
   ssinfo& ssi=static_cast<ssinfo&>(si);
   int nupost=ssi.n+(int)ci.nu;
   double nulampost=ci.nu*ci.lambda+ssi.sumy2;
   gen.set_df(nupost);

   return sqrt((nulampost)/gen.chi_square());
}
//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double sbrt::lm(sinfo& si)
{
   ssinfo& ssi=static_cast<ssinfo&>(si);
   double val;
   double nstar;
   double nudiv2;

   nudiv2=ci.nu/2.0;
   nstar=(ci.nu+ssi.n)/2.0;
   val = nudiv2*log(ci.nu*ci.lambda);
   val+= -(ci.nu+ssi.n)/2.0*log(ci.nu*ci.lambda+ssi.sumy2);
   val+= logam(nstar)-logam(nudiv2);

////cout << "ssi.n=" << ssi.n << " ssi.sumy2=" << ssi.sumy2 << endl;
   return val;
}
//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void sbrt::add_observation_to_suff(diterator& diter, sinfo& si)
{
   ssinfo& ssi=static_cast<ssinfo&>(si);
   ssi.n+=1;
   ssi.sumy2+=diter.gety()*diter.gety();
////cout << "add obs: " << ssi.n << " " << ssi.sumy2 << endl;
}
//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
void sbrt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   ssinfo& ssil=static_cast<ssinfo&>(sil);
   ssinfo& ssir=static_cast<ssinfo&>(sir);
   if(rank==0) { // MPI receive all the answers from the slaves
      MPI_Status status;
      ssinfo& tsil = (ssinfo&) *newsinfo();
      ssinfo& tsir = (ssinfo&) *newsinfo();
      char buffer[SIZE_UINT4];
      int position=0;
      unsigned int ln,rn;
      for(size_t i=1; i<=(size_t)tc; i++) {
         position=0;
         MPI_Recv(buffer,SIZE_UINT4,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&tsil.sumy2,1,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT4,&position,&tsir.sumy2,1,MPI_DOUBLE,MPI_COMM_WORLD);

         tsil.n=(size_t)ln;
         tsir.n=(size_t)rn;
         ssil+=tsil;
         ssir+=tsir;
////cout << ssil.n << ", " << ssil.sumy2 << " " << ssir.n << "," << ssir.sumy2 << endl;
      }
      delete &tsil;
      delete &tsir;
   }
   else // MPI send all the answers to root
   {
      char buffer[SIZE_UINT4];
      int position=0;  
      unsigned int ln,rn;
      ln=(unsigned int)ssil.n;
      rn=(unsigned int)ssir.n;
      MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);
      MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);
      MPI_Pack(&ssil.sumy2,1,MPI_DOUBLE,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);
      MPI_Pack(&ssir.sumy2,1,MPI_DOUBLE,buffer,SIZE_UINT4,&position,MPI_COMM_WORLD);

      MPI_Send(buffer,SIZE_UINT4,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }
#endif   
}
//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void sbrt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   unsigned int nvec[siv.size()];
   double sumy2vec[siv.size()];

   for(size_t i=0;i<siv.size();i++) { // on root node, these should be 0 because of newsinfo().
      ssinfo* ssi=static_cast<ssinfo*>(siv[i]);
      nvec[i]=(unsigned int)ssi->n;    // cast to int
      sumy2vec[i]=ssi->sumy2;
   }
// cout << "pre:" << siv[0]->n << " " << siv[1]->n << endl;
//// cout << "pre:" << nvec[0] << " " << sumy2vec[0] << endl;

   // MPI sum
   // MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE,&sumy2vec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   if(rank==0) {
      MPI_Status status;
      unsigned int tempnvec[siv.size()];
      double tempsumy2vec[siv.size()];

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
      // cast back to ssi
      for(size_t i=0;i<siv.size();i++) {
         ssinfo* ssi=static_cast<ssinfo*>(siv[i]);
         ssi->n=(size_t)nvec[i];    // cast back to size_t
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      // receive sumwy2vec, update and send back.
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempsumy2vec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            sumy2vec[j]+=tempsumy2vec[j];
      }
      request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&sumy2vec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
      }
      // cast back to ssi
      for(size_t i=0;i<siv.size();i++) {
         ssinfo* ssi=static_cast<ssinfo*>(siv[i]);
         ssi->sumy2=sumy2vec[i];
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
      MPI_Isend(&sumy2vec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
      // cast back to ssi
      for(size_t i=0;i<siv.size();i++) {
         ssinfo* ssi=static_cast<ssinfo*>(siv[i]);
         ssi->n=(size_t)nvec[i];    // cast back to size_t
      }
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;
      MPI_Recv(&sumy2vec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // update sumy2vec
      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         ssinfo* ssi=static_cast<ssinfo*>(siv[i]);
         ssi->sumy2=sumy2vec[i];
      }
   }

   // for(size_t i=0;i<siv.size();i++) {
   //    ssinfo* ssi=static_cast<ssinfo*>(siv[i]);
   //    ssi->n=(size_t)nvec[i];    // cast back to size_t
   //    ssi->sumy2=sumy2vec[i];
   // }
// cout << "reduced:" << siv[0]->n << " " << siv[1]->n << endl;
//// cout << "reduced:" << nvec[0] << " " << sumy2vec[0] << endl;
#endif
}
//--------------------------------------------------
//pr for brt
void sbrt::pr()
{
   std::cout << "***** sbrt object:\n";
   cout << "Conditioning info:" << endl;
   cout << "      dof:  nu=" << ci.nu << endl;
   cout << "    scale:  lambda=" << ci.lambda << endl;
   brt::pr();
}
//--------------------------------------------------
// compute the logarithm of the Gamma function
// people.sc.fsu.edu/~jburkardt/cpp_src/toms291/toms291.html
double sbrt::logam (double x)
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

