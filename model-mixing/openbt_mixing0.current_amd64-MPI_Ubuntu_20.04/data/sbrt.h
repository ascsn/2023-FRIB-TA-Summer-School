//     sbrt.h: Variance tree BT model class definition.
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


#ifndef GUARD_sbrt_h
#define GUARD_sbrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"


class ssinfo : public sinfo { //sufficient statistics (will depend on end node model)
public:
   ssinfo():sinfo(),sumy2(0.0) {}
   ssinfo(const ssinfo& is):sinfo(is),sumy2(is.sumy2) {}
   virtual ~ssinfo() {}  //need this so memory is properly freed in derived classes.
   double sumy2;
   // compound addition operator needed when adding suff stats
   virtual sinfo& operator+=(const sinfo& rhs) {
      sinfo::operator+=(rhs);
      const ssinfo& srhs=static_cast<const ssinfo&>(rhs);
      sumy2+=srhs.sumy2;
      return *this;
   }
   // assignment operator for suff stats
   virtual sinfo& operator=(const sinfo& rhs)
   {
      if(&rhs != this) {
         sinfo::operator=(rhs);
         const ssinfo& srhs=static_cast<const ssinfo&>(rhs);
         this->sumy2 = srhs.sumy2;
      }
      return *this;
   }
   // addition opertor is defined in terms of compound addition
   const ssinfo operator+(const ssinfo& other) const {
      ssinfo result = *this; //copy of myself.
      result += other;
      return result;
   }
};

class sbrt : public brt 
{
public:
   //--------------------
   //classes
   // tprior and mcmcinfo are same as in brt
   class cinfo { //parameters for end node model prior
   public:
      cinfo():nu(1.0),lambda(1.0) {}
      double nu; //dof of variance prior
      double lambda; //scale of variance prior
   };
   //--------------------
   //constructors/destructors
   sbrt():brt() { t.settheta(1.0); }
   sbrt(double itheta):brt() { t.settheta(itheta); }
   //--------------------
   //methods
   void draw(rn& gen);
   void draw_mpislave(rn& gen);
   void setci(double nu, double lambda) { ci.nu=nu; ci.lambda=lambda; }
   virtual double drawnodetheta(sinfo& si, rn& gen);
   virtual double lm(sinfo& si);
   virtual void add_observation_to_suff(diterator& diter, sinfo& si);
   virtual sinfo* newsinfo() { return new ssinfo; }
   virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
   virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new ssinfo); return *si; }
   virtual void local_mpi_reduce_allsuff(std::vector<sinfo*>& siv);
   virtual void local_mpi_sr_suffs(sinfo& sil, sinfo& sir);
   void pr();

   //--------------------
   //data
   //--------------------------------------------------
   //stuff that maybe should be protected
protected:
   //--------------------
   //model information
   cinfo ci; //conditioning info (e.g. other parameters and prior and end node models)
   //--------------------
   //data
   //--------------------
   //mcmc info
   //--------------------
   //methods
   double logam(double x);

};


#endif
