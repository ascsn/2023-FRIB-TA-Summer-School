//     singlepoisson.h: Poisson tree model class definition.
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


#ifndef GUARD_singlepoisson_h
#define GUARD_singlepoisson_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"


class singlepoissonsinfo : public sinfo { //sufficient statistics (will depend on end node model)
public:
   singlepoissonsinfo():sinfo(),sumy(0.0) {}
   singlepoissonsinfo(const singlepoissonsinfo& is):sinfo(is),sumy(is.sumy) {}
   virtual ~singlepoissonsinfo() {}  //need this so memory is properly freed in derived classes.
   double sumy;
   // compound addition operator needed when adding suff stats
   virtual sinfo& operator+=(const sinfo& rhs) {
      sinfo::operator+=(rhs);
      const singlepoissonsinfo& mrhs=static_cast<const singlepoissonsinfo&>(rhs);
      sumy+=mrhs.sumy;
      return *this;
   }
   // assignment operator for suff stats
   virtual sinfo& operator=(const sinfo& rhs)
   {
      if(&rhs != this) {
         sinfo::operator=(rhs);
         const singlepoissonsinfo& mrhs=static_cast<const singlepoissonsinfo&>(rhs);
         this->sumy = mrhs.sumy;
      }
      return *this;
   }
   // addition opertor is defined in terms of compound addition
   const singlepoissonsinfo operator+(const singlepoissonsinfo& other) const {
      singlepoissonsinfo result = *this; //copy of myself.
      result += other;
      return result;
   }
};

class singlepoissonbrt : public brt 
{
public:
   //--------------------
   //classes
   // tprior and mcmcinfo are same as in brt
   class cinfo { //parameters for end node model prior
   public:
      cinfo():alpha(0.5),beta(0.5) {}
      double alpha;
      double beta;
   };
   //--------------------
   //constructors/destructors
   singlepoissonbrt():brt() {}
   //--------------------
   //methods
   void draw(rn& gen);
   void draw_mpislave(rn& gen);
   void setci(double alpha, double beta) { ci.alpha=alpha; ci.beta=beta; }
   virtual double drawnodetheta(sinfo& si, rn& gen);
   virtual double lm(sinfo& si);
   virtual void add_observation_to_suff(diterator& diter, sinfo& si);
   virtual sinfo* newsinfo() { return new singlepoissonsinfo; }
   virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
   virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new singlepoissonsinfo); return *si; }
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
