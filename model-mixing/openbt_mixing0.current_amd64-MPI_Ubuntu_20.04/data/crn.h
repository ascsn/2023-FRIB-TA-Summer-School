//     crn.h: Random number generator class definition.
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


#ifndef CRN_H
#define CRN_H

#include "rn.h"
#include <random> 
#include <sstream>

class crn: public rn
{
//typedefs
   typedef std::default_random_engine genD;
   typedef std::normal_distribution<double> norD;
   typedef std::uniform_real_distribution<double> uniD;
   typedef std::chi_squared_distribution<double> chiD;
   typedef std::gamma_distribution<double> gamD;
public:
//constructor
   crn();
//virtual
   virtual ~crn();
   virtual double normal() {return (*nor)(*gen);}
   virtual double uniform() {return (*uni)(*gen);}
   virtual double chi_square() {return (*chi)(*gen);}
   virtual double gamma() {return (*gam)(*gen);}
   virtual void set_df(int df);
//get,set
   int get_df()  {return df;}
   void set_seed(int seed);
   void set_gam(double alpha,double beta);
   std::default_random_engine get_engine_state();
   void set_engine_state(std::stringstream& state);
private:
   int df;
   double alpha,beta;
   genD* gen;
   norD* nor;
   uniD* uni;
   chiD* chi;
   gamD* gam;
};

#endif //CRN_H

