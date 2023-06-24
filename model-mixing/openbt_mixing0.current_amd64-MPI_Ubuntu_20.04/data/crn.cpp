//     crn.h: Random number generator class methods.
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


#include "crn.h"

//--------------------------------------------------
crn::crn():df(0),alpha(0.5),beta(0.5),gen(0),nor(0),uni(0),chi(0),gam(0)
{
   gen = new genD;
   nor = new norD;
   uni = new uniD;
   gam = new gamD;
   gen->seed(99);
}
crn::~crn()
{
   delete gen;
   delete nor;
   delete uni;
   if(chi) delete chi;
   if(gam) delete gam;
}
//--------------------------------------------------
void crn::set_df(int df)
{
   if(df>0) {
      if(chi) delete chi;
      chi = new chiD(df);
   }
}
void crn::set_gam(double alpha,double beta)
{
   if(gam) delete gam;
   this->alpha=alpha;
   this->beta=beta;
   gam = new gamD(alpha,1.0/beta); //c++ uses scale, so 1/beta.
}
void crn::set_seed(int seed)
{
   // delete gen;
   // gen = new genD(seed);
   gen->seed(seed);
}
std::default_random_engine crn::get_engine_state()
{
   return *gen;
}
void crn::set_engine_state(std::stringstream& state)
{
   state >> (*gen);
}

