//     test.cpp: Random number generator testing/validation code.
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

using std::cout;
using std::endl;

#include "crn.h"

#ifdef _OPENMPI
#   include <mpi.h>
#endif

int main()
{
   cout << "Into test main for crn\n";

   crn gen;
   cout << "a normal from gen: " << gen.normal() << endl;

   crn gen1;
   gen1.set_seed(14);
   cout << "a normal from gen1: " << gen1.normal() << endl;

   int nd=1000;
   std::ofstream df("d.txt");
   for(int i=0;i<nd;i++) df << gen1.normal() << endl;

   gen.set_gam(1.0,1.0);
   cout << "a gamma from gen: " << gen.gamma() << endl;

   gen1.set_gam(5,100);
   cout << "a gamma from gen1: " << gen1.gamma() << endl;

   std::ofstream df2("dgam.txt");
   for(int i=0;i<nd;i++) df2 << gen1.gamma() << endl;


   return 0;
   
}
