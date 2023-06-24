//     test.cpp: BT tree class testing/validation code.
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

#include "crn.h"
#include "tree.h"
#include "brtfuns.h"

#ifdef _OPENMPI
#   include <mpi.h>
#endif


using std::cout;
using std::endl;

int main()
{
   /*
   //--------------------------------------------------
   // Test the tree for univariate theta
   //--------------------------------------------------
   cout << "*****into test for tree\n";

   crn gen;

   //--------------------------------------------------
   //make a simple tree

   tree t;
   cout << "** print out a null tree\n";
   t.pr();
      
   t.birth(1,0,50,-1.0,1.0);
   cout << "** print out a tree with one split\n";
   t.pr();
   
   t.birth(2,0,25,-1.5,-0.5);
   cout << "** print out a tree with two splits\n";
   t.pr();

   tree t2;
   t2.birth(1,0,33,-3,3);
   cout << "** tree 2\n";
   t2.pr();
   
   //make another tree - intialized by theta
   tree t3(0.5);
   t3.pr();
   
   tree::npv bots;
   tree st;

   st=t; //copy
   cout << "** supertree:\n";
   st.pr();

   cout << "** get bots:\n";
   st.getbots(bots);
   cout << "bots.size=" << bots.size() << endl;
   //collapse each tree j=1..m into the supertree
   for(size_t i=0;i<bots.size();i++) {
      cout << "iteration i=" << i << endl;
      collapsetree(st,bots[i],&t2); //mb[j]->t);
      st.pr();
   }
   bots.clear();

   cout << "** collapsed supertree:\n";
   st.pr();
   */

   //--------------------------------------------------
   //Test out the model mixing tree functions
   //--------------------------------------------------
   
   //Make simple null tree and print 
   tree tv1;
   cout << "~~~ Simple Tree with Vector Parameters: ~~~" << endl;
   tv1.pr_vec();
   cout << "***************************************" << endl;

   //Assign left and right vectors for terminal nodes in birth step
   Eigen::VectorXd thetavecl, thetavecr;
   thetavecl = Eigen::VectorXd::Random(tv1.k);
   thetavecr = Eigen::VectorXd::Random(tv1.k);

   //Performa a birth step on the root node
   tv1.birth(1,0,50,thetavecl,thetavecr);
   cout << "~~~ print out a tree with one split ~~~\n";
   tv1.pr_vec();
   cout << "***************************************" << endl;
   
   //Perform a second birth step 
   tv1.birth(2,0,25,thetavecl*2,thetavecr*1.5);
   cout << "~~~ print out a tree with two splits ~~~\n";
   tv1.pr_vec();
   cout << "***************************************" << endl;

   //Perform a death and insert new value of theta to collapsed node
   Eigen::VectorXd dtheta(2);
   dtheta << 0.2, 1.1;
   cout << "~~~ Perform a death at node 2 ~~~\n";
   tv1.death(2, dtheta);
   tv1.pr_vec();
   cout << "***************************************" << endl;

   //Reset tree to the root node with default parameters via a destructor
   cout << "~~~ Destruct the tree -- revert to root node ~~~\n";
   tv1.tonull();
   tv1.pr_vec();
   cout << "***************************************" << endl;

   //Test other constructors
   tree tv2(tv1); //Initialize new tree using existing tree
   cout << "~~~ New tree based on old tree ~~~\n";
   tv2.pr_vec();
   cout << "***************************************" << endl;

   Eigen::VectorXd itheta(3);
   itheta << 1.2, 0.8, 0.1; //populate a theta vector 
   tree tv3(itheta); //Initialize new tree with theta vector
   cout << "~~~ New tree initialized by itheta ~~~\n";
   tv3.pr_vec();
   cout << "***************************************" << endl;

   tree tv3_copy;
   tv3_copy = tv3;
   cout << "~~~ Copy Tree ~~~\n";
   tv3_copy.pr_vec();
   cout << "***************************************" << endl;

   Eigen::VectorXd itheta2(3);
   itheta2 << 2, 1, 0.5; //populate a theta vector
   cout << "~~~ Set Theta and Get Theta ~~~\n";
   tv3.setthetavec(itheta2);
   cout << "Updated theta vector: \n" << tv3.getthetavec() << endl;
   cout << "***************************************" << endl;

   //----------------------------------------------
   //Collapse a tree
   tree::npv bots;
   tree st;

   //Create two new trees
   Eigen::VectorXd v1(2), v2(2), v21(2), v22(2);
   v1 << 0.5, 0.78;
   v2 << 1.5, -0.9;
   v21 << 1.0,2.0;
   v22 << 0.9,0.8; 

   tree tc1(v1);
   tree tc2(v2);

   //Brith Step
   tc2.birth(1,0,50,v21,v22);

   st=tc1; //copy
   cout << "***** supertree:\n";
   st.pr_vec();

   cout << "***** get bots:\n";
   st.getbots(bots);
   cout << "bots.size=" << bots.size() << endl;
   //collapse each tree j=1..m into the supertree
   for(size_t i=0;i<bots.size();i++) {
      cout << "iteration i=" << i << endl;
      collapsetree_vec(st,bots[i],&tc2); 
      st.pr_vec();
   }
   bots.clear();

   cout << "***** collapsed supertree:\n";
   st.pr_vec();
   
   //----------------------------------------------
   //Tree to vector -- note with the vector parameter setup, otheta is of length nn*k (number.of.nodes x number.of.models)
   tree tv4; //get a new tree
   tv4.birth(1,0,50,v21,v22); //birth step
   cout << "***************************************" << endl;
   cout << "Construct new tree and perform a birth" << endl;
   tv4.pr_vec();

   //set up containers/pointers for the information
   int oid[3], oc[3], ov[3]; //tree size, nn = 3
   double otheta[3*2]; //k = 2
   
   tv4.treetovec(oid, ov, oc, otheta, 2);

   //Print Results
   cout << "***************************************" << endl;
   cout << "~~~ Tree to Vector Output ~~~" << endl; 
   for(int i = 0;i<3;i++){
      cout << "Out ID = " << oid[i] << " -- ";
      cout << "Out V = " << ov[i] << " -- ";
      cout << "Out C = " << oc[i] << " -- ";
      cout << "Out Theta = " << otheta[2*i] << ", " << otheta[2*i+1] << endl;
   }
   
   //----------------------------------------------
   //Vector to tree -- same thing as above, the theta here is nn*k
   tree tv5; //get a new tree
   cout << "***************************************" << endl;
   cout << "~~~ Tree to Vector Output ~~~" << endl;
   cout << "**** Initial Tree" << endl;
   tv5.pr_vec();

   //Use the vectors from previous example (tree to vector) to populate the tree
   tv5.vectotree(3,oid,ov,oc,otheta,2); //number nodes nn = 3 && number models k = 2
   cout << "**** Updated Tree" << endl;
   tv5.pr_vec();
   
   
   return 0;
}
