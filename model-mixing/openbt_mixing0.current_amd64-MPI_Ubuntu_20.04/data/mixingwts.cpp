    //     pred.cpp: Implement model prediction interface for OpenBT.
    //     Copyright (C) 2012-2018 Matthew T. Pratola
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


    #include <iostream>
    #include <string>
    #include <ctime>
    #include <sstream>

    #include <fstream>
    #include <vector>
    #include <limits>

    #include "Eigen/Dense"
    #include <Eigen/StdVector>

    #include "crn.h"
    #include "tree.h"
    #include "brt.h"
    #include "brtfuns.h"
    #include "dinfo.h"
    #include "mbrt.h"
    #include "ambrt.h"
    #include "psbrt.h"
    #include "tnorm.h"
    #include "mxbrt.h"
    #include "amxbrt.h"

    using std::cout;
    using std::endl;

    #define MODEL_BT 1
    #define MODEL_BINOMIAL 2
    #define MODEL_POISSON 3
    #define MODEL_BART 4
    #define MODEL_HBART 5
    #define MODEL_PROBIT 6
    #define MODEL_MODIFIEDPROBIT 7
    #define MODEL_MIXBART 9 //Skipped 8 because MERCK is 8 in cli.cpp


    // Draw predictive realizations at the prediciton points, xp.
    int main(int argc, char* argv[])
    {
    std::string folder("");

    if(argc>1)
    {
        //argument on the command line is path to config file.
        folder=std::string(argv[1]);
        folder=folder+"/";
    }


    //--------------------------------------------------
    //process args
    std::ifstream conf(folder+"config.mxwts");
    std::string modelname;
    int modeltype;
    std::string xicore;
    std::string xwcore;
    std::string fitcore;

    //model name, xi and xp
    conf >> modelname;
    conf >> modeltype;
    conf >> xicore;
    conf >> xwcore;
    conf >> fitcore;

    //number of saved draws and number of trees
    size_t nd;
    size_t m;
    size_t mh;

    conf >> nd;
    conf >> m;
    conf >> mh;

    //number of predictors
    size_t p, k;
    conf >> p;
    conf >> k;
  
    //thread count
    int tc;
    conf >> tc;
    conf.close();

    //MPI initialization
    int mpirank=0;
    #ifdef _OPENMPI
    int mpitc;
    MPI_Init(NULL,NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpitc);
    #ifndef SILENT
    cout << "\nMPI: node " << mpirank << " of " << mpitc << " processes." << endl;
    #endif
    if(tc<=1){
        cout << "Error: tc=" << tc << endl;
        MPI_Finalize();
        return 0; //need at least 2 processes! 
    } 
    if(tc!=mpitc) {
        cout << "Error: tc does not match mpitc" <<  endl;
        MPI_Finalize();
        return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
    }
    // #else
    //    if(tc!=1) return 0; //serial mode should have no slave threads!
    #endif


    //--------------------------------------------------
    // Banner
    if(mpirank==0) {
        cout << endl;
        cout << "-----------------------------------" << endl;
        cout << "OpenBT extract mixing weights CLI" << endl;
        cout << "Loading config file at " << folder << endl;
    }

    //--------------------------------------------------
    //read in xp.
    std::vector<double> xp;
    double xtemp;
    size_t np;
    //   std::string folder("."+modelname+"/");
    std::stringstream xfss;
    std::string xfs;
    xfss << folder << xwcore << mpirank;
    xfs=xfss.str();
    std::ifstream xf(xfs);
    while(xf >> xtemp)
        xp.push_back(xtemp);
    np = xp.size()/p;
    #ifndef SILENT
    cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;
    #endif
    
    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    xi.resize(p);

    for(size_t i=0;i<p;i++) {
        std::vector<double> xivec;
        double xitemp;

        std::stringstream xifss;
        std::string xifs;
        xifss << folder << xicore << (i+1);
        xifs=xifss.str();
        std::ifstream xif(xifs);
        while(xif >> xitemp)
            xivec.push_back(xitemp);
        xi[i]=xivec;
    }
    #ifndef SILENT
    cout << "&&& made xinfo\n";
    #endif

    //summarize input variables:
    #ifndef SILENT
    if(mpirank==0)
        for(size_t i=0;i<p;i++)
        {
            cout << "Variable " << i << " has numcuts=" << xi[i].size() << " : ";
            cout << xi[i][0] << " ... " << xi[i][xi[i].size()-1] << endl;
        }
    #endif

    //--------------------------------------------------
    // set up amxbrt object
    amxbrt axb(m); //Sets number of trees
    axb.setk(k);
    axb.setxi(&xi); //set the cutpoints for this model object
    
    //load from file
    #ifndef SILENT
    if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
    #endif
    size_t ind,im,imh;
    std::ifstream imf(folder + modelname + fitcore);
    imf >> ind;
    imf >> im;
    imf >> imh;
    #ifdef _OPENMPI
    if(nd!=ind) { cout << "Error loading posterior trees"<< "nd = " << nd << " -- ind = " << ind << endl; MPI_Finalize(); return 0; }
    if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; MPI_Finalize(); return 0; }
    if(mh!=imh) { cout << "Error loading posterior trees" << "mh = " << m << " -- imh = " << imh<< endl; MPI_Finalize(); return 0; }
    #else
    if(nd!=ind) { cout << "Error loading posterior trees" << "nd = " << nd << " -- ind = " << ind << endl; return 0; }
    if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; return 0; }
    if(mh!=imh) { cout << "Error loading posterior trees" << "mh = " << mh << " -- imh = " << imh<< endl; return 0; }
    #endif

    size_t temp=0;
    imf >> temp;
    std::vector<int> e_ots(temp);
    for(size_t i=0;i<temp;i++) imf >> e_ots.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_oid(temp);
    for(size_t i=0;i<temp;i++) imf >> e_oid.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_ovar(temp);
    for(size_t i=0;i<temp;i++) imf >> e_ovar.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_oc(temp);
    for(size_t i=0;i<temp;i++) imf >> e_oc.at(i);

    temp=0;
    imf >> temp;
    std::vector<double> e_otheta(temp);
    for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_otheta.at(i);

    imf.close();

    //Create dinfo for predictions. The fp is just a place holder and is not needed in this script
    double *fp = new double[np];
    dinfo dip;
    dip.x = &xp[0]; dip.y=fp; dip.p = p; dip.n=np; dip.tc=1;

    //Create Eigen objects to store the weight posterior draws -- these are just the thetas for each bottom node
    mxd wts_iter(k,np); //Eigen matrix to store the weights at each iteration -- will be reset to zero prior to running get wts method  
    mxd wts_draw(nd,np); //Eigen matrix to hold posterior draws for each model weight -- used when writing to the file for ease of notation
    std::vector<mxd, Eigen::aligned_allocator<mxd>> wts_list(k); //An std vector of dim k -- each element is an nd X np eigen matrix

    //mxd theta_iter(k,m); //Eigen matrix to store the terminal node parameters at each iteration for the first obs on the root -- will be reset to zero prior to running get wts method  
    //mxd theta_draw(nd,m); //Eigen matrix to hold posterior draws for each terminal node for first obs on root -- used when writing to the file for ease of notation
    //std::vector<mxd, Eigen::aligned_allocator<mxd>> theta_list(k); //An std vector of dim k -- each element is an nd X m eigen matrix

    //Initialize wts_list -- the vector of eigen matrices which will hold the nd X np weight draws
    for(size_t i=0; i<k; i++){
        wts_list[i] = mxd::Zero(nd,np);
        //theta_list[i] = mxd::Zero(nd,m);
    }

    // Temporary vectors used for loading one model realization at a time.
    std::vector<int> onn(m,1);
    std::vector<std::vector<int> > oid(m, std::vector<int>(1));
    std::vector<std::vector<int> > ov(m, std::vector<int>(1));
    std::vector<std::vector<int> > oc(m, std::vector<int>(1));
    std::vector<std::vector<double> > otheta(m, std::vector<double>(1));
    
    // Draw realizations of the posterior predictive.
    size_t curdx=0;
    size_t cumdx=0;
    #ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
    #endif

    // Mean trees first
    if(mpirank==0) cout << "Collecting posterior model weights" << endl;
    for(size_t i=0;i<nd;i++) {
        curdx=0;
        for(size_t j=0;j<m;j++) {
            onn[j]=e_ots.at(i*m+j);
            oid[j].resize(onn[j]);
            ov[j].resize(onn[j]);
            oc[j].resize(onn[j]);
            otheta[j].resize(onn[j]*k);
            for(size_t l=0;l<(size_t)onn[j];l++) {
            oid[j][l]=e_oid.at(cumdx+curdx+l);
            ov[j][l]=e_ovar.at(cumdx+curdx+l);
            oc[j][l]=e_oc.at(cumdx+curdx+l);
            for(size_t r=0;r<k;r++){
                otheta[j][l*k+r]=e_otheta.at((cumdx+curdx+l)*k+r);
            }
            
            }
            curdx+=(size_t)onn[j];
        }
        cumdx+=curdx;

        //Load the current tree structure by using the above vectors
        axb.loadtree_vec(0,m,onn,oid,ov,oc,otheta); 
        //if(mpirank == 0) axb.pr_vec();

        //Get the current posterior draw of the weights
        wts_iter = mxd::Zero(k,np);
        axb.get_mix_wts(&dip, &wts_iter);
        
        //Get terminal node parameters for the 1st pt on the node -- remove later
        //theta_iter = mxd::Zero(k,m);
        //axb.get_mix_theta(&dip, &theta_iter);
        
        //Store these weights into the Vector of Eigen Matrices
        for(size_t j = 0; j<k; j++){
            wts_list[j].row(i) = wts_iter.row(j); //populate the ith row of each wts_list[j] matrix (ith post draw) for model weight j
            //theta_list[j].row(i) = theta_iter.row(j); //populate the ith row of each theta_list[j] matrix (ith post draw) for term node parameter j 
        }

    }
    #ifdef _OPENMPI
    if(mpirank==0) {
        tend=MPI_Wtime();
        cout << "Posterior predictive draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
    }
    #endif
    // Save the draws.
    if(mpirank==0) cout << "Saving posterior weight draws...";
    for(size_t l = 0; l<k; l++){
        std::ofstream omf(folder + modelname + ".w" + std::to_string(l+1) + "draws" + std::to_string(mpirank));
        wts_draw = wts_list[l];
        for(size_t i=0;i<nd;i++) {
            for(size_t j=0;j<np;j++)
                omf << std::scientific << wts_draw(i,j) << " ";
            omf << endl;
        }
        omf.close();
    }
    
    //Save the terminal node draws for the 1st iteration on each root
    /*
    for(size_t l = 0; l<k; l++){
        std::ofstream omf(folder + modelname + ".tnp" + std::to_string(l+1) + "draws" + std::to_string(mpirank));
        theta_draw = theta_list[l];
        for(size_t i=0;i<nd;i++) {
            for(size_t j=0;j<m;j++)
                omf << std::scientific << theta_draw(i,j) << " ";
            omf << endl;
        }
        omf.close();
    }
    */

    if(mpirank==0) cout << " done." << endl;

    //-------------------------------------------------- 
    // Cleanup.
    #ifdef _OPENMPI
    MPI_Finalize();
    #endif

    return 0;
    }

