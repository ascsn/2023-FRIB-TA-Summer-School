#include <chrono>
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
#define MODEL_MERCK_TRUNCATED 8
#define MODEL_MIXBART 9
#define MODEL_MIXEMULATE 10


int main(int argc, char* argv[])
{
    std::string folder("");

    if(argc>1)
    {
        std::string confopt("--conf");
    if(confopt.compare(argv[1])==0) {
#ifdef _OPENMPI
        return 101;
#else
        return 100;
#endif
        }

    //otherwise argument on the command line is path to conifg file.
    folder=std::string(argv[1]);
    folder=folder+"/";
    }


    //-----------------------------------------------------------
    //random number generation
    crn gen;
    gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()));

    //--------------------------------------------------
    //process args
    std::ifstream conf(folder+"config");

    // model type
    int modeltype;
    conf >> modeltype;

    //Number of models
    int nummodels;
    conf >> nummodels;

    // extract the cores for the mix model and simulator inputs
    std::string xcore,ycore,score,ccore;
    double means,base,baseh,power,powerh,lam,nu;
    size_t m,mh;

    std::vector<std::string> xcore_list,ycore_list,score_list,ccore_list;
    std::vector<double> means_list,base_list,baseh_list,power_list,powerh_list, lam_list, nu_list;
    std::vector<size_t> m_list, mh_list;
    
    // Get model mixing cores & inputs then get each emulator cores & inputs 
    for(int i=0;i<=nummodels;i++){
        // Get xcore, ycore, score, and ccore
        conf >> xcore;
        conf >> ycore;
        conf >> score;
        conf >> ccore;

        // Store into respective lists
        xcore_list.push_back(xcore);
        ycore_list.push_back(ycore);
        score_list.push_back(score);
        ccore_list.push_back(ccore);

        // Data means
        conf >> means;
        means_list.push_back(means);

        // Tree sizes
        conf >> m;
        conf >> mh;
        m_list.push_back(m);
        mh_list.push_back(mh);

        // Tree prior hyperparameters
        conf >> base;
        conf >> baseh;
        conf >> power;
        conf >> powerh;
        base_list.push_back(base);
        baseh_list.push_back(baseh);
        power_list.push_back(power);
        powerh_list.push_back(powerh);

        // Variance Prior
        conf >> lam;
        conf >> nu;
        lam_list.push_back(lam);
        nu_list.push_back(nu);


        // Prints
        /*
        cout << "xcore = " << xcore << endl;
        cout << "ycore = " << ycore << endl;
        cout << "score = " << score << endl;
        cout << "ccore = " << ccore << endl;
        cout << "data.mean = " << means << endl;
        cout << "m = " << m << endl;
        cout << "mh = " << mh << endl;
        cout << "base = " << base << endl;
        cout << "baseh = " << baseh << endl;
        cout << "power = " << power << endl;
        cout << "powerh = " << powerh << endl;
        cout << "lam = " << lam << endl;
        cout << "nu = " << nu << endl;
        */
    }

    // Get the design columns per emulator
    std::vector<std::vector<size_t>> x_cols_list(nummodels, std::vector<size_t>(1));
    std::vector<size_t> xcols, pvec;
    size_t p, xcol;
    for(int i=0;i<nummodels;i++){
        conf >> p;
        pvec.push_back(p);
        x_cols_list[i].resize(p);
        for(size_t j = 0; j<p; j++){
            conf >> xcol;
            x_cols_list[i][j] = xcol;
        }

    }

    // Get the id root computer and field obs ids per emulator
    // std::string idcore;
    // conf >> idcore;

    // MCMC properties
    size_t nd, nburn, nadapt, adaptevery;
    conf >> nd;
    conf >> nburn;
    conf >> nadapt;
    conf >> adaptevery;

    // Get tau and beta for terminal node priors
    std::vector<double> tau_emu_list;
    double tau_disc, tau_wts, tau_emu, beta_disc, beta_wts;
    conf >> tau_disc; //discrepancy tau
    conf >> tau_wts; // wts tau
    for(int i=0;i<nummodels;i++){
        conf >> tau_emu; // emulator tau
        tau_emu_list.push_back(tau_emu);
    }
    conf >> beta_disc;
    conf >> beta_wts;

    //control
    double pbd, pb, pbdh, pbh;
    double stepwpert, stepwperth;
    double probchv, probchvh;
    int tc;
    size_t minnumbot;
    size_t minnumboth;
    size_t printevery;
    std::string xicore;
    std::string modelname;
    conf >> pbd;
    conf >> pb;
    conf >> pbdh;
    conf >> pbh;
    conf >> stepwpert;
    conf >> stepwperth;
    conf >> probchv;
    conf >> probchvh;
    conf >> minnumbot;
    conf >> minnumboth;
    conf >> printevery;
    conf >> xicore;
    conf >> tc;
    conf >> modelname;

    bool dopert=true;
    bool doperth=true;
    if(probchv<0) dopert=false;
    if(probchvh<0) doperth=false;
    
    // summary statistics yes/no
    bool summarystats = false;
    std::string summarystats_str;
    conf >> summarystats_str;
    if(summarystats_str=="TRUE"){ summarystats = true; }
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
    if(tc<=1) return 0; //need at least 2 processes!
    if(tc!=mpitc) return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
    // #else
    //    if(tc!=1) return 0; //serial mode should have no slave threads!
#endif

    //--------------------------------------------------
    // Banner
    if(mpirank==0) {
        cout << endl;
        cout << "-----------------------------------" << endl;
        cout << "OpenBT model mixing interface" << endl;
        cout << "Loading config file at " << folder << endl;
    }

    //--------------------------------------------------
    //read in y for mixing and z's for emulation
    std::vector<std::vector<double>> y_list(ycore_list.size(), std::vector<double>(1));
    std::vector<double> y;
    std::vector<size_t> nvec(nummodels+1,0);
    double ytemp;
    size_t n=0;
    std::stringstream yfss;
    std::string yfs;
    std::ifstream yf;

    for(size_t i=0;i<ycore_list.size();i++){
    if(y.size()>0){y.clear();} //clear the contents of the y vector
    #ifdef _OPENMPI
        if(mpirank>0) { //only load data on slaves
    #endif
        yfss << folder << ycore_list[i] << mpirank;
        yfs=yfss.str();
        yf.open(yfs);
        while(yf >> ytemp)
            y.push_back(ytemp);
        n=y.size();
        // Store into the vectors
        //nvec.push_back(n);
        nvec[i]=n;
        //cout << "nvec[i] = " << nvec[i] << endl;       
        y_list[i].resize(n);
        y_list[i] = y;
        // Append the field obs to the computer ouput when i > 0....not this second portion of the vector will be updated each iteration of mcmc
        if(i>0){
            y_list[i].insert(y_list[i].end(),y_list[0].begin(),y_list[0].end());    
        }

        //reset stream variables
        yfss.str("");
        yf.close();
    
    #ifndef SILENT
        cout << "node " << mpirank << " loaded " << n << " from " << yfs <<endl;
    #endif
    #ifdef _OPENMPI
    }
    #endif
    }

    //--------------------------------------------------
    //read in x 
    std::vector<std::vector<double>> x_list(xcore_list.size(), std::vector<double>(1));
    std::vector<std::vector<double>> xv_list; // x inputs for the variance model (stores )
    std::vector<double> x;
    std::stringstream xfss;
    std::string xfs;
    std::ifstream xf;     
    double xtemp;
    p = 0;
    for(size_t i = 0;i<xcore_list.size();i++){
#ifdef _OPENMPI
        if(mpirank>0) {
#endif
        if(x.size() > 0){x.clear();}
        xfss << folder << xcore_list[i] << mpirank;
        xfs=xfss.str();
        xf.open(xfs);
        while(xf >> xtemp){
            x.push_back(xtemp);
            //cout << "model = " << i << "---- x = " << xtemp << endl;
        }
        p = x.size()/nvec[i];
        if(i == 0){
            pvec.insert(pvec.begin(), p);
        }
        // Store into the vectors 
        x_list[i].resize(nvec[i]*pvec[i]);
        x_list[i] = x;

        //Update nvec[i] when i>0 (this accounts for the step of adding field obs to the emulation data sets, which happens later)
        if(i>0){
            nvec[i] = nvec[i] + nvec[0];
            //cout << "nvec[i] = " << nvec[i] << endl; 
        }

        //reset stream variables
        xfss.str("");
        xf.close();
#ifndef SILENT
        cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << p << " from " << xfs << endl;
#endif
#ifdef _OPENMPI
        }
        int tempp = (unsigned int) pvec[i];
        MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if(mpirank>0 && pvec[i] != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA" << endl; MPI_Finalize(); return 0;}
        pvec[i]=(size_t)tempp;
#endif
    }

    // Set xv_list = x_list. This preserves the original separation of field anc computer inputs
    // xv_list is the set of inputs passed into the variance models
    if(mpirank>0){
        xv_list = x_list;
    }

    //--------------------------------------------------
    // Update x_lists[1] to x_lists[nummodels] to include the field inputs
    std::vector<std::vector<double>> xf_list(nummodels);
    size_t xcolsize = 0;
    xcol = 0;
    // Get the appropriate x columns
    if(mpirank > 0){
        for(size_t i=0;i<nummodels;i++){
            xcolsize = x_cols_list[i].size(); //x_cols_list is nummodel dimensional -- only for emulators
            for(size_t j=0;j<nvec[0];j++){
                for(size_t k=0;k<xcolsize;k++){
                    xcol = x_cols_list[i][k] - 1;
                    xf_list[i].push_back(x_list[0][j*xcolsize + xcol]); //xf_list is nummodel dimensional -- only for emulators
                    //cout << "model = " << i+1 << "--x = " << x_list[0][j*xcolsize + xcol] << endl;
                }
            }

            // Add the field obs data to the computer obs data (x_list is not used in dimix but it is convenient to update here in this loop)
            x_list[i+1].insert(x_list[i+1].end(),xf_list[i].begin(),xf_list[i].end()); //x_list is nummodel+1  dimensional -- for mixing & emulators
            //cout << "x_list[i+1].size = " << x_list[i+1].size() << endl;

            /*
            for(size_t k=0;k<x_list[i+1].size();k++){
                cout << "x_list[i+1][k] = " << x_list[i+1][k] << endl;
            }
            */
        }
    }


    //--------------------------------------------------
    //dinfo
    std::vector<dinfo> dinfo_list(nummodels+1);
    for(int i=0;i<=nummodels;i++){
        dinfo_list[i].n=0;dinfo_list[i].p=pvec[i],dinfo_list[i].x = NULL;dinfo_list[i].y=NULL;dinfo_list[i].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0){ 
#endif 
            dinfo_list[i].n=nvec[i]; dinfo_list[i].x = &x_list[i][0]; dinfo_list[i].y = &y_list[i][0];
            //cout <<  "dinfo_list[i].n = " << dinfo_list[i].n << endl;
#ifdef _OPENMPI
        }
#endif
    }

    /*   
    diterator diter1(&dinfo_list[1]);
    cout << "dinfo_list[1].n = " << dinfo_list[1].n << endl;
    for(;diter1<diter1.until();diter1++) {
        cout << "gety = " <<  diter1.gety() << endl;
        //cout << "getx = " <<  diter1.getx() << endl;       
    }
    
    diterator diter2(&dinfo_list[2]);
    cout << "dinfo_list[2].n = " << dinfo_list[2].n << endl;
    for(;diter2<diter2.until();diter2++) {
        cout << "gety = " <<  diter2.gety() << endl;
        //cout << "getx = " <<  diter2.getx() << endl;       
    }
    */
    
    //--------------------------------------------------
    //read in sigmav  -- same as above.
    std::vector<std::vector<double>> sigmav_list(score_list.size(), std::vector<double>(1));
    std::vector<double> sigmav;
    std::vector<size_t> nsigvec;
    std::vector<dinfo> disig_list(nummodels+1);
    std::vector<double*> sig_vec(nummodels+1);
    std::stringstream sfss;
    std::string sfs;
    std::ifstream sf;
    double stemp;
    size_t nsig=0;
    for(int i=0;i<=nummodels;i++){
#ifdef _OPENMPI
        if(mpirank>0) { //only load data on slaves
#endif
        sigmav.clear(); // clear the vector of any contents
        sfss << folder << score_list[i] << mpirank;
        sfs=sfss.str();
        sf.open(sfs);
        while(sf >> stemp)
            sigmav.push_back(stemp);
        nsig=sigmav.size();
        // Store the results in the vector
        sigmav_list[i].resize(nsig);
        sigmav_list[i] = sigmav;
        //reset stream variables
        sfss.str("");
        sf.close();
#ifndef SILENT
        cout << "node " << mpirank << " loaded " << nsig << " from " << sfs <<endl;
#endif
#ifdef _OPENMPI
        if(nvec[i]!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; MPI_Finalize(); return 0; }
        }
#else
        if(nvec[i]!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; return 0; }
#endif

        sig_vec[i]=&sigmav_list[i][0];
        disig_list[i].n=0; disig_list[i].p=pvec[i]; disig_list[i].x=NULL; disig_list[i].y=NULL; disig_list[i].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0) { 
#endif
        if(i>0){
            // Emulators
            disig_list[i].n=(nvec[i]-nvec[0]); // number of model runs   
            disig_list[i].x=&xv_list[i][0]; // inputs for those model runs --- we want to get predictions of variance for these runs
            disig_list[i].y=sig_vec[i]; // sig_vec that keeps track of variance for model runs AND field obs. This is tied into ambm_list, hence both elements are needed
            // NOTE: length of sig_vec[i] != nvec[i]-nvec[0] (nor does it equal the number of rows in computer design matrix)
            // This is a hack to get things to work in the existing framewkork
        }else{
            //Mixer -- xv_list[0] = x_list[0] ... just using xv_list here for consistency
            disig_list[i].n=nvec[0];disig_list[i].x=&xv_list[i][0];disig_list[i].y=sig_vec[i];    
        }

#ifdef _OPENMPI
        }
#endif
    }
    
    //--------------------------------------------------
    // read in the initial change of variable rank correlation matrix
    std::vector<std::vector<std::vector<double>>> chgv_list;
    std::vector<std::vector<double>> chgv;
    std::vector<double> cvvtemp;
    double cvtemp;
    std::stringstream chgvfss;
    std::string chgvfs;
    std::ifstream chgvf;
    std::vector<int*> lwr_vec(nummodels+1, new int[tc]);
    std::vector<int*> upr_vec(nummodels+1, new int[tc]);

    for(int k=0;k<=nummodels;k++){
        chgvfss << folder << ccore_list[k];
        chgvfs=chgvfss.str();
        chgvf.open(chgvfs);
        for(size_t i=0;i<dinfo_list[k].p;i++) {
            cvvtemp.clear();
            for(size_t j=0;j<dinfo_list[k].p;j++) {
                chgvf >> cvtemp;
                cvvtemp.push_back(cvtemp);
            }
            chgv.push_back(cvvtemp);
        }
        chgv_list.push_back(chgv);
        //reset stream variables
        chgvfss.str("");
        chgvf.close();
#ifndef SILENT
    cout << "mpirank=" << mpirank << ": change of variable rank correlation matrix loaded:" << endl;
#endif
    }
    // if(mpirank==0) //print it out:
    //    for(size_t i=0;i<di.p;i++) {
    //       for(size_t j=0;j<di.p;j++)
    //          cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
    //       cout << endl;
    //    }

    //--------------------------------------------------
    // decide what variables each slave node will update in change-of-variable proposals.
#ifdef _OPENMPI
    //int* lwr=new int[tc];
    //int* upr=new int[tc];
    for(int j=0;j<=nummodels;j++){
        lwr_vec[j][0]=-1; upr_vec[j][0]=-1;
        for(size_t i=1;i<(size_t)tc;i++) { 
            lwr_vec[j][i]=-1; upr_vec[j][i]=-1; 
            calcbegend(pvec[j],i-1,tc-1,&lwr_vec[j][i],&upr_vec[j][i]);
            if(pvec[j]>1 && lwr_vec[j][i]==0 && upr_vec[j][i]==0) { lwr_vec[j][i]=-1; upr_vec[j][i]=-1; }
        }
#ifndef SILENT
        if(mpirank>0) cout << "Slave node " << mpirank << " will update variables " << lwr_vec[j][mpirank] << " to " << upr_vec[j][mpirank]-1 << endl;
#endif
    }
#endif
    
    //--------------------------------------------------
    //make xinfo
    std::vector<xinfo> xi_list(nummodels+1);
    std::vector<double> xivec;
    std::stringstream xifss;
    std::string xifs;
    std::ifstream xif;
    double xitemp;
    size_t ind = 0;

    for(int j=0;j<=nummodels;j++){
        xi_list[j].resize(pvec[j]);
        for(size_t i=0;i<pvec[j];i++) {
            // Get the next column in the x_cols_list -- important since emulators may have different inputs
            if(j>0){
                ind = (size_t)x_cols_list[j-1][i]; 
            }else{
                ind = i+1;
            }
            xifss << folder << xicore << (ind); 
            xifs=xifss.str();
            xif.open(xifs);
            while(xif >> xitemp){
                xivec.push_back(xitemp);
            }
            xi_list[j][i]=xivec;
            //Reset file strings
            xifss.str("");
            xif.close();
            xivec.clear();
        }
#ifndef SILENT
        cout << "&&& made xinfo\n";
#endif

    //summarize input variables:
#ifndef SILENT
        for(size_t i=0;i<p;i++){
            cout << "Variable " << i << " has numcuts=" << xi_list[j][i].size() << " : ";
            cout << xi_list[j][i][0] << " ... " << xi_list[j][i][xi_list[j][i].size()-1] << endl;
        }
#endif
    }
    
    //--------------------------------------------------
    //Load master list of ids for emulators, designates field obs vs computer output
    /*
    std::vector<std::vector<std::string>> id_list(nummodels);
    std::stringstream idfss;
    std::string idfs;
    std::ifstream idf;
    std::string idtemp;
    
    idfss << folder << idcore << mpirank;
    idfs=idfss.str();
    idf.open(idfs);
    for(int i=0;i<nummodels;i++){
        for(int j=0;j<nvec[i+1];j++){
            idf >> idtemp;
            id_list[i].push_back(idtemp);
        }
    }
    

    //--------------------------------------------------
    //Copy the original z value for all field obs -- do per emulation dataset
    //This is essential when getting the contribution of the field data for the emulator
    //By design, the original z's for field obs are the observed y values.
    std::vector<std::vector<double>> zf_list(nummodels); //Copy of the original observed data
    std::vector<std::vector<int>> zfidx_list(nummodels); //contains row indexes corresponding to whcih row each field obs belongs in per emu dataset
    std::vector<std::vector<double>> xf_emu_list(xcore_list.size(), std::vector<double>(1)); //copy of the x's for the field obs
    for(int i=0;i<nummodels;i++){
        for(int j=0;j<id_list[i].size();j++){
            if(id_list[i][j] == "f"){
                zf_list[i].push_back(y_list[i+1][j]);
                zfidx_list[i].push_back(j);
                for(int k=0;k<pvec[i+1];k++){
                    xf_emu_list[i].push_back(x_list[i][j*pvec[i+1] + k]);
                }
            }
        }
    }
    */
    //--------------------------------------------------
    //Set up model objects and MCMC
    //--------------------------------------------------    
    ambrt *ambm_list[nummodels]; //additive mean bart emulators
    psbrt *psbm_list[nummodels]; //product variance for bart emulators
    amxbrt axb(m_list[0]); // additive mean mixing bart
    psbrt pxb(mh_list[0],lam_list[0]); //product model for mixing variance
    std::vector<dinfo> dips_list(nummodels+1); //dinfo for psbrt objects
    std::vector<double*> r_list(nummodels+1); //residual list
    double opm; //variance info
    double lambda; //variance info
    finfo fi;
    size_t tempn; // used when defining dips_list for emulators
    int l = 0; //Used for indexing emulators
    nu = 1.0; //reset nu to 1, previosuly defined earlier in program

    //Initialize the model mixing bart objects
    if(mpirank > 0){
        fi = mxd::Ones(nvec[0], nummodels+1); //dummy initialize to matrix of 1's -- n0 x K+1 (1st column is discrepancy)
    }
    //cutpoints
    axb.setxi(&xi_list[0]);   
    //function output information
    axb.setfi(&fi, nummodels+1);
    //data objects
    axb.setdata_mix(&dinfo_list[0]);  //set the data
    //thread count
    axb.settc(tc-1);      //set the number of slaves when using MPI.
    //mpi rank
#ifdef _OPENMPI
    axb.setmpirank(mpirank);  //set the rank when using MPI.
    axb.setmpicvrange(lwr_vec[0],upr_vec[0]); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior
    axb.settp(base_list[0], //the alpha parameter in the tree depth penalty prior
            power_list[0]     //the beta parameter in the tree depth penalty prior
            );
    //MCMC info
    axb.setmi(
            pbd,  //probability of birth/death
            pb,  //probability of birth
            minnumbot,    //minimum number of observations in a bottom node
            dopert, //do perturb/change variable proposal?
            stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv_list[0]  //initialize the change of variable correlation matrix.
            );

    //Set prior information
    mxd prior_precision(nummodels+1,nummodels+1);
    vxd prior_mean(nummodels+1);
    prior_precision = (1/(tau_wts*tau_wts))*mxd::Identity(nummodels+1,nummodels+1);
    prior_precision(0,0) = (1/(tau_disc*tau_disc));
    prior_mean = beta_wts*vxd::Ones(nummodels+1);
    prior_mean(0) = beta_disc;
    
    //Sets the model priors for the functions if they are different
    axb.setci(prior_precision, prior_mean, sig_vec[0]);

    //--------------------------------------------------
    //setup psbrt object
    //make di for psbrt object
    dips_list[0].n=0; dips_list[0].p=pvec[0]; dips_list[0].x=NULL; dips_list[0].y=NULL; dips_list[0].tc=tc;
    for(int j=0;j<=nummodels;j++) r_list[j] = NULL;
    //double *r = NULL;
#ifdef _OPENMPI
    if(mpirank>0) {
#endif
    r_list[0] = new double[nvec[0]]; 
    //r = new double[nvec[0]];
    for(size_t i=0;i<nvec[0];i++) r_list[0][i]=sigmav_list[0][i];
    //for(size_t i=0;i<nvec[0];i++) r[i]=sigmav_list[0][i];
    dips_list[0].x=&xv_list[0][0]; dips_list[0].y=r_list[0]; dips_list[0].n=nvec[0];
    //dips_list[0].x=&x_list[0][0]; dips_list[0].y=r; dips_list[0].n=nvec[0];
#ifdef _OPENMPI
    }
#endif

    //Variance infomration
    opm=1.0/((double)mh_list[0]);
    nu=2.0*pow(nu_list[0],opm)/(pow(nu_list[0],opm)-pow(nu_list[0]-2.0,opm));
    lambda=pow(lam_list[0],opm);
 
    //cutpoints
    pxb.setxi(&xi_list[0]);    //set the cutpoints for this model object
    //data objects
    pxb.setdata(&dips_list[0]);  //set the data
    //thread count
    pxb.settc(tc-1); 
    //mpi rank
#ifdef _OPENMPI
    pxb.setmpirank(mpirank);  //set the rank when using MPI.
    pxb.setmpicvrange(lwr_vec[0],upr_vec[0]); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior
    pxb.settp(baseh_list[0], //the alpha parameter in the tree depth penalty prior
            powerh_list[0]     //the beta parameter in the tree depth penalty prior
            );
    pxb.setmi(
            pbdh,  //probability of birth/death
            pbh,  //probability of birth
            minnumboth,    //minimum number of observations in a bottom node
            doperth, //do perturb/change variable proposal?
            stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv_list[0]  //initialize the change of variable correlation matrix.
            );
    pxb.setci(nu,lambda);

    //Initialize the emulation bart objects
    for(int j=1;j<=nummodels;j++){
        // Set l for indexing
        l = j-1;
        //Redefine the class instance
        ambm_list[l] = new ambrt(m_list[j]);     
        //cutpoints
        ambm_list[l]->setxi(&xi_list[j]);
        //data objects
        ambm_list[l]->setdata(&dinfo_list[j]);        
        //thread count
        ambm_list[l]->settc(tc-1);      //set the number of slaves when using MPI.
        //mpi rank
    #ifdef _OPENMPI
        ambm_list[l]->setmpirank(mpirank);  //set the rank when using MPI.
        ambm_list[l]->setmpicvrange(lwr_vec[j],upr_vec[j]); //range of variables each slave node will update in MPI change-of-var proposals.
    #endif
        //tree prior
        ambm_list[l]->settp(base_list[j], //the alpha parameter in the tree depth penalty prior
                    power_list[j]     //the beta parameter in the tree depth penalty prior
                    );
        //MCMC info
        ambm_list[l]->setmi(
                pbd,  //probability of birth/death
                pb,  //probability of birth
                minnumbot,    //minimum number of observations in a bottom node
                dopert, //do perturb/change variable proposal?
                stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
                probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
                &chgv_list[j]  //initialize the change of variable correlation matrix.
                );
        ambm_list[l]->setci(tau_emu_list[l],sig_vec[j]);
        
        //--------------------------------------------------
        //setup psbrt object
        psbm_list[l] = new psbrt(mh_list[j]);

        //make di for psbrt object
        opm=1.0/((double)mh_list[j]);
        nu=2.0*pow(nu_list[j],opm)/(pow(nu_list[j],opm)-pow(nu_list[j]-2.0,opm));
        lambda=pow(lam_list[j],opm);

        //make dips info
        tempn = 0;
        dips_list[j].n=0; dips_list[j].p=p; dips_list[j].x=NULL; dips_list[j].y=NULL; dips_list[j].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0) {
#endif      
            tempn = nvec[j] - nvec[0];
            r_list[j] = new double[tempn];
            for(size_t i=0;i<tempn;i++) r_list[j][i]=sigmav_list[j][i];
            dips_list[j].x=&xv_list[j][0]; dips_list[j].y=r_list[j]; dips_list[j].n=tempn;
            /*
            if(j == 2){
                diterator diter2(&dips_list[2]);
                cout << "dinfo_list[2].n = " << dinfo_list[2].n << endl;
                for(;diter2<diter2.until();diter2++) {
                    //cout << "gety = " <<  diter2.gety() << endl;
                    cout << "getx = " <<  diter2.getx() << endl;       
                }
            }
            */

#ifdef _OPENMPI
        }
#endif
        //cutpoints
        psbm_list[l]->setxi(&xi_list[j]);    //set the cutpoints for this model object
        //data objects
        psbm_list[l]->setdata(&dips_list[j]);  //set the data
        //thread count
        psbm_list[l]->settc(tc-1); 
        //mpi rank
        #ifdef _OPENMPI
        psbm_list[l]->setmpirank(mpirank);  //set the rank when using MPI.
        psbm_list[l]->setmpicvrange(lwr_vec[j],upr_vec[j]); //range of variables each slave node will update in MPI change-of-var proposals.
        #endif
        //tree prior
        psbm_list[l]->settp(baseh_list[j], //the alpha parameter in the tree depth penalty prior
                powerh_list[j]     //the beta parameter in the tree depth penalty prior
                );
        psbm_list[l]->setmi(
                pbdh,  //probability of birth/death
                pbh,  //probability of birth
                minnumboth,    //minimum number of observations in a bottom node
                doperth, //do perturb/change variable proposal?
                stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
                probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
                &chgv_list[j]  //initialize the change of variable correlation matrix.
                );
        psbm_list[l]->setci(nu,lambda);
    }

    //-------------------------------------------------- 
    // MCMC
    //-------------------------------------------------- 
    // Method Wrappers
    brtMethodWrapper faxb(&brt::f,axb);
    brtMethodWrapper fpxb(&brt::f,pxb);
    //brtMethodWrapper *fambm_list[nummodels];
    //brtMethodWrapper *fpsbm_list[nummodels];
    
    // Define containers -- similar to those in cli.cpp, except now we iterate over K+1 bart objects
    std::vector<std::vector<int>> onn_list(nummodels+1, std::vector<int>(nd,1));
    std::vector<std::vector<std::vector<int>>> oid_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> ovar_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> oc_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<double>>> otheta_list(nummodels+1, std::vector<std::vector<double>>(nd, std::vector<double>(1)));
    
    std::vector<std::vector<int>> snn_list(nummodels+1, std::vector<int>(nd,1));
    std::vector<std::vector<std::vector<int>>> sid_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> svar_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> sc_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<double>>> stheta_list(nummodels+1, std::vector<std::vector<double>>(nd, std::vector<double>(1)));
  
    // Initialization of objects
    for(int i=0;i<=nummodels;i++){
        onn_list[i].resize(nd*m_list[i],1);
        oid_list[i].resize(nd*m_list[i], std::vector<int>(1));
        ovar_list[i].resize(nd*m_list[i], std::vector<int>(1));
        oc_list[i].resize(nd*m_list[i], std::vector<int>(1));
        otheta_list[i].resize(nd*m_list[i], std::vector<double>(1));
        

        snn_list[i].resize(nd*mh_list[i],1);
        sid_list[i].resize(nd*mh_list[i], std::vector<int>(1));
        svar_list[i].resize(nd*mh_list[i], std::vector<int>(1));
        sc_list[i].resize(nd*mh_list[i], std::vector<int>(1));
        stheta_list[i].resize(nd*mh_list[i], std::vector<double>(1));
        if(i>0){
            //fambm_list[i-1] = new brtMethodWrapper(&brt::f,*ambm_list[i-1]);
            //fpsbm_list[i-1] = new brtMethodWrapper(&brt::f,*psbm_list[i-1]);
        }
     }

    // dinfo for predictions
    std::vector<dinfo> dimix_list(nummodels);
    std::vector<double*> fmix_list(nummodels);
    for(int i=0;i<nummodels;i++){
        // Initialize class objects
        if(mpirank > 0){
            fmix_list[i] = new double[nvec[0]];
            dimix_list[i].y=fmix_list[i];
            dimix_list[i].p = pvec[i+1]; 
            dimix_list[i].n=nvec[0];
            dimix_list[i].tc=1;
            dimix_list[i].x = &xf_list[i][0];
        }else{
            fmix_list[i] = NULL;
            dimix_list[i].y = NULL;
            dimix_list[i].x = NULL;
            dimix_list[i].p = pvec[i+1]; 
            dimix_list[i].n=0;
            dimix_list[i].tc=1;
        }
    }

    // Items used to extract weights from the mix bart model
    mxd wts_iter; //matrix to store current set of weights
    std::vector<double> mixprednotj;
    
    double *fw = NULL;
    dinfo diw;
    if(mpirank>0){
        // Resize objects and define diw
        wts_iter.resize(nummodels+1,nvec[0]);
        mixprednotj.resize(nvec[0],0);
        fw = new double[nvec[0]];
        diw.x = &x_list[0][0]; diw.y=fw; diw.p = pvec[0]; diw.n=nvec[0]; diw.tc=1;
    }else{
        diw.x = NULL; diw.y=NULL; diw.p = pvec[0]; diw.n=0; diw.tc=1;
    }

    // Start the MCMC
#ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
    if(mpirank==0) cout << "Starting MCMC..." << endl;
#else
    cout << "Starting MCMC..." << endl;
#endif

    // Initialize finfo using predictions from each emulator
    if(mpirank > 0){
        for(int j=0;j<nummodels;j++){
            ambm_list[j]->predict(&dimix_list[j]);
            for(size_t k=0;k<dimix_list[j].n;k++){
                fi(k,j+1) = fmix_list[j][k] + means_list[j+1]; //fmix_list is K dimensional and the others are K+1 dimensional (hence j vs. j+1)
            }   
        }
    }

    // Adapt Stage in the MCMC
    diterator diter0(&dips_list[0]);
    for(size_t i=0;i<nadapt;i++) { 
        // Print adapt step number
        if((i % printevery) ==0 && mpirank==0) cout << "Adapt iteration " << i << endl;
#ifdef _OPENMPI  
        // Model Mixing step
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}
        // Get the current model mixing weights   
        if(mpirank>0){
            // draw new weight matrix
            wts_iter = mxd::Zero(nummodels+1,dimix_list[0].n); //resets wt matrix
            axb.get_mix_wts(&diw, &wts_iter);  
        } 
        
        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            // Get re-weighted field observations
            if(mpirank > 0){
                // rest mixprednotj
                mixprednotj.clear();
                mixprednotj.resize(dimix_list[0].n, 0);
                for(int k=0;k<nummodels;k++){    
                    // Get the mixed prediction
                    if(k!=j){
                        for(size_t l=0;l<dimix_list[j].n;l++){
                            mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                            //if(i%1000 == 0)cout << "mixprednotj[l] = " << mixprednotj[l] << endl;
                        }
                    }
                }
                // add the discrepancy and get the weighted field obs
                for(size_t l=0;l<dimix_list[j].n;l++){
                    mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                    y_list[j+1][nvec[j+1] - nvec[0] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l) - means_list[j+1];
                    //cout << "wts_iter(j+1,l) = " << wts_iter(j+1,l) << "---- y_list[j+1][nvec[j+1] - nvec[0] + l] = " << y_list[j+1][nvec[j+1]- nvec[0] + l] << endl;
                    // Now update the appropriate sigmav_list[j+1] entries where the last nvec[0] entries are reweighted field obs
                    sigmav_list[j+1][nvec[j+1]-nvec[0]+l] = sigmav_list[0][l]/wts_iter(j+1,l);
                    //if(i%1000 == 0)cout << "y_list[j+1][nvec[j+1]- nvec[0] + l] = " << y_list[j+1][nvec[j+1]- nvec[0] + l] << endl;
                }
            }
            
            // Update emulator
            if(mpirank==0){ambm_list[j]->draw(gen);} else {ambm_list[j]->draw_mpislave(gen);}
            
            if(mpirank>0){
                //Update finfo column 
                ambm_list[j]->predict(&dimix_list[j]);
                for(size_t l=0;l<dimix_list[j].n;l++){
                    fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; // f_mix is only K dimensional -- hence using j as its index
                    //cout << "fi(l,j+1) = " << fi(l,j+1) << endl; 
                }
            }           
        } 
#else
        // Model Mixing step
        axb.drawvec(gen);
        
        // Get the current model mixing weights    
        wts_iter = mxd::Zero(nummodels+1,dimix_list[0].n); //resets wt matrix
        axb.get_mix_wts(&diw, &wts_iter);  

        // Emulation Steps
        for(int j=0;j<nummodels;j++){
            // reset mixprednotj
            mixprednotj.clear();
            mixprednotj.resize(dimix_list[0].n, 0);

            for(int k=0;k<nummodels;k++){
                // Get the mixed prediction
                if(k!=j){
                    for(int l=0;l<nvec[0];l++){
                        mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                    }
                }
            }
            
            // add the discrepancy and get the weighted field obs
            for(int l=0;l<nvec[0];l++){
                mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                y_list[j+1][nvec[j+1] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l)- means_list[j+1];
                // Now update the appropriate sigmav_list[j+1] entries where the last nvec[0] entries are reweighted field obs
                sigmav_list[j+1][nvec[j+1]-nvec[0]+l] = sigmav_list[0][l]/wts_iter(j+1,l);
            }
            ambm_list[j]->drawvec(gen);
            
            // Update finfo column
            ambm_list[j]->predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; 
            }
        }
#endif
    // *** Think about moving this into mpi sections since we use mpirank > 0 below
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                if(mpirank>0){
                    ambm_list[j-1]->predict(&dips_list[j]);
                    for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}    
                    //dips_list[j] -= dinfo_list[j]; // Get residuals
                    //dips_list[j] *= -1;
                }else{
                    dips_list[j] = dinfo_list[j];
                }
                if((i+1)%adaptevery==0 && mpirank==0){ambm_list[j-1]->adapt();}
            }else{
                // Model Mixing
                //dips_list[0] = dinfo_list[0];
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];}
                dips_list[0] -= faxb;
                if((i+1)%adaptevery==0 && mpirank==0){axb.adapt();}
            }
        }

#ifdef _OPENMPI
        // Draw the variances
        // Model Mixing
        if(mpirank==0) pxb.draw(gen); else pxb.draw_mpislave(gen);
        //if(mpirank>0){cout << "sigdraw[0][0] = " << sigmav_list[0][0] << endl;} 

        // Emulators
        for(int j=0;j<nummodels;j++){
            if(mpirank==0) psbm_list[j]->draw(gen); else psbm_list[j]->draw_mpislave(gen);
        }

#else
        // Draw for model mixing
        pxb.draw(gen);        
        // Draw for emulators
        for(int j=0;j<nummodels;j++){psbm_list[j]->draw(gen);}
#endif 
        for(int j=0;j<=nummodels;j++){
            if(j>0){
                // Update the variance for the computer model runs
                psbm_list[j-1]->predict(&disig_list[j]);
                
                // Now update the rest of sigmav_list[j]...the last nvec[0] entries are field obs
                for(size_t l=0;l<nvec[0];l++){sigmav_list[j][nvec[j]-nvec[0]+l] = sigmav_list[0][l]; }
                if((i+1)%adaptevery==0 && mpirank==0) psbm_list[j-1]->adapt();
            }else{
                disig_list[0] = fpxb;
                if((i+1)%adaptevery==0 && mpirank==0) pxb.adapt();
            }
            
        }
    }


    // Burn-in Stage in the MCMC
    for(size_t i=0;i<nburn;i++) { 
        // Print burn step number
        if((i % printevery) ==0 && mpirank==0) cout << "Burn iteration " << i << endl;
#ifdef _OPENMPI  
        // Model Mixing step
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}
        // Get the current model mixing weights   
        if(mpirank>0){
            wts_iter = mxd::Zero(nummodels+1,dimix_list[0].n); //resets wt matrix
            axb.get_mix_wts(&diw, &wts_iter);  
        } 
        
        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            // Get re-weighted field observations
            if(mpirank > 0){
                // rest mixprednotj
                mixprednotj.clear();
                mixprednotj.resize(dimix_list[0].n, 0);
                for(int k=0;k<nummodels;k++){    
                    // Get the mixed prediction
                    if(k!=j){
                        for(size_t l=0;l<dimix_list[j].n;l++){
                            mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                            //cout << "mixprednotj[l] = " << mixprednotj[l] << endl;
                        }
                    }
                }
                // add the discrepancy and get the weighted field obs
                for(size_t l=0;l<dimix_list[j].n;l++){
                    mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                    y_list[j+1][nvec[j+1] - nvec[0] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l)- means_list[j+1];
                    // Now update the appropriate sigmav_list[j+1] entries where the last nvec[0] entries are reweighted field obs
                    sigmav_list[j+1][nvec[j+1]-nvec[0]+l] = sigmav_list[0][l]/wts_iter(j+1,l);
                }
            }
            
            // Update emulator
            if(mpirank==0){ambm_list[j]->draw(gen);} else {ambm_list[j]->draw_mpislave(gen);}
            if(mpirank>0){
                //Update finfo column 
                ambm_list[j]->predict(&dimix_list[j]);
                for(size_t l=0;l<dimix_list[j].n;l++){
                    fi(l,j+1) = fmix_list[j][l] + means_list[j+1];
                    //cout << "fi(l,j+1) = " << fi(l,j+1) << endl; 
                }
            }           
        } 
#else
        // Model Mixing step
        axb.drawvec(gen);
        
        // Get the current model mixing weights    
        wts_iter = mxd::Zero(nummodels+1,dimix_list[0].n); //resets wt matrix
        axb.get_mix_wts(&diw, &wts_iter);  

        // Emulation Steps
        for(int j=0;j<nummodels;j++){
            // rest mixprednotj
            mixprednotj.clear();
            mixprednotj.resize(dimix_list[0].n, 0);
            for(int k=0;k<nummodels;k++){
                // Get the mixed prediction
                if(k!=j){
                    for(int l=0;l<nvec[0];l++){
                        mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                    }
                }
            }
            // add the discrepancy and get the weighted field obs
            for(int l=0;l<nvec[0];l++){
                mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                y_list[j+1][nvec[j+1] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l)- means_list[j+1];
                sigmav_list[j+1][nvec[j+1]-nvec[0]+l] = sigmav_list[0][l]/wts_iter(j+1,l);
            }
            ambm_list[j]->drawvec(gen);
            
            // Update finfo column
            ambm_list[j]->predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; 
            }
        }
#endif
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                if(mpirank>0){
                    ambm_list[j-1]->predict(&dips_list[j]);
                    for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}
                    //dips_list[j] -= dinfo_list[j];
                    //dips_list[j] *= -1;
                }else{
                    dips_list[j] = dinfo_list[j];
                }
            }else{
                // Model Mixing
                //dips_list[0] = dinfo_list[0];
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];} 
                dips_list[0] -= faxb;
            }
        }

#ifdef _OPENMPI
        // Draw the variances
        // Model Mixing
        if(mpirank==0) pxb.draw(gen); else pxb.draw_mpislave(gen);

        // Emulators
        for(int j=0;j<nummodels;j++){
            if(mpirank==0) psbm_list[j]->draw(gen); else psbm_list[j]->draw_mpislave(gen);
        }

#else
        // Draw for model mixing
        pxb.draw(gen);        
        // Draw for emulators
        for(int j=0;j<nummodels;j++){psbm_list[j]->draw(gen);}
#endif 
        for(int j=0;j<=nummodels;j++){
            if(j>0){
                // Update the variance for the computer model runs
                psbm_list[j-1]->predict(&disig_list[j]);
                
                // Now update the rest of sigmav_list[j]...the last nvec[0] entries are field obs
                for(size_t l=0;l<nvec[0];l++){sigmav_list[j][nvec[j]-nvec[0]+l] = sigmav_list[0][l]; }
            }else{
                disig_list[0] = fpxb;
            }
            
        }
    }

    for(size_t i=0;i<nd;i++) { 
        // Print burn step number
        if((i % printevery) ==0 && mpirank==0) cout << "Draw iteration " << i << endl;
#ifdef _OPENMPI  
        // Model Mixing step
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}
        // Get the current model mixing weights   
        if(mpirank>0){
            wts_iter = mxd::Zero(nummodels+1,dimix_list[0].n); //resets wt matrix
            axb.get_mix_wts(&diw, &wts_iter);  
        } 
        
        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            // Get re-weighted field observations
            if(mpirank > 0){
                // rest mixprednotj
                mixprednotj.clear();
                mixprednotj.resize(dimix_list[0].n, 0);
                for(int k=0;k<nummodels;k++){    
                    // Get the mixed prediction
                    if(k!=j){
                        for(size_t l=0;l<dimix_list[j].n;l++){
                            mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                            //cout << "mixprednotj[l] = " << mixprednotj[l] << endl;
                        }
                    }
                }
                // add the discrepancy and get the weighted field obs
                for(size_t l=0;l<dimix_list[j].n;l++){
                    mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                    y_list[j+1][nvec[j+1] - nvec[0] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l)- means_list[j+1];
                    sigmav_list[j+1][nvec[j+1]-nvec[0]+l] = sigmav_list[0][l]/wts_iter(j+1,l);
                }
            }
            
            // Update emulator
            if(mpirank==0){ambm_list[j]->draw(gen);} else {ambm_list[j]->draw_mpislave(gen);}
            if(mpirank>0){
                //Update finfo column 
                ambm_list[j]->predict(&dimix_list[j]);
                for(size_t l=0;l<dimix_list[j].n;l++){
                    fi(l,j+1) = fmix_list[j][l] + means_list[j+1];
                    //cout << "fi(l,j+1) = " << fi(l,j+1) << endl; 
                }
            }           
        } 
#else
        // Model Mixing step
        axb.drawvec(gen);
        
        // Get the current model mixing weights    
        wts_iter = mxd::Zero(nummodels+1,dimix_list[0].n); //resets wt matrix
        axb.get_mix_wts(&diw, &wts_iter);  

        // Emulation Steps
        for(int j=0;j<nummodels;j++){
            // rest mixprednotj
            mixprednotj.clear();
            mixprednotj.resize(dimix_list[0].n, 0);
            for(int k=0;k<nummodels;k++){
                // Get the mixed prediction
                if(k!=j){
                    for(int l=0;l<nvec[0];l++){
                        mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                    }
                }
            }
            // add the discrepancy and get the weighted field obs
            for(int l=0;l<nvec[0];l++){
                mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                y_list[j+1][nvec[j+1] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l)- means_list[j+1];
                sigmav_list[j+1][nvec[j+1]-nvec[0]+l] = sigmav_list[0][l]/wts_iter(j+1,l);
            }
            ambm_list[j]->drawvec(gen);
            
            // Update finfo column
            ambm_list[j]->predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; 
            }
        }
#endif
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                if(mpirank>0){
                    ambm_list[j-1]->predict(&dips_list[j]);
                    for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}
                    //dips_list[j] -= dinfo_list[j];
                    //dips_list[j] *= -1;
                }else{
                    dips_list[j] = dinfo_list[j];
                }
            }else{
                // Model Mixing
                //dips_list[0] = dinfo_list[0];
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];} 
                dips_list[0] -= faxb;
            }
        }

#ifdef _OPENMPI
        // Draw the variances
        // Model Mixing
        if(mpirank==0) pxb.draw(gen); else pxb.draw_mpislave(gen);

        // Emulators
        for(int j=0;j<nummodels;j++){
            if(mpirank==0) psbm_list[j]->draw(gen); else psbm_list[j]->draw_mpislave(gen);
        }

#else
        // Draw for model mixing
        pxb.draw(gen);        
        // Draw for emulators
        for(int j=0;j<nummodels;j++){psbm_list[j]->draw(gen);}
#endif 
        for(int j=0;j<=nummodels;j++){
            if(j>0){
                // Update the variance for the computer model runs
                psbm_list[j-1]->predict(&disig_list[j]);
                
                // Now update the rest of sigmav_list[j]...the last nvec[0] entries are field obs
                for(size_t l=0;l<nvec[0];l++){sigmav_list[j][nvec[j]-nvec[0]+l] = sigmav_list[0][l]; }
            }else{
                disig_list[0] = fpxb;
                
            }
            
        }


        // Save Tree to vector format
        if(mpirank==0) {
            //axb.pr_vec();
            axb.savetree_vec(i,m_list[0],onn_list[0],oid_list[0],ovar_list[0],oc_list[0],otheta_list[0]); 
            pxb.savetree(i,mh_list[0],snn_list[0],sid_list[0],svar_list[0],sc_list[0],stheta_list[0]);
            for(int j=1;j<=nummodels;j++){
                ambm_list[j-1]->savetree(i,m_list[j],onn_list[j],oid_list[j],ovar_list[j],oc_list[j],otheta_list[j]);
                psbm_list[j-1]->savetree(i,mh_list[j],snn_list[j],sid_list[j],svar_list[j],sc_list[j],stheta_list[j]); 
            }
        }
    }

// Writing data to output files
#ifdef _OPENMPI
    if(mpirank==0) {
        tend=MPI_Wtime();
    cout << "Training time was " << (tend-tstart)/60.0 << " minutes." << endl;
    }
#endif
    //Flatten posterior trees to a few (very long) vectors so we can just pass pointers
    //to these vectors back to R (which is much much faster than copying all the data back).
    if(mpirank==0) {
        cout << "Returning posterior, please wait...";
        // Instantiate containers
        std::vector<std::vector<int>*> e_ots(nummodels+1); //=new std::vector<int>(nd*m);
        std::vector<std::vector<int>*> e_oid(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_ovar(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_oc(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<double>*> e_otheta(nummodels+1); //=new std::vector<double>;
        std::vector<std::vector<int>*> e_sts(nummodels+1); //=new std::vector<int>(nd*mh);
        std::vector<std::vector<int>*> e_sid(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_svar(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_sc(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<double>*> e_stheta(nummodels+1); //=new std::vector<double>;

        // Initialize containers with pointers
        for(int j=0;j<=nummodels;j++){
            e_ots[j]=new std::vector<int>(nd*m_list[j]);
            e_oid[j]=new std::vector<int>;
            e_ovar[j]=new std::vector<int>;
            e_oc[j]=new std::vector<int>;
            e_otheta[j]=new std::vector<double>;
            e_sts[j]=new std::vector<int>(nd*mh_list[j]);
            e_sid[j]=new std::vector<int>;
            e_svar[j]=new std::vector<int>;
            e_sc[j]=new std::vector<int>;
            e_stheta[j]=new std::vector<double>;
        }

        // Loop through each model and store in appropriate outfile 
        for(size_t i=0;i<nd;i++)
            for(int l=0;l<=nummodels;l++) {
                m = m_list[l];
                for(size_t j=0;j<m;j++){ 
                    e_ots[l]->at(i*m+j)=static_cast<int>(oid_list[l][i*m+j].size());
                    e_oid[l]->insert(e_oid[l]->end(),oid_list[l][i*m+j].begin(),oid_list[l][i*m+j].end());
                    e_ovar[l]->insert(e_ovar[l]->end(),ovar_list[l][i*m+j].begin(),ovar_list[l][i*m+j].end());
                    e_oc[l]->insert(e_oc[l]->end(),oc_list[l][i*m+j].begin(),oc_list[l][i*m+j].end());
                    e_otheta[l]->insert(e_otheta[l]->end(),otheta_list[l][i*m+j].begin(),otheta_list[l][i*m+j].end());
                }
            }
        for(size_t i=0;i<nd;i++)
            for(int l=0;l<=nummodels;l++) {
                mh = mh_list[l];
                for(size_t j=0;j<mh;j++){
                    e_sts[l]->at(i*mh+j)=static_cast<int>(sid_list[l][i*mh+j].size());
                    e_sid[l]->insert(e_sid[l]->end(),sid_list[l][i*mh+j].begin(),sid_list[l][i*mh+j].end());
                    e_svar[l]->insert(e_svar[l]->end(),svar_list[l][i*mh+j].begin(),svar_list[l][i*mh+j].end());
                    e_sc[l]->insert(e_sc[l]->end(),sc_list[l][i*mh+j].begin(),sc_list[l][i*mh+j].end());
                    e_stheta[l]->insert(e_stheta[l]->end(),stheta_list[l][i*mh+j].begin(),stheta_list[l][i*mh+j].end());
                }
            }

        //write out to file
        std::ofstream omf;
        std::string ofile;
        //std::ofstream omf_mix(folder + modelname + ".fitmix");
        //std::ofstream omf_emu(folder + modelname + ".fitemu");
        
        for(int j=0;j<=nummodels;j++){
            // Open the mixing for emulation file
            if(j == 0){
                ofile = folder + modelname + ".fitmix";
                omf.open(ofile);
                cout << "Saving mixing trees..." << endl;
            }else if(j == 1){
                ofile = folder + modelname + ".fitemulate";
                omf.open(ofile); //opened at first emulator -- kept open until very end
                cout << "Saving emulation trees..." << endl;
            }

            omf << nd << endl;
            omf << m_list[j] << endl;
            omf << mh_list[j] << endl;
            omf << e_ots[j]->size() << endl;
            for(size_t i=0;i<e_ots[j]->size();i++) omf << e_ots[j]->at(i) << endl;
            omf << e_oid[j]->size() << endl;
            for(size_t i=0;i<e_oid[j]->size();i++) omf << e_oid[j]->at(i) << endl;
            omf << e_ovar[j]->size() << endl;
            for(size_t i=0;i<e_ovar[j]->size();i++) omf << e_ovar[j]->at(i) << endl;
            omf << e_oc[j]->size() << endl;
            for(size_t i=0;i<e_oc[j]->size();i++) omf << e_oc[j]->at(i) << endl;
            omf << e_otheta[j]->size() << endl;
            for(size_t i=0;i<e_otheta[j]->size();i++) omf << std::scientific << e_otheta[j]->at(i) << endl;
            omf << e_sts[j]->size() << endl;
            for(size_t i=0;i<e_sts[j]->size();i++) omf << e_sts[j]->at(i) << endl;
            omf << e_sid[j]->size() << endl;
            for(size_t i=0;i<e_sid[j]->size();i++) omf << e_sid[j]->at(i) << endl;
            omf << e_svar[j]->size() << endl;
            for(size_t i=0;i<e_svar[j]->size();i++) omf << e_svar[j]->at(i) << endl;
            omf << e_sc[j]->size() << endl;
            for(size_t i=0;i<e_sc[j]->size();i++) omf << e_sc[j]->at(i) << endl;
            omf << e_stheta[j]->size() << endl;
            for(size_t i=0;i<e_stheta[j]->size();i++) omf << std::scientific << e_stheta[j]->at(i) << endl;
            
            // Close the mixing file before saving the emulation trees
            if(j == 0){
                omf.close();
            }
        }
        // Close the emulation text file
        omf.close();
        cout << " done." << endl;
    }
    //-------------------------------------------------- 
    // Cleanup.
#ifdef _OPENMPI
    //delete[] lwr_vec; //make pointer friendly
    //delete[] upr_vec; //make pointer friendly
    MPI_Finalize();
#endif
    return 0;
}