//     test_brtvp.cpp: Base BT model class with vector parameters test/validation code.

#include <iostream>
#include <fstream>

#include "Eigen/Dense"

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"

#ifdef _OPENMPI
#   include <mpi.h>
#endif

using std::cout;
using std::endl;

int main(){
    cout << "\n*****into test for brt\n";
    cout << "\n\n";

    crn gen;
    gen.set_seed(199);

    int tc=4; //thread count for OpenMP

    //--------------------------------------------------
    //read in y
    std::vector<double> y;
    double ytemp;

    std::ifstream yf("y.txt");
    while(yf >> ytemp)
        y.push_back(ytemp);
    size_t n = y.size();
    cout << "n from y.txt: " << n <<endl;

    //--------------------------------------------------
    //read in x
    std::vector<double> x;
    double xtemp;
    size_t p;
    p=1;

    std::ifstream xf("x.txt");
    while(xf >> xtemp){
        x.push_back(xtemp);
    }
    
    if(x.size() != n*p) {
        cout << "error: input x file has wrong number of values\n";
        return 1;
    }
    cout << "n,p: " << n << ", " << p << endl;

    //--------------------------------------------------
    //read in f
    std::vector<double> f;
    double ftemp;
    size_t k; //number of columns in f
    k=2;

    std::ifstream ff("f.txt");
    while(ff >> ftemp){
        f.push_back(ftemp);
    }
    
    if(f.size() != n*k) {
        cout << "error: input f file has wrong number of values\n";
        return 1;
    }
    cout << "n,k: " << n << ", " << k << endl;
    
    //--------------------------------------------------
    //Make dinfo and diterator
    dinfo di;
    di.n=n;di.p=p,di.x = &x[0];di.tc=tc;
    di.y = &y[0];

    diterator diter(&di);
    /*
    cout << "Output of first diter: \n" << "X = " << diter.getx() << " i = " << diter.geti() <<  " *diter = "  << *diter << endl;

    for(;diter<diter.until();diter++){
        cout << diter.getx() << "-------" << *diter << endl;
    }
    */

    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    size_t nc=100;
    makexinfo(p,n,&x[0],xi,nc); //use the 1st column with x[0]

    //prxi(xi);
    //--------------------------------------------------
    //make finfo -- need to read in and store f formally, just using same x from above for now
    finfo fi;
    makefinfo(k, n, &f[0], fi);
    cout << fi << endl;

    //--------------------------------------------------
    // read in the initial change of variable rank correlation matrix
    std::vector<std::vector<double> > chgv;
    std::vector<double> cvvtemp;
    double cvtemp;
    std::ifstream chgvf("chgv.txt");
    for(size_t i=0;i<di.p;i++) {
        cvvtemp.clear();
        for(size_t j=0;j<di.p;j++) {
            chgvf >> cvtemp;
            cvvtemp.push_back(cvtemp);
        }
        chgv.push_back(cvvtemp);
    }
    cout << "change of variable rank correlation matrix loaded:" << endl;
    for(size_t i=0;i<di.p;i++) {
        for(size_t j=0;j<di.p;j++)
            cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
        cout << endl;
    }

    //--------------------------------------------------
    //brt Example 1:
    cout << "\n******************************************" << endl;
    cout << "\n Make a brt object and print it out\n";
    brt bm;
    cout << "\nbefore init:\n";
    bm.pr_vec();
    //cutpoints
    bm.setxi(&xi);    //set the cutpoints for this model object
    //set fi
    bm.setfi(&fi,k); //set the function values 
    //data objects
    bm.setdata_mix(&di);  //set the data...not sure if we need to used setdata_mix() since this should only be used at the start of mcmc
    //thread count
    bm.settc(tc);      //set the number of threads when using OpenMP, etc.
    //tree prior
    bm.settp(0.95, //the alpha parameter in the tree depth penalty prior
            1.0     //the beta parameter in the tree depth penalty prior
            );

    //MCMC info
    bm.setmi(0.7,  //probability of birth/death
         0.5,  //probability of birth
         5,    //minimum number of observations in a bottom node 
         true, //do perturb/change variable proposal?
         0.2,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         0.2,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
    cout << "\nafter init:\n";
    bm.pr_vec();
    

    //--------------------------------------------------
    //brt Example 2:
    cout << "\n******************************************" << endl;
    cout << "\nTry some draws of brt and print it out\n";
    cout << "\n1 draw:\n";
    bm.drawvec(gen);
    bm.pr_vec();
    
    size_t nd=1000;
    cout << "\n" << nd << " draws:\n";
    for(size_t i=0;i<nd;i++){
        bm.drawvec(gen);
        //bm.pr_vec();
    } 
    bm.pr_vec();

    //--------------------------------------------------
    //Example 3: Test the setf_mix & setr_mix and compare to setf & setr
    cout << "\n******************************************" << endl;
    cout << "Before setf_mix ... " << bm.f(2) << endl;
    bm.setf_mix();
    cout << "After setf_mix ... " << bm.f(2) << endl;

    cout << "Before setr_mix ... " << bm.r(2) << endl;
    bm.setr_mix();
    cout << "After setr_mix ... " << bm.r(2) << endl;
    
    //--------------------------------------------------
    //Example 4: Work with sinfo
    cout << "\n******************************************" << endl;
    std::vector<sinfo> siv(2);
    std::cout << "testing vectors of sinfos\n";
    std::cout << siv[0].n << ", " << siv[1].n << std::endl;

    siv.clear();
    siv.resize(2);
    std::cout << siv[0].n << ", " << siv[1].n << std::endl;

    //--------------------------------------------------
    //Example 5: Run an MCMC
    cout << "\n******************************************" << endl;
    size_t tuneevery=250;
    size_t tune=5000;
    size_t burn=5000;
    size_t draws=5000;
    brt b;

    b.setxi(&xi);
    b.setfi(&fi, k);
    b.setdata_mix(&di);
    b.settc(tc);
    //Setmi -- pbd,pb,minperbot,dopert,pertalpha,pchgv,chgv
    b.setmi(0.8,0.5,5,true,0.1,0.2,&chgv);

    // tune the sampler   
    for(size_t i=0;i<tune;i++){
        b.drawvec(gen);
        if((i+1)%tuneevery==0){
            b.adapt();
        }
    }

    b.t.pr_vec();
    // run some burn-in, tuning is fixed now
    for(size_t i=0;i<burn;i++){
        b.drawvec(gen);
    }

    // draw from the posterior
    // After burn-in, turn on statistics if we want them:
    cout << "Collecting statistics" << endl;
    b.setstats(true);
    // then do the draws
    for(size_t i=0;i<draws;i++){
        b.drawvec(gen);
    }

    // summary statistics
    unsigned int varcount[p];
    unsigned int totvarcount=0;
    for(size_t i=0;i<p;i++) varcount[i]=0;
    unsigned int tmaxd=0;
    unsigned int tmind=0;
    double tavgd=0.0;

    b.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
    for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
    tavgd/=(double)(draws);

    cout << "Average tree depth: " << tavgd << endl;
    cout << "Maximum tree depth: " << tmaxd << endl;
    cout << "Minimum tree depth: " << tmind << endl;
    cout << "Variable perctg:    ";
    for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
    cout << endl;
    cout << "                    ";
    for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
    cout << endl;

    //--------------------------------------------------
    //Example 6: Save and load tree
    brt b1, b2; //new brt objects
    b1.setxi(&xi); b1.setfi(&fi,k); b1.setdata_mix(&di); b1.settc(tc); b1.settp(0.95,1.0); b1.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);
    b2.setxi(&xi); b2.setfi(&fi,k); b2.setdata_mix(&di); b2.settc(tc); b2.settp(0.95,1.0); b2.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);

    //Populate the brt objects -- look through to get larger objects
    for(int i=0; i<20;i++){
        b1.drawvec(gen);
        b2.drawvec(gen);
    }
    
    //Print the bart objects
    cout << "\n~~~Print brt 1~~~" << endl;
    b1.pr_vec();
    cout << "\n~~~Print brt 2~~~" << endl;
    b2.pr_vec();

    //Save tree's 1 and 2 using brt object;
    std::vector<int> nn_vec(2); //2 bart models -- length of vector = 2
    std::vector<std::vector<int>> id_vec(2), v_vec(2), c_vec(2); //2 bart models -- number of rows = 2 
    std::vector<std::vector<double>> theta_vec(k); //length of each vector is nn*k 

    b1.savetree_vec(0,1,nn_vec,id_vec,v_vec,c_vec,theta_vec); //0th iteration 
    b2.savetree_vec(1,1,nn_vec,id_vec,v_vec,c_vec,theta_vec); //1st iteration

    //Print the saved tree vectors
    cout << "\n****** Print nn vector" << endl;
    for(int i=0; i<2;i++){
        cout << "--------------------------------------" << endl;
        cout << "Brt " << i+1 << ": \n" << "nn = " << nn_vec[i] << endl;
        for(int j=0;j<id_vec[i].size();j++){
            cout << "id = " << id_vec[i][j] << " -- " << "(v,c) = " << "(" << v_vec[i][j] << "," << c_vec[i][j] << ") -- " << "thetavec = ";
            for(int l = 0; l<k; l++){
                cout << theta_vec[i][j*k+l] << ", ";
            }
            cout << endl;
        }    
    }

    //Load a tree -- use the containers from above
    brt b11, b22;
    b11.setxi(&xi); b11.setfi(&fi,k); b11.setdata_mix(&di); b11.settc(tc); b11.settp(0.95,1.0); b11.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);
    b22.setxi(&xi); b22.setfi(&fi,k); b22.setdata_mix(&di); b22.settc(tc); b22.settp(0.95,1.0); b22.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);
    b11.loadtree_vec(0,1,nn_vec,id_vec,v_vec,c_vec,theta_vec);
    b22.loadtree_vec(1,1,nn_vec,id_vec,v_vec,c_vec,theta_vec);
    
    //Print the bart objects from the loading process
    cout << "\n*******Loading Trees*******" << endl;
    cout << "\n~~~Print brt 1~~~" << endl;
    b11.t.pr_vec();
    cout << "\n~~~Print brt 2~~~" << endl;
    b22.t.pr_vec();


    /*
    tree t1,t2; //New Trees
    brt b1; //New brt object
    int kk = 3;
    vxd theta(k), thetar(k), thetal(k);
    thetar << 1.9, 2.4, 3.1;
    thetal << 4.3, 2.2, 3.1;
    theta << 0.9, 0.4, -1.2;

    //Set theta for tree 1
    t1.setthetavec(theta);

    //brith for tree 2
    t2.birth(1, 0, 2, thetal, thetar);
    */

    //--------------------------------------------------
    //--------------------------------------------------
    //Extra 
    //cout << x[44] << endl;

    
}