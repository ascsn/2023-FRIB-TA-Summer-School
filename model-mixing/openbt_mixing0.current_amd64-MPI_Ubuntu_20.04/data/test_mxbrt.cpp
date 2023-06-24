#include <iostream>
#include <fstream>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mxbrt.h"

//Include Eigen library
#include "Eigen/Dense"

int main(){
    //-------------------------------------------------------
    //---Choose to run example 1,2,3,or 4 
    //-------------------------------------------------------


    //-------------------------------------------------------
    //---Read in Data for mxbrt examples
    //-------------------------------------------------------
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

    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    size_t nc=100; //100
    makexinfo(p,n,&x[0],xi,nc); //use the 1st column with x[0]

    prxi(xi);
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
    //Make test set information
    //Read in test data
    //read in x
    std::vector<double> x_test;
    int n_test;
    std::ifstream xf2("xtest.txt");
    while(xf2 >> xtemp){
        x_test.push_back(xtemp);
    }
    n_test = x_test.size()/p;
    if(x_test.size() != n_test*p) {
        cout << "error: input x file has wrong number of values\n";
        return 1;
    }
    cout << "n_test,p: " << n_test << ", " << p << endl;

    //read in f
    std::vector<double> f_test;
    std::ifstream ff2("ftest.txt");
    while(ff2 >> ftemp){
        f_test.push_back(ftemp);
    }
    
    if(f_test.size() != n_test*k) {
        cout << "error: input f file has wrong number of values\n";
        return 1;
    }
    cout << "n_test,k: " << n_test << ", " << k << endl;
    
    //Make dinfo and diterator
    dinfo di_test;
    std::vector<double> y_test(n_test); //empty y
    for(int j=0;j<n_test;j++){y_test.push_back(0);}
    di_test.n=n_test;di_test.p=p,di_test.x = &x_test[0];di_test.tc=tc;di_test.y = &y_test[0];    
    
    diterator diter_test(&di_test);

    //Make finfo
    finfo fi_test;
    makefinfo(k,n_test, &f_test[0], fi_test);

    //-------------------------------------------------------
    //Example 1 -- Test mxsinfo
    //-------------------------------------------------------
    //Initialize matrix and vector
    Eigen::MatrixXd F;
    Eigen::VectorXd v;
    double yy = 10.2;

    F = Eigen::MatrixXd::Random(k,k);
    v = Eigen::VectorXd::Random(k);

    //Work with sinfo object
    sinfo s;
    std::cout << s.n << std::endl; //Prints out 0 as expected
    s.n = 10; //Change the sample size to 10

    //Try to create mxinfo objects
    mxsinfo mx1; //constructor 1
    mxsinfo mx2(s, k, F, v, yy); //Constructor 2
    mxsinfo mx3(mx2); //Constructor 3

    //See what they print
    mx1.print_mx();
    mx2.print_mx();
    mx3.print_mx();

    //Work with operators
    std::cout << "---Compound Addition Operator" << std::endl;
    mx3 += mx2;
    mx3.print_mx();

    std::cout << "---Addition and Equality Operator" << std::endl;    
    mxsinfo mx4 = mx2 + mx3; //Add two mxsinfo objects 
    mx4.print_mx();
    
    //-------------------------------------------------------
    //Example 2 -- Create an mxbrt object -- Use a fixed variance
    //-------------------------------------------------------
    /*
    cout << "\n\n-----------------------------------------" << endl;
    cout << "Example 2: Work with a mxbrt object \n" << endl;
    
    //Initialize prior parameters
    double *sig = new double[di.n];
    double tau = 0.5; //.... (1/B)*0.5/k .... B = sqrt(m*f1^2 + m*f2^2) .... m = 1 here
    double beta0 = 0.55; //..... median(y)/((median(f1) + median(f2))*m) .... m = 1 here
    std::vector<double> fitted(n), predicted(n_test);
    for(size_t i=0;i<di.n;i++) sig[i]=0.03; //True error std = 0.03
    dinfo di_predict;
    di_predict.n=n_test;di_predict.p=p,di_predict.x = &x_test[0];di_predict.tc=tc;di_predict.y = &predicted[0];

    //First mix bart object with basic constructor
    mxbrt mxb; 
    cout << "****Initial Object" << endl; 
    mxb.pr_vec();
    mxb.setxi(&xi);    //set the cutpoints for this model object
    //function output 
    mxb.setfi(&fi,k); //set function output for this model object
    //data objects
    mxb.setdata_mix(&di);  //set the data for model mixing
    //thread count
    mxb.settc(tc);      //set the number of threads when using OpenMP, etc.
    //tree prior
    mxb.settp(0.95, //the alpha parameter in the tree depth penalty prior
            0.75     //the beta parameter in the tree depth penalty prior
            );
    //MCMC info
    mxb.setmi(
            0.5,  //probability of birth/death
            0.5,  //probability of birth
            3,    //minimum number of observations in a bottom node
            true, //do perturb/change variable proposal?
            0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv  //initialize the change of variable correlation matrix.
            );
    mxb.setci(tau,beta0,sig);

    cout << "\n*****After init:\n";
    mxb.pr_vec();

    cout << "-----------------------------------" << endl;
    cout << "Test Individual Functions involved in draw: \n\n" << endl; 
    cout << "mxbrt lm = " << mxb.lm(mx3) << endl;
    cout << "mxbrt drawnodethetavec: " << mxb.drawnodethetavec(mx3, gen) << endl;
    cout << "mxbrt birth/death: \n" << endl;
    mxb.bd_vec(gen);
    mxb.pr_vec();

    cout << "-----------------------------------" << endl;
    cout << "\n-----------------------------------" << endl;
    size_t nd = 20000;
    size_t nadapt=5000;
    size_t adaptevery=500;
    size_t nburn=1000;

    for(size_t i=0;i<nadapt;i++) { mxb.drawvec(gen); if((i+1)%adaptevery==0) mxb.adapt(); }
    for(size_t i=0;i<nburn;i++) mxb.drawvec(gen); 
    
    cout << "\n*****After "<< nd << " draws:\n";
    cout << "Collecting statistics" << endl;
    mxb.setstats(true);
    for(int i = 0; i<nd; i++){
        //Sample tree and theta
        mxb.drawvec(gen);
        if((i % 2500) ==0){
            cout << "***Draw " << i << "\n" << endl;
            //mxb.pr_vec();
        } 
        
        //Updated fitted values
        for(size_t j=0;j<n;j++) fitted[j]+=mxb.f(j)/nd;
        
        //Predictions
        mxb.predict_mix(&di_test, &fi_test);
        di_predict += di_test;
    }    

    //Take the prediction average
    di_predict/=((double)nd);

    // summary statistics
    unsigned int varcount[p];
    unsigned int totvarcount=0;
    for(size_t i=0;i<p;i++) varcount[i]=0;
    unsigned int tmaxd=0;
    unsigned int tmind=0;
    double tavgd=0.0;

    mxb.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
    for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
    tavgd/=(double)(nd);

    cout << "Average tree depth: " << tavgd << endl;
    cout << "Maximum tree depth: " << tmaxd << endl;
    cout << "Minimum tree depth: " << tmind << endl;
    cout << "Variable perctg:    ";
    for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
    cout << endl;
    cout << "                    ";
    for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
    cout << endl;

    cout << "Print Fitted Values" << endl;
    for(int i = 0; i<n; i++){
        cout << "X = " << x[i] << " -- Y = " << y[i] <<" -- Fitted = " << fitted[i] << " -- Error = " << fitted[i] - y[i] << endl;
    }

    //Write Fitted values to a file
    std::ofstream outdata;
    outdata.open("fit_mxb2.txt"); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    for(int i = 0; i<n; i++){
        outdata << fitted[i] <<  endl;
    }
    outdata.close();

    //Write all data values to a file
    std::ofstream outpred;
    outpred.open("predict_mxb2.txt"); // opens the file
    if( !outpred ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    for(int i = 0; i<n_test; i++){
        outpred << predicted[i] << endl;
    }
    outpred.close();
    */

    //-------------------------------------------------------
    //Example 3 -- Create an mxbrt object -- Constant but unknown error variance
    //-------------------------------------------------------
    /*
    cout << "\n\n-----------------------------------------" << endl;
    cout << "Example 3: Work with a mxbrt object with unknown and constant variance \n" << endl;
    
    //Initialize prior parameters
    double *sig = new double[di.n];
    double tau = 0.5; //.... (1/B)*0.5/k .... B = sqrt(m*f1^2 + m*f2^2) .... m = 1 here
    double beta0 = 0.55; //..... median(y)/((median(f1) + median(f2))*m) .... m = 1 here
    double nu = 5.0;
    double lambda = 0.01; //0.01
    std::vector<double> fitted(n), predicted(n_test);
    for(size_t i=0;i<di.n;i++) sig[i]=0.03; //True error std = 0.03
    dinfo di_predict;
    di_predict.n=n_test;di_predict.p=p,di_predict.x = &x_test[0];di_predict.tc=tc;di_predict.y = &predicted[0];

    //First mix bart object with basic constructor
    mxbrt mxb; 
    cout << "****Initial Object" << endl; 
    mxb.pr_vec();
    mxb.setxi(&xi);    //set the cutpoints for this model object
    mxb.setfi(&fi,k); //set function output for this model object
    mxb.setdata_mix(&di);  //set the data for model mixing
    mxb.settc(tc);      //set the number of threads when using OpenMP, etc.
    mxb.settp(0.95,0.75); //the alpha and beta parameters in the tree depth penalty prior
    mxb.setmi(
            0.5,  //probability of birth/death
            0.5,  //probability of birth
            1,    //minimum number of observations in a bottom node
            true, //do perturb/change variable proposal?
            0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv  //initialize the change of variable correlation matrix.
            );
    mxb.setci(tau,beta0,sig); //Set conditioning information for prior mean
    mxb.setvi(nu, lambda); //Set conditioning information for prior varaince

    cout << "\n*****After init:\n";
    mxb.pr_vec();

    cout << "-----------------------------------" << endl;
    cout << "\n-----------------------------------" << endl;
    size_t nd = 20000;
    size_t nadapt=5000;
    size_t adaptevery=500;
    size_t nburn=1000;

    for(size_t i=0;i<nadapt;i++) { mxb.drawvec(gen); mxb.drawsigma(gen); if((i+1)%adaptevery==0) mxb.adapt(); }
    for(size_t i=0;i<nburn;i++) {mxb.drawvec(gen); mxb.drawsigma(gen); }
    
    //Initialize the sigma posterior txt file
    std::ofstream outsig;
    outsig.open("postsig_mxb2.txt"); // opens the file
    outsig.close(); // closes the file


    cout << "\n*****After "<< nd << " draws:\n";
    cout << "Collecting statistics" << endl;
    mxb.setstats(true);
    for(int i = 0; i<nd; i++){
        //Draw theta and a tree
        mxb.drawvec(gen);

        //Draw Sigma and save the last 25% of draws
        mxb.drawsigma(gen);

        if(i > nd*0.75){
            outsig.open("postsig_mxb2.txt", std::ios_base::app); // opens the file
            outsig << mxb.getsigma() << endl;
            outsig.close(); // closes the file
        }
        

        if((i % 2500) ==0){cout << "***Draw " << i << "\n" << endl;} 
        
        //Get Fitted Values 
        for(size_t j=0;j<n;j++) fitted[j]+=mxb.f(j)/nd;

        //get Predictions
        mxb.predict_mix(&di_test, &fi_test);
        di_predict += di_test;
    }    
    
    mxb.pr_vec();

    //Take the prediction average
    di_predict/=((double)nd);
    
    // summary statistics
    unsigned int varcount[p];
    unsigned int totvarcount=0;
    for(size_t i=0;i<p;i++) varcount[i]=0;
    unsigned int tmaxd=0;
    unsigned int tmind=0;
    double tavgd=0.0;

    mxb.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
    for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
    tavgd/=(double)(nd);

    cout << "Average tree depth: " << tavgd << endl;
    cout << "Maximum tree depth: " << tmaxd << endl;
    cout << "Minimum tree depth: " << tmind << endl;
    cout << "Variable perctg:    ";
    for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
    cout << endl;
    cout << "                    ";
    for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
    cout << endl;


    cout << "Print Fitted Values" << endl;
    for(int i = 0; i<n; i++){
        cout << "X = " << x[i] << " -- Y = " << y[i] <<" -- Fitted = " << fitted[i] << " -- Error = " << fitted[i] - y[i] << endl;
    }

    //Write Fitted values to a file
    std::ofstream outdata;
    outdata.open("fit_mxb2_sig.txt"); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    for(int i = 0; i<n; i++){
        outdata << fitted[i] <<  endl;
    }
    outdata.close();

    //Write all data values to a file
    std::ofstream outpred;
    outpred.open("predict_mxb2_sig.txt"); // opens the file
    if( !outpred ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    for(int i = 0; i<n_test; i++){
        outpred << predicted[i] << endl;
    }
    outpred.close();
    */
    //-------------------------------------------------------
    //Example 4 -- Save and load the tree
    //-------------------------------------------------------
    //Initialize prior parameters
    double *sig = new double[di.n];
    double tau = 0.5; 
    double beta0 = 0.55; 
    double nu = 5.0;
    double lambda = 0.01; //0.01
    for(size_t i=0;i<di.n;i++) sig[i]=0.03; //True error std = 0.03
    
    //Setup the objects
    mxbrt b1, b2;
    b1.setxi(&xi); b1.setfi(&fi,k); b1.setdata_mix(&di); b1.settc(tc); b1.settp(0.95,1.0); b1.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);b1.setci(tau,beta0,sig);b1.setvi(nu, lambda);
    b2.setxi(&xi); b2.setfi(&fi,k); b2.setdata_mix(&di); b2.settc(tc); b2.settp(0.95,1.0); b2.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);b2.setci(tau,beta0,sig);b2.setvi(nu, lambda);

    //Populate the brt objects -- look through to get larger objects
    for(int i=0; i<20;i++){b1.drawvec(gen);}
    for(int i=0; i<25;i++){b2.drawvec(gen);}
    
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
    mxbrt b11, b22;
    b11.setxi(&xi); b11.setfi(&fi,k); b11.setdata_mix(&di); b11.settc(tc); b11.settp(0.95,1.0); b11.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);b11.setci(tau,beta0,sig);b11.setvi(nu, lambda);
    b22.setxi(&xi); b22.setfi(&fi,k); b22.setdata_mix(&di); b22.settc(tc); b22.settp(0.95,1.0); b22.setmi(0.7,0.5,5,true,0.2,0.2,&chgv);b22.setci(tau,beta0,sig);b22.setvi(nu, lambda);
    b11.loadtree_vec(0,1,nn_vec,id_vec,v_vec,c_vec,theta_vec);
    b22.loadtree_vec(1,1,nn_vec,id_vec,v_vec,c_vec,theta_vec);
    
    //Print the bart objects from the loading process
    cout << "\n*******Loading Trees*******" << endl;
    cout << "\n~~~Print brt 1~~~" << endl;
    b11.t.pr_vec();
    cout << "\n~~~Print brt 2~~~" << endl;
    b22.t.pr_vec();
    
    return 0;

}
