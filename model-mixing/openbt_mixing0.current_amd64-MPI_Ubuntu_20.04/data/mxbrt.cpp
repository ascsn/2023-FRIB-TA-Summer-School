#include <iostream>

//header files from OpenBT 
#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"
#include "mxbrt.h"

//Include the Eigen header files
#include "Eigen/Dense"
#include <Eigen/StdVector>

//--------------------------------------------------
//a single iteration of the MCMC for brt model
void mxbrt::drawvec(rn& gen){
    //cout << "enter drawvec" << endl; 
    //All the usual steps
    brt::drawvec(gen);

    // Update the in-sample predicted vector
    setf_mix();

    // Update the in-sample residual vector
    setr_mix();
}

//--------------------------------------------------
//slave controller for draw when using MPI
void mxbrt::drawvec_mpislave(rn& gen){
    //All the usual steps
    brt::drawvec_mpislave(gen);

    // Update the in-sample predicted vector
    setf_mix();

    // Update the in-sample residual vector
    setr_mix();
}

//--------------------------------------------------
//draw theta for a single bottom node for the brt model
vxd mxbrt::drawnodethetavec(sinfo& si, rn& gen){
    //initialize variables
    mxsinfo& mxsi=static_cast<mxsinfo&>(si);
    mxd I(k,k), Sig_inv(k,k), Sig(k,k), Ev(k,k), E(k,k), Sp(k,k), Prior_Sig_Inv(k,k);
    vxd muhat(k), evals(k), stdnorm(k), betavec(k);
    double t2 = ci.tau*ci.tau;
    
    I = Eigen::MatrixXd::Identity(k,k); //Set identity matrix
    betavec = ci.beta0*Eigen::VectorXd::Ones(k); //Set the prior mean vector

    //Set the prior precision matrix 
    if(ci.diffpriors){
        Prior_Sig_Inv = ci.invtau2_matrix;
        betavec = ci.beta_vec;    
    }else{
        //Assuming the same prior settings for tau2
        if(nsprior){
            // Compute beta vector and scale by 1/m (stored in c1.beta0) 
            betavec = mxsi.sump/mxsi.n;
            betavec = betavec*ci.beta0;

            // Compute the prior precision using the value of t2
            Prior_Sig_Inv = 1/(t2)*I;
            
        }else{
            // Compute the prior precision for stationary prior
            Prior_Sig_Inv = (1/t2)*I;
        }
    }

    //Compute the covariance
    Sig_inv = mxsi.sumffw + Prior_Sig_Inv; //Get inverse covariance matrix
    Sig = Sig_inv.llt().solve(I); //Invert Sig_inv with Cholesky Decomposition
    //Compute the mean vector
    muhat = Sig*(mxsi.sumfyw + Prior_Sig_Inv*betavec); //Get posterior mean -- may be able to simplify this calculation (k*ci.beta0/(ci.tau*ci.tau)) 
    //Spectral Decomposition of Covaraince Matrix Sig -- maybe move to a helper function
    //--Get eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Sig); //Get eigenvectors and eigenvalues
    if(eigensolver.info() != Eigen::ComputationInfo::Success) abort(); //Checks if any errors occurred
    evals = eigensolver.eigenvalues(); //Get vector of eigenvalues
    Ev = eigensolver.eigenvectors(); //Get matrix of eigenvectors

    //--Get sqrt of eigen values and store into a matrix    
    E = mxd::Zero(k,k); //Set as matrix of 0's
    E.diagonal() = evals.array().sqrt(); //Diagonal Matrix of sqrt of eigen values
    
    //--Compute Spectral Decomposition
    Sp = Ev*E*Ev.transpose();

    //Generate MVN Random vector
    //--Get vector of standard normal normal rv's
    for(size_t i=0; i<k;i++){
        stdnorm(i) = gen.normal(); 
    }
    
    //Print out matrix algebra step-by-step
    /*
    std::cout << "\nAll matrix Calculations:" << std::endl;
    std::cout << "Sig_inv = \n" << Sig_inv << std::endl;
    std::cout << "\n Sig = \n" << Sig << std::endl;
    std::cout << "\n muhat = \n" << muhat << std::endl;
    std::cout << "\n Ev = \n" << Ev << std::endl;
    std::cout << "\n evals = \n" << evals << std::endl;
    std::cout << "\n E = \n" << E << std::endl;
    std::cout << "\n Sp = \n" << Sp << std::endl;
    std::cout << "\n thetavec = " << std::endl;
    */
        
    //--Generate the MVN vector
    return muhat + Sp*stdnorm;
}

//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double mxbrt::lm(sinfo& si){
    mxsinfo& mxsi=static_cast<mxsinfo&>(si);
    mxd Sig_inv(k,k), Sig(k,k), I(k,k);
    vxd betavec(k), beta_t2(k); 
    double t2 = ci.tau*ci.tau;
    double suml; //sum of log determinent 
    double sumq; //sum of quadratic term
    double sumpl; //sum of prior precision log determinant
    
    I = mxd::Identity(k,k); //Set identity matrix

    //Compute lm value if no discrepancy is specified and each weight has the same prior
    if(!nsprior && !ci.diffpriors){
        //prior mean vector
        betavec = ci.beta0*vxd::Ones(k);

        //Get covariance matrix
        Sig = mxsi.sumffw + I/t2; //get Sig matrix
        Sig_inv = Sig.llt().solve(I); //Get Sigma_inv
        
        //Compute Log determinent
        mxd L(Sig.llt().matrixL()); //Cholesky decomp and store the L matrix
        suml = 2*(L.diagonal().array().log().sum()); //The log determinent is the same as 2*sum(log(Lii)) --- Lii = diag element of L

        //Now work on the exponential terms
        sumq = mxsi.sumyyw - (mxsi.sumfyw + betavec/t2).transpose()*Sig_inv*(mxsi.sumfyw + betavec/t2) + k*ci.beta0*ci.beta0/t2;

        //Get sum of prior precision log determinant
        sumpl = k*log(t2);

    }else if(nsprior && !ci.diffpriors){
        //Get prior inverse covariance and mean vector
        mxd Prior_Sig_Inv(k,k);
        
        //Get beta vector and scale by 1/m 
        betavec = mxsi.sump/mxsi.n;
        betavec = betavec*ci.beta0;

        //Get the prior precision
        Prior_Sig_Inv = (1/t2)*I;
        
        //Get posterior covariance matrix (Sig_inv)
        Sig = mxsi.sumffw + Prior_Sig_Inv; //get Sig matrix
        Sig_inv = Sig.llt().solve(I); //Get Sigma_inv
        
        //Compute Log determinent
        mxd L(Sig.llt().matrixL()); //Cholesky decomp and store the L matrix
        suml = 2*(L.diagonal().array().log().sum()); //The log determinent is the same as 2*sum(log(Lii)) --- Lii = diag element of L

        //Now work on the exponential terms
        beta_t2 = Prior_Sig_Inv*betavec;
        sumq = mxsi.sumyyw - (mxsi.sumfyw + beta_t2).transpose()*Sig_inv*(mxsi.sumfyw + beta_t2) + betavec.transpose()*beta_t2;

        //Get sum of prior precision log determinant
        sumpl = -Prior_Sig_Inv.diagonal().array().log().sum(); //The negative appears here because we factor out -0.5 in the final sum       

    }else if(ci.diffpriors){
        //No discrpeancy and different priors for weights
        //prior mean vector
        betavec = ci.beta_vec;

        //Get covariance matrix
        Sig = mxsi.sumffw + ci.invtau2_matrix; //get Sig matrix
        Sig_inv = Sig.llt().solve(I); //Get Sigma_inv
        
        //Compute Log determinent
        mxd L(Sig.llt().matrixL()); //Cholesky decomp and store the L matrix
        suml = 2*(L.diagonal().array().log().sum()); //The log determinent is the same as 2*sum(log(Lii)) --- Lii = diag element of L

        //Now work on the exponential terms
        beta_t2 = ci.invtau2_matrix*betavec; //saves on doing this calculation 3 separate times 
        sumq = mxsi.sumyyw - (mxsi.sumfyw + beta_t2).transpose()*Sig_inv*(mxsi.sumfyw + beta_t2) + betavec.transpose()*beta_t2;

        //Get sum of prior precision log determinant
        sumpl = -ci.invtau2_matrix.diagonal().array().log().sum();    
    }
    return -0.5*(suml + sumq + sumpl);

}

//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void mxbrt::add_observation_to_suff(diterator& diter, sinfo& si){
    //Declare variables 
    mxsinfo& mxsi=static_cast<mxsinfo&>(si);
    //mxsi.resize(k);
    double w, yy;
    mxd ff(k,k);
    vxd fy(k), pv(k), pv2(k);
    
    //Assign values
    w=1.0/(ci.sigma[*diter]*ci.sigma[*diter]);
    ff = (*fi).row(*diter).transpose()*(*fi).row(*diter);
    fy = (*fi).row(*diter).transpose()*diter.gety();
    yy = diter.gety()*diter.gety();

    //Work with scaled precision vector for discrepancy model if true
    if(nsprior){
        pv = vxd::Zero(k);
        for(size_t l = 0; l<k; l++){
            //pv(l) = (*fdelta_sd)(*diter,l)*(*fdelta_sd)(*diter,l);
            pv(l) = 1/((*fisd)(*diter,l)*(*fisd)(*diter,l));
        }
        //Constrain pv to the simplex and get pv2 using symmetry about 0.5
        pv = pv/pv.sum();
        
        //Update sufficient stats
        mxsi.sump+=pv;
    }else{
        //Set as 1's by default when no discrepancy is used
        mxsi.sump = vxd::Ones(k);
}
    
    //Update sufficient stats for nodes
    mxsi.n+=1;
    mxsi.sumffw+=w*ff;
    mxsi.sumfyw+=w*fy;
    mxsi.sumyyw+=w*yy;
    //mxsi.print_mx();

}

//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
void mxbrt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir){
#ifdef _OPENMPI
   mxsinfo& msil=static_cast<mxsinfo&>(sil);
   mxsinfo& msir=static_cast<mxsinfo&>(sir);
   int buffer_size = 10*SIZE_UINT6*k*k;
   if(rank==0) { // MPI receive all the answers from the slaves
        MPI_Status status;
        mxsinfo& tsil = (mxsinfo&) *newsinfo();
        mxsinfo& tsir = (mxsinfo&) *newsinfo();
        char buffer[buffer_size];
        int position=0;
        unsigned int ln,rn;
        
        //Cast the matrices and vectors to arrays
        double sumffw_rarray[k*k], sumffw_larray[k*k]; //arrays for right and left matrix suff stats
        double sumfyw_rarray[k], sumfyw_larray[k]; //arrays for right and left vector suff stats
        matrix_to_array(tsil.sumffw, &sumffw_larray[0]); //function defined in brtfuns
        matrix_to_array(tsir.sumffw, &sumffw_rarray[0]); //function defined in brtfuns
        vector_to_array(tsil.sumfyw, &sumfyw_larray[0]); //function defined in brtfuns
        vector_to_array(tsir.sumfyw, &sumfyw_rarray[0]); //function defined in brtfuns
        
        //Cast the prior precision vectors to arrays - if discrepancy is true
        double sump_rarray[k], sump_larray[k];
        if(nsprior){
            vector_to_array(tsil.sump, &sump_larray[0]); //function defined in brtfuns
            vector_to_array(tsir.sump, &sump_rarray[0]); //function defined in brtfuns
        }
         
        for(size_t i=1; i<=(size_t)tc; i++) {
            position=0;
            MPI_Recv(buffer,buffer_size,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
            MPI_Unpack(buffer,buffer_size,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&sumffw_larray,k*k,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&sumffw_rarray,k*k,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&sumfyw_larray,k,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&sumfyw_rarray,k,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsil.sumyyw,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsir.sumyyw,1,MPI_DOUBLE,MPI_COMM_WORLD);
            if(nsprior){
                //Unpack prior precision if specified
                MPI_Unpack(buffer,buffer_size,&position,&sump_larray,k,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&sump_rarray,k,MPI_DOUBLE,MPI_COMM_WORLD);

                //convert back to a vector
                array_to_vector(tsil.sump,&sump_larray[0]);    
                array_to_vector(tsir.sump,&sump_rarray[0]);
            }


            //convert sumffw_larray/sumffw_rarray to a matrix defined in tsil/tsir
            array_to_matrix(tsil.sumffw,&sumffw_larray[0]);
            array_to_matrix(tsir.sumffw,&sumffw_rarray[0]);

            //convert sumfyw_larray/sumfyw_rarray to a vector defined in tsil/tsir
            array_to_vector(tsil.sumfyw,&sumfyw_larray[0]);    
            array_to_vector(tsir.sumfyw,&sumfyw_rarray[0]);
            
            tsil.n=(size_t)ln;
            tsir.n=(size_t)rn;
            msil+=tsil;
            msir+=tsir;

        }
      delete &tsil;
      delete &tsir;
   }
   else // MPI send all the answers to root
   {
    char buffer[buffer_size];
    int position=0;  
    unsigned int ln,rn;

    //Cast the matrices and vectors to arrays
    double sumffw_rarray[k*k], sumffw_larray[k*k]; //arrays for right and left matrix suff stats
    double sumfyw_rarray[k], sumfyw_larray[k]; //arrays for right and left vector suff stats
    matrix_to_array(msil.sumffw, &sumffw_larray[0]); //function defined in brtfuns
    matrix_to_array(msir.sumffw, &sumffw_rarray[0]); //function defined in brtfuns
    vector_to_array(msil.sumfyw, &sumfyw_larray[0]); //function defined in brtfuns
    vector_to_array(msir.sumfyw, &sumfyw_rarray[0]); //function defined in brtfuns
    
    //Cast vectors for prior precision if specified
    double sump_rarray[k], sump_larray[k];
    if(nsprior){
        vector_to_array(msil.sump, &sump_larray[0]); //function defined in brtfuns
        vector_to_array(msir.sump, &sump_rarray[0]); //function defined in brtfuns
    }

    ln=(unsigned int)msil.n;
    rn=(unsigned int)msir.n;
    MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
    MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
    MPI_Pack(&sumffw_larray,k*k,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
    MPI_Pack(&sumffw_rarray,k*k,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
    MPI_Pack(&sumfyw_larray,k,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
    MPI_Pack(&sumfyw_rarray,k,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
    MPI_Pack(&msil.sumyyw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
    MPI_Pack(&msir.sumyyw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
    if(nsprior){
        MPI_Pack(&sump_larray,k,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&sump_rarray,k,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
    }

    MPI_Send(buffer,buffer_size,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }
#endif   
}

//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void mxbrt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv){
#ifdef _OPENMPI
    //Construct containers for suff stats -- each ith element in the container is a suff stat for the ith node
    unsigned int nvec[siv.size()];
    double sumyywvec[siv.size()];
    double sumffwvec[k*k*siv.size()]; //An array that contains siv.size() matricies of kXk dimension that are flattened (by row).
    double sumfywvec[k*siv.size()]; //An array that contains siv.size() vectors of k dimension.
    double sumpvec[k*siv.size()]; //An array that contains siv.size() vectors of k dimension that are flattened (by row)

    std::vector<mxd, Eigen::aligned_allocator<mxd>> sumffwvec_em(siv.size()); //Vector of Eigen Matrixes kXk dim
    std::vector<vxd, Eigen::aligned_allocator<vxd>> sumfywvec_ev(siv.size()); //Vector of Eigen Vectors k dim
    std::vector<vxd, Eigen::aligned_allocator<vxd>> sumpvec_em(siv.size()); //Vector of Eigen Vectors k dim

    for(size_t i=0;i<siv.size();i++) { // on root node, these should be 0 because of newsinfo().
        mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
        nvec[i]=(unsigned int)mxsi->n;    // cast to int
        sumyywvec[i]=mxsi->sumyyw;
        sumfywvec_ev[i]=mxsi->sumfyw;
        sumffwvec_em[i]=mxsi->sumffw;
        sumpvec_em[i]=mxsi->sump;

        //Cast the current matrix to an array -- keep in mind to location given the stat for each node is stacked in the same array
        matrix_to_array(sumffwvec_em[i], &sumffwvec[i*k*k]);

        //cast the current vector to an array
        vector_to_array(sumfywvec_ev[i], &sumfywvec[i*k]);
        
        //cast the current prior precision vector to an array
        vector_to_array(sumpvec_em[i], &sumpvec[i*k]);


    }

    // MPI sum
    // MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE,&sumwvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE,&sumwyvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    if(rank==0) {
        //std::cout << "Enter with rank 0 ---- " << std::endl;
        MPI_Status status;
        unsigned int tempnvec[siv.size()];
        double tempsumyywvec[siv.size()];
        double tempsumffwvec[k*k*siv.size()]; //An array that contains siv.size() matricies of kXk dimension that are flattened (by row).
        double tempsumfywvec[k*siv.size()]; //An array that contains siv.size() vectors of k dimension.
        double tempsumpvec[k*siv.size()]; //An array that contains siv.size() vectors of k dimension that are flattened.

        // receive nvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempnvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                nvec[j]+=tempnvec[j];
            }
        }
        MPI_Request *request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->n=(size_t)nvec[i];    // cast back to size_t
            //std::cout << "nvec[" << i << "] = " << nvec[i] << std::endl;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumyywvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumyywvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++)
                sumyywvec[j]+=tempsumyywvec[j];
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumyywvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->sumyyw=sumyywvec[i];
            //std::cout << "sumyyw[" << i << "] = " << sumyywvec[i] << std::endl;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumfywvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumfywvec,k*siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<k*siv.size();j++){
                sumfywvec[j]+=tempsumfywvec[j]; //add temp vector to the sufficient stat
            }
        }
        //for(size_t j=0; j<k*siv.size(); j++){cout << "sumfywvec = " << sumfywvec[j] << endl;} //delete later
        
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumfywvec,k*siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        
        // cast back to mxsi
        Eigen::VectorXd tempsumfyw(k); //temp vector
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumfyw = Eigen::VectorXd::Zero(k); //initialize as zero vector before populating each node's suff stat 
            
            //Turn the ith section of k elements in the array into an Eigen VextorXd
            for(size_t l=0; l<k; l++){
                tempsumfyw(l) = sumfywvec[i*k+l]; 
            }
            //std::cout << "tempsumfyw = \n" << tempsumfyw.transpose() << std::endl;
            mxsi->sumfyw=tempsumfyw;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumffwvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumffwvec,k*k*siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<(k*k*siv.size());j++){
                sumffwvec[j]+=tempsumffwvec[j];
            }
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumffwvec,k*k*siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to mxsi
        Eigen::MatrixXd tempsumffw(k,k); //temp matrix
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumffw = Eigen::MatrixXd::Zero(k,k); //initialize as zero matrix before populating each node's suff stat
            
            //cast the specific k*k elements from the array into the suff stat matrix
            array_to_matrix(tempsumffw, &sumffwvec[i*k*k]);
            //std::cout << "tempsumffw = \n" << tempsumffw << std::endl; 
            mxsi->sumffw=tempsumffw;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        //For discrepancy models
        if(nsprior){
            //receive sump, update, and send back
            for(size_t i=1; i<=(size_t)tc; i++) {
                MPI_Recv(&tempsumpvec,k*siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
                for(size_t j=0;j<k*siv.size();j++){
                    sumpvec[j]+=tempsumpvec[j];
                }
            }
            request=new MPI_Request[tc];
            for(size_t i=1; i<=(size_t)tc; i++) {
                MPI_Isend(&sumpvec,k*siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
            }
            // cast back to mxsi
            Eigen::VectorXd tempsump(k); //temp matrix
            for(size_t i=0;i<siv.size();i++) {
                mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
                tempsump = Eigen::VectorXd::Zero(k); //initialize as zero vector before populating each node's suff stat
                
                //cast the specific k elements from the array into the suff stat vector
                array_to_vector(tempsump, &sumpvec[i*k]);
                //std::cout << "tempsump = \n" << tempsump << std::endl; 
                mxsi->sump=tempsump;
            }
            MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
            delete[] request;

        }
    }   
    else {
        //std::cout << "Enter with rank " << rank << " ---- " << std::endl;
        MPI_Request *request=new MPI_Request;
        MPI_Status status;

        // send/recv nvec      
        MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&nvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumyywvec, update nvec, receive sumyywvec
        request=new MPI_Request;
        MPI_Isend(&sumyywvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->n=(size_t)nvec[i];    // cast back to size_t
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumyywvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumfywvec, update sumyywvec, receive sumfywvec
        request=new MPI_Request;
        MPI_Isend(&sumfywvec,k*siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->sumyyw=sumyywvec[i];
            //std::cout << "sumyy["<<i<<"] = "<<sumyywvec[i]<<std::endl;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumfywvec,k*siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumffwvec, update sumfywvec, receive sumffwvec
        request=new MPI_Request;
        MPI_Isend(&sumffwvec,k*k*siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to mxsi
        Eigen::VectorXd tempsumfyw(k); //temp vector
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumfyw = Eigen::VectorXd::Zero(k); //initialize as zero vector before populating each node's suff stat 
            
            //Turn the ith section of k elements in the array into an Eigen VextorXd
            for(size_t l=0; l<k; l++){
                tempsumfyw(l) = sumfywvec[i*k+l]; 
            }
            mxsi->sumfyw=tempsumfyw;
            //std::cout << "tempsumfy["<<i<<"] = "<<tempsumfyw.transpose()<<std::endl;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumffwvec,k*k*siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        if(nsprior){
            // send sumpvec 
            request=new MPI_Request;
            MPI_Isend(&sumpvec,k*siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        }

        // update sumffwvec
        // cast back to mxsi
        Eigen::MatrixXd tempsumffw(k,k); //temp matrix
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumffw = Eigen::MatrixXd::Zero(k,k); //initialize as zero matrix before populating each node's suff stat
            
            //cast the specific k*k elements from the array into the suff stat matrix
            array_to_matrix(tempsumffw, &sumffwvec[i*k*k]); 
            mxsi->sumffw=tempsumffw;
            //std::cout << "tempsumffw["<<i<<"] = \n"<<tempsumffw<<std::endl;
        }

        if(nsprior){
            //receive and update sumpvec if using discrepancy
            MPI_Wait(request,MPI_STATUSES_IGNORE);
            delete request;
            MPI_Recv(&sumpvec,k*siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

            //cast sumpvec back to mxsi
            Eigen::VectorXd tempsump(k); //temp vector
            for(size_t i=0;i<siv.size();i++) {
                mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
                tempsump = Eigen::VectorXd::Zero(k); //initialize as zero vector before populating each node's suff stat
                
                //cast the specific k elements from the array into the suff stat matrix
                array_to_vector(tempsump, &sumpvec[i*k]); 
                mxsi->sump=tempsump;
                //std::cout << "tempsump["<<i<<"] = \n"<<tempsump<<std::endl;
            }

        }
    }
    // cout << "reduced:" << siv[0]->n << " " << siv[1]->n << endl;
#endif
}

//--------------------------------------------------
//Print mxbrt object
void mxbrt::pr_vec()
{
   std::cout << "***** mxbrt object:\n";
   std::cout << "Conditioning info:" << std::endl;
   std::cout << "   mean:   tau =" << ci.tau << std::endl;
   std::cout << "   mean:   beta0 =" << ci.beta0 << std::endl;
   if(!ci.sigma)
     std::cout << "         sigma=[]" << std::endl;
   else
     std::cout << "         sigma=[" << ci.sigma[0] << ",...," << ci.sigma[di->n-1] << "]" << std::endl;
   brt::pr_vec();
}

//--------------------------------------------------
//Sample from the std deviation posterior -- using Gibbs sampler under a homscedastic variance assumption
void mxbrt::drawsigma(rn& gen){
    double sumr2 = 0.0;
    int n; 
    int ntot=0;
    n = resid.size();
    if(n == 0){n = 1;} //used for root node when using mpi
    getsumr2(sumr2,ntot);
    /*
    //Get sum of residuals squared
    for(int i = 0; i<n;i++){
        sumr2 = sumr2 + resid[i]*resid[i];
        //std::cout << resid[i] << " --- " << resid[i]*resid[i] << std::endl;
    }
    */
   
    //Get nu*lambda and nu
    double nulampost=ci.nu*ci.lambda+sumr2;
    double nupost = ntot + (int)ci.nu;

    //When MPI is active -- reset so we draw the same sigma for every processor
    #ifdef _OPENMPI
        mpi_resetrn(gen);
    #endif

    //std::cout << "Before --- *ci.sigma = " <<  *(ci.sigma) << std::endl;

    //Generate new inverse scaled chi2 rv
    gen.set_df(nupost); //set df's
    double newsig = sqrt((nulampost)/gen.chi_square()); 
    
    for(int i = 0;i<n;i++){
        ci.sigma[i] = newsig; 
    }
    
    /*
    std::vector<double> newsigvec(n,newsig);
    
    double* newsigptrvec; 
    double* newsigptr;
    newsigptrvec = &newsigvec[0];
    newsigptr = &newsig;  

    //Set the sigma value -- n = 0 on node 0 hence we need this conditional
    if(n>0){
        setci(ci.tau, ci.beta0, newsigptrvec);
    }else{
        setci(ci.tau, ci.beta0, newsigptr);
    }
    */    

    

    /*
    std::cout << "n=" << n << " -- sumr2=" << sumr2 << std::endl;
    std::cout << "Chi2 = " <<  gen.chi_square() << std::endl;
    std::cout << "nulambda post = " <<  nulampost << std::endl;
    std::cout << "Value = " << sqrt((nulampost)/gen.chi_square()) << std::endl;
    *(ci.sigma) = sqrt((nulampost)/gen.chi_square());
    std::cout << "After --- *ci.sigma = " <<  *(ci.sigma) << std::endl;
    */

}

//--------------------------------------------------
//getsumr2 -- calls local functions similar to allsuff
void mxbrt::getsumr2(double &sumr2, int &n){
    #ifdef _OPENMPI
        local_mpi_getsumr2(sumr2,n);
    #else
        local_getsumr2(sumr2,n);
    #endif
}

//--------------------------------------------------
//local_getsumr2 -- performs the calculation for getting the resid sum of squares
void mxbrt::local_getsumr2(double &sumr2, int &n){
    //get n
    n = resid.size();

    //Get sum of residuals squared
    for(int i = 0; i<n;i++){
        sumr2 = sumr2 + resid[i]*resid[i];
        //std::cout << resid[i] << " --- " << resid[i]*resid[i] << std::endl;
    }
} 

//--------------------------------------------------
//local_mpi_getsumr2 -- performs the calculation for getting the resid sum of squares
void mxbrt::local_mpi_getsumr2(double &sumr2, int &n){
#ifdef _OPENMPI
    if(rank==0) {
        //reduce all the sumr2  across the processes (nodes)
        local_mpi_reduce_getsumr2(sumr2, n);
    }
    else
    {
        //compute the sumr2 for this process 
        local_getsumr2(sumr2,n);
        //std::cout << "local_getsumr2 --- mpirank = " << rank << " --- sumr2,n = " << sumr2 << "," << n << std::endl; 

        //reduce all the sumr2  across the processes (nodes)
        local_mpi_reduce_getsumr2(sumr2, n);
    }
    //std::cout << "After MPI Comm--- mpirank = " << rank << " --- sumr2,n = " << sumr2 << "," << n << std::endl;
#endif
}

//--------------------------------------------------
//local_mpi_reduce_getsumr2 -- the MPI communication part of local_mpi_getsrumr2.
void mxbrt::local_mpi_reduce_getsumr2(double &sumr2, int &n)
{
#ifdef _OPENMPI
    double sr2 = sumr2; //this should be zero on root 
    int nval = n; //this is zero on the root
    if(rank==0) {
        MPI_Status status;
        double tempsr2;
        unsigned int tempnval;
        
        //Receive, Send, and update sr2
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsr2,1,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            sr2 += tempsr2;
        }

        MPI_Request *request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sr2,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }

        //set sumr2 to the value
        sumr2 = sr2;

        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        //Receive, Send and update nval
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempnval,1,MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            nval += tempnval;
        }

        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&nval,1,MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
        }

        //set nval to the value
        n = (int)nval;

        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;
    }
    else {

        //Send and receive sumr2
        MPI_Request *request=new MPI_Request;
        MPI_Status status;

        MPI_Isend(&sr2,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;

        MPI_Recv(&sr2,1,MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        
        //send nval
        request=new MPI_Request[tc];
        MPI_Isend(&nval,1,MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
        
        //update sumr2 to the value
        sumr2 = sr2;
        
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;

        //Receive nval
        MPI_Recv(&nval,1,MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        
        //set sumr2 to the value
        n = nval;
    }

    // cast back to size_t
    // for(size_t i=0;i<siv.size();i++)
    //    siv[i]->n=(size_t)nvec[i];
// cout << "reduced:" << siv[0]->n << " " << siv[1]->n << endl;
#endif
}

/*
//--------------------------------------------------
//Model Discrepancy
//--------------------------------------------------
//draw individual model discrepancy
void mxbrt::drawdelta(rn& gen){
    //Get the number of obs on this node
    size_t n = (*fdelta_mean).rows();

    //Reset fdelta to zero
    *fdelta = mxd::Zero(n,k);

    //Draw random variables and update delta
    for(size_t i = 0; i<n; i++){
        for(size_t j=0; j<k;j++){
           (*fdelta)(i,j) = (*fdelta_mean)(i,j) + (*fdelta_sd)(i,j)*gen.normal(); 
        }
    }
}
*/