#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "utility.h"
#include "json_io.h"
#include "utility_rcpp.h"

// [[Rcpp::export]]
Rcpp::List xbart_predict(arma::mat X, double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    // Result Container
    matrix<double> yhats_test_xinfo;
    size_t N_sweeps = (*trees).size();
    size_t M = (*trees)[0].size();
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    NormalModel *model = new NormalModel();

    // Predict
    model->predict_std(Xpointer, N, p, M, N_sweeps,
                       yhats_test_xinfo, *trees);

    // Convert back to Rcpp
    Rcpp::NumericMatrix yhats(N, N_sweeps);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            yhats(i, j) = yhats_test_xinfo[j][i];
        }
    }

    return Rcpp::List::create(Rcpp::Named("yhats") = yhats);
}

// [[Rcpp::export]]
Rcpp::List xbart_predict_full(arma::mat X, double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    size_t N_sweeps = (*trees).size();
    size_t M = (*trees)[0].size();

    std::vector<double> output_vec(N * N_sweeps * M);

    NormalModel *model = new NormalModel();

    // Predict
    model->predict_whole_std(Xpointer, N, p, M, N_sweeps,
                       output_vec, *trees);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);
    output.attr("dim") = Rcpp::Dimension(N, N_sweeps, M);


    return Rcpp::List::create(Rcpp::Named("yhats") = output);
}


// [[Rcpp::export]]
Rcpp::List gp_predict_old(arma::mat y, arma::mat X, arma::mat Xtest, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{
    // return leaf id for train and test samples
    // return max active variable for each test samples

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    Rcpp::NumericMatrix Xtest_std(N_test, p);
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtest_std(i, j) = Xtest(i, j);
        }
    }
    double *Xtestpointer = &Xtest_std[0];

     // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    // Result Container
    matrix<double> yhats_test_xinfo;
    size_t N_sweeps = (*trees).size();
    size_t M = (*trees)[0].size();
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);


    // get leaf id for all train data
    std::vector<double> train_id(N * N_sweeps * M );
    tree::tree_p bn; // pointer to bottom node
    for (size_t i = 0; i < N; i++)
    {
        for (size_t sweep_ind = 0; sweep_ind < N_sweeps; sweep_ind++){
            
            for (size_t tree_ind = 0; tree_ind < M; tree_ind++){
                // loop over observations
                // tree search
                bn = (*trees)[sweep_ind][tree_ind].search_bottom_std(Xpointer, i, p, N);
                train_id[i + sweep_ind * N + tree_ind * N * N_sweeps] = bn->getID();
            }
        }        
    }

    // get leaf id, mu and active variable for test data
    std::vector<double> test_theta(N_test * N_sweeps * M );
    std::vector<size_t> test_id(N_test * N_sweeps * M );
    std::vector<std::vector<size_t>> test_var(N_test * N_sweeps * M);
    double theta;
    size_t leaf_id;
    std::vector<bool> active_var(p);
    std::vector<size_t> active_set;

    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t sweep_ind = 0; sweep_ind < N_sweeps; sweep_ind++){
            
            for (size_t tree_ind = 0; tree_ind < M; tree_ind++){
                // loop over observations
                // // tree search
                active_set.clear();
                std::fill(active_var.begin(), active_var.end(), false);
                (*trees)[sweep_ind][tree_ind].get_gp_info(Xtestpointer, i, p, N_test, active_var, theta, leaf_id);
                test_id[i + sweep_ind * N_test + tree_ind * N_test * N_sweeps] = leaf_id;
                test_theta[i + sweep_ind * N_test + tree_ind * N_test * N_sweeps] = theta;

                for (size_t v = 0; v < p; v++){
                    if (active_var[v]) { active_set.push_back(v); }
                }                              
                test_var[i + sweep_ind * N_test + tree_ind * N_test * N_sweeps] = active_set;
            }
        }        
    }

    Rcpp::NumericVector train_id_rcpp = Rcpp::wrap(train_id);
    Rcpp::NumericVector test_id_rcpp = Rcpp::wrap(test_id);
    Rcpp::NumericVector test_theta_rcpp = Rcpp::wrap(test_theta);
    train_id_rcpp.attr("dim") = Rcpp::Dimension(N, N_sweeps, M);
    test_id_rcpp.attr("dim") = Rcpp::Dimension(N_test, N_sweeps, M);
    test_theta_rcpp.attr("dim") = Rcpp::Dimension(N_test, N_sweeps, M);
    
    Rcpp::NumericVector active_var_rcpp(p);
    arma::field<Rcpp::NumericVector> test_var_rcpp(N_test * N_sweeps * M);
    for (size_t i = 0; i < N_test; i++)
    {
        
        for (size_t sweep_ind = 0; sweep_ind < N_sweeps; sweep_ind++){
            
            for (size_t tree_ind = 0; tree_ind < M; tree_ind++){

                test_var_rcpp(i * N_sweeps * M + sweep_ind * M + tree_ind ) = Rcpp::wrap(test_var[i + sweep_ind * N_test + tree_ind * N_test * N_sweeps]);;
           }
        }        
    }

    return Rcpp::List::create(
        Rcpp::Named("train_id") = train_id_rcpp,
        Rcpp::Named("test_id") = test_id_rcpp,
        Rcpp::Named("yhat_test") = test_theta_rcpp,
        Rcpp::Named("active_var") = test_var_rcpp,
        Rcpp::Named("num_sweeps") = N_sweeps,
        Rcpp::Named("num_trees") = M
        );

}

// [[Rcpp::export]]
Rcpp::List gp_predict(arma::mat y, arma::mat X, arma::mat Xtest, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt, Rcpp::NumericVector resid, double theta, double tau, size_t p_categorical = 0)
{
    // should be able to run in parallel
    cout << "predict with gaussian process" << endl;
    // if (parallel && (nthread == 0))
    // {
    //     // if turn on parallel and do not sepicifiy number of threads
    //     // use max - 1, leave one out
    //     nthread = omp_get_max_threads() - 1;
    // }

    // if (parallel)
    // {
    //     omp_set_num_threads(nthread);
    //     cout << "Running in parallel with " << nthread << " threads." << endl;
    // }
    // else
    // {
    //     cout << "Running with single thread." << endl;
    // }
    
    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;
    // number of continuous variables
    size_t p_continuous = p - p_categorical; // only work for continuous for now

    matrix<size_t> Xorder_std;
    ini_matrix(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    rcpp_to_std2(y, X, Xtest, y_std, y_mean, X_std, Xtest_std, Xorder_std);


    matrix<size_t> Xtestorder_std;
    ini_matrix(Xtestorder_std, N_test, p);

    // Create Xtestorder
    arma::umat Xtestorder(Xtest.n_rows, Xtest.n_cols);
    for (size_t i = 0; i < Xtest.n_cols; i++)
    {
        Xtestorder.col(i) = arma::sort_index(Xtest.col(i));
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtestorder_std[j][i] = Xtestorder(i, j);
        }
    }

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;
    size_t num_sweeps = (*trees).size();
    size_t num_trees = (*trees)[0].size();

    // initialize X_struct
    std::vector<double> initial_theta(1, y_mean / (double)num_trees);
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));
    std::unique_ptr<X_struct> xtest_struct(new X_struct(Xtestpointer, &y_std, N_test, Xtestorder_std, p_categorical, p_continuous, &initial_theta, num_trees));
    x_struct->n_y = N;
    xtest_struct->n_y = N_test;

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N_test, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++){
        std::fill(yhats_test_xinfo[i].begin(), yhats_test_xinfo[i].end(), 0.0);
    }

    std::vector<bool> active_var(p);
    std::fill(active_var.begin(), active_var.end(), false);

    // get residuals
    matrix<std::vector<double>> residuals;
    ini_matrix(residuals, num_trees, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++){
        for (size_t j = 0; j < num_trees; j++){
            residuals[i][j].resize(N);
            for (size_t k = 0; k < N; k++){
                residuals[i][j][k] =  resid(k + i * N + j * num_sweeps * N);
            }
        }
    }
    x_struct->set_resid(residuals);

    // // define model
    // NormalModel *model = new NormalModel(1, 1, 1, 1, 1, false, 1, 1, 1);
    // // cout << "after define model " << model->tau << " " << model->tau_mean << endl;
    // model->setNoSplitPenality(1);

    // // State settings
    // std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, false, 0, 5, 20, p, Xpointer, num_sweeps, false, &y_std, 1.0, 250, y_mean, 0, model->dim_residual, 12, true)); //last input is nthread, need update
    // std::unique_ptr<State> state_test(new NormalState(Xtestpointer, Xtestorder_std, N_test, p, num_trees, p_categorical, p_continuous, false, 0, 5, 20, p, Xtestpointer, num_sweeps, false, &y_std, 1.0, 250, y_mean, 0, model->dim_residual, 12, true)); //last input is nthread, need update

    // mcmc loop
    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {
            // cout << "sweeps = " << sweeps << ", tree_ind = " << tree_ind << endl;
            (*trees)[sweeps][tree_ind].gp_predict_from_root(Xorder_std, x_struct, x_struct->X_counts, x_struct->X_num_unique, 
            Xtestorder_std, xtest_struct, xtest_struct->X_counts, xtest_struct->X_num_unique, 
            yhats_test_xinfo, active_var, p_categorical, sweeps, tree_ind, theta, tau);

        }
    }

    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);
    
    return Rcpp::List::create( Rcpp::Named("yhats_test") = yhats_test);

}

// [[Rcpp::export]]
Rcpp::List xbart_multinomial_predict(arma::mat X, double y_mean, size_t num_class, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt, arma::vec iteration)
{

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    size_t iteration_len = iteration.n_elem;

    std::vector<size_t> iteration_vec(iteration_len);

    for (size_t i = 0; i < iteration_len; i++)
    {
        iteration_vec[i] = iteration(i);
    }

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    // Result Container
    matrix<double> yhats_test_xinfo;
    size_t N_sweeps = (*trees).size();
    size_t N_trees = (*trees)[0].size();
    ini_xinfo(yhats_test_xinfo, N, iteration_len);

    std::vector<double> output_vec(iteration_len * N * num_class);
    std::vector<size_t> output_leaf_index(iteration_len * N * N_trees);

    LogitModel *model = new LogitModel();

    model->dim_residual = num_class;

    // Predict
    model->predict_std_standalone(Xpointer, N, p, N_trees, N_sweeps, yhats_test_xinfo, *trees, output_vec, iteration_vec, output_leaf_index);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);
    output.attr("dim") = Rcpp::Dimension(iteration_len, N, num_class);
    Rcpp::NumericVector index = Rcpp::wrap(output_leaf_index);
    index.attr("dim") = Rcpp::Dimension(iteration_len, N, N_trees);
    
    return Rcpp::List::create(Rcpp::Named("yhats") = output, Rcpp::Named("leaf_index") = index);
}

// [[Rcpp::export]]
Rcpp::List xbart_multinomial_predict_3D(arma::mat X, double y_mean, size_t num_class, Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt, arma::vec iteration)
{
    // used to predict for a three dimensional matrix of trees, for separate tree model of multinomial classification
    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    size_t iteration_len = iteration.n_elem;

    std::vector<size_t> iteration_vec(iteration_len);

    for (size_t i = 0; i < iteration_len; i++)
    {
        iteration_vec[i] = iteration(i);
    }

    // Trees
    std::vector<std::vector<std::vector<tree>>> *trees = tree_pnt;

    // Result Container
    vector<vector<double>> yhats_test_xinfo;
    size_t N_sweeps = (*trees)[0].size();
    size_t N_trees = (*trees)[0][0].size();

    ini_xinfo(yhats_test_xinfo, N, iteration_len);

    std::vector<double> output_vec(iteration_len * N * num_class);

    LogitModelSeparateTrees *model = new LogitModelSeparateTrees();

    model->dim_residual = num_class;

    // Predict
    model->predict_std_standalone(Xpointer, N, p, N_trees, N_sweeps, yhats_test_xinfo, *trees, output_vec, iteration_vec, 0);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);

    output.attr("dim") = Rcpp::Dimension(iteration_len, N, num_class);

    return Rcpp::List::create(Rcpp::Named("yhats") = output);
}

// [[Rcpp::export]]
Rcpp::StringVector r_to_json(double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{
    // push two dimensional matrix of trees to json
    Rcpp::StringVector result(1);
    std::vector<std::vector<tree>> *trees = tree_pnt;
    json j = get_forest_json(*trees, y_mean);
    result[0] = j.dump(4);
    return result;
}

// [[Rcpp::export]]
Rcpp::List json_to_r(Rcpp::StringVector json_string_r)
{
    // load json to a two dimensional matrix of trees
    std::vector<std::string> json_string(json_string_r.size());
    //std::string json_string = json_string_r(0);
    json_string[0] = json_string_r(0);
    double y_mean;

    // Create trees
    vector<vector<tree>> *trees2 = new std::vector<vector<tree>>();

    // Load
    from_json_to_forest(json_string[0], *trees2, y_mean);

    // tree::npv bv;
    // (*trees2)[0][0].getbots(bv); //get bottom nodes
    // for (size_t i = 0; i < bv.size(); i++){
    //     cout << bv[i]->getv() << " " << bv[i]->getID() << endl;
    // }

    // Define External Pointer
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    return Rcpp::List::create(Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean));
}

// [[Rcpp::export]]
Rcpp::StringVector r_to_json_3D(Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt)
{
    // push 3 dimensional matrix of trees to json, used for separate trees in multinomial classification
    Rcpp::StringVector result(1);
    std::vector<std::vector<std::vector<tree>>> *trees = tree_pnt;
    json j = get_forest_json_3D(*trees);
    result[0] = j.dump(4);
    return result;
}

// [[Rcpp::export]]
Rcpp::List json_to_r_3D(Rcpp::StringVector json_string_r)
{
    // push json to 3 dimensional matrix of trees, used for separate trees in multinomial classification
    std::vector<std::string> json_string(json_string_r.size());
    //std::string json_string = json_string_r(0);
    json_string[0] = json_string_r(0);

    // Create trees
    vector<vector<vector<tree>>> *trees2 = new std::vector<std::vector<vector<tree>>>();

    // Load
    from_json_to_forest_3D(json_string[0], *trees2);

    // Define External Pointer
    Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt(trees2, true);

    return Rcpp::List::create(Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt));
}
