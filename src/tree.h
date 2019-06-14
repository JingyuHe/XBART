/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#ifndef GUARD_tree_h
#define GUARD_tree_h

#include <map>
#include <cmath>
#include <cstddef>
#include "common.h"
#include "sample_int_crank.h"
#include "model.h"

//
#include "json.h"
// for convenience
using json = nlohmann::json;

void unique_value_count(const double *Xpointer, xinfo_sizet &Xorder_std, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique);

void cumulative_sum_std(std::vector<double> &y_cumsum, std::vector<double> &y_cumsum_inv, double &y_sum, double *y, xinfo_sizet &Xorder, size_t &i, size_t &N);

// void calc_suff_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, double &suff_stat, bool adaptive_cutpoint);

//--------------------------------------------------
//BART likelihood function
//--------------------------------------------------
//generate a vector of integers
// arma::uvec range(size_t start, size_t end); Removed

//--------------------------------------------------
//info contained in a node, used by input operator
struct node_info
{
    std::size_t id; //node id
    std::size_t v;  //variable
    double c;       //cut point // different from BART
    std::vector<double> theta_vector;
};

//--------------------------------------------------
class tree
{
public:
    // std::vector<double> theta_vector;
    std::vector<double> theta_vector;

    // suff_Stat save nodewise sufficient statistics
    // such as mean of y, sum of y squares
    // hard code as a vector of length 2 for now
    std::vector<double> suff_stat;

    //typedefs--------------------
    typedef tree *tree_p;
    typedef const tree *tree_cp;
    typedef std::vector<tree_p> npv;
    typedef std::vector<tree_cp> cnpv;
    //friends--------------------
    friend std::istream &operator>>(std::istream &, tree &);
    //  friend void update_sufficient_stat(tree& tree, arma::mat& y, arma::mat& X, tree::npv& bv, tree::npv& bv2, double& tau, double& sigma, double& alpha, double& beta);
    //contructors,destructors--------------------
    tree() : suff_stat(2, 0.0), theta_vector(1, 0.0), sig(0.0), v(0), c(0), p(0), l(0), r(0), prob_split(0.0), prob_leaf(0.0), drawn_ind(0), y_mean(0.0), N_Xorder(0), depth(0) {}
    tree(const tree &n) : suff_stat(2, 0.0), theta_vector(1, 0.0), sig(0.0), v(0), c(0), p(0), l(0), r(0), prob_split(0.0), prob_leaf(0.0), drawn_ind(0), y_mean(0.0), N_Xorder(0), depth(0) { cp(this, &n); }
    tree(double itheta) : suff_stat(2, 0.0), theta_vector(itheta, 0.0), sig(0.0), v(0), c(0), p(0), l(0), r(0), prob_split(0.0), prob_leaf(0.0), drawn_ind(0), y_mean(0.0), N_Xorder(0), depth(0) {}
    tree(size_t num_classes, const tree_p parent) : suff_stat(2, 0.0), theta_vector(num_classes, 0.0), sig(0.0), v(0), c(0), p(parent), l(0), r(0), prob_split(0.0), prob_leaf(0.0), drawn_ind(0), y_mean(0.0), N_Xorder(0), depth(0) {}

    void tonull(); //like a "clear", null tree has just one node
    ~tree() { tonull(); }
    //operators----------
    tree &operator=(const tree &);
    //interface--------------------
    //set
    void settheta(std::vector<double> theta_vector) { this->theta_vector = theta_vector; }

    void setv(size_t v) { this->v = v; }
    void setc(size_t c) { this->c = c; }
    void setdepth(size_t depth) { this->depth = depth; }
    //get
    std::vector<double> gettheta_vector() const { return theta_vector; }

    double getsig() const { return sig; }
    double getprob_split() const { return prob_split; }
    double getprob_leaf() const { return prob_leaf; }
    size_t getv() const { return v; }
    double getc() const { return c; }
    size_t getdepth() const { return depth; }

    size_t getN_Xorder() const { return N_Xorder; }
    void setN_Xorder(size_t N_Xorder) { this->N_Xorder = N_Xorder; }

    double gety_mean() const { return y_mean; }
    void sety_mean(double y_mean) { this->y_mean = y_mean; }

    // size_t getsplit_var() const {return split_var; }
    // size_t getsplit_point() const {return split_point; }
    // bool getno_split() const {return no_split;}

    // void setno_split(bool no_split) {this->no_split = no_split; }
    // void setsplit_var(size_t split_var) {this->split_var = split_var;}
    // void setsplit_point(size_t split_point) {this->split_point = split_point;}

    tree_p getp() { return p; }
    tree_p getl() { return l; }
    tree_p getr() { return r; }
    //tree functions--------------------
    tree_p getptr(size_t nid); //get node pointer from node id, 0 if not there
    void pr(bool pc = true);   //to screen, pc is "print children"
    size_t treesize();         //number of nodes in tree
    size_t nnogs();            //number of nog nodes (no grandchildren nodes)
    size_t nbots();            //number of bottom nodes

    void getbots(npv &bv);        //get bottom nodes
    void getnogs(npv &nv);        //get nog nodes (no granchildren)
    void getnodes(npv &v);        //get vector of all nodes
    void getnodes(cnpv &v) const; //get vector of all nodes (const)
    tree_p gettop();              // get pointer to the top node

    friend void calculate_loglikelihood_continuous(std::vector<double> &loglike, const std::vector<size_t> &subset_vars, size_t &N_Xorder, xinfo_sizet &Xorder_std, const double &y_sum, const double &beta, const double &alpha, size_t &depth, const size_t &p, double &tau, double &sigma2, double &loglike_max, Model *model, std::unique_ptr<FitInfo> &fit_info);

    friend void BART_likelihood_all(xinfo_sizet &Xorder_std, double tau, double sigma, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, const std::vector<size_t> &subset_vars, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, std::unique_ptr<FitInfo> &fit_info, tree_p tree_pointer);

    friend void BART_likelihood_update_old_tree(xinfo_sizet &Xorder_std, double tau, double sigma, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, const std::vector<size_t> &subset_vars, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, std::unique_ptr<FitInfo> &fit_info, tree_p tree_pointer);

    friend void calculate_loglikelihood_continuous(std::vector<double> &loglike, const std::vector<size_t> &subset_vars, size_t &N_Xorder, xinfo_sizet &Xorder_std, const double &beta, const double &alpha, const size_t &p, double &tau, double &sigma2, double &loglike_max, Model *model, std::unique_ptr<FitInfo> &fit_info, tree_p tree_pointer);

    friend void calculate_loglikelihood_categorical(std::vector<double> &loglike, size_t &loglike_start, const std::vector<size_t> &subset_vars, size_t &N_Xorder, xinfo_sizet &Xorder_std, const double &beta, const double &alpha, const size_t &p, double &tau, double &sigma2, double &loglike_max, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, size_t &total_categorical_split_candidates, std::unique_ptr<FitInfo> &fit_info, tree_p tree_pointer);

    friend void calculate_likelihood_no_split(std::vector<double> &loglike, size_t &N_Xorder, const double &beta, const double &alpha, const size_t &p, double &tau, double &sigma2, double &loglike_max, Model *model, size_t &total_categorical_split_candidates, std::unique_ptr<FitInfo> &fit_info, tree_p tree_pointer);

    void grow_from_root(std::unique_ptr<FitInfo> &fit_info, size_t max_depth, double tau, double sigma, double alpha, double beta, bool draw_mu, xinfo_sizet &Xorder_std, std::vector<double> &mtry_weight_current_tree, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, const size_t &tree_ind, bool sample_weights_flag);

    void update_split_prob(std::unique_ptr<FitInfo> &fit_info, double y_mean, size_t depth, size_t max_depth, double tau, double sigma, double alpha, double beta, bool draw_mu, xinfo_sizet &Xorder_std, std::vector<double> &mtry_weight_current_tree, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, const size_t &tree_ind, bool sample_weights_flag);

    friend void split_xorder_std_continuous(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, size_t N_y, size_t p, double &yleft_mean, double &yright_mean, const double &y_mean, double &yleft_sq_sum, double &yright_sq_sum, const double &y_sq_sum, Model *model, std::unique_ptr<FitInfo> &fit_info);

    friend void split_xorder_std_categorical(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, size_t N_y, size_t p, double &yleft_mean, double &yright_mean, const double &y_mean, double &yleft_sq_sum, double &yright_sq_sum, const double &y_sq_sum, std::vector<size_t> &X_counts_left, std::vector<size_t> &X_counts_right, std::vector<size_t> &X_num_unique_left, std::vector<size_t> &X_num_unique_right, std::vector<size_t> &X_counts, Model *model, std::unique_ptr<FitInfo> &fit_info);

    double transition_prob();

    double log_like_tree(double sigma2, double tau);

    tree_p bn(double *x, xinfo &xi); //find Bottom Node, original BART version
    tree_p bn_std(double *x);        // find Bottom Node, std version, compare
    tree_p search_bottom_std(const double *X, const size_t &i, const size_t &p, const size_t &N);
    void rg(size_t v, size_t *L, size_t *U); //recursively find region [L,U] for var v
    //node functions--------------------
    size_t nid() const; //nid of a node
    // size_t depth();     //depth of a node
    char ntype(); //node type t:top, b:bot, n:no grandchildren i:interior (t can be b)
    bool isnog();

    json to_json();
    void from_json(json &j3, size_t num_classes);

    // #ifndef NoRcpp
    // #endif
private:
    size_t depth;

    double sig;
    //rule: left if x[v] < xinfo[v][c]
    size_t v; //index of variable to split

    double c;

    double prob_split; // posterior of the chose split points, by Bayes rule

    double prob_leaf; // posterior of the leaf parameter

    size_t drawn_ind; // index drawn when sampling cutpoints (in the total likelihood + nosplit vector)

    size_t N_Xorder; // number of data points in this node, for debugging use

    double y_mean; // average of y in current node, for debugging use

    //tree structure
    tree_p p; //parent
    tree_p l; //left child
    tree_p r; //right child
    //utiity functions
    void cp(tree_p n, tree_cp o); //copy tree
};

std::istream &operator>>(std::istream &, tree &);
std::ostream &operator<<(std::ostream &, const tree &);

void predict_from_tree(tree &tree, const double *X_std, size_t N, size_t p, std::vector<double> &output, Model *model);

void predict_from_datapointers(const double *X_std, size_t N, size_t M, std::vector<double> &output, matrix<std::vector<double> *> &data_pointers, Model *model);

#endif