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

void split_xorder_std_continuous(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, const double *X_std, size_t N_y, size_t p, size_t p_continuous, size_t p_categorical, double &yleft_mean, double &yright_mean, const double &y_mean, std::vector<double> &y_std);

void split_xorder_std_categorical(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, const double *X_std, size_t N_y, size_t p, size_t p_continuous, size_t p_categorical, double &yleft_mean, double &yright_mean, const double &y_mean, std::vector<double> &y_std, std::vector<size_t> &X_counts_left, std::vector<size_t> &X_counts_right, std::vector<size_t> &X_num_unique_left, std::vector<size_t> &X_num_unique_right, std::vector<size_t> &X_counts, std::vector<double> &X_values, std::vector<size_t> &variable_ind);

void unique_value_count(const double *Xpointer, xinfo_sizet &Xorder_std, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique);

void unique_value_count2(const double *Xpointer, xinfo_sizet &Xorder_std, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, size_t &p_categorical, size_t &p_continuous);

void BART_likelihood_adaptive_std_mtry_old(double y_sum, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars);

void BART_likelihood_adaptive_std_mtry_old_categorical(double y_sum, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, const Model* model);

void BART_likelihood_all(double y_sum, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, size_t &p_categorical, size_t &p_continuous, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, const Model *model, std::mt19937 &gen);

void cumulative_sum_std(std::vector<double> &y_cumsum, std::vector<double> &y_cumsum_inv, double &y_sum, double *y, xinfo_sizet &Xorder, size_t &i, size_t &N);

void calculate_loglikelihood_continuous(std::vector<double> &loglike, const std::vector<size_t> &subset_vars, size_t &N_Xorder, size_t &Nmin, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double &y_sum, const double &beta, const double &alpha, size_t &depth, const size_t &p, size_t &p_continuous, size_t &Ncutpoints, double &tau, double &sigma2, double &loglike_max, const Model *model);

void calculate_loglikelihood_categorical(std::vector<double> &loglike, size_t &loglike_start, const std::vector<size_t> &subset_vars, size_t &N_Xorder, size_t &N_min, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double &y_sum, const double &beta, const double &alpha, size_t &depth, const size_t &p, const size_t &p_continuous, size_t &p_categorical, size_t &Ncutpoints, double &tau, double &sigma2, double &loglike_max, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, const Model *model);

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
  double theta;   //theta
  std::vector<double> theta_vector;
};

//--------------------------------------------------
class tree
{
public:
  //typedefs--------------------
  typedef tree *tree_p;
  typedef const tree *tree_cp;
  typedef std::vector<tree_p> npv;
  typedef std::vector<tree_cp> cnpv;
  //friends--------------------
  friend std::istream &operator>>(std::istream &, tree &);
  //  friend void update_sufficient_stat(tree& tree, arma::mat& y, arma::mat& X, tree::npv& bv, tree::npv& bv2, double& tau, double& sigma, double& alpha, double& beta);
  //contructors,destructors--------------------
  tree() : theta(0.0), theta_noise(0.0), theta_vector(1, 0.0), sig(0.0), v(0), c(0), p(0), l(0), r(0) {}
  tree(const tree &n) : theta(0.0), theta_noise(0.0), theta_vector(1, 0.0), sig(0.0), v(0), c(0), p(0), l(0), r(0) { cp(this, &n); }
  tree(double itheta) : theta(itheta), theta_noise(0.0), theta_vector(1, 0.0), sig(0.0), v(0), c(0), p(0), l(0), r(0) {}
  tree(size_t num_classes) : theta(0.0), theta_noise(0.0), theta_vector(num_classes, 0.0), sig(0.0), v(0), c(0), p(0), l(0), r(0) {}

  void tonull(); //like a "clear", null tree has just one node
  ~tree() { tonull(); }
  //operators----------
  tree &operator=(const tree &);
  //interface--------------------
  //set
  void settheta(std::vector<double> theta_vector) { this->theta_vector = theta_vector; }

  void setv(size_t v) { this->v = v; }
  void setc(size_t c) { this->c = c; }
  //get
  std::vector<double> gettheta_vector() const { return theta_vector; }

  double getsig() const { return sig; }
  size_t getv() const { return v; }
  double getc() const { return c; }
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
  //  void grow_tree(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, double tau, double sigma, double alpha, double beta);

  void grow_tree_adaptive_abarth_train(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints,
                                       double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel,
                                       std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, double &run_time,
                                       bool &use_all, std::vector<double> &mtry_weight_current_tree, std::vector<double> &split_count_current_tree, const Model *model, std::mt19937 &gen);

  //     void grow_tree_adaptive_std_mtrywithinnode(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, double &run_time, Rcpp::IntegerVector &var_index_candidate, bool &use_all, Rcpp::NumericMatrix &split_count_all_tree, Rcpp::NumericVector &mtry_weight_current_tree, Rcpp::NumericVector &split_count_current_tree);

  void grow_tree_adaptive_std_mtrywithinnode(double y_mean, size_t depth, size_t max_depth, size_t Nmin,
                                             size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu,
                                             bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry,
                                             double &run_time, bool &use_all, xinfo &split_count_all_tree, std::vector<double> &mtry_weight_current_tree,
                                             std::vector<double> &split_count_current_tree, const Model *model, std::mt19937 &gen);

  void grow_tree_adaptive_std_mtrywithinnode_categorical(double y_mean, size_t depth, size_t max_depth, size_t Nmin,
                                                         size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel,
                                                         std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, double &run_time,
                                                         bool &use_all, xinfo &split_count_all_tree, std::vector<double> &mtry_weight_current_tree,
                                                         std::vector<double> &split_count_current_tree, std::vector<double> &X_values, std::vector<size_t> &X_counts,
                                                         std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, const Model *model, std::mt19937 &gen);

  void grow_tree_adaptive_std_all(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints,
                                  double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel,
                                  std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, bool &use_all,
                                  xinfo &split_count_all_tree, std::vector<double> &mtry_weight_current_tree,
                                  std::vector<double> &split_count_current_tree, bool &categorical_variables, size_t &p_categorical,
                                  size_t &p_continuous, std::vector<double> &X_values, std::vector<size_t> &X_counts,
                                  std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, Model *model,
                                  matrix<tree::tree_p> &data_pointers, const size_t &tree_ind, std::mt19937 &gen);

  tree_p bn(double *x, xinfo &xi); //find Bottom Node, original BART version
  tree_p bn_std(double *x);        // find Bottom Node, std version, compare
  //tree_p search_bottom(arma::mat &Xnew, const size_t &i);
  tree_p search_bottom_std(const double *X, const size_t &i, const size_t &p, const size_t &N);
  //tree_p search_bottom_test(arma::mat &Xnew, const size_t &i, const double *X_std, const size_t &p, const size_t &N);
  void rg(size_t v, size_t *L, size_t *U); //recursively find region [L,U] for var v
  //node functions--------------------
  size_t nid() const; //nid of a node
  size_t depth();     //depth of a node
  char ntype();       //node type t:top, b:bot, n:no grandchildren i:interior (t can be b)
  bool isnog();

#ifndef NoRcpp
  //REMOVED :
  //  Rcpp::List tree2list(xinfo &xi, double center = 0., double scale = 1.); // create an efficient list from a single tree
  //tree list2tree(Rcpp::List&, xinfo& xi); // create a tree from a list and an xinfo
  //   Rcpp::IntegerVector tree2count(size_t nvar); // for one tree, count the number of branches for each variable
#endif
private:
  double theta; //univariate double parameter
  double theta_noise;
  std::vector<double> theta_vector;

  double sig;
  //rule: left if x[v] < xinfo[v][c]
  size_t v; //index of variable to split
  double c;

  //////////////////////////
  // size_t c;  // split point
  //////////////////////////
  // different from BART package, they use index, we save raw split value

  //tree structure
  tree_p p; //parent
  tree_p l; //left child
  tree_p r; //right child
  //utiity functions
  void cp(tree_p n, tree_cp o); //copy tree
};
std::istream &operator>>(std::istream &, tree &);
std::ostream &operator<<(std::ostream &, const tree &);

#endif