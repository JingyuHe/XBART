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
#include "utility.h"

// [[Rcpp::plugins(cpp11)]]
//--------------------------------------------------
//xinfo xi, then xi[v][c] is the c^{th} cutpoint for variable v.
//left if x[v] < xi[v][c]
void split_xorder_std(xinfo_sizet& Xorder_left_std, xinfo_sizet& Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet& Xorder_std, const double* X_std, size_t N_y, size_t p);


void split_xorder_std_newXorder(xinfo_sizet& Xorder_left_std, xinfo_sizet& Xorder_right_std, const size_t& split_var, const size_t& split_point, const xinfo_sizet& Xorder_std, xinfo_sizet& Xorder_next_index, const double* X_std, size_t N_y, size_t p, size_t& N_Xorder, size_t& N_Xorder_left, size_t& N_Xorder_right, std::vector<size_t>& Xorder_firstline, std::vector<size_t>& Xorder_left_firstline, std::vector<size_t>& Xorder_right_firstline, xinfo_sizet& Xorder_full);


//--------------------------------------------------
//find best split variable and value, CART
void split_error(const arma::umat& Xorder, arma::vec& y, arma::uvec& best_split, arma::vec& least_error);
void BART_likelihood(const arma::umat& Xorder, arma::mat& y, arma::vec& loglike, double tau, double sigma, size_t depth, size_t Nmin, double alpha, double beta);
void BART_likelihood_adaptive(const arma::umat& Xorder, arma::mat& y, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool& no_split, size_t & split_var, size_t & split_point, bool parallel);



void BART_likelihood_adaptive_std(std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool& no_split, size_t & split_var, size_t & split_point, bool parallel, std::vector<size_t>& subset_vars);



void BART_likelihood_adaptive_std_mtry(std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool& no_split, size_t & split_var, size_t & split_point, bool parallel, const std::vector<size_t>& subset_vars);


void cumulative_sum_std(std::vector<double>& y_cumsum, std::vector<double>& y_cumsum_inv, double& y_sum, double* y, xinfo_sizet& Xorder, size_t& i, size_t& N);
//--------------------------------------------------
//split Xorder matrix for two subnodes
void split_xorder(arma::umat& Xorder_left, arma::umat& Xorder_right, arma::umat& Xorder, arma::mat& X, size_t split_var, size_t split_point);
//--------------------------------------------------
//BART likelihood function
arma::vec BART_likelihood_function(arma::vec& n1, arma::vec& n2, arma::vec& s1, arma::vec& s2, double& tau, double& sigma, double& alpha, double& penalty);
//--------------------------------------------------
//generate a vector of integers
arma::uvec range(size_t start, size_t end);





//--------------------------------------------------
//info contained in a node, used by input operator
struct node_info {
   std::size_t id; //node id
   std::size_t v;  //variable
   double c;  //cut point // different from BART
   double theta;   //theta
};

//--------------------------------------------------
class tree {
public:
   //typedefs--------------------
   typedef tree* tree_p;
   typedef const tree* tree_cp;
   typedef std::vector<tree_p> npv;
   typedef std::vector<tree_cp> cnpv;
   //friends--------------------
   friend std::istream& operator>>(std::istream&, tree&);
  //  friend void update_sufficient_stat(tree& tree, arma::mat& y, arma::mat& X, tree::npv& bv, tree::npv& bv2, double& tau, double& sigma, double& alpha, double& beta);
   //contructors,destructors--------------------
   tree(): theta(0.0),theta_noise(0.0),sig(0.0),v(0),c(0),p(0),l(0),r(0) {}
   tree(const tree& n): theta(0.0),theta_noise(0.0),sig(0.0),v(0),c(0),p(0),l(0),r(0) {cp(this,&n);}
   tree(double itheta): theta(itheta),theta_noise(0.0),sig(0.0),v(0),c(0),p(0),l(0),r(0) {}
   void tonull(); //like a "clear", null tree has just one node
   ~tree() {tonull();}
   //operators----------
   tree& operator=(const tree&);
   //interface--------------------
   //set
   void settheta(double theta) {this->theta=theta;}
   void setv(size_t v) {this->v = v;}
   void setc(size_t c) {this->c = c;}
   //get
   double gettheta() const {return theta;}
   double gettheta_noise() const {return theta_noise;}
   double getsig() const {return sig;}
   size_t getv() const {return v;}
   double getc() const {return c;}
   tree_p getp() {return p;}
   tree_p getl() {return l;}
   tree_p getr() {return r;}
   //tree functions--------------------
   tree_p getptr(size_t nid); //get node pointer from node id, 0 if not there
   void pr(bool pc=true); //to screen, pc is "print children"
   size_t treesize(); //number of nodes in tree
   size_t nnogs();    //number of nog nodes (no grandchildren nodes)
   size_t nbots();    //number of bottom nodes
   bool birth(size_t nid, size_t v, size_t c, double thetal, double thetar);
   bool death(size_t nid, double theta);
   void birthp(tree_p np,size_t v, size_t c, double thetal, double thetar);
   void deathp(tree_p nb, double theta);
   void getbots(npv& bv);         //get bottom nodes
   void getnogs(npv& nv);         //get nog nodes (no granchildren)
   void getnodes(npv& v);         //get vector of all nodes
   void getnodes(cnpv& v) const;  //get vector of all nodes (const)
   tree_p gettop(); // get pointer to the top node
  //  void grow_tree(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, double tau, double sigma, double alpha, double beta);
   void grow_tree(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu);
   void grow_tree_adaptive(arma::mat& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel);


   void grow_tree_adaptive_std(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, double* split_var_count_pointer, size_t& mtry, const std::vector<size_t>& subset_vars, double& run_time);

   void grow_tree_adaptive_std_newXorder(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double>& y_std, xinfo_sizet& Xorder_std, xinfo_sizet& Xorder_next_index, std::vector<size_t>& Xorder_firstline, const double* X_std, double* split_var_count_pointer, size_t& mtry, const std::vector<size_t>& subset_vars, xinfo_sizet& Xorder_full);

   void grow_tree_adaptive_onestep(arma::mat& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel);

   void grow_tree_adaptive_onestep_std(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, std::vector<double>& split_var_count);


   void prune_regrow(arma::mat& y, double y_mean, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double& tau, double& sigma, double& alpha, double& beta, bool draw_sigma, bool draw_mu, bool parallel);
   void one_step_grow(arma::mat& y, double y_mean, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double& tau, double& sigma, double& alpha, double& beta, bool draw_sigma, bool draw_mu, bool parallel);
   void one_step_prune(arma::mat& y, double y_mean, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double& tau, double& sigma, double& alpha, double& beta, bool draw_sigma, bool draw_mu, bool parallel);
   void sample_theta(arma::mat& y, arma::mat& X, double& tau, double& sigma, bool draw_mu);
   tree_p bn(double *x,xinfo& xi); //find Bottom Node, original BART version
   tree_p bn_std(double *x); // find Bottom Node, std version, compare
   tree_p search_bottom(arma::mat& Xnew, const size_t& i);
   tree_p search_bottom_std(const double* X, const size_t& i, const size_t& p, const size_t& N);
   tree_p search_bottom_test(arma::mat& Xnew, const size_t& i, const double* X_std, const size_t& p, const size_t& N);
   void rg(size_t v, size_t* L, size_t* U); //recursively find region [L,U] for var v
   //node functions--------------------
   size_t nid() const; //nid of a node
   size_t depth();  //depth of a node
   char ntype(); //node type t:top, b:bot, n:no grandchildren i:interior (t can be b)
   bool isnog();
#ifndef NoRcpp
  Rcpp::List tree2list(xinfo& xi, double center=0., double scale=1.); // create an efficient list from a single tree
  //tree list2tree(Rcpp::List&, xinfo& xi); // create a tree from a list and an xinfo
  Rcpp::IntegerVector tree2count(size_t nvar); // for one tree, count the number of branches for each variable
#endif
private:
   double theta; //univariate double parameter
   double theta_noise;
   double sig;
   //rule: left if x[v] < xinfo[v][c]
   size_t v;  //index of variable to split
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
   void cp(tree_p n,  tree_cp o); //copy tree
};
std::istream& operator>>(std::istream&, tree&);
std::ostream& operator<<(std::ostream&, const tree&);

#endif
