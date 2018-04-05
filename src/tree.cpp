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

#include "tree.h"
#include <RcppArmadilloExtensions/sample.h>


//--------------------
// node id
size_t tree::nid() const 
{
   if(!p) return 1; //if you don't have a parent, you are the top
   if(this==p->l) return 2*(p->nid()); //if you are a left child
   else return 2*(p->nid())+1; //else you are a right child
}
//--------------------
tree::tree_p tree::getptr(size_t nid)
{
   if(this->nid() == nid) return this; //found it
   if(l==0) return 0; //no children, did not find it
   tree_p lp = l->getptr(nid);
   if(lp) return lp; //found on left
   tree_p rp = r->getptr(nid);
   if(rp) return rp; //found on right
   return 0; //never found it
}
//--------------------
//add children to  bot node nid
bool tree::birth(size_t nid,size_t v, size_t c, double thetal, double thetar)
{
   tree_p np = getptr(nid);
   if(np==0) {
      cout << "error in birth: bottom node not found\n";
      return false; //did not find note with that nid
   }
   if(np->l!=0) {
      cout << "error in birth: found node has children\n";
      return false; //node is not a bottom node
   }

   //add children to bottom node np
   tree_p l = new tree;
   l->theta=thetal;
   tree_p r = new tree;
   r->theta=thetar;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;

   return true;
}
//--------------------
//depth of node
size_t tree::depth()
{
   if(!p) return 0; //no parents
   else return (1+p->depth());
}
//--------------------
//tree size
size_t tree::treesize()
{
   if(l==0) return 1;  //if bottom node, tree size is 1
   else return (1+l->treesize()+r->treesize());
}
//--------------------
//node type
char tree::ntype()
{
   //t:top, b:bottom, n:no grandchildren, i:internal
   if(!p) return 't';
   if(!l) return 'b';
   if(!(l->l) && !(r->l)) return 'n';
   return 'i';
}
//--------------------
//print out tree(pc=true) or node(pc=false) information
void tree::pr(bool pc) 
{
   size_t d = depth();
   size_t id = nid();

   size_t pid;
   if(!p) pid=0; //parent of top node
   else pid = p->nid();

   std::string pad(2*d,' ');
   std::string sp(", ");
   if(pc && (ntype()=='t'))
      cout << "tree size: " << treesize() << std::endl;
   cout << pad << "(id,parent): " << id << sp << pid;
   cout << sp << "(v,c): " << v << sp << c;
   cout << sp << "theta: " << theta;
   cout << sp << "type: " << ntype();
   cout << sp << "depth: " << depth();
   cout << sp << "pointer: " << this << std::endl;

   if(pc) {
      if(l) {
         l->pr(pc);
         r->pr(pc);
      }
   }
}
//--------------------
//kill children of  nog node nid
bool tree::death(size_t nid, double theta)
{
   tree_p nb = getptr(nid);
   if(nb==0) {
      cout << "error in death, nid invalid\n";
      return false;
   }
   if(nb->isnog()) {
      delete nb->l;
      delete nb->r;
      nb->l=0;
      nb->r=0;
      nb->v=0;
      nb->c=0;
      nb->theta=theta;
      return true;
   } else {
      cout << "error in death, node is not a nog node\n";
      return false;
   }
}
//--------------------
//is the node a nog node
bool tree::isnog() 
{
   bool isnog=true;
   if(l) {
      if(l->l || r->l) isnog=false; //one of the children has children.
   } else {
      isnog=false; //no children
   }
   return isnog;
}
//--------------------
size_t tree::nnogs() 
{
   if(!l) return 0; //bottom node
   if(l->l || r->l) { //not a nog
      return (l->nnogs() + r->nnogs());
   } else { //is a nog
      return 1;
   }
}
//--------------------
size_t tree::nbots() 
{
   if(l==0) { //if a bottom node
      return 1;
   } else {
      return l->nbots() + r->nbots();
   }
}
//--------------------
//get bottom nodes
void tree::getbots(npv& bv)
{
   if(l) { //have children
      l->getbots(bv);
      r->getbots(bv);
   } else {
      bv.push_back(this);
   }
}
//--------------------
//get nog nodes
void tree::getnogs(npv& nv)
{
   if(l) { //have children
      if((l->l) || (r->l)) {  //have grandchildren
         if(l->l) l->getnogs(nv);
         if(r->l) r->getnogs(nv);
      } else {
         nv.push_back(this);
      }
   }
}
//--------------------
//get all nodes
void tree::getnodes(npv& v)
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
void tree::getnodes(cnpv& v)  const
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
//--------------------
tree::tree_p tree::bn(double *x,xinfo& xi)
{
   if(l==0) return this; //no children
   if(x[v] <= xi[v][c]) {
       // if smaller than or equals to the cutpoint, go to left child

      return l->bn(x,xi);
   } else {
       // if greater than cutpoint, go to right child
      return r->bn(x,xi);
   }
}

tree::tree_p tree::search_bottom(arma::mat& Xnew){
    // cout << "c value" << c << endl;
    if(l == 0){
        // cout << "return this" << endl;
        return this;
        } // no children
    if(arma::as_scalar(Xnew.col(v)) <= c){

        return l -> search_bottom(Xnew);  // if smaller or equal cut point, go to left node
    } else {

        return r -> search_bottom(Xnew);
    }
}
//--------------------
//find region for a given variable
void tree::rg(size_t v, int* L, int* U)
{
   if(this->p==0)  {
      return;
   }
   if((this->p)->v == v) { //does my parent use v?
      if(this == p->l) { //am I left or right child
         if((int)(p->c) <= (*U)) *U = (p->c)-1;
         p->rg(v,L,U);
      } else {
         if((int)(p->c) >= *L) *L = (p->c)+1;
         p->rg(v,L,U);
      }
   } else {
      p->rg(v,L,U);
   }
}
//--------------------
//cut back to one node
void tree::tonull()
{
   size_t ts = treesize();
   //loop invariant: ts>=1
   while(ts>1) { //if false ts=1
      npv nv;
      getnogs(nv);
      for(size_t i=0;i<nv.size();i++) {
         delete nv[i]->l;
         delete nv[i]->r;
         nv[i]->l=0;
         nv[i]->r=0;
      }
      ts = treesize(); //make invariant true
   }
   theta=0.0;
   v=0;c=0;
   p=0;l=0;r=0;
}
//--------------------
//copy tree tree o to tree n
void tree::cp(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
{
   if(n->l) {
      cout << "cp:error node has children\n";
      return;
   }

   n->theta = o->theta;
   n->v = o->v;
   n->c = o->c;

   if(o->l) { //if o has children
      n->l = new tree;
      (n->l)->p = n;
      cp(n->l,o->l);
      n->r = new tree;
      (n->r)->p = n;
      cp(n->r,o->r);
   }
}
//--------------------------------------------------
//operators
tree& tree::operator=(const tree& rhs)
{
   if(&rhs != this) {
      tonull(); //kill left hand side (this)
      cp(this,&rhs); //copy right hand side to left hand side
   }
   return *this;
}
//--------------------------------------------------
//functions
std::ostream& operator<<(std::ostream& os, const tree& t)
{
   tree::cnpv nds;
   t.getnodes(nds);
   os << nds.size() << std::endl;
   for(size_t i=0;i<nds.size();i++) {
    //  os << " a new node "<< endl;
      os << nds[i]->nid() << " ";
      os << nds[i]->getv() << " ";
      os << nds[i]->getc() << " ";
      os << nds[i]->gettheta() << std::endl;
   }
   return os;
}
std::istream& operator>>(std::istream& is, tree& t)
{
   size_t tid,pid; //tid: id of current node, pid: parent's id
   std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id
   size_t nn; //number of nodes

   t.tonull(); // obliterate old tree (if there)

   //read number of nodes----------
   is >> nn;
   if(!is) {
      //cout << ">> error: unable to read number of nodes" << endl;
      return is;
   }

//    cout << "number of nodes " << nn << endl;


    // The idea is to dump string to a lot of node_info structure first, then link them as a tree, by nid

   //read in vector of node information----------
   std::vector<node_info> nv(nn);
   for(size_t i=0;i!=nn;i++) {
      is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta;
      if(!is) {
         //cout << ">> error: unable to read node info, on node  " << i+1 << endl;
         return is;
      }
   }

   
   //first node has to be the top one
   pts[1] = &t; //careful! this is not the first pts, it is pointer of id 1.
   t.setv(nv[0].v); t.setc(nv[0].c); t.settheta(nv[0].theta);
   t.p=0;


    // cout << "nvszie " << nv.size() << endl;

   //now loop through the rest of the nodes knowing parent is already there.
   for(size_t i=1;i!=nv.size();i++) {
      tree::tree_p np = new tree;
      np->v = nv[i].v; np->c=nv[i].c; np->theta=nv[i].theta;
      tid = nv[i].id;
      pts[tid] = np;
      pid = tid/2;
    if(tid % 2 == 0) { //left child has even id
         pts[pid]->l = np;
      } else {
         pts[pid]->r = np;        
      }
      np->p = pts[pid];
   }
   return is;
}
//--------------------
//add children to bot node *np
void tree::birthp(tree_p np,size_t v, size_t c, double thetal, double thetar)
{
   tree_p l = new tree;
   l->theta=thetal;
   tree_p r = new tree;
   r->theta=thetar;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;
}
//--------------------
//kill children of  nog node *nb
void tree::deathp(tree_p nb, double theta)
{
   delete nb->l;
   delete nb->r;
   nb->l=0;
   nb->r=0;
   nb->v=0;
   nb->c=0;
   nb->theta=theta;
}




// void tree::grow_tree(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, int depth, int max_depth, int Nmin, double tau, double sigma){
void tree::grow_tree(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, int depth, int max_depth, int Nmin, double tau, double sigma, double alpha, double beta){
    // this function grow the tree greedly
    // always pick the best split point
    theta = y_mean;

    if(Xorder.n_rows <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    int N = Xorder.n_cols;
    arma::uvec best_split(Xorder.n_cols);
    arma::vec least_error(Xorder.n_cols); 
    split_error_2(Xorder, y, best_split, least_error, tau, sigma, depth, alpha, beta);


    int split_var = arma::index_max(least_error); // maximize likelihood
    double split_point = best_split(split_var);
    if(split_point == 0){
        return;
    }
    if(split_point == Xorder.n_rows - 1){
        return;
    }
    
    this -> v = split_var;

    this -> c =  X(Xorder(split_point, split_var), split_var);


    // if smaller than or equals to the cutpoint, go to left child
    // strictly greater than cutpoint, go to right child

    arma::umat Xorder_left = arma::zeros<arma::umat>(split_point + 1, Xorder.n_cols);
    arma::umat Xorder_right = arma::zeros<arma::umat>(Xorder.n_rows - split_point - 1, Xorder.n_cols);

    split_xorder(Xorder_left, Xorder_right, Xorder, X, split_var, split_point);


    double yleft_mean = arma::as_scalar(arma::mean(y(Xorder_left.col(split_var))));
    double yright_mean = arma::as_scalar(arma::mean(y(Xorder_right.col(split_var))));

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree(y, yleft_mean, Xorder_left, X, depth, max_depth, Nmin, tau, sigma, alpha, beta);
    tree::tree_p rchild = new tree();
    rchild->grow_tree(y, yright_mean, Xorder_right, X, depth, max_depth, Nmin, tau, sigma, alpha, beta);
    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;

    return;
}





void tree::grow_tree_2(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, int depth, int max_depth, int Nmin, double tau, double sigma, double alpha, double beta){
    // this function is more randomized
    // sample from several best split points

    // theta = y_mean * Xorder.n_cols / pow(sigma, 2) * 1.0 / (1.0 / pow(tau, 2) + Xorder.n_cols / pow(sigma, 2));

    theta = y_mean / pow(sigma, 2) * 1.0 / (1.0 / pow(tau, 2) + 1.0 / pow(sigma, 2));

    if(Xorder.n_rows <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    int N = Xorder.n_rows;
    int p = Xorder.n_cols;
    arma::umat best_split = arma::zeros<arma::umat>(Xorder.n_rows, Xorder.n_cols);
    arma::mat loglike = arma::zeros<arma::mat>(Xorder.n_rows, Xorder.n_cols); 

    
    split_error_3(Xorder, y, best_split, loglike, tau, sigma, depth, alpha, beta);

    // now we have 5 best split points for each variable
    // combine them and sample from top 10 split points
    // need to normalize the error vector
    // arma::accu : sum of all elements
    // arma::uvec least_error_vec = 1.0 / arma::accu(least_error) * arma::sort_index(arma::vectorise(least_error), "descend");

    
    // arma::vec least_error_vec = 1.0 / arma::accu(least_error) * arma::vectorise(least_error);
    
    // cout << loglike << endl;

    // convert log likelihood to probability
    // uniformly sample from it

    loglike.row(loglike.n_rows - 1) = loglike.row(loglike.n_rows - 1) - log(p);

    arma::vec loglike_vec = arma::vectorise(loglike);
    loglike_vec = loglike_vec - max(loglike_vec);
    loglike_vec = exp(loglike_vec);
    loglike_vec = loglike_vec / arma::as_scalar(arma::sum(loglike_vec));

    // print out probability of top 5 split points
    arma::vec templog = arma::sort(loglike_vec, "descend");

    Rcpp::IntegerVector temp_ind = Rcpp::seq_len(loglike_vec.n_elem) - 1;

    int ind = Rcpp::RcppArmadillo::sample(temp_ind, 1, false, loglike_vec)[0];

    int split_var = ind / loglike.n_rows;

    int split_point = ind % loglike.n_rows;

    // if(split_point == 0){
    //     return;
    // }
    
    if(split_point == Xorder.n_rows - 1){
        // cout << "early termination" << endl;
        return;
    }
    
    this -> v = split_var;

    this -> c =  X(Xorder(split_point, split_var), split_var);

    arma::umat Xorder_left = arma::zeros<arma::umat>(split_point + 1, Xorder.n_cols);
    arma::umat Xorder_right = arma::zeros<arma::umat>(Xorder.n_rows - split_point - 1, Xorder.n_cols);

    split_xorder(Xorder_left, Xorder_right, Xorder, X, split_var, split_point);


    double yleft_mean = arma::as_scalar(arma::mean(y(Xorder_left.col(split_var))));
    double yright_mean = arma::as_scalar(arma::mean(y(Xorder_right.col(split_var))));

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_2(y, yleft_mean, Xorder_left, X, depth, max_depth, Nmin, tau, sigma, alpha, beta);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_2(y, yright_mean, Xorder_right, X, depth, max_depth, Nmin, tau, sigma, alpha, beta);
    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;

    return;
}





void split_xorder(arma::umat& Xorder_left, arma::umat& Xorder_right, arma::umat& Xorder, arma::mat& X, int split_var, int split_point){
    // preserve order of other variables
    int N = Xorder.n_rows;
    int left_ix = 0;
    int right_ix = 0;
    for(int i = 0; i < Xorder.n_cols; i ++){
            left_ix = 0;
            right_ix = 0;
            for(int j = 0; j < N; j ++){
                // loop over all observations
                if(X(Xorder(j, i), split_var) <= X(Xorder(split_point, split_var), split_var)){
                    Xorder_left(left_ix, i) = Xorder(j, i);
                    left_ix = left_ix + 1;
                }else{
                    Xorder_right(right_ix, i) = Xorder(j, i);
                    right_ix = right_ix + 1;
                // }

            }
        }
    }
    return;
}



arma::vec BART_likelihood(arma::vec& n1, arma::vec& n2, arma::vec& s1, arma::vec& s2, double& tau, double& sigma, double& alpha, double& penalty){
    // log - likelihood of BART model
    // n1 is number of observations in group 1
    // s1 is sum of group 1
    arma::vec result;
    double sigma2 = pow(sigma, 2);
    arma::vec n1tau = n1 * tau;
    arma::vec n2tau = n2 * tau;
    result = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(s1, 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(s2, 2)/(sigma2 * (n2tau + sigma2));
    // result(result.n_elem - 1) = result(result.n_elem - 1) - penalty;
    // double temp = result.min();
    // result(0) = temp;
    // result(result.n_elem - 1) = temp;

    // the last entry is probability of no split
    // alpha is the prior probability of split, multiply it
    // result = result + log(alpha);
    result(result.n_elem - 1) = result(result.n_elem - 1) + log(1.0 - alpha) - log(alpha);
    return result;
}




void split_error(const arma::umat& Xorder, arma::vec& y, arma::uvec& best_split, arma::vec& least_error){
    // regular CART algorithm, compute sum of squared loss error

    int N = Xorder.n_rows;
    int p = Xorder.n_cols;
    // arma::mat errormat = arma::zeros(N, p);
    // loop over all variables and observations and compute error

    double y_error = arma::as_scalar(arma::sum(pow(y(Xorder.col(0)) - arma::mean(y(Xorder.col(0))), 2)));

    double ee;
    double temp_error = y_error;
    arma::vec y_cumsum(y.n_elem);
    arma::vec y2_cumsum(y.n_elem);

    y_cumsum = arma::cumsum(y(Xorder.col(0)));
    y2_cumsum = arma::cumsum(pow(y(Xorder.col(0)), 2));

    double y_sum = y_cumsum(y_cumsum.n_elem - 1);
    double y2_sum = y2_cumsum(y2_cumsum.n_elem - 1);

    arma::vec y2 = pow(y, 2);
    for(int i = 0; i < p; i++){ // loop over variables 
        temp_error = 100.0;
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y2_cumsum = arma::cumsum(pow(y(Xorder.col(i)), 2));
        for(int j = 1; j < N - 1; j++){ // loop over cutpoints

            ee = y2_cumsum(j) - pow(y_cumsum(j), 2) / (double) (j+ 1) + y2_sum - y2_cumsum(j) - pow((y_sum - y_cumsum(j)), 2) / (double) (N - j - 1) ;

            if(ee < temp_error || temp_error == 100.0){
                best_split(i) = j; // Xorder(j,i) coordinate;
                temp_error = ee;
                least_error(i) = ee;
            }
        }
    }
    return;
}




void split_error_2(const arma::umat& Xorder, arma::vec& y, arma::uvec& best_split, arma::vec& least_error, double tau, double sigma, double depth, double alpha, double beta){
    // compute BART posterior (loglikelihood + logprior penalty)
    // greedy 

    int N = Xorder.n_rows;
    int p = Xorder.n_cols;

    double y_error = arma::as_scalar(arma::sum(pow(y(Xorder.col(0)) - arma::mean(y(Xorder.col(0))), 2)));

    double ee;
    
    arma::vec y_cumsum;

    double y_sum;

    arma::vec y_cumsum_inv;

    arma::vec ind1 = arma::linspace(1, N, N);
    arma::vec ind2 = arma::linspace(N, 1, N);
    arma::vec temp_error;

    double penalty = log(alpha) - beta * log(1.0 + depth);
    for(int i = 0; i < p; i++){ // loop over variables 
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y_sum = y_cumsum(y_cumsum.n_elem - 1);
        y_cumsum_inv = y_sum - y_cumsum;

        temp_error = BART_likelihood(ind1, ind2, y_cumsum, y_cumsum_inv, tau, sigma, alpha, penalty);
        temp_error(arma::span(1, N-2)) = temp_error(arma::span(1, N - 2)) + penalty;
        
        best_split(i) = arma::index_max(temp_error); // maximize likelihood
        least_error(i) = arma::max(temp_error);
    }
    return;
}



void split_error_3(const arma::umat& Xorder, arma::vec& y, arma::umat& best_split, arma::mat& loglike, double tau, double sigma, double depth, double alpha, double beta){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized


    int N = Xorder.n_rows;
    int p = Xorder.n_cols;

    double y_error = arma::as_scalar(arma::sum(pow(y(Xorder.col(0)) - arma::mean(y(Xorder.col(0))), 2)));

    double ee;
    
    arma::vec y_cumsum;

    double y_sum;

    arma::vec y_cumsum_inv;

    arma::vec ind1 = arma::linspace(1, N, N);
    arma::vec ind2 = arma::linspace(N-1, 0, N);
    arma::vec temp_likelihood;
    arma::uvec temp_ind;

    double penalty = log(alpha) - beta * log(1.0 + depth);

    for(int i = 0; i < p; i++){ // loop over variables 
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y_sum = y_cumsum(y_cumsum.n_elem - 1);
        y_cumsum_inv = y_sum - y_cumsum;

        loglike.col(i) = BART_likelihood(ind1, ind2, y_cumsum, y_cumsum_inv, tau, sigma, alpha, penalty);
        // temp_likelihood(arma::span(1, N-2)) = temp_likelihood(arma::span(1, N - 2)) + penalty;
        // temp_ind = arma::sort_index(temp_likelihood, "descend"); // decreasing order, pick the largest value
        // best_split(i) = arma::index_max(temp_error); // maximize likelihood
        // best_split.col(i) = temp_ind;
        // loglike.col(i) = temp_likelihood(best_split.col(i));
        
    }
    // add penalty term
    loglike.row(N - 1) = loglike.row(N - 1) - beta * log(1.0 + depth) + beta * log(depth);
    

    return;
}


arma::uvec range(int start, int end){
    // generate integers from start to end
    int N = end - start;
    arma::uvec output(N);
    for(int i = 0; i < N; i ++){
        output(i) = start + i;
    }
    return output;
}





#ifndef NoRcpp   
// instead of returning y.test, let's return trees
// this conveniently avoids the need for x.test
// loosely based on pr() 
// create an efficient list from a single tree
// tree2list calls itself recursively
Rcpp::List tree::tree2list(xinfo& xi, double center, double scale) {
  Rcpp::List res;

  // five possible scenarios
  if(l) { // tree has branches
    //double cut=xi[v][c];
    size_t var=v, cut=c;

    var++; cut++; // increment from 0-based (C) to 1-based (R) array index

    if(l->l && r->l)         // two sub-trees
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=1,
			     Rcpp::Named("left")= l->tree2list(xi, center, scale),
			     Rcpp::Named("right")=r->tree2list(xi, center, scale));   
    else if(l->l && !(r->l)) // left sub-tree and right terminal
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=2,
			     Rcpp::Named("left")= l->tree2list(xi, center, scale),
			     Rcpp::Named("right")=r->gettheta()*scale+center);    
    else if(!(l->l) && r->l) // left terminal and right sub-tree
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=3,
			     Rcpp::Named("left")= l->gettheta()*scale+center,
			     Rcpp::Named("right")=r->tree2list(xi, center, scale));
    else                     // no sub-trees 
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=0,
			     Rcpp::Named("left")= l->gettheta()*scale+center,
			     Rcpp::Named("right")=r->gettheta()*scale+center);
  }
  else // no branches
    res=Rcpp::List::create(Rcpp::Named("var")=0, // var=0 means root
			   //Rcpp::Named("cut")=0.,
			   Rcpp::Named("cut")=0,
			   Rcpp::Named("type")=0,
			   Rcpp::Named("left") =theta*scale+center,
			   Rcpp::Named("right")=theta*scale+center);

  return res;
}

// for one tree, count the number of branches for each variable
Rcpp::IntegerVector tree::tree2count(size_t nvar) {
  Rcpp::IntegerVector res(nvar);

  if(l) { // tree branches
    res[v]++;
    
    if(l->l) res+=l->tree2count(nvar); // if left sub-tree
    if(r->l) res+=r->tree2count(nvar); // if right sub-tree
  } // else no branches and nothing to do

  return res;
}
#endif

