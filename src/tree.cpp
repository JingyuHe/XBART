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
#include "treefuns.h"
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
//get pointer to the top tree
tree::tree_p tree::gettop()
{
    if(!p){
        return this; 
    }else{
        return p->gettop();
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

   // original BART function, v and c are index of split point in xinfo& xi


   if(l==0) return this; //no children
   if(x[v] <= xi[v][c]) {
       // if smaller than or equals to the cutpoint, go to left child

      return l->bn(x,xi);
   } else {
       // if greater than cutpoint, go to right child
      return r->bn(x,xi);
   }
}

tree::tree_p tree::bn_std(double *x)
{
    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly

    if(l==0) return this;
    if(x[v] <= c){
        return l-> bn_std(x);
    }else{
        return r->bn_std(x);
    }
}


tree::tree_p tree::search_bottom(arma::mat& Xnew){

    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly

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
void tree::rg(size_t v, size_t* L, size_t* U)
{
   if(this->p==0)  {
      return;
   }
   if((this->p)->v == v) { //does my parent use v?
      if(this == p->l) { //am I left or right child
         if((size_t)(p->c) <= (*U)) *U = (p->c)-1;
         p->rg(v,L,U);
      } else {
         if((size_t)(p->c) >= *L) *L = (p->c)+1;
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







void tree::grow_tree(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, double tau, double sigma, double alpha, double beta, arma::vec& residual, bool draw_sigma, bool draw_mu){
    // this function is more randomized
    // sample from several best split points

    // tau is prior VARIANCE, do not take squares
    
    if(draw_mu == true){
        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + Xorder.n_rows / pow(sigma, 2))) * Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;
    }else{
        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2));

        this->theta_noise = this->theta; // identical to theta
    }
    // this->theta_noise = this->theta;


    // cout << "ok" << endl;    

    if(draw_sigma == true){
        tree::tree_p top_p = this->gettop();

        // draw sigma use residual of noisy theta
        arma::vec reshat = residual - fit_new_theta_noise( * top_p, X);
        sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4)))));
    }

    this->sig = sigma;

    // theta = y_mean / pow(sigma, 2) * 1.0 / (1.0 / pow(tau, 2) + 1.0 / pow(sigma, 2));

    // cout << Xorder.n_rows << endl;
    if(Xorder.n_rows <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    // arma::umat best_split = arma::zeros<arma::umat>(Xorder.n_rows, Xorder.n_cols);
    


    arma::vec loglike_vec((N - 1) * p + 1);

    BART_likelihood(Xorder, y, loglike_vec, tau, sigma, depth, Nmin, alpha, beta);
    Rcpp::IntegerVector temp_ind = Rcpp::seq_len(loglike_vec.n_elem) - 1;
    size_t ind = Rcpp::RcppArmadillo::sample(temp_ind, 1, false, loglike_vec)[0];
    size_t split_var = ind / (N - 1);
    size_t split_point = ind % (N - 1);

    if(ind == loglike_vec.n_elem - 1){
        // cout << "early termination" << endl;
        return;
    }

    
    this -> v = split_var;

    this -> c =  X(Xorder(split_point, split_var), split_var);

    // if(split_point + 1 < Nmin || Xorder.n_rows - split_point - 1 < Nmin){
    //     return;
    // }


    arma::umat Xorder_left = arma::zeros<arma::umat>(split_point + 1, Xorder.n_cols);
    arma::umat Xorder_right = arma::zeros<arma::umat>(Xorder.n_rows - split_point - 1, Xorder.n_cols);

    split_xorder(Xorder_left, Xorder_right, Xorder, X, split_var, split_point);



    double yleft_mean = arma::as_scalar(arma::mean(y(Xorder_left.col(split_var))));
    double yright_mean = arma::as_scalar(arma::mean(y(Xorder_right.col(split_var))));

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree(y, yleft_mean, Xorder_left, X, depth, max_depth, Nmin, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu);
    tree::tree_p rchild = new tree();
    rchild->grow_tree(y, yright_mean, Xorder_right, X, depth, max_depth, Nmin, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu);
    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;



    return;
}


void tree::grow_tree_adaptive(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, double tau, double sigma, double alpha, double beta, arma::vec& residual, bool draw_sigma, bool draw_mu){
    // this function is more randomized
    // sample from several best split points

    // tau is prior VARIANCE, do not take squares
    
    if(draw_mu == true){
        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + Xorder.n_rows / pow(sigma, 2))) * Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;
    }else{
        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2));

        this->theta_noise = this->theta; // identical to theta
    }
    // this->theta_noise = this->theta;


    // cout << "ok" << endl;    

    if(draw_sigma == true){
        tree::tree_p top_p = this->gettop();

        // draw sigma use residual of noisy theta
        arma::vec reshat = residual - fit_new_theta_noise( * top_p, X);
        sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4)))));
    }

    this->sig = sigma;

    // theta = y_mean / pow(sigma, 2) * 1.0 / (1.0 / pow(tau, 2) + 1.0 / pow(sigma, 2));

    // cout << Xorder.n_rows << endl;
    if(Xorder.n_rows <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    // arma::umat best_split = arma::zeros<arma::umat>(Xorder.n_rows, Xorder.n_cols);
    


    arma::vec loglike_vec((N - 1) * p + 1);

    BART_likelihood_adaptive(Xorder, y, loglike_vec, tau, sigma, depth, Nmin, alpha, beta);
    
    Rcpp::IntegerVector temp_ind = Rcpp::seq_len(loglike_vec.n_elem) - 1;
    size_t ind = Rcpp::RcppArmadillo::sample(temp_ind, 1, false, loglike_vec)[0];
    size_t split_var = ind / (N - 1);
    size_t split_point = ind % (N - 1);

    if(ind == loglike_vec.n_elem - 1){
        // cout << "early termination" << endl;
        return;
    }

    
    this -> v = split_var;

    this -> c =  X(Xorder(split_point, split_var), split_var);

    // if(split_point + 1 < Nmin || Xorder.n_rows - split_point - 1 < Nmin){
    //     return;
    // }


    arma::umat Xorder_left = arma::zeros<arma::umat>(split_point + 1, Xorder.n_cols);
    arma::umat Xorder_right = arma::zeros<arma::umat>(Xorder.n_rows - split_point - 1, Xorder.n_cols);

    split_xorder(Xorder_left, Xorder_right, Xorder, X, split_var, split_point);



    double yleft_mean = arma::as_scalar(arma::mean(y(Xorder_left.col(split_var))));
    double yright_mean = arma::as_scalar(arma::mean(y(Xorder_right.col(split_var))));

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree(y, yleft_mean, Xorder_left, X, depth, max_depth, Nmin, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu);
    tree::tree_p rchild = new tree();
    rchild->grow_tree(y, yright_mean, Xorder_right, X, depth, max_depth, Nmin, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu);
    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;



    return;
}


void tree::grow_tree_std(double* y, double& y_mean, xinfo_sizet& Xorder, double* X, size_t N, size_t p, size_t depth, size_t max_depth, size_t Nmin, double tau, double sigma, double alpha, double beta, double* residual, bool draw_sigma, bool draw_mu){

    // X is a p * N matrix of data, stacked by row, the (i, j)th entry of the matrix is  *(X+p*i+j)

    if(draw_mu == true){
        this->theta = y_mean * N / pow(sigma, 2) / (1.0 / tau + N / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N / pow(sigma, 2))) * Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;
    }else{
        this->theta = y_mean * N / pow(sigma, 2) / (1.0 / tau + N / pow(sigma, 2));

        this->theta_noise = this->theta; // identical to theta
    }

    if(draw_sigma == true){
        tree::tree_p top_p = this->gettop();
        std::vector<double> fptemp(N);
        // draw sigma use residual of noisy theta
        fit_noise_std( * top_p, p, N, X, fptemp);
        // arma::vec reshat = residual - fit_new_theta_noise( * top_p, X);
        // sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4)))));
    }
    this->sig = sigma;

    // theta = y_mean / pow(sigma, 2) * 1.0 / (1.0 / pow(tau, 2) + 1.0 / pow(sigma, 2));
    if(N <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }


    std::vector<double> loglike_vec((N - 1) * p + 1);

    BART_likelihood_std(N, p, Xorder, y, loglike_vec, tau, sigma, depth, alpha, beta);


    double loglike_vec_max = *std::max_element(loglike_vec.begin(), loglike_vec.end());
    for(size_t i = 0; i < loglike_vec.size(); i ++ ){
        // take exponents and normalize probability vector
        loglike_vec[i] = exp(loglike_vec[i] - loglike_vec_max);
    }

    // sample from multinomial distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(loglike_vec.begin(), loglike_vec.end());
    // sample one index of split point
    size_t ind = d(gen); 

    // find corresponding split point index in Xorder
    size_t split_var = ind / (N - 1);
    size_t split_point = ind % (N - 1);


    if(ind == loglike_vec.size() - 1){
        // cout << "early termination" << endl;
        return;
    }

    // save split variable and value to tree object
    this -> v = split_var;

    // Xorder[split_var][split_point]
    this -> c =  *(X + p * split_var + Xorder[split_var][split_point]);


    xinfo_sizet Xorder_left = ini_xinfo_sizet(split_point + 1, p);
    xinfo_sizet Xorder_right = ini_xinfo_sizet(N - split_point - 1, p);


    split_xorder_std(Xorder_left, Xorder_right, Xorder, X, split_var, split_point, N , p);


    double yleft_mean;// = arma::as_scalar(arma::mean(y(Xorder_left.col(split_var))));
    double yright_mean;// = arma::as_scalar(arma::mean(y(Xorder_right.col(split_var))));

    size_t N_left = split_point + 1;
    size_t N_right = N - split_point - 1;

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_std(y, yleft_mean, Xorder_left, X, N_left, p, depth, max_depth, Nmin, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_std(y, yright_mean, Xorder_right, X, N_right, p, depth, max_depth, Nmin, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu);
    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;

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
      res=Rcpp::List::create(Rcpp::Named("var")=(size_t)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(size_t)cut,
			     Rcpp::Named("type")=1,
			     Rcpp::Named("left")= l->tree2list(xi, center, scale),
			     Rcpp::Named("right")=r->tree2list(xi, center, scale));   
    else if(l->l && !(r->l)) // left sub-tree and right terminal
      res=Rcpp::List::create(Rcpp::Named("var")=(size_t)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(size_t)cut,
			     Rcpp::Named("type")=2,
			     Rcpp::Named("left")= l->tree2list(xi, center, scale),
			     Rcpp::Named("right")=r->gettheta()*scale+center);    
    else if(!(l->l) && r->l) // left terminal and right sub-tree
      res=Rcpp::List::create(Rcpp::Named("var")=(size_t)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(size_t)cut,
			     Rcpp::Named("type")=3,
			     Rcpp::Named("left")= l->gettheta()*scale+center,
			     Rcpp::Named("right")=r->tree2list(xi, center, scale));
    else                     // no sub-trees 
      res=Rcpp::List::create(Rcpp::Named("var")=(size_t)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(size_t)cut,
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

