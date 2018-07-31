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


tree::tree_p tree::search_bottom(arma::mat& Xnew, const size_t& i){

    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly
    // only look at the i-th row

    if(l == 0){
        return this;
        } // no children
    if(arma::as_scalar(Xnew(i, v)) <= c){

        return l -> search_bottom(Xnew, i);  // if smaller or equal cut point, go to left node
    } else {

        return r -> search_bottom(Xnew, i);
    }
}

tree::tree_p tree::search_bottom_std(const double* X, const size_t& i, const size_t& p, const size_t& N){
    // X is a matrix, std vector of vectors, stack by column, N rows and p columns
    // i is index of row in X to predict
    if(l == 0){
        return this;
    }
    // X[v][i], v-th column and i-th row
    // if(X[v][i] <= c){
    if(*(X + N * v + i) <= c){
        return l -> search_bottom_std(X, i, p, N);
    }else{
        return r -> search_bottom_std(X, i, p, N);
    }
}


tree::tree_p tree::search_bottom_test(arma::mat& Xnew, const size_t& i, const double* X_std, const size_t& p, const size_t& N){

    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly
    // only look at the i-th row

    if(l == 0){
        return this;
        } // no children){

    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly
    // only look at the i-th row

    if(l == 0){
        return this;
        } // no children
    if(arma::as_scalar(Xnew(i, v)) <= c){

        return l -> search_bottom(Xnew, i);  // if smaller or equal cut point, go to left node
    } else {

        return r -> search_bottom(Xnew, i);
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







void tree::grow_tree(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu){
    // this function is more randomized
    // sample from several best split points
    // tau is prior VARIANCE, do not take squares

    // set up random device
    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);

    
    if(draw_mu == true){
        
        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + Xorder.n_rows / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;

    }else{
        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2));

        this->theta_noise = this->theta; // identical to theta
    }


    if(draw_sigma == true){
        tree::tree_p top_p = this->gettop();

        // draw sigma use residual of noisy theta
        arma::vec reshat = y - fit_new_theta_noise( * top_p, X);
        // sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4)))));

        std::gamma_distribution<double> gamma_samp((reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4));

        sigma = 1.0 / gamma_samp(generator);
    }

    this->sig = sigma;
    
    if(Xorder.n_rows <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    arma::vec loglike_vec((N - 1) * p + 1);

    BART_likelihood(Xorder, y, loglike_vec, tau, sigma, depth, Nmin, alpha, beta);
    Rcpp::IntegerVector temp_ind = Rcpp::seq_len(loglike_vec.n_elem) - 1;

    std::discrete_distribution<> d2(loglike_vec.begin(), loglike_vec.end());
    size_t ind = d2(generator);
    // size_t ind = Rcpp::RcppArmadillo::sample(temp_ind, 1, false, loglike_vec)[0];
    size_t split_var = ind / (N - 1);
    size_t split_point = ind % (N - 1);

    if(ind == loglike_vec.n_elem - 1){
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
    lchild->grow_tree(y, yleft_mean, Xorder_left, X, depth, max_depth, Nmin, tau, sigma, alpha, beta, draw_sigma, draw_mu);
    tree::tree_p rchild = new tree();
    rchild->grow_tree(y, yright_mean, Xorder_right, X, depth, max_depth, Nmin, tau, sigma, alpha, beta, draw_sigma, draw_mu);
    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;
    return;
}




void tree::grow_tree_adaptive(arma::mat& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel){

    // grow a tree, users can control number of split points
    
    if(Xorder.n_rows <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    // tau is prior VARIANCE, do not take squares

    // set up random device
    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);
    
    if(draw_mu == true){

        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + Xorder.n_rows / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;

    }else{

        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2));
        this->theta_noise = this->theta; // identical to theta

    }


    if(draw_sigma == true){

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta
        arma::vec reshat = y - fit_new_theta_noise( * top_p, X);
        // sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4)))));

        std::gamma_distribution<double> gamma_samp((reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4));
        sigma = 1.0 / gamma_samp(generator);
    
    }

    this->sig = sigma;
    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    size_t ind;
    size_t split_var;
    size_t split_point;

    bool no_split = false;

    BART_likelihood_adaptive(Xorder, y, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel);

    if(no_split == true){
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
    lchild->grow_tree_adaptive(y, yleft_mean, Xorder_left, X, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive(y, yright_mean, Xorder_right, X, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel);

    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;

    return;
}



void tree::grow_tree_adaptive_std(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, double* split_var_count_pointer, size_t& mtry, const std::vector<size_t>& subset_vars){


    // grow a tree, users can control number of split points

    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if(N_Xorder <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);
    
    if(draw_mu == true){

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;

    }else{

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        this->theta_noise = this->theta; // identical to theta

    }


    if(draw_sigma == true){

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta
        
        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std( * top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    
    }

    this->sig = sigma;
    bool no_split = false;


    BART_likelihood_adaptive_std_mtry(y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars);

    if(no_split == true){
        return;
    }


    this -> v = split_var;
    this -> c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);


    split_var_count_pointer[split_var] ++;
    

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    split_xorder_std(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p);

    // free(Xorder_std);

    double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_std(yleft_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_left_std, X_std, split_var_count_pointer, mtry, subset_vars);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_std(yright_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_right_std, X_std, split_var_count_pointer, mtry, subset_vars);

    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;

    return;
}




void tree::grow_tree_adaptive_onestep(arma::mat& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel){

    // similar to grow_tree_adaptive, users can control number of split points
    // but only grow the tree to one layer deeper
    
    if(Xorder.n_rows <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    // tau is prior VARIANCE, do not take squares
    
    // set up random device
    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);

    if(draw_mu == true){

        this->theta = y_mean * Xorder.n_rows / pow(sigma, 2) / (1.0 / tau + Xorder.n_rows / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + Xorder.n_rows / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;

    }else{

        this->theta = y_mean;
        this->theta_noise = this->theta; // identical to theta
    }


    if(draw_sigma == true){

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta
        arma::vec reshat = y - fit_new_theta_noise( * top_p, X);
        // sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4)))));

        std::gamma_distribution<double> gamma_samp((reshat.n_elem + 16) / 2.0, 2.0 / as_scalar(sum(pow(reshat, 2)) + 4));
        sigma = 1.0 / gamma_samp(generator);

    }

    this->sig = sigma;


    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    size_t ind;
    size_t split_var;
    size_t split_point;

    bool no_split = false; // if true, do not split at current node

    BART_likelihood_adaptive(Xorder, y, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel);

    if(no_split == true){
        return;
    }

    
    this -> v = split_var;

    this -> c =  X(Xorder(split_point, split_var), split_var);

    arma::umat Xorder_left = arma::zeros<arma::umat>(split_point + 1, Xorder.n_cols);
    arma::umat Xorder_right = arma::zeros<arma::umat>(Xorder.n_rows - split_point - 1, Xorder.n_cols);

    split_xorder(Xorder_left, Xorder_right, Xorder, X, split_var, split_point);

    double yleft_mean = arma::as_scalar(arma::mean(y(Xorder_left.col(split_var))));
    double yright_mean = arma::as_scalar(arma::mean(y(Xorder_right.col(split_var))));


    // one step grow, both left and right child are set to a new NULL tree

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    // lchild->grow_tree_adaptive(y, yleft_mean, Xorder_left, X, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);
    tree::tree_p rchild = new tree();
    // rchild->grow_tree_adaptive(y, yright_mean, Xorder_right, X, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);

    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;
    lchild -> theta = yleft_mean;
    rchild -> theta = yright_mean;

    return;
}




void tree::grow_tree_adaptive_onestep_std(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, std::vector<double>& split_var_count){


    // grow a tree, users can control number of split points

    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if(N_Xorder <= Nmin){
        return;
    }

    if(depth >= max_depth - 1){
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);
    
    if(draw_mu == true){

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta ;

    }else{

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        this->theta_noise = this->theta; // identical to theta

    }


    if(draw_sigma == true){

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta
        
        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std( * top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    
    }

    this->sig = sigma;
    bool no_split = false;


    std::vector<size_t> subset_vars(p);

    // generate a vector {0, 1, 2, 3, ...., p - 1};
    std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);

    BART_likelihood_adaptive_std_mtry(y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars);

    if(no_split == true){
        return;
    }


    this -> v = split_var;
    this -> c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);


    split_var_count[split_var] ++;
    

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    split_xorder_std(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p);

    double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    // lchild->grow_tree_adaptive_std(yleft_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_left_std, X_std, split_var_count);
    tree::tree_p rchild = new tree();
    // rchild->grow_tree_adaptive_std(yright_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_right_std, X_std, split_var_count);

    lchild -> p = this;
    rchild -> p = this;
    this -> l = lchild;
    this -> r = rchild;

    return;
}




void split_xorder(arma::umat& Xorder_left, arma::umat& Xorder_right, arma::umat& Xorder, arma::mat& X, size_t split_var, size_t split_point){

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes

    // preserve order of other variables
    size_t N = Xorder.n_rows;
    size_t left_ix = 0; // index of current row for Xorder_left
    size_t right_ix = 0;    // index of current column for Xorder_right
    double cutpoint = X(Xorder(split_point, split_var), split_var);
    for(size_t i = 0; i < Xorder.n_cols; i ++){
        // loop over variables
        left_ix = 0;    // reset to 0, start from first row of Xorder_left
        right_ix = 0;   // reset to 0, start from first row of Xorder_right
        for(size_t j = 0; j < N; j ++){
            // loop over all observations
            if(X(Xorder(j, i), split_var) <= cutpoint){
                Xorder_left(left_ix, i) = Xorder(j, i);
                left_ix = left_ix + 1;
            }else{
                Xorder_right(right_ix, i) = Xorder(j, i);
                right_ix = right_ix + 1;   
            }
        }
    }
    return;
}


void split_xorder_std(xinfo_sizet& Xorder_left_std, xinfo_sizet& Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet& Xorder_std, const double* X_std, size_t N_y, size_t p){

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes

    // preserve order of other variables
    size_t N_Xorder = Xorder_std[0].size();
    size_t left_ix = 0;
    size_t right_ix = 0;
    
    // for(size_t i = 0; i < p; i ++){
    //     left_ix = 0;
    //     right_ix = 0;
    //     for(size_t j = 0; j < N_Xorder; j ++){
    //         // loop over all observations
    //         if(X(Xorder(j, i), split_var) <= X(Xorder(split_point, split_var), split_var)){
    //             Xorder_left(left_ix, i) = Xorder(j, i);
    //             left_ix = left_ix + 1;
    //         }else{
    //             Xorder_right(right_ix, i) = Xorder(j, i);
    //             right_ix = right_ix + 1;   
    //         }
    //     }
    // }


    double cutvalue = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    for(size_t i = 0; i < p; i ++ ){
        // loop over variables
        left_ix = 0;
        right_ix = 0;
        const double * temp_pointer = X_std + N_y * split_var;
        for(size_t j = 0; j < N_Xorder; j ++){
            // Xorder(j, i), jth row and ith column
            // look at X(Xorder(j, i), split_var)
            // X[split_var][Xorder[i][j]]
            // X[split_var][Xorder[split_var][split_point]]
            if( *(temp_pointer + Xorder_std[i][j])<= cutvalue){
                // copy a row
                // for(size_t k = 0; k < p; k ++){
                //     Xorder_left_std[i][left_ix];// = Xorder_std[i][j];
                //     left_ix = left_ix + 1;
                // }
                Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                left_ix = left_ix + 1;
            }else{
                // for(size_t k = 0; k < p; k ++){
                //     // Xorder_right[i][right_ix] = Xorder[i][j];
                //     right_ix = right_ix + 1;
                // }
                Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                right_ix = right_ix + 1;
            }

        }

    }


    return;
}


void BART_likelihood(const arma::umat& Xorder, arma::mat& y, arma::vec& loglike, double tau, double sigma, size_t depth, size_t Nmin, double alpha, double beta){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // faster than split_error_3
    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    
    arma::vec y_cumsum;

    double y_sum;

    arma::vec y_cumsum_inv;

    arma::vec n1tau = tau * arma::linspace(1, N - 1, N - 1);
    arma::vec n2tau = tau * arma::linspace(N-1, 1, N - 1);
    arma::vec temp_likelihood;
    arma::uvec temp_ind;

    double sigma2 = pow(sigma, 2);
    
    // double penalty = log(alpha) - beta * log(1.0 + depth);

    for(size_t i = 0; i < p; i++){ // loop over variables 
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y_sum = y_cumsum(y_cumsum.n_elem - 1);
        y_cumsum_inv = y_sum - y_cumsum;  // redundant copy!

        // loglike.col(i) = BART_likelihood(ind1, ind2, y_cumsum, y_cumsum_inv, tau, sigma, alpha, penalty);
        loglike(arma::span(i * (N - 1), i * (N - 1) + N - 2)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));
        // temp_likelihood(arma::span(1, N-2)) = temp_likelihood(arma::span(1, N - 2)) + penalty;
        // temp_ind = arma::sort_index(temp_likelihood, "descend"); // decreasing order, pick the largest value
        // best_split(i) = arma::index_max(temp_error); // maximize likelihood
        // best_split.col(i) = temp_ind;
        // loglike.col(i) = temp_likelihood(best_split.col(i));
        
    }
    loglike(loglike.n_elem - 1) = - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);
    // add penalty term
    // loglike.row(N - 1) = loglike.row(N - 1) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);
    
    loglike = loglike - max(loglike);
    loglike = exp(loglike);
    loglike = loglike / arma::as_scalar(arma::sum(loglike));

    // if((N - 1) > 2 * Nmin){
    //     for(size_t i = 0; i < p; i ++ ){
    //         // delete some candidates, otherwise size of the new node can be smaller than Nmin
    //         loglike(arma::span(i * (N - 1), i * (N - 1) + Nmin)).fill(0);
    //         loglike(arma::span(i * (N - 1) + N - 2 - Nmin, i * (N - 1) + N - 2)).fill(0);
    //     }
    // }
    return;
}


void BART_likelihood_adaptive(const arma::umat& Xorder, arma::mat& y, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool& no_split, size_t & split_var, size_t & split_point, bool parallel){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    size_t ind;


    std::vector< std::vector<size_t> > Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);
    for(size_t i = 0; i < p; i ++ ){
        for(size_t j = 0; j < N; j ++ ){
            Xorder_std[i][j] = Xorder(j, i);
        }
    }


    double y_sum;

    double sigma2 = pow(sigma, 2);
    
    
    if( N  <= Ncutpoints + 1 + 2 * Nmin){
        // cout << "all points" << endl;
        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points
        arma::vec n1tau = tau * arma::linspace(1, N - 1, N - 1);
        arma::vec n2tau = tau * arma::linspace(N - 1, 1, N - 1);
        arma::vec loglike((N - 1) * p + 1);
        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates       
        // note that the first Nmin and last Nmin cannot be splitpoint candidate


        // y_sum = arma::as_scalar(arma::sum(y(Xorder.col(0))));
        // arma::uvec candidate_index(N);
        // seq_gen(0, N - 1, N - 1, candidate_index);
        // std::vector<double> loglike2(loglike.n_elem);
        

        if(parallel == false){
            arma::vec y_cumsum(y.n_elem);
            arma::vec y_cumsum_inv(y.n_elem);
            // arma::vec temp_likelihood((N - 1) * p + 1);
            // arma::uvec temp_ind((N - 1) * p + 1);

            for(size_t i = 0; i < p; i++){ // loop over variables 
                y_cumsum = arma::cumsum(y.rows(Xorder.col(i)));
                y_sum = y_cumsum(y_cumsum.n_elem - 1);
                y_cumsum_inv = y_sum - y_cumsum;  // redundant copy!
                loglike(arma::span(i * (N - 1), i * (N - 1) + N - 2)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));   
            }

            // std::vector<size_t> candidate_index_2(candidate_index.n_elem);
            // std::vector<double> y_cumsum_2(N);
            // std::vector<double> y_cumsum_inv_2(N);
            
            // for(size_t m = 0; m < candidate_index.n_elem; m ++ ){
            //     candidate_index_2[m] = candidate_index[m];
            // }

            // std::vector<double> Y_sort(N);
            // double* ypointer;
            // double n1tau2;
            // double n2tau2;


            // for(size_t iii = 0; iii < p; iii ++ ){

            //     for(size_t q = 0;  q < N; q++ ){
            //         Y_sort[q] = y(Xorder(q, iii));
            //     }
            //     ypointer = &Y_sort[0];  

            //     std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum_2.begin());

            //     for(size_t k = 0; k < N; k ++ ){
            //         y_cumsum_inv_2[k] = y_sum - y_cumsum_2[k];
            //     }

            //     // calculate_y_cumsum_std(ypointer, N, y_sum, candidate_index_2, y_cumsum_2, y_cumsum_inv_2);

            //     for(size_t j = 0; j < N - 1; j ++ ){
            //         // loop over all possible cutpoints
            //         n1tau2 = (j + 1) * tau; // number of points on left side (x <= cutpoint)
            //         n2tau2 = N * tau - n1tau2; // number of points on right side (x > cutpoint)

            //         loglike2[(N - 1) * iii + j] = - 0.5 * log(n1tau2 + sigma2) - 0.5 * log(n2tau2 + sigma2) + 0.5 * tau * pow(y_cumsum_2[j], 2) / (sigma2 * (n1tau2 + sigma2)) + 0.5 * tau * pow(y_cumsum_inv_2[j], 2) / (sigma2 * (n2tau2 + sigma2));
            //     }
            // }


        }else{
            
            likelihood_evaluation_fullset like_parallel_full(y, Xorder, loglike, sigma2, tau, N, n1tau, n2tau);
            parallelFor(0, p, like_parallel_full);
            
        }

        // loglike(loglike.n_elem - 1) = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);

        loglike(loglike.n_elem - 1) = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike2[loglike2.size() - 1] = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);


        loglike = loglike - max(loglike);
        loglike = exp(loglike);
        loglike = loglike / arma::as_scalar(arma::sum(loglike));

        if((N - 1) > 2 * Nmin){
            for(size_t i = 0; i < p; i ++ ){
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                loglike(arma::span(i * (N - 1), i * (N - 1) + Nmin)).fill(0);
                loglike(arma::span(i * (N - 1) + N - 2 - Nmin, i * (N - 1) + N - 2)).fill(0);
            }
        }else{
            no_split = true;
            return;
        }


        // Rcpp::IntegerVector temp_ind2 = Rcpp::seq_len(loglike.n_elem) - 1;
        // ind = Rcpp::RcppArmadillo::sample(temp_ind2, 1, false, loglike)[0];


        std::vector<double> loglike_vec(loglike.n_elem);
        for(size_t i = 0; i < loglike.n_elem; i ++ ){
            loglike_vec[i] = loglike(i);
        }


        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(loglike_vec.begin(), loglike_vec.end());
        // sample one index of split point
        ind = d(gen); 


        split_var = ind / (N - 1);
        split_point = ind % (N - 1);

        if(ind == (N - 1) * p){no_split = true;}

        if((N - 1)<= 2 * Nmin){
            no_split = true;
        }




    }else{
        
        // cout << "some points " << endl;
        y_sum = arma::as_scalar(arma::sum(y(Xorder.col(0))));
        arma::vec loglike(Ncutpoints * p + 1);
        // otherwise, simplify calculate, use only Ncutpoints splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate
        arma::uvec candidate_index(Ncutpoints);
        // seq_gen(2, N - 2, Ncutpoints, candidate_index); // row index in Xorder to be candidates
        seq_gen(Nmin, N - Nmin, Ncutpoints, candidate_index);
        // cout << "tau" << tau << endl;
        arma::vec n1tau = tau * (1.0 + arma::conv_to<arma::vec>::from(candidate_index)); // plus 1 because the index starts from 0, we want count of observations
        arma::vec n2tau = tau * N  - n1tau;
                
        std::vector<double> loglike2(loglike.n_elem);

        // compute cumulative sum of chunks
        if(parallel == false){
            arma::vec y_cumsum(Ncutpoints);
            arma::vec y_cumsum_inv(Ncutpoints);
            arma::vec y_sort(N);

            std::vector<size_t> candidate_index_2(candidate_index.n_elem);
            std::vector<double> y_cumsum_2(Ncutpoints);
            std::vector<double> y_cumsum_inv_2(Ncutpoints);
            
            for(size_t m = 0; m < candidate_index.n_elem; m ++ ){
                candidate_index_2[m] = candidate_index[m];
            }
        
            for(size_t i = 0; i < p; i ++ ){

                y_sort = y(Xorder.col(i));

                calculate_y_cumsum(y_sort, y_sum, candidate_index, y_cumsum, y_cumsum_inv);
                
                loglike(arma::span(i * Ncutpoints, i * Ncutpoints + Ncutpoints - 1)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum, 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv, 2)/(sigma2 * (n2tau + sigma2));

            }

            // STD part

                std::vector<double> Y_sort(Xorder.n_rows);
                double* ypointer;
                double n1tau2;
                double n2tau2;
                double Ntau = Xorder.n_rows * tau;

                for(size_t iii = 0; iii < p; iii ++ ){

                    for(size_t q = 0;  q < Xorder.n_rows; q++ ){
                        Y_sort[q] = y(Xorder_std[iii][q]);
                    }
                    ypointer = &Y_sort[0];  

                    calculate_y_cumsum_std(ypointer, N, y_sum, candidate_index_2, y_cumsum_2, y_cumsum_inv_2);

                    // std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum_2.begin());

                    // for(size_t k = 0; k < Xorder.n_rows; k ++ ){
                    //     y_cumsum_inv_2[k] = y_sum - y_cumsum_2[k];
                    // }

                    for(size_t j = 0; j < Ncutpoints; j ++ ){
                        // loop over all possible cutpoints
                        n1tau2 = (candidate_index_2[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                        n2tau2 = Ntau - n1tau2; // number of points on right side (x > cutpoint)

                        loglike2[(Ncutpoints) * iii + j] = - 0.5 * log(n1tau2 + sigma2) - 0.5 * log(n2tau2 + sigma2) + 0.5 * tau * pow(y_cumsum_2[j], 2) / (sigma2 * (n1tau2 + sigma2)) + 0.5 * tau * pow(y_cumsum_inv_2[j], 2) / (sigma2 * (n2tau2 + sigma2));

                    }
                }
            
        
        }else{
            likelihood_evaluation_subset like_parallel(y, Xorder, candidate_index, loglike, sigma2, tau, y_sum, Ncutpoints, N, n1tau, n2tau);
            parallelFor(0, p, like_parallel);
        }

        // loglike(loglike.n_elem - 1) = log(Ncutpoints) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);

        loglike(loglike.n_elem - 1) = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        loglike2[loglike2.size() - 1] = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);


        // cout << loglike(loglike.n_elem - 1) << "  " << loglike2[loglike2.size() - 1] << endl;

        // take exponential, normalize values
        // double loglike2_max = *std::max_element(loglike2.begin(), loglike2.end());
        // for(size_t ii = 0; ii < loglike2.size(); ii ++ ){
        //     loglike2[ii] = exp(loglike2[ii] - loglike2_max);
        // }

        // cout << "loglike" << endl;
        // cout << loglike << endl;
        // cout << "loglike 2 " << endl;
        // cout << loglike2 << endl;
        // cout << "---------" << endl;


    
        loglike = loglike - max(loglike);
        loglike = exp(loglike);
        loglike = loglike / arma::as_scalar(arma::sum(loglike));


        // Rcpp::IntegerVector temp_ind2 = Rcpp::seq_len(loglike.n_elem) - 1;  // sample candidate ! start from 0
        // ind = Rcpp::RcppArmadillo::sample(temp_ind2, 1, false, loglike)[0];


        // cout << "prob of selected " << loglike(ind) << " prob of no split " << loglike(loglike.n_elem - 1) << " max prob " << max(loglike) << endl;


        // copy from armadillo to std for sampling use
        std::vector<double> loglike_vec(loglike.n_elem);
        for(size_t i = 0; i < loglike.n_elem; i ++ ){
            loglike_vec[i] = loglike(i);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(loglike_vec.begin(), loglike_vec.end());
        // // sample one index of split point
        ind = d(gen); 


        split_var = ind / Ncutpoints;

        split_point = candidate_index(ind % Ncutpoints);

        if(ind == (Ncutpoints) * p){no_split = true;}

    }

    // cout << "select variable " << split_var << endl;
    return;
}



void BART_likelihood_adaptive_std(std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool& no_split, size_t & split_var, size_t & split_point, bool parallel, std::vector<size_t>& subset_vars){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t ind;
    size_t N_Xorder = N;

    double y_sum;
    double sigma2 = pow(sigma, 2);
    

    if( N  <= Ncutpoints + 1 + 2 * Nmin){

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates       
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double* ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;
        std::vector<double> loglike((N_Xorder - 1) * p + 1);
        std::vector<double> y_cumsum(N_Xorder);
        std::vector<double> y_cumsum_inv(N_Xorder);



        if(parallel == false){

            for(size_t i = 0; i < p; i++){ 
                // loop over variables 
                for(size_t q = 0;  q < N_Xorder; q++ ){
                    Y_sort[q] = y_std[Xorder_std[i][q]];
                }
                ypointer = &Y_sort[0];  

                std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum.begin());

                y_sum = y_cumsum[y_cumsum.size() - 1]; // last one 

                for(size_t k = 0; k < N_Xorder; k ++ ){
                    y_cumsum_inv[k] = y_sum - y_cumsum[k];
                }

                for(size_t j = 0; j < N_Xorder - 1; j ++ ){
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau; // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder-1) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));
                }
            }
            
        }else{
            
            // parallel computing 

            // likelihood_evaluation_fullset like_parallel_full(y, Xorder, loglike, sigma2, tau, N, n1tau, n2tau);
            // parallelFor(0, p, like_parallel_full);
            
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for(size_t ii = 0; ii < loglike.size(); ii ++ ){
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

   
        if((N - 1) > 2 * Nmin){
            for(size_t i = 0; i < p; i ++ ){
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);

            }

        }else{
            no_split = true;
            return;
        }


        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(loglike.begin(), loglike.end());
        // sample one index of split point

        ind = d(gen); 

        split_var = ind / (N - 1);
        split_point = ind % (N - 1);

        if(ind == (N - 1) * p){no_split = true;}

        if((N - 1)<= 2 * Nmin){
            no_split = true;

        }


    
    }else{
        
        std::vector<double> loglike(Ncutpoints * p + 1);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);
        std::vector<double> y_cumsum_inv(Ncutpoints);
        


        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        if(parallel == false){

            std::vector<double> Y_sort(N_Xorder);
            double* ypointer;
            double n1tau;
            double n2tau;
            double Ntau = N_Xorder * tau;
            for(size_t iii = 0; iii < p; iii ++ ){

                for(size_t q = 0;  q < N_Xorder; q++ ){
                    Y_sort[q] = y_std[Xorder_std[iii][q]];
                }
                ypointer = &Y_sort[0];  
                
                if(iii == 0){
                    y_sum = sum_vec(Y_sort);

                }

                calculate_y_cumsum_std(ypointer, N, y_sum, candidate_index, y_cumsum, y_cumsum_inv);

                for(size_t j = 0; j < Ncutpoints; j ++ ){
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau; // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints) * iii + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

                }
            }
        
        }else{

            // parallel computing

            // likelihood_evaluation_subset like_parallel(y, Xorder, candidate_index, loglike, sigma2, tau, y_sum, Ncutpoints, N, n1tau, n2tau);
            // parallelFor(0, p, like_parallel);
        }


        loglike[loglike.size() - 1] = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for(size_t ii = 0; ii < loglike.size(); ii ++ ){
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen); 

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if(ind == (Ncutpoints) * p){no_split = true;}

    }

    return;
}




void BART_likelihood_adaptive_std_mtry(std::vector<double>& y_std, xinfo_sizet& Xorder_std, const double* X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool& no_split, size_t & split_var, size_t & split_point, bool parallel, const std::vector<size_t>& subset_vars){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t ind;
    size_t N_Xorder = N;

    double y_sum;
    double sigma2 = pow(sigma, 2);
    

    if( N  <= Ncutpoints + 1 + 2 * Nmin){

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates       
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double* ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // initialize log likelihood at -INFINITY
        std::vector<double> loglike((N_Xorder - 1) * p + 1, -INFINITY);
        std::vector<double> y_cumsum(N_Xorder);
        std::vector<double> y_cumsum_inv(N_Xorder);


        // std::vector<double> loglike_2(loglike.size(), -INFINITY);



        if(parallel == false){

            // for(size_t i = 0; i < p; i++){ 
            for(auto&& i : subset_vars){
                // loop over variables 
                for(size_t q = 0;  q < N_Xorder; q++ ){
                    Y_sort[q] = y_std[Xorder_std[i][q]];
                }
                ypointer = &Y_sort[0];  

                std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum.begin());

                y_sum = y_cumsum[y_cumsum.size() - 1]; // last one 

                for(size_t k = 0; k < N_Xorder; k ++ ){
                    y_cumsum_inv[k] = y_sum - y_cumsum[k];
                }

                for(size_t j = 0; j < N_Xorder - 1; j ++ ){
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau; // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder-1) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));
                }
            }


            
        }else{
            
            // parallel computing 

            likelihood_fullset_std like_parallel_full(y_std, Xorder_std, N_Xorder, subset_vars, tau, Ntau, sigma2, loglike);
            parallelFor(0, subset_vars.size(), like_parallel_full);
            
        }


        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // normalize loglike, take exp to likelihood
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for(size_t ii = 0; ii < loglike.size(); ii ++ ){
            // if a variable is not selected, take exp will becomes 0
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

   
        if((N - 1) > 2 * Nmin){
            // for(size_t i = 0; i < p; i ++ ){
            for(auto&& i : subset_vars){
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);

            }

        }else{
            no_split = true;
            return;
        }


        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(loglike.begin(), loglike.end());
        // sample one index of split point

        ind = d(gen); 

        split_var = ind / (N - 1);
        split_point = ind % (N - 1);

        if(ind == (N - 1) * p){no_split = true;}

        if((N - 1)<= 2 * Nmin){
            no_split = true;

        }


    
    }else{
        
        // initialize loglikelihood at -INFINITY
        std::vector<double> loglike(Ncutpoints * p + 1, -INFINITY);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);
        std::vector<double> y_cumsum_inv(Ncutpoints);
        


        // std::vector<double> loglike_2(loglike.size(), -INFINITY);


        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        
        double Ntau = N_Xorder * tau;

        if(parallel == false){

            bool firstrun = true;   // flag of the first loop
            std::vector<double> Y_sort(N_Xorder);
            double* ypointer;
            double n1tau;
            double n2tau;
            // for(size_t i = 0; i < p; i ++ ){
            for(auto&& i : subset_vars){

                for(size_t q = 0;  q < N_Xorder; q++ ){
                    Y_sort[q] = y_std[Xorder_std[i][q]];
                }
                ypointer = &Y_sort[0];  
                
                if(firstrun){
                    y_sum = sum_vec(Y_sort);
                    firstrun = false;
                }

                calculate_y_cumsum_std(ypointer, Y_sort.size(), y_sum, candidate_index, y_cumsum, y_cumsum_inv);

                for(size_t j = 0; j < Ncutpoints; j ++ ){
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau; // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

                }
            }



        
        }else{

            // parallel computing


            likelihood_subset_std like_parallel(y_std, Xorder_std, N_Xorder, Ncutpoints, subset_vars, tau, sigma2, candidate_index, loglike);
            parallelFor(0, subset_vars.size(), like_parallel);
        }


        // no split option
        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike_2[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike_2 = loglike_2 - loglike;
        
        // cout << sum_squared(loglike_2) << endl;
        // loglike = loglike_2;

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        // double loglike_2_max = *std::max_element(loglike_2.begin(), loglike_2.end());
        for(size_t ii = 0; ii < loglike.size(); ii ++ ){
            loglike[ii] = exp(loglike[ii] - loglike_max);
            // loglike_2[ii] = exp(loglike_2[ii] - loglike_2_max);
        }
        // loglike_2 = loglike_2 - loglike;
        // cout << sum_squared(loglike_2) << endl;


        // loglike = loglike_2;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen); 

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if(ind == (Ncutpoints) * p){no_split = true;}

    }

    return;
}




void cumulative_sum_std(std::vector<double>& y_cumsum, std::vector<double>& y_cumsum_inv, double& y_sum, double* y, xinfo_sizet& Xorder, size_t& i, size_t& N){
    // y_cumsum is the output cumulative sum
    // y is the original data
    // Xorder is sorted index matrix
    // i means take the i-th column of Xorder
    // N is length of y and y_cumsum
    if(N > 1){
        y_cumsum[0] = y[Xorder[i][0]];
        for(size_t j = 1; j < N; j++){
            y_cumsum[j] = y_cumsum[j - 1] + y[Xorder[i][j]];
        }
    }else{
        y_cumsum[0] = y[Xorder[i][0]];
    }
    y_sum = y_cumsum[N - 1];

    for(size_t j = 1; j < N; j ++){
        y_cumsum_inv[j] = y_sum - y_cumsum[j];
    }
    return;
}


arma::uvec range(size_t start, size_t end){
    // generate integers from start to end
    size_t N = end - start;
    arma::uvec output(N);
    for(size_t i = 0; i < N; i ++){
        output(i) = start + i;
    }
    return output;
}


void tree::prune_regrow(arma::mat& y, double y_mean, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double& tau, double& sigma, double& alpha, double& beta, bool draw_sigma, bool draw_mu,bool parallel){


    tree::npv bv;       // vector of pointers to bottom nodes
    tree::npv bv2;      // vector of pointers to nodes without grandchild
    this->getbots(bv);
    this->getnogs(bv2);

    // create a map object, key is pointer to endnodes and value is sufficient statistics
    std::map< tree::tree_p, arma::vec > sufficient_stat;
    size_t N_endnodes = bv.size();
    size_t N_obs = y.n_elem;
    for(size_t i = 0; i < N_endnodes; i ++ ){
        // initialize the object
        sufficient_stat[bv[i]] = arma::zeros<arma::vec>(2);
        // 2 dimension vector, first element for counts, second element for sum of y fall in that node
    }

    // set up random device
    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);

    // create a map to save *index* of ys fall into each endnodes
    std::map<tree::tree_p, std::vector<size_t> > node_y_ind;

    arma::vec y_ind(N_obs); // same length as y, the value is ID of end nodes associated with
    tree::tree_p temp_pointer;

    // loop over observations
    for(size_t i = 0; i < N_obs; i ++ ){
        temp_pointer = this->search_bottom(X, i);
        // if(sufficient_stat.count(temp_pointer)){ // for protection, since the map is intialized, not necessary
            // update sufficient statistics
            // if statement for protection
            // y_ind[i] = temp_pointer->nid();
            sufficient_stat[temp_pointer][0] += 1;      // add one for count
            sufficient_stat[temp_pointer][1] += arma::as_scalar(y.row(i));   // sum of y, add it to the sum
        // }
        // if(node_y_ind.count(temp_pointer)){
            node_y_ind[temp_pointer].push_back(i);      // push the index to node_y_ind vector
        // }
    }


    // update theta (mu) of all other nodes
    for(size_t ii = 0; ii < bv.size(); ii ++ ){
                    bv[ii]->theta = sufficient_stat[bv[ii]][1] / sufficient_stat[bv[ii]][0] * sufficient_stat[bv[ii]][0] / pow(sigma, 2) / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];
    }
    

    if(1){
        //////////////////////////////////////////////////////////////
        // prune the tree, looks to be correct
        //////////////////////////////////////////////////////////////
        double left_loglike = 0.0;
        double right_loglike = 0.0;
        double total_loglike = 0.0;
        double sigma2 = pow(sigma, 2);
        bool test;
        // arma::vec loglike(2);
        std::vector<double> loglike(2);       // only two elements, collapse or not
        // Rcpp::IntegerVector temp_ind2 = Rcpp::seq_len(2) - 1;
        size_t ind ;

        size_t left_nid;
        size_t right_nid;
        size_t current_nid;


        size_t keep_count = 0;


        // while(0.95 * bv2.size() > keep_count && bv2.size() > 1){
        // while(bv2.size() > 8){
            // stop loop untile 95% ends node cannot be collapsed, might prune too much

            bv2.clear();    // clear the vector of no grandchild nodes  
            this->getnogs(bv2);      
            keep_count = 0; // count of nodes not collapsed

            cout << " number of no grands " << bv2.size() << endl;

            // cout << bv2.size() << endl;
            for(size_t i = 0; i < bv2.size(); i ++ ){
                
                left_loglike = - 0.5 * log(sufficient_stat[bv2[i]->l][0] * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(sufficient_stat[bv2[i]->l][1], 2) / (sigma2 * (sufficient_stat[bv2[i]->l][0] * tau + sigma2)) - beta * log(1.0 + bv2[i]->l->depth()) + beta * log(bv2[i]->l->depth()) + log(1.0 - alpha) - log(alpha);
                right_loglike = - 0.5 * log(sufficient_stat[bv2[i]->r][0] * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(sufficient_stat[bv2[i]->r][1], 2) / (sigma2 * (sufficient_stat[bv2[i]->r][0] * tau + sigma2)) - beta * log(1.0 + bv2[i]->r->depth()) + beta * log(bv2[i]->r->depth()) + log(1.0 - alpha) - log(alpha);
                total_loglike = - 0.5 * log((sufficient_stat[bv2[i]->l][0] + sufficient_stat[bv2[i]->r][0]) * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow((sufficient_stat[bv2[i]->l][1] + sufficient_stat[bv2[i]->r][1]), 2) / (sigma2 * ((sufficient_stat[bv2[i]->l][0] + sufficient_stat[bv2[i]->r][0]) * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + bv2[i]->depth(), -1.0 * beta)) - log(alpha) + beta * log(1.0 + bv2[i]->depth()); //- beta * log(1.0 + bv2[i]->depth()) + beta * log(bv2[i]->depth()) + log(1.0 - alpha) - log(alpha);

                if(total_loglike > left_loglike + right_loglike){
                    loglike[0] = 1.0;
                    loglike[1] = exp(left_loglike + right_loglike - total_loglike);
                }else{
                    loglike[0] = exp(total_loglike - left_loglike - right_loglike);
                    loglike[1] = 1.0;                    
                }

                std::random_device rd;
                std::mt19937 gen(rd());
                std::discrete_distribution<> d(loglike.begin(), loglike.end());
                // // sample one index of split point
                ind = d(gen); 
                // if ind == 0, collapse the current node
                if(ind == 0){

                    // collapse two child node, create a new key in node_y_ind for the parent
                    // the current no grandchild node becomes a new end node
                    std::merge(node_y_ind[bv2[i]->l].begin(), node_y_ind[bv2[i]->l].end(), node_y_ind[bv2[i]->r].begin(), node_y_ind[bv2[i]->r].end(), std::back_inserter(node_y_ind[bv2[i]]));

                    // also need to update sufficient_stat map
                    sufficient_stat[bv2[i]] = sufficient_stat[bv2[i]->l] + sufficient_stat[bv2[i]->r];

                    free(bv2[i]->l);
                    free(bv2[i]->r);
                    bv2[i]->l = 0;
                    bv2[i]->r = 0;

                }else{
                    keep_count ++ ;
                }
        }
    }



    if(1){

        //////////////////////////////////////////////////////////////
        // regrow the tree
        //////////////////////////////////////////////////////////////

        // update list of bottom nodes
        bv.clear();
        this->getbots(bv);      // get bottom nodes

        size_t node_id;

        arma::uvec y_ind_subnode;

        arma::uvec temp_ind;    // index for y in the current nodes

        arma::mat temp_X;       // X values

        arma::umat temp_Xorder; // Xorder values

        double temp_y_mean;     // mean of y 

        // size_t new_maxdepth;


        arma::mat temp_y;

        
        // cout << "number of end nodes "<< bv.size()  <<endl;

        for(size_t i = 0; i < bv.size(); i ++ ){
            // loop over all endnodes

            // if(node_y_ind[bv[i]].size() > Nmin){

            // create Xorder for the subnode

                temp_ind.set_size(node_y_ind[bv[i]].size());

                for(size_t j = 0; j < node_y_ind[bv[i]].size(); j ++ ){
                    // copy indicies of y falls in node bv[i]
                    temp_ind[j] = node_y_ind[bv[i]][j];
                } 

                temp_X = X.rows(temp_ind);          // take corresponding rows of X

                temp_Xorder.set_size(temp_X.n_rows, temp_X.n_cols);         // sort corresponding X, create order matrix

                // cout << "aaaaaaa !!!" << temp_X.n_cols << endl;
                for(size_t t = 0; t < temp_X.n_cols; t++){
                    // cout << "t " << t << endl;
                    temp_Xorder.col(t) = arma::sort_index(temp_X.col(t));
                }

                // cout << "ok 3" << endl;
                temp_y_mean = arma::as_scalar(mean(y.rows(temp_ind)));

                temp_y = y.rows(temp_ind);

                // if(max_depth > bv[i]->depth()){
                //     new_maxdepth = max_depth - bv[i]->depth();
                // }else{
                //     new_maxdepth = 0;
                // }
                bv[i]->grow_tree_adaptive(temp_y, temp_y_mean, temp_Xorder, temp_X, bv[i]->depth(), max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel);
            // }

        }
        cout << "after regrows " << this->treesize() << endl;

        
    }
                                cout << "------------------------------------" << endl;


    return;
}




void tree::one_step_grow(arma::mat& y, double y_mean, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double& tau, double& sigma, double& alpha, double& beta, bool draw_sigma, bool draw_mu, bool parallel){

    tree::npv bv;       // vector of pointers to bottom nodes
    tree::npv bv2;      // vector of pointers to nodes without grandchild
    this->getbots(bv);
    this->getnogs(bv2);

    // create a map object, key is pointer to endnodes and value is sufficient statistics
    std::map< tree::tree_p, arma::vec > sufficient_stat;
    size_t N_endnodes = bv.size();
    size_t N_obs = y.n_elem;
    for(size_t i = 0; i < N_endnodes; i ++ ){
        // initialize the object
        sufficient_stat[bv[i]] = arma::zeros<arma::vec>(2);
        // 2 dimension vector, first element for counts, second element for sum of y fall in that node
    }

    // set up random device
    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);


    // create a map to save *index* of ys fall into each endnodes
    std::map<tree::tree_p, std::vector<size_t> > node_y_ind;

    arma::vec y_ind(N_obs); // same length as y, the value is ID of end nodes associated with
    tree::tree_p temp_pointer;

    // loop over observations
    for(size_t i = 0; i < N_obs; i ++ ){
        temp_pointer = this->search_bottom(X, i);
        // if(sufficient_stat.count(temp_pointer)){ // for protection, since the map is intialized, not necessary
            // update sufficient statistics
            // if statement for protection
            // y_ind[i] = temp_pointer->nid();
            sufficient_stat[temp_pointer][0] += 1;      // add one for count
            sufficient_stat[temp_pointer][1] += arma::as_scalar(y.row(i));   // sum of y, add it to the sum
        // }
        // if(node_y_ind.count(temp_pointer)){
            node_y_ind[temp_pointer].push_back(i);      // push the index to node_y_ind vector
        // }
    }


    //////////////////////////////////////////////////////////////
    // regrow the tree
    //////////////////////////////////////////////////////////////
    // update list of bottom nodes
    bv.clear();
    this->getbots(bv);      // get bottom nodes

    size_t node_id;

    arma::uvec y_ind_subnode;

    arma::uvec temp_ind;    // index for y in the current nodes

    arma::mat temp_X;       // X values

    arma::umat temp_Xorder; // Xorder values

    double temp_y_mean;     // mean of y 

    arma::mat temp_y;

    size_t i = 0;

    // randomly select one leaf to grow

    std::vector<double> prob(bv.size() , 1.0 / (double) bv.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(prob.begin(), prob.end());
    // // sample one index of split point
    i = d(gen);     

    temp_ind.set_size(node_y_ind[bv[i]].size());

    for(size_t j = 0; j < node_y_ind[bv[i]].size(); j ++ ){
        // copy indicies of y falls in node bv[i]
        temp_ind[j] = node_y_ind[bv[i]][j];
    }

    temp_X = X.rows(temp_ind);          // take corresponding rows of X

    temp_Xorder.set_size(temp_X.n_rows, temp_X.n_cols);         // sort corresponding X, create order matrix

    for(size_t t = 0; t < temp_X.n_cols; t++){
        temp_Xorder.col(t) = arma::sort_index(temp_X.col(t));
    }

    temp_y_mean = arma::as_scalar(mean(y.rows(temp_ind)));

    temp_y = y.rows(temp_ind);

    bv[i]->grow_tree_adaptive_onestep(temp_y, temp_y_mean, temp_Xorder, temp_X, bv[i]->depth(), max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel);

    if(draw_mu){
        // update theta (mu) of all other nodes
        for(size_t ii = 0; ii < bv.size(); ii ++ ){
            if(ii!=i){
                bv[ii]->theta = sufficient_stat[bv[ii]][1] / sufficient_stat[bv[ii]][0] * sufficient_stat[bv[ii]][0] / pow(sigma, 2) / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];
            }
        }
    }

    return;
}




void tree::one_step_prune(arma::mat& y, double y_mean, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double& tau, double& sigma, double& alpha, double& beta, bool draw_sigma, bool draw_mu, bool parallel){


    tree::npv bv;       // vector of pointers to bottom nodes
    tree::npv bv2;      // vector of pointers to nodes without grandchild
    this->getbots(bv);
    this->getnogs(bv2);

    // create a map object, key is pointer to endnodes and value is sufficient statistics
    std::map< tree::tree_p, arma::vec > sufficient_stat;
    size_t N_endnodes = bv.size();
    size_t N_obs = y.n_elem;
    for(size_t i = 0; i < N_endnodes; i ++ ){
        // initialize the object
        sufficient_stat[bv[i]] = arma::zeros<arma::vec>(2);
        // 2 dimension vector, first element for counts, second element for sum of y fall in that node
    }

    // set up random device
    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0,1.0);

    // create a map to save *index* of ys fall into each endnodes
    std::map<tree::tree_p, std::vector<size_t> > node_y_ind;

    arma::vec y_ind(N_obs); // same length as y, the value is ID of end nodes associated with
    tree::tree_p temp_pointer;

    // loop over observations
    for(size_t i = 0; i < N_obs; i ++ ){
        temp_pointer = this->search_bottom(X, i);
        sufficient_stat[temp_pointer][0] += 1;      // add one for count
        sufficient_stat[temp_pointer][1] += arma::as_scalar(y.row(i));   // sum of y, add it to the sum

        node_y_ind[temp_pointer].push_back(i);      // push the index to node_y_ind vector
        
    }


    //////////////////////////////////////////////////////////////
    // prune the tree, looks to be correct
    //////////////////////////////////////////////////////////////
    double left_loglike = 0.0;
    double right_loglike = 0.0;
    double total_loglike = 0.0;
    double sigma2 = pow(sigma, 2);
    bool test;
    // arma::vec loglike(2);
    std::vector<double> loglike(2);       // only two elements, collapse or not
    // Rcpp::IntegerVector temp_ind2 = Rcpp::seq_len(2) - 1;
    size_t ind ;

    size_t left_nid;
    size_t right_nid;
    size_t current_nid;

    bv2.clear();    // clear the vector of no grandchild nodes  
    this->getnogs(bv2);      

    // randomly sample one nograndchild node to collapse

    std::vector<double> prob(bv2.size() , 1.0 / (double) bv2.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(prob.begin(), prob.end());
    // // sample one index of split point
    size_t i = d(gen);   

    size_t current_depth = bv2[i]->depth();


    // left_loglike = - 0.5 * log(sufficient_stat[bv2[i]->l][0] * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(sufficient_stat[bv2[i]->l][1], 2) / (sigma2 * (sufficient_stat[bv2[i]->l][0] * tau + sigma2)) - beta * log(1.0 + bv2[i]->l->depth()) + beta * log(bv2[i]->l->depth()) + log(1.0 - alpha) - log(alpha);
    // right_loglike = - 0.5 * log(sufficient_stat[bv2[i]->r][0] * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(sufficient_stat[bv2[i]->r][1], 2) / (sigma2 * (sufficient_stat[bv2[i]->r][0] * tau + sigma2)) - beta * log(1.0 + bv2[i]->r->depth()) + beta * log(bv2[i]->r->depth()) + log(1.0 - alpha) - log(alpha);

    // total_loglike = - 0.5 * log((sufficient_stat[bv2[i]->l][0] + sufficient_stat[bv2[i]->r][0]) * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow((sufficient_stat[bv2[i]->l][1] + sufficient_stat[bv2[i]->r][1]), 2) / (sigma2 * ((sufficient_stat[bv2[i]->l][0] + sufficient_stat[bv2[i]->r][0]) * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + bv2[i]->depth(), -1.0 * beta)) - log(alpha) + beta * log(1.0 + bv2[i]->depth());//- beta * log(1.0 + bv2[i]->depth()) + beta * log(bv2[i]->depth()) + log(1.0 - alpha) - log(alpha);


    left_loglike = - 0.5 * log(sufficient_stat[bv2[i]->l][0] * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(sufficient_stat[bv2[i]->l][1], 2) / (sigma2 * (sufficient_stat[bv2[i]->l][0] * tau + sigma2)); 
    right_loglike = - 0.5 * log(sufficient_stat[bv2[i]->r][0] * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(sufficient_stat[bv2[i]->r][1], 2) / (sigma2 * (sufficient_stat[bv2[i]->r][0] * tau + sigma2));

    total_loglike = - 0.5 * log((sufficient_stat[bv2[i]->l][0] + sufficient_stat[bv2[i]->r][0]) * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow((sufficient_stat[bv2[i]->l][1] + sufficient_stat[bv2[i]->r][1]), 2) / (sigma2 * ((sufficient_stat[bv2[i]->l][0] + sufficient_stat[bv2[i]->r][0]) * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + current_depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + current_depth);//- beta * log(1.0 + bv2[i]->depth()) + beta * log(bv2[i]->depth()) + log(1.0 - alpha) - log(alpha);



        // loglike[0] = total_loglike;
        // loglike[1] = left_loglike + right_loglike;
        // loglike = exp(loglike - max(loglike));
        // ind = Rcpp::RcppArmadillo::sample(temp_ind2, 1, false, loglike)[0];

    if(total_loglike > left_loglike + right_loglike){
        loglike[0] = 1.0;
        loglike[1] = exp(left_loglike + right_loglike - total_loglike);
    }else{
        loglike[0] = exp(total_loglike - left_loglike - right_loglike);
        loglike[1] = 1.0;                    
    }

    std::discrete_distribution<> d2(loglike.begin(), loglike.end());
        // // sample one index of split point
    ind = d2(gen); 

    // update theta (mu) of all other nodes
    if(draw_mu){
        for(size_t ii = 0; ii < bv.size(); ii ++ ){
            if(ii!=i){
                bv[ii]->theta = sufficient_stat[bv[ii]][1] / sufficient_stat[bv[ii]][0] * sufficient_stat[bv[ii]][0] / pow(sigma, 2) / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2))) * normal_samp(generator);//Rcpp::rnorm(1, 0, 1)[0];
            }
        }
    }
    if(ind == 0){
        // collapse two child node, create a new key in node_y_ind for the parent
        // the current no grandchild node becomes a new end node
        std::merge(node_y_ind[bv2[i]->l].begin(), node_y_ind[bv2[i]->l].end(), node_y_ind[bv2[i]->r].begin(), node_y_ind[bv2[i]->r].end(), std::back_inserter(node_y_ind[bv2[i]]));

        // also need to update sufficient_stat map
        sufficient_stat[bv2[i]] = sufficient_stat[bv2[i]->l] + sufficient_stat[bv2[i]->r];

        free(bv2[i]->l);
        free(bv2[i]->r);
        bv2[i]->l = 0;
        bv2[i]->r = 0;
        // cout << bv2[i]->theta << endl;

    }


    return;
}




void tree::sample_theta(arma::mat& y, arma::mat& X, double& tau, double& sigma, bool draw_mu){

    if(draw_mu){
        tree::npv bv;       // vector of pointers to bottom nodes
        tree::npv bv2;      // vector of pointers to nodes without grandchild
        this->getbots(bv);
        this->getnogs(bv2);

        // set up random device
        std::default_random_engine generator;
        std::normal_distribution<double> normal_samp(0.0,1.0);

        // create a map object, key is pointer to endnodes and value is sufficient statistics
        std::map< tree::tree_p, arma::vec > sufficient_stat;
        size_t N_endnodes = bv.size();
        size_t N_obs = y.n_elem;
        for(size_t i = 0; i < N_endnodes; i ++ ){
            // initialize the object
            sufficient_stat[bv[i]] = arma::zeros<arma::vec>(2);
            // 2 dimension vector, first element for counts, second element for sum of y fall in that node
        }

        // create a map to save *index* of ys fall into each endnodes
        std::map<tree::tree_p, std::vector<size_t> > node_y_ind;

        arma::vec y_ind(N_obs); // same length as y, the value is ID of end nodes associated with
        tree::tree_p temp_pointer;

        // loop over observations
        for(size_t i = 0; i < N_obs; i ++ ){
            temp_pointer = this->search_bottom(X, i);
                sufficient_stat[temp_pointer][0] += 1;      // add one for count
                sufficient_stat[temp_pointer][1] += arma::as_scalar(y.row(i));   // sum of y, add it to the sum
            // }
            // if(node_y_ind.count(temp_pointer)){
                node_y_ind[temp_pointer].push_back(i);      // push the index to node_y_ind vector
            // }
        }
        // update theta (mu) of all other nodes
        for(size_t ii = 0; ii < bv.size(); ii ++ ){
            bv[ii]->theta = sufficient_stat[bv[ii]][1] / sufficient_stat[bv[ii]][0] * sufficient_stat[bv[ii]][0] / pow(sigma, 2) / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + sufficient_stat[bv[ii]][0] / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];
        }
    
    }
    return;
}




// arma::vec BART_likelihood_function(arma::vec& n1, arma::vec& n2, arma::vec& s1, arma::vec& s2, double& tau, double& sigma, double& alpha, double& penalty){
//     // log - likelihood of BART model
//     // n1 is number of observations in group 1
//     // s1 is sum of group 1
//     arma::vec result;
//     double sigma2 = pow(sigma, 2);
//     arma::vec n1tau = n1 * tau;
//     arma::vec n2tau = n2 * tau;
//     result = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(s1, 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(s2, 2)/(sigma2 * (n2tau + sigma2));
//     // result(result.n_elem - 1) = result(result.n_elem - 1) - penalty;
//     // double temp = result.min();
//     // result(0) = temp;
//     // result(result.n_elem - 1) = temp;

//     // the last entry is probability of no split
//     // alpha is the prior probability of split, multiply it
//     // result = result + log(alpha);
//     result(result.n_elem - 1) = result(result.n_elem - 1) + log(1.0 - alpha) - log(alpha);
//     return result;
// }


// void split_error(const arma::umat& Xorder, arma::vec& y, arma::uvec& best_split, arma::vec& least_error){
//     // regular CART algorithm, compute sum of squared loss error

//     size_t N = Xorder.n_rows;
//     size_t p = Xorder.n_cols;
//     // arma::mat errormat = arma::zeros(N, p);
//     // loop over all variables and observations and compute error

//     double y_error = arma::as_scalar(arma::sum(pow(y(Xorder.col(0)) - arma::mean(y(Xorder.col(0))), 2)));

//     double ee;
//     double temp_error = y_error;
//     arma::vec y_cumsum(y.n_elem);
//     arma::vec y2_cumsum(y.n_elem);

//     y_cumsum = arma::cumsum(y(Xorder.col(0)));
//     y2_cumsum = arma::cumsum(pow(y(Xorder.col(0)), 2));

//     double y_sum = y_cumsum(y_cumsum.n_elem - 1);
//     double y2_sum = y2_cumsum(y2_cumsum.n_elem - 1);

//     arma::vec y2 = pow(y, 2);
//     for(size_t i = 0; i < p; i++){ // loop over variables 
//         temp_error = 100.0;
//         y_cumsum = arma::cumsum(y(Xorder.col(i)));
//         y2_cumsum = arma::cumsum(pow(y(Xorder.col(i)), 2));
//         for(size_t j = 1; j < N - 1; j++){ // loop over cutpoints

//             ee = y2_cumsum(j) - pow(y_cumsum(j), 2) / (double) (j+ 1) + y2_sum - y2_cumsum(j) - pow((y_sum - y_cumsum(j)), 2) / (double) (N - j - 1) ;

//             if(ee < temp_error || temp_error == 100.0){
//                 best_split(i) = j; // Xorder(j,i) coordinate;
//                 temp_error = ee;
//                 least_error(i) = ee;
//             }
//         }
//     }
//     return;
// }





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

