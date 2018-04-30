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
    


    arma::vec loglike((N - 1) * p + 1);

    size_t ind;
    size_t split_var;
    size_t split_point;

    BART_likelihood(Xorder, y, tau, sigma, depth, Nmin, alpha, beta, ind, split_var, split_point);


    if(ind == (N - 1) * p + 1 - 1){
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


void tree::grow_tree_adaptive(arma::vec& y, double y_mean, arma::umat& Xorder, arma::mat& X, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, arma::vec& residual, bool draw_sigma, bool draw_mu){
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

    size_t ind;
    size_t split_var;
    size_t split_point;

    BART_likelihood_adaptive(Xorder, y, loglike_vec, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, ind, split_var, split_point);
    

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






void split_xorder(arma::umat& Xorder_left, arma::umat& Xorder_right, arma::umat& Xorder, arma::mat& X, size_t split_var, size_t split_point){
    // preserve order of other variables
    size_t N = Xorder.n_rows;
    size_t left_ix = 0;
    size_t right_ix = 0;
    for(size_t i = 0; i < Xorder.n_cols; i ++){
            left_ix = 0;
            right_ix = 0;
            for(size_t j = 0; j < N; j ++){
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


void split_xorder_std(xinfo_sizet& Xorder_left, xinfo_sizet& Xorder_right, xinfo_sizet& Xorder, double *  X, size_t split_var, size_t split_point, size_t N, size_t p){
    // N is number of rows for Xorder
    size_t left_ix = 0;
    size_t right_ix = 0;
    for(size_t i = 0; i < p; i ++ ){
        left_ix = 0;
        right_ix = 0;
        for(size_t j = 0; j < N; j ++){
            // Xorder(j, i), jth row and ith column
            // look at X(Xorder(j, i), split_var)
            // X[split_var][Xorder[i][j]]
            // X[split_var][Xorder[split_var][split_point]]
            if( *(X + p * split_var + Xorder[i][j])<= *(X + p * split_var + Xorder[split_var][split_point])){
                // copy a row
                for(size_t k = 0; k < p; k ++){
                    Xorder_left[i][left_ix] = Xorder[i][j];
                    left_ix = left_ix + 1;
                }
            }else{
                for(size_t k = 0; k < p; k ++){
                    Xorder_right[i][right_ix] = Xorder[i][j];
                    right_ix = right_ix + 1;
                }
            }
        }
    }
    return;
}





arma::vec BART_likelihood_function(arma::vec& n1, arma::vec& n2, arma::vec& s1, arma::vec& s2, double& tau, double& sigma, double& alpha, double& penalty){
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

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
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
    for(size_t i = 0; i < p; i++){ // loop over variables 
        temp_error = 100.0;
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y2_cumsum = arma::cumsum(pow(y(Xorder.col(i)), 2));
        for(size_t j = 1; j < N - 1; j++){ // loop over cutpoints

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





void BART_likelihood(const arma::umat& Xorder, arma::vec& y, double tau, double sigma, size_t depth, size_t Nmin, double alpha, double beta, size_t& ind, size_t& split_var, size_t& split_point){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // faster than split_error_3
    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    arma::vec loglike((N - 1) * p + 1);
    arma::vec y_cumsum(y.n_elem);

    double y_sum;

    arma::vec y_cumsum_inv(y.n_elem);

    arma::vec n1tau = tau * arma::linspace(1, N - 1, N - 1);
    arma::vec n2tau = tau * arma::linspace(N-1, 1, N - 1);
    arma::vec temp_likelihood((N - 1) * p + 1);
    arma::uvec temp_ind((N - 1) * p + 1);

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

    if((N - 1) > 2 * Nmin){
        for(size_t i = 0; i < p; i ++ ){
            // delete some candidates, otherwise size of the new node can be smaller than Nmin
            loglike(arma::span(i * (N - 1), i * (N - 1) + Nmin)).fill(0);
            loglike(arma::span(i * (N - 1) + N - 2 - Nmin, i * (N - 1) + N - 2)).fill(0);
        }
    }

    Rcpp::IntegerVector temp_ind2 = Rcpp::seq_len(loglike.n_elem) - 1;
    ind = Rcpp::RcppArmadillo::sample(temp_ind2, 1, false, loglike)[0];
    split_var = ind / (N - 1);
    split_point = ind % (N - 1);

    return;
}





void BART_likelihood_adaptive(const arma::umat& Xorder, arma::vec& y, arma::vec& loglike, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, size_t& ind, size_t & split_var, size_t & split_point){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // faster than split_error_3
    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    

    double y_sum;

    arma::vec n1tau = tau * arma::linspace(1, N - 1, N - 1);
    arma::vec n2tau = tau * arma::linspace(N-1, 1, N - 1);

    double sigma2 = pow(sigma, 2);
    
    if( N - 1 - 2 * Nmin <= Ncutpoints){
        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates       
        // note that the first Nmin and last Nmin cannot be splitpoint candidate
        arma::vec y_cumsum(y.n_elem);
        arma::vec y_cumsum_inv(y.n_elem);
        arma::vec temp_likelihood((N - 1) * p + 1);
        arma::uvec temp_ind((N - 1) * p + 1);

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

        if((N - 1) > 2 * Nmin){
            for(size_t i = 0; i < p; i ++ ){
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                loglike(arma::span(i * (N - 1), i * (N - 1) + Nmin)).fill(0);
                loglike(arma::span(i * (N - 1) + N - 2 - Nmin, i * (N - 1) + N - 2)).fill(0);
            }
        }

        Rcpp::IntegerVector temp_ind2 = Rcpp::seq_len(loglike.n_elem) - 1;
        ind = Rcpp::RcppArmadillo::sample(temp_ind2, 1, false, loglike)[0];
        split_var = ind / (N - 1);
        split_point = ind % (N - 1);

    }else{
        // otherwise, simplify calculate, use only Ncutpoints splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate
        arma::uvec candidate_index(Ncutpoints);
        seq_gen(Nmin, N - 1 - Nmin, Ncutpoints, candidate_index); // row index in Xorder to be candidates
        // compute cumulative sum of chunks
        arma::vec y_cumsum_chunk(Ncutpoints+1);
        cumsum_chunk(y, candidate_index, y_cumsum_chunk);
        arma::vec y_cumsum(Ncutpoints);
        arma::vec y_cumsum_inv(Ncutpoints);

        for(size_t i = 0; i < p; i ++ ){
            calculate_y_cumsum(y_cumsum_chunk, y_cumsum, y_cumsum_inv);
            // loglike() = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));
        }


    }
    return;
}



void BART_likelihood_std(size_t N, size_t p, xinfo_sizet& Xorder, double* y, std::vector<double>& loglike, double& tau, double& sigma, size_t& depth, double& alpha, double& beta){
    std::vector<double> y_cumsum(N);
    double y_sum;
    std::vector<double> y_cumsum_inv(N);
    std::vector<double> n1tau;
    std::vector<double> n2tau;
    std::vector<size_t> temp_ind;
    double sigma2 = pow(sigma, 2);
    for(size_t i = 0; i < p; i ++){
        // calculate cumulative sum, reorder y as the i-th column of Xorder matrix (i-th variable)
        cumulative_sum_std(y_cumsum, y_cumsum_inv, y_sum, y, Xorder, i, N);
        y_sum = y_cumsum[N - 1]; // the last one
        // y_cumsum_inv = 
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

