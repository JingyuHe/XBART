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
#include <chrono>

using namespace std;
using namespace chrono;

//--------------------
// node id
size_t tree::nid() const
{
    if (!p)
        return 1; //if you don't have a parent, you are the top
    if (this == p->l)
        return 2 * (p->nid()); //if you are a left child
    else
        return 2 * (p->nid()) + 1; //else you are a right child
}
//--------------------
tree::tree_p tree::getptr(size_t nid)
{
    if (this->nid() == nid)
        return this; //found it
    if (l == 0)
        return 0; //no children, did not find it
    tree_p lp = l->getptr(nid);
    if (lp)
        return lp; //found on left
    tree_p rp = r->getptr(nid);
    if (rp)
        return rp; //found on right
    return 0;      //never found it
}
//--------------------
//add children to  bot node nid
bool tree::birth(size_t nid, size_t v, size_t c, double thetal, double thetar)
{
    tree_p np = getptr(nid);
    if (np == 0)
    {
        cout << "error in birth: bottom node not found\n";
        return false; //did not find note with that nid
    }
    if (np->l != 0)
    {
        cout << "error in birth: found node has children\n";
        return false; //node is not a bottom node
    }

    //add children to bottom node np
    tree_p l = new tree;
    l->theta = thetal;
    tree_p r = new tree;
    r->theta = thetar;
    np->l = l;
    np->r = r;
    np->v = v;
    np->c = c;
    l->p = np;
    r->p = np;

    return true;
}
//--------------------
//depth of node
size_t tree::depth()
{
    if (!p)
        return 0; //no parents
    else
        return (1 + p->depth());
}
//--------------------
//tree size
size_t tree::treesize()
{
    if (l == 0)
        return 1; //if bottom node, tree size is 1
    else
        return (1 + l->treesize() + r->treesize());
}
//--------------------
//node type
char tree::ntype()
{
    //t:top, b:bottom, n:no grandchildren, i:internal
    if (!p)
        return 't';
    if (!l)
        return 'b';
    if (!(l->l) && !(r->l))
        return 'n';
    return 'i';
}
//--------------------
//print out tree(pc=true) or node(pc=false) information
void tree::pr(bool pc)
{
    size_t d = depth();
    size_t id = nid();

    size_t pid;
    if (!p)
        pid = 0; //parent of top node
    else
        pid = p->nid();

    std::string pad(2 * d, ' ');
    std::string sp(", ");
    if (pc && (ntype() == 't'))
        cout << "tree size: " << treesize() << std::endl;
    cout << pad << "(id,parent): " << id << sp << pid;
    cout << sp << "(v,c): " << v << sp << c;
    cout << sp << "theta: " << theta;
    cout << sp << "type: " << ntype();
    cout << sp << "depth: " << depth();
    cout << sp << "pointer: " << this << std::endl;

    if (pc)
    {
        if (l)
        {
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
    if (nb == 0)
    {
        cout << "error in death, nid invalid\n";
        return false;
    }
    if (nb->isnog())
    {
        delete nb->l;
        delete nb->r;
        nb->l = 0;
        nb->r = 0;
        nb->v = 0;
        nb->c = 0;
        nb->theta = theta;
        return true;
    }
    else
    {
        cout << "error in death, node is not a nog node\n";
        return false;
    }
}
//--------------------
//is the node a nog node
bool tree::isnog()
{
    bool isnog = true;
    if (l)
    {
        if (l->l || r->l)
            isnog = false; //one of the children has children.
    }
    else
    {
        isnog = false; //no children
    }
    return isnog;
}
//--------------------
size_t tree::nnogs()
{
    if (!l)
        return 0; //bottom node
    if (l->l || r->l)
    { //not a nog
        return (l->nnogs() + r->nnogs());
    }
    else
    { //is a nog
        return 1;
    }
}
//--------------------
size_t tree::nbots()
{
    if (l == 0)
    { //if a bottom node
        return 1;
    }
    else
    {
        return l->nbots() + r->nbots();
    }
}
//--------------------
//get bottom nodes
void tree::getbots(npv &bv)
{
    if (l)
    { //have children
        l->getbots(bv);
        r->getbots(bv);
    }
    else
    {
        bv.push_back(this);
    }
}
//--------------------
//get nog nodes
void tree::getnogs(npv &nv)
{
    if (l)
    { //have children
        if ((l->l) || (r->l))
        { //have grandchildren
            if (l->l)
                l->getnogs(nv);
            if (r->l)
                r->getnogs(nv);
        }
        else
        {
            nv.push_back(this);
        }
    }
}
//--------------------
//get pointer to the top tree
tree::tree_p tree::gettop()
{
    if (!p)
    {
        return this;
    }
    else
    {
        return p->gettop();
    }
}
//--------------------
//get all nodes
void tree::getnodes(npv &v)
{
    v.push_back(this);
    if (l)
    {
        l->getnodes(v);
        r->getnodes(v);
    }
}
void tree::getnodes(cnpv &v) const
{
    v.push_back(this);
    if (l)
    {
        l->getnodes(v);
        r->getnodes(v);
    }
}
//--------------------
tree::tree_p tree::bn(double *x, xinfo &xi)
{

    // original BART function, v and c are index of split point in xinfo& xi

    if (l == 0)
        return this; //no children
    if (x[v] <= xi[v][c])
    {
        // if smaller than or equals to the cutpoint, go to left child

        return l->bn(x, xi);
    }
    else
    {
        // if greater than cutpoint, go to right child
        return r->bn(x, xi);
    }
}

tree::tree_p tree::bn_std(double *x)
{
    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly

    if (l == 0)
        return this;
    if (x[v] <= c)
    {
        return l->bn_std(x);
    }
    else
    {
        return r->bn_std(x);
    }
}

tree::tree_p tree::search_bottom(arma::mat &Xnew, const size_t &i)
{

    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly
    // only look at the i-th row

    if (l == 0)
    {
        return this;
    } // no children
    if (arma::as_scalar(Xnew(i, v)) <= c)
    {

        return l->search_bottom(Xnew, i); // if smaller or equal cut point, go to left node
    }
    else
    {

        return r->search_bottom(Xnew, i);
    }
}

tree::tree_p tree::search_bottom_std(const double *X, const size_t &i, const size_t &p, const size_t &N)
{
    // X is a matrix, std vector of vectors, stack by column, N rows and p columns
    // i is index of row in X to predict
    if (l == 0)
    {
        return this;
    }
    // X[v][i], v-th column and i-th row
    // if(X[v][i] <= c){
    if (*(X + N * v + i) <= c)
    {
        return l->search_bottom_std(X, i, p, N);
    }
    else
    {
        return r->search_bottom_std(X, i, p, N);
    }
}

tree::tree_p tree::search_bottom_test(arma::mat &Xnew, const size_t &i, const double *X_std, const size_t &p, const size_t &N)
{

    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly
    // only look at the i-th row

    if (l == 0)
    {
        return this;
    } // no children){

    // v is variable to split, c is raw value
    // not index in xinfo, so compare x[v] with c directly
    // only look at the i-th row

    if (l == 0)
    {
        return this;
    } // no children
    if (arma::as_scalar(Xnew(i, v)) <= c)
    {

        return l->search_bottom(Xnew, i); // if smaller or equal cut point, go to left node
    }
    else
    {

        return r->search_bottom(Xnew, i);
    }
}

//--------------------
//find region for a given variable
void tree::rg(size_t v, size_t *L, size_t *U)
{
    if (this->p == 0)
    {
        return;
    }
    if ((this->p)->v == v)
    { //does my parent use v?
        if (this == p->l)
        { //am I left or right child
            if ((size_t)(p->c) <= (*U))
                *U = (p->c) - 1;
            p->rg(v, L, U);
        }
        else
        {
            if ((size_t)(p->c) >= *L)
                *L = (p->c) + 1;
            p->rg(v, L, U);
        }
    }
    else
    {
        p->rg(v, L, U);
    }
}
//--------------------
//cut back to one node
void tree::tonull()
{
    size_t ts = treesize();
    //loop invariant: ts>=1
    while (ts > 1)
    { //if false ts=1
        npv nv;
        getnogs(nv);
        for (size_t i = 0; i < nv.size(); i++)
        {
            delete nv[i]->l;
            delete nv[i]->r;
            nv[i]->l = 0;
            nv[i]->r = 0;
        }
        ts = treesize(); //make invariant true
    }
    theta = 0.0;
    v = 0;
    c = 0;
    p = 0;
    l = 0;
    r = 0;
}
//--------------------
//copy tree tree o to tree n
void tree::cp(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
{
    if (n->l)
    {
        cout << "cp:error node has children\n";
        return;
    }

    n->theta = o->theta;
    n->v = o->v;
    n->c = o->c;

    if (o->l)
    { //if o has children
        n->l = new tree;
        (n->l)->p = n;
        cp(n->l, o->l);
        n->r = new tree;
        (n->r)->p = n;
        cp(n->r, o->r);
    }
}
//--------------------------------------------------
//operators
tree &tree::operator=(const tree &rhs)
{
    if (&rhs != this)
    {
        tonull();       //kill left hand side (this)
        cp(this, &rhs); //copy right hand side to left hand side
    }
    return *this;
}
//--------------------------------------------------
//functions
std::ostream &operator<<(std::ostream &os, const tree &t)
{
    tree::cnpv nds;
    t.getnodes(nds);
    os << nds.size() << std::endl;
    for (size_t i = 0; i < nds.size(); i++)
    {
        //  os << " a new node "<< endl;
        os << nds[i]->nid() << " ";
        os << nds[i]->getv() << " ";
        os << nds[i]->getc() << " ";
        os << nds[i]->gettheta() << std::endl;
    }
    return os;
}
std::istream &operator>>(std::istream &is, tree &t)
{
    size_t tid, pid;                    //tid: id of current node, pid: parent's id
    std::map<size_t, tree::tree_p> pts; //pointers to nodes indexed by node id
    size_t nn;                          //number of nodes

    t.tonull(); // obliterate old tree (if there)

    //read number of nodes----------
    is >> nn;
    if (!is)
    {
        //cout << ">> error: unable to read number of nodes" << endl;
        return is;
    }

    // The idea is to dump string to a lot of node_info structure first, then link them as a tree, by nid

    //read in vector of node information----------
    std::vector<node_info> nv(nn);
    for (size_t i = 0; i != nn; i++)
    {
        is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta;
        if (!is)
        {
            //cout << ">> error: unable to read node info, on node  " << i+1 << endl;
            return is;
        }
    }

    //first node has to be the top one
    pts[1] = &t; //careful! this is not the first pts, it is pointer of id 1.
    t.setv(nv[0].v);
    t.setc(nv[0].c);
    t.settheta(nv[0].theta);
    t.p = 0;

    // cout << "nvszie " << nv.size() << endl;

    //now loop through the rest of the nodes knowing parent is already there.
    for (size_t i = 1; i != nv.size(); i++)
    {
        tree::tree_p np = new tree;
        np->v = nv[i].v;
        np->c = nv[i].c;
        np->theta = nv[i].theta;
        tid = nv[i].id;
        pts[tid] = np;
        pid = tid / 2;
        if (tid % 2 == 0)
        { //left child has even id
            pts[pid]->l = np;
        }
        else
        {
            pts[pid]->r = np;
        }
        np->p = pts[pid];
    }
    return is;
}
//--------------------
//add children to bot node *np
void tree::birthp(tree_p np, size_t v, size_t c, double thetal, double thetar)
{
    tree_p l = new tree;
    l->theta = thetal;
    tree_p r = new tree;
    r->theta = thetar;
    np->l = l;
    np->r = r;
    np->v = v;
    np->c = c;
    l->p = np;
    r->p = np;
}
//--------------------
//kill children of  nog node *nb
void tree::deathp(tree_p nb, double theta)
{
    delete nb->l;
    delete nb->r;
    nb->l = 0;
    nb->r = 0;
    nb->v = 0;
    nb->c = 0;
    nb->theta = theta;
}

void tree::grow_tree_adaptive_std(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double *split_var_count_pointer, size_t &mtry, const std::vector<size_t> &subset_vars, double &run_time)
{

    // grow a tree, users can control number of split points

    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if (N_Xorder <= Nmin)
    {
        return;
    }

    if (depth >= max_depth - 1)
    {
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    if (draw_mu == true)
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta;
    }
    else
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        this->theta_noise = this->theta; // identical to theta
    }

    if (draw_sigma == true)
    {

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta

        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std(*top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    }

    this->sig = sigma;
    bool no_split = false;

    BART_likelihood_adaptive_std_mtry_old(y_mean * N_Xorder, y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars);

    if (no_split == true)
    {
        return;
    }

    this->v = split_var;
    this->c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);

    split_var_count_pointer[split_var]++;

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    // system_clock::time_point start;
    // system_clock::time_point end;
    // start = system_clock::now();
    // split_xorder_std_old(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p);
    // double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    // double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);
    // end = system_clock::now();
    // auto duration = duration_cast<microseconds>(end - start);
    // double running_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    //     cout << " ----- ---- " << endl;
    //     cout << "running time 1 " << duration.count() << endl;

    auto start = system_clock::now();
    double yleft_mean_std = 0.0;
    double yright_mean_std = 0.0;

    split_xorder_std(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p, yleft_mean_std, yright_mean_std, y_mean, y_std);

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
    double running_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    // duration = duration_cast<microseconds>(end - start);
    // cout << "running time 2 " << duration.count() << endl;
    // free(Xorder_std);
    // cout<< "left " << yleft_mean_std << " " << yleft_mean2 << endl;
    // cout<< "right "<< yright_mean_std << " " << yright_mean2 << endl;

    double running_time_left = 0.0;
    double running_time_right = 0.0;

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_std(yleft_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_left_std, X_std, split_var_count_pointer, mtry, subset_vars, running_time_left);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_std(yright_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_right_std, X_std, split_var_count_pointer, mtry, subset_vars, running_time_right);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    run_time = run_time + running_time + running_time_left + running_time_right;

    return;
}


void tree::grow_tree_adaptive_std_mtrywithinnode(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, double &run_time, Rcpp::IntegerVector &var_index_candidate, bool &use_all, Rcpp::NumericMatrix& split_count_all_tree, Rcpp::NumericVector &mtry_weight_current_tree, Rcpp::NumericVector &split_count_current_tree)
{

    // grow a tree, users can control number of split points

    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if (N_Xorder <= Nmin)
    {
        return;
    }

    if (depth >= max_depth - 1)
    {
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    if (draw_mu == true)
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta;
    }
    else
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        this->theta_noise = this->theta; // identical to theta
    }

    if (draw_sigma == true)
    {

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta

        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std(*top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    }

    this->sig = sigma;
    bool no_split = false;

    std::vector<size_t> subset_vars;

    if(use_all){    
        subset_vars.resize(p);
        std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);
    }else{
        // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));
        subset_vars = Rcpp::as<std::vector<size_t> >(sample(var_index_candidate, mtry, false, mtry_weight_current_tree));
    }

    BART_likelihood_adaptive_std_mtry_old(y_mean * N_Xorder, y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars);

    if (no_split == true)
    {
        return;
    }

    this->v = split_var;
    this->c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);

    // split_var_count_pointer[split_var]++;

    split_count_current_tree[split_var] = split_count_current_tree[split_var] + 1;

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    // system_clock::time_point start;
    // system_clock::time_point end;
    // start = system_clock::now();
    // split_xorder_std_old(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p);
    // double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    // double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);
    // end = system_clock::now();
    // auto duration = duration_cast<microseconds>(end - start);
    // double running_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    //     cout << " ----- ---- " << endl;
    //     cout << "running time 1 " << duration.count() << endl;

    auto start = system_clock::now();
    double yleft_mean_std = 0.0;
    double yright_mean_std = 0.0;

    split_xorder_std(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p, yleft_mean_std, yright_mean_std, y_mean, y_std);

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
    double running_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    // duration = duration_cast<microseconds>(end - start);
    // cout << "running time 2 " << duration.count() << endl;
    // free(Xorder_std);
    // cout<< "left " << yleft_mean_std << " " << yleft_mean2 << endl;
    // cout<< "right "<< yright_mean_std << " " << yright_mean2 << endl;

    double running_time_left = 0.0;
    double running_time_right = 0.0;

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_std_mtrywithinnode(yleft_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_left_std, X_std, mtry, running_time_left, var_index_candidate, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_std_mtrywithinnode(yright_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_right_std, X_std, mtry, running_time_right, var_index_candidate, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    run_time = run_time + running_time + running_time_left + running_time_right;

    return;
}




void tree::grow_tree_adaptive_std_all(double y_mean, double y_sum, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double *split_var_count_pointer, size_t &mtry, const std::vector<size_t> &subset_vars, double &run_time, xinfo_sizet &Xorder_next_index, xinfo_sizet &Xorder_full, std::vector<size_t> &Xorder_firstline, double &old_time, double &new_time)
{

    // grow a tree, users can control number of split points

    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if (N_Xorder <= Nmin)
    {
        return;
    }

    if (depth >= max_depth - 1)
    {
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    if (draw_mu == true)
    {

        theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        theta_noise = theta;
    }
    else
    {

        theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        theta_noise = theta; // identical to theta
    }

    if (draw_sigma == true)
    {

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta

        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std(*top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    }

    sig = sigma;
    bool no_split = false;

    // double cutvalue = 0.0;
    // double cutvalue2 ;

    // cout << "ok1 " << endl;
    BART_likelihood_adaptive_std_mtry_all(y_mean * N_Xorder, y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, v, split_point, parallel, subset_vars, Xorder_full, Xorder_next_index, Xorder_firstline, N_y, c, old_time, new_time);
    // cout << "ok2 " << endl;
    if (no_split == true)
    {
        return;
    }

    // this->v = split_var;
    // v = split_var;
    split_var = v;
    // c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    // cutvalue = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    // c = cutvalue;
    // cutvalue2 = cutvalue;
    // this->c = cutvalue2;
    // if(cutvalue != this->c){
    // cout << "cut value  " << cutvalue << "  " << c << " variable " << split_var << endl;
    // }
    split_var_count_pointer[split_var]++;

    size_t N_Xorder_left = split_point + 1;
    size_t N_Xorder_right = N_Xorder - split_point - 1;

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    // system_clock::time_point start;
    // system_clock::time_point end;
    // start = system_clock::now();
    // split_xorder_std_old(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p);
    // double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    // double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);
    // end = system_clock::now();
    // auto duration = duration_cast<microseconds>(end - start);
    // double running_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    //     cout << " ----- ---- " << endl;
    //     cout << "running time 1 " << duration.count() << endl;

    std::vector<size_t> Xorder_right_firstline(p, 0);
    std::vector<size_t> Xorder_left_firstline(p, 0);

    // std::fill(Xorder_right_firstline.begin(), Xorder_right_firstline.end(), 0);
    // std::fill(Xorder_left_firstline.begin(), Xorder_left_firstline.end(), 0);

    auto start = system_clock::now();
    double yleft_mean_std = 0.0;
    double yright_mean_std = 0.0;
    split_xorder_std(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p, yleft_mean_std, yright_mean_std, y_mean, y_std);
    auto end = system_clock::now();
    auto duration = duration_cast<std::chrono::nanoseconds>(end - start);
    old_time = old_time + duration.count() / 1000;

    start = system_clock::now();
    split_xorder_std_newXorder_old(this->c, split_var, split_point, Xorder_next_index, X_std, N_y, p, N_Xorder, N_Xorder_left, N_Xorder_right, Xorder_firstline, Xorder_left_firstline, Xorder_right_firstline, Xorder_full, yleft_mean_std, yright_mean_std, y_mean, y_std);
    end = system_clock::now();
    duration = duration_cast<std::chrono::nanoseconds>(end - start);
    new_time = new_time + duration.count() / 1000;

    // duration = duration_cast<microseconds>(end - start);
    // cout << "running time 2 " << duration.count() << endl;
    // free(Xorder_std);
    // cout<< "left " << yleft_mean_std << " " << yleft_mean2 << endl;
    // cout<< "right "<< yright_mean_std << " " << yright_mean2 << endl;
    double yleft_sum = yleft_mean_std * N_Xorder_left;
    double yright_sum = yright_mean_std * N_Xorder_right;

    double running_time_left = 0.0;
    double running_time_right = 0.0;

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_std_all(yleft_mean_std, yleft_sum, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_left_std, X_std, split_var_count_pointer, mtry, subset_vars, running_time_left, Xorder_next_index, Xorder_full, Xorder_left_firstline, old_time, new_time);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_std_all(yright_mean_std, yright_sum, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_right_std, X_std, split_var_count_pointer, mtry, subset_vars, running_time_right, Xorder_next_index, Xorder_full, Xorder_right_firstline, old_time, new_time);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    // run_time = run_time  + running_time_left + running_time_right;

    return;
}

void tree::grow_tree_adaptive_std_newXorder(double y_mean, double y_sum, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, size_t N_Xorder, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_next_index, std::vector<size_t> &Xorder_firstline, const double *X_std, double *split_var_count_pointer, size_t &mtry, const std::vector<size_t> &subset_vars, xinfo_sizet &Xorder_full)
{

    // grow a tree, users can control number of split points

    // size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_full.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if (N_Xorder <= Nmin)
    {
        return;
    }

    if (depth >= max_depth - 1)
    {
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    if (draw_mu == true)
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta;
    }
    else
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        this->theta_noise = this->theta; // identical to theta
    }

    if (draw_sigma == true)
    {

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta

        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std(*top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    }

    this->sig = sigma;
    bool no_split = false;

    double cutvalue = 0.0;
    BART_likelihood_adaptive_std_mtry_newXorder(y_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, N_Xorder, alpha, beta, no_split, split_var, split_point, parallel, subset_vars, Xorder_full, Xorder_next_index, Xorder_firstline, N_y, cutvalue, y_sum);

    if (no_split == true)
    {
        return;
    }

    this->v = split_var;
    // this -> c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    // this -> c = *(X_std + N_y * split_var + Xorder_full[split_var][current_index]);

    this->c = cutvalue;
    // cout << "real split point " << this-> c <<"   " << cutvalue <<  endl;

    split_var_count_pointer[split_var]++;

    size_t N_Xorder_left = split_point + 1;
    size_t N_Xorder_right = N_Xorder - split_point - 1;

    // xinfo_sizet Xorder_left_std;
    // xinfo_sizet Xorder_right_std;
    // ini_xinfo_sizet(Xorder_left_std, N_Xorder_left, p);
    // ini_xinfo_sizet(Xorder_right_std, N_Xorder_right, p);

    // std::vector<size_t> Xorder_firstline(p);
    std::vector<size_t> Xorder_right_firstline(p, 0);
    std::vector<size_t> Xorder_left_firstline(p, 0);

    // std::fill(Xorder_right_firstline.begin(), Xorder_right_firstline.end(), 0);
    // std::fill(Xorder_left_firstline.begin(), Xorder_left_firstline.end(), 0);

    // cout << "OK 1" << endl;

    double yleft_mean_std = 0.0;
    double yright_mean_std = 0.0;

    // cout << "ok 1" << endl;
    split_xorder_std_newXorder(this->c, split_var, split_point, Xorder_next_index, X_std, N_y, p, N_Xorder, N_Xorder_left, N_Xorder_right, Xorder_firstline, Xorder_left_firstline, Xorder_right_firstline, Xorder_full, yleft_mean_std, yright_mean_std, y_mean, y_std);

    double yleft_sum = yleft_mean_std * N_Xorder_left;
    double yright_sum = yright_mean_std * N_Xorder_right;
    // cout << "OK 2" << endl;

    // free(Xorder_std)

    // double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    // double yleft_mean_std = subnode_mean_newXorder(y_std, Xorder_full, Xorder_next_index, split_var, Xorder_left_firstline, N_Xorder_left);
    // double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);

    // double yright_mean_std = subnode_mean_newXorder(y_std, Xorder_full, Xorder_next_index, split_var, Xorder_right_firstline, N_Xorder_right);

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_std_newXorder(yleft_mean_std, yleft_sum, depth, max_depth, Nmin, Ncutpoints, N_Xorder_left, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_next_index, Xorder_left_firstline, X_std, split_var_count_pointer, mtry, subset_vars, Xorder_full);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_std_newXorder(yright_mean_std, yright_sum, depth, max_depth, Nmin, Ncutpoints, N_Xorder_right, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_next_index, Xorder_right_firstline, X_std, split_var_count_pointer, mtry, subset_vars, Xorder_full);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    return;
}

void split_xorder_std(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, const double *X_std, size_t N_y, size_t p, double &yleft_mean, double &yright_mean, const double &y_mean, std::vector<double> &y_std)
{

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes

    // preserve order of other variables
    size_t N_Xorder = Xorder_std[0].size();
    size_t left_ix = 0;
    size_t right_ix = 0;
    size_t N_Xorder_left = Xorder_left_std[0].size();
    size_t N_Xorder_right = Xorder_right_std[0].size();

    // if the left side is smaller, we only compute sum of it
    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    yleft_mean = 0.0;
    yright_mean = 0.0;

    double cutvalue = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    for (size_t i = 0; i < p; i++)
    {
        // loop over variables
        left_ix = 0;
        right_ix = 0;
        const double *temp_pointer = X_std + N_y * split_var;
        if (i == split_var)
        {
            if (compute_left_side)
            {
                for (size_t j = 0; j < N_Xorder; j++)
                {
                    if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                    {
                        yleft_mean = yleft_mean + y_std[Xorder_std[split_var][j]];

                        Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                        left_ix = left_ix + 1;
                    }
                    else
                    {

                        Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                        right_ix = right_ix + 1;
                    }
                }
            }
            else
            {
                for (size_t j = 0; j < N_Xorder; j++)
                {
                    if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                    {

                        Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                        left_ix = left_ix + 1;
                    }
                    else
                    {
                        yright_mean = yright_mean + y_std[Xorder_std[split_var][j]];

                        Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                        right_ix = right_ix + 1;
                    }
                }
            }
        }
        else
        {
            for (size_t j = 0; j < N_Xorder; j++)
            {
                if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                {
                    Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                    left_ix = left_ix + 1;
                }
                else
                {

                    Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                    right_ix = right_ix + 1;
                }
            }
        }
    }

    if (compute_left_side)
    {
        yright_mean = (y_mean * N_Xorder - yleft_mean) / N_Xorder_right;
        yleft_mean = yleft_mean / N_Xorder_left;
    }
    else
    {
        yleft_mean = (y_mean * N_Xorder - yright_mean) / N_Xorder_left;
        yright_mean = yright_mean / N_Xorder_right;
    }

    // yright_mean = yright_mean / N_Xorder_right;
    // yleft_mean = yleft_mean / N_Xorder_left;

    return;
}

void split_xorder_std_old(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, const double *X_std, size_t N_y, size_t p)
{

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes

    // preserve order of other variables
    size_t N_Xorder = Xorder_std[0].size();
    size_t left_ix = 0;
    size_t right_ix = 0;

    double cutvalue = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    for (size_t i = 0; i < p; i++)
    {
        // loop over variables
        left_ix = 0;
        right_ix = 0;
        const double *temp_pointer = X_std + N_y * split_var;
        for (size_t j = 0; j < N_Xorder; j++)
        {
            // Xorder(j, i), jth row and ith column
            // look at X(Xorder(j, i), split_var)
            // X[split_var][Xorder[i][j]]
            // X[split_var][Xorder[split_var][split_point]]
            if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
            {
                // copy a row
                // for(size_t k = 0; k < p; k ++){
                //     Xorder_left_std[i][left_ix];// = Xorder_std[i][j];
                //     left_ix = left_ix + 1;
                // }
                Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                left_ix = left_ix + 1;
            }
            else
            {
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

void split_xorder_std_newXorder(const double &cutvalue, const size_t &split_var, const size_t &split_point, xinfo_sizet &Xorder_next_index, const double *X_std, size_t N_y, size_t p, size_t &N_Xorder, size_t &N_Xorder_left, size_t &N_Xorder_right, std::vector<size_t> &Xorder_firstline, std::vector<size_t> &Xorder_left_firstline, std::vector<size_t> &Xorder_right_firstline, xinfo_sizet &Xorder_std_full, double &y_left_mean, double &y_right_mean, const double &y_mean, std::vector<double> &y_std)
{

    y_left_mean = 0.0;
    y_right_mean = 0.0;

    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes
    size_t left_ix = 0;
    size_t right_ix = 0;

    const double *temp_pointer = X_std + N_y * split_var;

    size_t current_index = 0;
    size_t left_previous_index = 0;
    size_t right_previous_index = 0;
    size_t next_index = 0;

    size_t count = 0;

    left_ix = 0;
    right_ix = 0;

    // cout << "inside ok 1" << endl;

    for (size_t i = 0; i < p; i++)
    {
        left_ix = 0;
        right_ix = 0;
        current_index = Xorder_firstline[i];
        left_previous_index = current_index;
        right_previous_index = current_index;
        next_index = current_index;

        // cout << "inside ok 2" << endl;

        while (next_index < UINT_MAX)
        {

            next_index = Xorder_next_index[i][current_index];

            // cout << "inside ok 3" << " " << next_index <<   endl;

            if (*(temp_pointer + Xorder_std_full[i][current_index]) <= cutvalue)
            {
                // cout << "left " << endl;

                if (i == split_var && compute_left_side)
                {
                    y_left_mean = y_left_mean + y_std[Xorder_std_full[split_var][current_index]];
                }

                if (left_ix == 0)
                {
                    // cout << "left first " << endl;
                    Xorder_left_firstline[i] = current_index;
                    left_previous_index = current_index;
                    current_index = next_index;
                    left_ix++;
                }
                else
                {
                    Xorder_next_index[i][left_previous_index] = current_index;
                    left_previous_index = current_index;
                    current_index = next_index;
                    left_ix++;
                }
            }
            else
            {

                if (i == split_var && (!compute_left_side))
                {
                    y_right_mean = y_right_mean + y_std[Xorder_std_full[split_var][current_index]];
                }
                // cout << "right " << endl;
                if (right_ix == 0)
                {
                    // cout << "right first " << endl;
                    Xorder_right_firstline[i] = current_index;
                    right_previous_index = current_index;
                    current_index = next_index;
                    right_ix++;
                }
                else
                {
                    Xorder_next_index[i][right_previous_index] = current_index;
                    right_previous_index = current_index;
                    current_index = next_index;
                    right_ix++;
                }
            }
        }

        if (left_ix >= N_Xorder_left)
        {
            Xorder_next_index[i][left_previous_index] = UINT_MAX;
        }
        if (right_ix >= N_Xorder_right)
        {
            Xorder_next_index[i][right_previous_index] = UINT_MAX;
        }
    }

    if (compute_left_side)
    {
        y_right_mean = (y_mean * N_Xorder - y_left_mean) / N_Xorder_right;
        y_left_mean = y_left_mean / N_Xorder_left;
    }
    else
    {
        y_left_mean = (y_mean * N_Xorder - y_right_mean) / N_Xorder_left;
        y_right_mean = y_right_mean / N_Xorder_right;
    }

    // for(size_t i = 0; i < p; i ++ ){
    //     // cout << Xorder_left_std[i][0] << "   " << Xorder_std[i][Xorder_left_firstline[i]]<< endl;
    //     cout << Xorder_next_index[i] << endl;
    // }

    // // check that Xorder_left_firstline is correct
    // for(size_t i = 0; i < p; i ++ ){
    //     cout << Xorder_right_std[i][0] << "   " << Xorder_std[i][Xorder_right_firstline[i]]<< endl;
    //     // cout << Xorder_next_index[i] << endl;
    // }

    // cout << "first " << Xorder_right_firstline << endl;

    // cout << Xorder_firstline << endl;

    // cout << "---------------------- RIGHT " << endl;

    // for(size_t j = 0; j < p; j ++ ){
    //     cout << "variable " << j << endl;
    //     cout << Xorder_right_std[j] << endl;

    //     current_index = Xorder_right_firstline[j];
    //     while(current_index < 10000){
    //         cout << Xorder_std_full[j][current_index] << ", ";
    //         current_index = Xorder_next_index[j][current_index];
    //     }

    //     cout << endl;

    // }

    // cout << "---------------------- LEFT " << endl;

    // for(size_t j = 0; j < p; j ++ ){
    //     cout << "variable " << j << endl;
    //     cout << Xorder_left_std[j] << endl;

    //     current_index = Xorder_left_firstline[j];
    //     while(current_index < 10000){
    //         cout << Xorder_std_full[j][current_index] << "    ";
    //         current_index = Xorder_next_index[j][current_index];
    //     }

    //     cout << endl;

    // }

    // cout << "---------------------- Xorder NEXT " << endl;
    // for(size_t j = 0; j < p ; j ++ ){
    //     cout << Xorder_next_index[j] << endl;
    // }

    return;
}

void BART_likelihood_adaptive_std(std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, std::vector<size_t> &subset_vars)
{
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

    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double *ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;
        std::vector<double> loglike((N_Xorder - 1) * p + 1);
        std::vector<double> y_cumsum(N_Xorder);
        std::vector<double> y_cumsum_inv(N_Xorder);

        if (parallel == false)
        {

            for (size_t i = 0; i < p; i++)
            {
                // loop over variables
                for (size_t q = 0; q < N_Xorder; q++)
                {
                    Y_sort[q] = y_std[Xorder_std[i][q]];
                }
                ypointer = &Y_sort[0];

                std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum.begin());

                y_sum = y_cumsum[y_cumsum.size() - 1]; // last one

                for (size_t k = 0; k < N_Xorder; k++)
                {
                    y_cumsum_inv[k] = y_sum - y_cumsum[k];
                }

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder - 1) * i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_evaluation_fullset like_parallel_full(y, Xorder, loglike, sigma2, tau, N, n1tau, n2tau);
            // parallelFor(0, p, like_parallel_full);
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        if ((N - 1) > 2 * Nmin)
        {
            for (size_t i = 0; i < p; i++)
            {
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
            }
        }
        else
        {
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

        if (ind == (N - 1) * p)
        {
            no_split = true;
        }

        if ((N - 1) <= 2 * Nmin)
        {
            no_split = true;
        }
    }
    else
    {

        std::vector<double> loglike(Ncutpoints * p + 1);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);
        std::vector<double> y_cumsum_inv(Ncutpoints);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        if (parallel == false)
        {

            std::vector<double> Y_sort(N_Xorder);
            double *ypointer;
            double n1tau;
            double n2tau;
            double Ntau = N_Xorder * tau;
            for (size_t iii = 0; iii < p; iii++)
            {

                for (size_t q = 0; q < N_Xorder; q++)
                {
                    Y_sort[q] = y_std[Xorder_std[iii][q]];
                }
                ypointer = &Y_sort[0];

                if (iii == 0)
                {
                    y_sum = sum_vec(Y_sort);
                }

                calculate_y_cumsum_std(ypointer, N, y_sum, candidate_index, y_cumsum, y_cumsum_inv);

                for (size_t j = 0; j < Ncutpoints; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;                   // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints)*iii + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_evaluation_subset like_parallel(y, Xorder, candidate_index, loglike, sigma2, tau, y_sum, Ncutpoints, N, n1tau, n2tau);
            // parallelFor(0, p, like_parallel);
        }

        loglike[loglike.size() - 1] = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if (ind == (Ncutpoints)*p)
        {
            no_split = true;
        }
    }

    return;
}

void BART_likelihood_adaptive_std_mtry(std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars)
{
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

    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double *ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // initialize log likelihood at -INFINITY
        std::vector<double> loglike((N_Xorder - 1) * p + 1, -INFINITY);
        std::vector<double> y_cumsum(N_Xorder);
        std::vector<double> y_cumsum_inv(N_Xorder);

        // std::vector<double> loglike_2(loglike.size(), -INFINITY);

        system_clock::time_point start;
        system_clock::time_point end;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ;

        if (parallel == false)
        {

            // for(size_t i = 0; i < p; i++){
            for (auto &&i : subset_vars)
            {
                // loop over variables

                // start = system_clock::now();
                // for(size_t q = 0;  q < N_Xorder; q++ ){
                //     Y_sort[q] = y_std[Xorder_std[i][q]];
                // }
                // ypointer = &Y_sort[0];
                // std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum.begin());
                // end = system_clock::now();

                // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                // // cout << "partial_sum " << duration.count() << endl;

                // start = system_clock::now();
                // compute_partial_sum(y_std, Xorder_std, i, y_cumsum);
                // end = system_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                //     // cout << "function call " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

                // start = system_clock::now();

                for (size_t q = 1; q < N_Xorder; q++)
                {
                    y_cumsum[q] = y_cumsum[q - 1] + y_std[Xorder_std[i][q]];
                }

                // end = system_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                //     cout << "no function" << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

                y_sum = y_cumsum[y_cumsum.size() - 1]; // last one

                for (size_t k = 0; k < N_Xorder; k++)
                {
                    y_cumsum_inv[k] = y_sum - y_cumsum[k];
                }

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder - 1) * i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

                    // loglike[(N_Xorder-1) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2)) + 2 * 0.5 * log(sigma2);
                }
            }
        }
        else
        {

            // parallel computing

            likelihood_fullset_std like_parallel_full(y_std, Xorder_std, N_Xorder, subset_vars, tau, Ntau, sigma2, loglike);
            parallelFor(0, subset_vars.size(), like_parallel_full);
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth) + 0.5 * log(sigma2);

        // normalize loglike, take exp to likelihood
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            // if a variable is not selected, take exp will becomes 0
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        if ((N - 1) > 2 * Nmin)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
            }
        }
        else
        {
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

        if (ind == (N - 1) * p)
        {
            no_split = true;
        }

        if ((N - 1) <= 2 * Nmin)
        {
            no_split = true;
        }
    }
    else
    {

        // initialize loglikelihood at -INFINITY
        std::vector<double> loglike(Ncutpoints * p + 1, -INFINITY);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);
        std::vector<double> y_cumsum_inv(Ncutpoints);

        // std::vector<double> y_cumsum2(Ncutpoints);

        // std::vector<double> loglike_2(loglike.size(), -INFINITY);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        double Ntau = N_Xorder * tau;

        if (parallel == false)
        {

            bool firstrun = true; // flag of the first loop
            std::vector<double> Y_sort(N_Xorder);
            double *ypointer;
            double n1tau;
            double n2tau;
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {

                // for(size_t q = 0;  q < N_Xorder; q++ ){
                //     Y_sort[q] = y_std[Xorder_std[i][q]];
                // }
                // ypointer = &Y_sort[0];

                // if(firstrun){
                //     y_sum = sum_vec(Y_sort);
                //     firstrun = false;
                // }

                // calculate_y_cumsum_std(ypointer, Y_sort.size(), y_sum, candidate_index, y_cumsum, y_cumsum_inv);

                // std::fill(y_cumsum.begin(), y_cumsum.end(), 0.0);
                compute_partial_sum_adaptive(y_std, candidate_index, y_cumsum, Xorder_std, i);

                y_sum = y_cumsum[y_cumsum.size() - 1]; // last one

                for (size_t k = 0; k < Ncutpoints; k++)
                {
                    y_cumsum_inv[k] = y_sum - y_cumsum[k];
                }

                // cout << y_cumsum[1] <<" " <<  y_cumsum2[1] << endl;

                for (size_t j = 0; j < Ncutpoints; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;                   // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints)*i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

                    // loglike[(Ncutpoints) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2)) + 2 * 0.5 * log(sigma2);
                }
            }
        }
        else
        {

            // parallel computing

            likelihood_subset_std like_parallel(y_std, Xorder_std, N_Xorder, Ncutpoints, subset_vars, tau, sigma2, candidate_index, loglike);
            parallelFor(0, subset_vars.size(), like_parallel);
        }

        // no split option
        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth) + 0.5 * log(sigma2);

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        // double loglike_2_max = *std::max_element(loglike_2.begin(), loglike_2.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if (ind == (Ncutpoints)*p)
        {
            no_split = true;
        }
    }

    return;
}

void BART_likelihood_adaptive_std_mtry_old(double y_sum, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars)
{
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

    double y_sum2;
    double sigma2 = pow(sigma, 2);

    double loglike_max = -INFINITY;

    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double *ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // initialize log likelihood at -INFINITY
        std::vector<double> loglike((N_Xorder - 1) * p + 1, -INFINITY);
        std::vector<double> y_cumsum(N_Xorder);
        // std::vector<double> y_cumsum_inv(N_Xorder);

        if (parallel == false)
        {

            // for(size_t i = 0; i < p; i++){
            for (auto &&i : subset_vars)
            {

                y_cumsum[0] = y_std[Xorder_std[i][0]];
                // y_cumsum_inv[0] = y_sum - y_cumsum[0];
                for (size_t q = 1; q < N_Xorder; q++)
                {
                    y_cumsum[q] = y_cumsum[q - 1] + y_std[Xorder_std[i][q]];
                    // y_cumsum_inv[q] = y_sum - y_cumsum[q];
                }

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder - 1) * i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    if (loglike[(N_Xorder - 1) * i + j] > loglike_max)
                    {
                        loglike_max = loglike[(N_Xorder - 1) * i + j];
                    }
                }
            }
        }
        else
        {

            // parallel computing

            likelihood_fullset_std like_parallel_full(y_std, Xorder_std, N_Xorder, subset_vars, tau, Ntau, sigma2, loglike);
            parallelFor(0, subset_vars.size(), like_parallel_full);
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        if (loglike[loglike.size() - 1] > loglike_max)
        {
            loglike_max = loglike[loglike.size() - 1];
        }

        // if(loglike_max != *std::max_element(loglike.begin(), loglike.end())){
        //     cout << "loglike_max " <<  loglike_max << " " <<  *std::max_element(loglike.begin(), loglike.end()) << endl;
        // }
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            // if a variable is not selected, take exp will becomes 0
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        if ((N - 1) > 2 * Nmin)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
            }
        }
        else
        {
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

        if (ind == (N - 1) * p)
        {
            no_split = true;
        }

        if ((N - 1) <= 2 * Nmin)
        {
            no_split = true;
        }
    }
    else
    {

        // initialize loglikelihood at -INFINITY
        std::vector<double> loglike(Ncutpoints * p + 1, -INFINITY);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);
        // std::vector<double> y_cumsum_inv(Ncutpoints);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        double Ntau = N_Xorder * tau;

        // double y_sum2;

        if (parallel == false)
        {

            bool firstrun = true; // flag of the first loop
            std::vector<double> Y_sort(N_Xorder);
            double *ypointer;
            double n1tau;
            double n2tau;
            for (auto &&i : subset_vars)
            {

                size_t ind = 0;
                y_cumsum[0] = 0.0;
                // size_t N_Xorder = Xorder_std[0].size();

                // cout << y_sum << " " << y_sum2 << endl;
                for (size_t q = 0; q < N_Xorder; q++)
                {
                    // cout << ind << " " << Ncutpoints << endl;
                    if (q <= candidate_index[ind])
                    {
                        y_cumsum[ind] = y_cumsum[ind] + y_std[Xorder_std[i][q]];
                    }
                    else
                    {

                        if (ind < Ncutpoints - 1)
                        {
                            // y_cumsum_inv[ind] = y_sum - y_cumsum[ind];
                            ind++;
                            y_cumsum[ind] = y_cumsum[ind - 1] + y_std[Xorder_std[i][q]];
                        }
                        else
                        {
                            // have done cumulative sum, do no care about elements after index of last entry of candiate_index
                            break;
                        }
                    }
                }

                // y_cumsum_inv[Ncutpoints - 1] = y_sum - y_cumsum[Ncutpoints - 1];

                for (size_t j = 0; j < Ncutpoints; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;                   // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints)*i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    if (loglike[(Ncutpoints)*i + j] > loglike_max)
                    {
                        loglike_max = loglike[(Ncutpoints)*i + j];
                    }
                }
            }
        }
        else
        {

            // parallel computing

            likelihood_subset_std like_parallel(y_std, Xorder_std, N_Xorder, Ncutpoints, subset_vars, tau, sigma2, candidate_index, loglike);
            parallelFor(0, subset_vars.size(), like_parallel);
        }

        // no split option
        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        if (loglike[loglike.size() - 1] > loglike_max)
        {
            loglike_max = loglike[loglike.size() - 1];
        }

        // normalize loglike
        // double loglike_max = *std::max_element(loglike.begin(), loglike.end());

        // if(loglike_max != *std::max_element(loglike.begin(), loglike.end())){
        //     cout << "loglike_max " << loglike_max << " " <<  *std::max_element(loglike.begin(), loglike.end()) << endl;
        // }

        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if (ind == (Ncutpoints)*p)
        {
            no_split = true;
        }
    }

    return;
}

void BART_likelihood_adaptive_std_mtry_all(double y_sum, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, const xinfo_sizet &Xorder_full, xinfo_sizet &Xorder_next_index, std::vector<size_t> Xorder_firstline, size_t &N_y, double &cutvalue, double &old_time, double &new_time)
{
    // cout << "-------------------------------" << endl;
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

    double y_sum2;
    double sigma2 = pow(sigma, 2);

    double loglike_max = -INFINITY;
    system_clock::time_point start;
    system_clock::time_point end;
    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {
        // cout << " all " << endl;

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        // std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double *ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // initialize log likelihood at -INFINITY
        std::vector<double> loglike((N_Xorder - 1) * p + 1, -INFINITY);
        std::vector<double> y_cumsum(N_Xorder);

        xinfo possible_cutpoints;

        ini_xinfo(possible_cutpoints, N_Xorder, p);

        // xinfo possible_cutpoints2;
        // std::vector<double> Y_sort(N_Xorder);
        std::vector<double> y_cumsum2(N_Xorder);
        // ini_xinfo(possible_cutpoints2, N_Xorder, p);
        // std::vector<double> y_cumsum_inv(N_Xorder);

        size_t current_index;
        size_t temp_index;
        if (parallel == false)
        {

            // for(size_t i = 0; i < p; i++){
            for (auto &&i : subset_vars)
            {

                // start = system_clock::now();

                y_cumsum[0] = y_std[Xorder_std[i][0]];
                // y_cumsum_inv[0] = y_sum - y_cumsum[0];
                for (size_t q = 1; q < N_Xorder; q++)
                {
                    y_cumsum[q] = y_cumsum[q - 1] + y_std[Xorder_std[i][q]];
                    // y_cumsum_inv[q] = y_sum - y_cumsum[q];
                }
                // end = system_clock::now();
                // auto duration = duration_cast<std::chrono::nanoseconds>(end - start);
                // // cout << "old " << duration.count() << endl;
                // old_time = old_time + duration.count() / 1000 ;

                // start = system_clock::now();

                current_index = Xorder_firstline[i];
                y_cumsum2[0] = y_std[Xorder_full[i][current_index]];
                possible_cutpoints[i][0] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                current_index = Xorder_next_index[i][current_index];
                temp_index = 1;
                while (current_index < UINT_MAX)
                {
                    possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                    y_cumsum2[temp_index] = y_cumsum2[temp_index - 1] + y_std[Xorder_full[i][current_index]];
                    current_index = Xorder_next_index[i][current_index];
                    temp_index++;
                }
                // end = system_clock::now();

                // auto duration2 = duration_cast<std::chrono::nanoseconds>(end - start);
                // cout << "old " << duration.count() << endl;
                // new_time = new_time + duration2.count() / 1000 ;

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder - 1) * i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    if (loglike[(N_Xorder - 1) * i + j] > loglike_max)
                    {
                        loglike_max = loglike[(N_Xorder - 1) * i + j];
                    }
                }
            }
        }
        else
        {

            // parallel computing

            likelihood_fullset_std like_parallel_full(y_std, Xorder_std, N_Xorder, subset_vars, tau, Ntau, sigma2, loglike);
            parallelFor(0, subset_vars.size(), like_parallel_full);
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        if (loglike[loglike.size() - 1] > loglike_max)
        {
            loglike_max = loglike[loglike.size() - 1];
        }

        // if(loglike_max != *std::max_element(loglike.begin(), loglike.end())){
        //     cout << "loglike_max " <<  loglike_max << " " <<  *std::max_element(loglike.begin(), loglike.end()) << endl;
        // }
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            // if a variable is not selected, take exp will becomes 0
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        if ((N - 1) > 2 * Nmin)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
            }
        }
        else
        {
            no_split = true;
            cutvalue = 0;
            // cout << "no split " << endl;
            return;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(loglike.begin(), loglike.end());
        // sample one index of split point

        ind = d(gen);

        split_var = ind / (N - 1);
        split_point = ind % (N - 1);

        if (ind == (N - 1) * p)
        {
            no_split = true;
        }
        else
        {
            cutvalue = possible_cutpoints[split_var][split_point];
        }

        if ((N - 1) <= 2 * Nmin)
        {
            no_split = true;
        }
        // cout << " fine " << endl;

        // cout << "cutvalue " << cutvalue << " " << *(X_std + N_y * split_var + Xorder_std[split_var][split_point]) << endl;
    }
    else
    {
        // cout << " not all " << endl;

        // initialize loglikelihood at -INFINITY
        std::vector<double> loglike(Ncutpoints * p + 1, -INFINITY);
        std::vector<size_t> candidate_index(Ncutpoints);
        // std::vector<size_t> candidate_index2(Ncutpoints + 1); // first element is always 0
        std::vector<double> y_cumsum(Ncutpoints);
        std::vector<double> y_cumsum2(Ncutpoints);
        // std::vector<double> y_cumsum_inv(Ncutpoints);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);
        // seq_gen_std_2(Nmin, N - Nmin, Ncutpoints, candidate_index2);
        // candidate_index2[0] = -1;

        double Ntau = N_Xorder * tau;

        // double y_sum2;
        xinfo possible_cutpoints;

        ini_xinfo(possible_cutpoints, Ncutpoints, p);

        xinfo possible_cutpoints2;

        ini_xinfo(possible_cutpoints2, Ncutpoints, p);

        size_t current_index;
        size_t previous_index;
        size_t temp_index;
        size_t count;
        std::vector<double> Y_sort(N_Xorder);

        if (parallel == false)
        {

            bool firstrun = true; // flag of the first loop
            std::vector<double> Y_sort(N_Xorder);
            double *ypointer;
            double n1tau;
            double n2tau;
            for (auto &&i : subset_vars)
            {

                // start = system_clock::now();
                // for (size_t q = 0; q < N_Xorder; q++)
                // {
                //     Y_sort[q] = y_std[Xorder_std[i][q]];
                // }

                // size_t ind = 0;
                // y_cumsum[0] = 0.0;
                // // size_t N_Xorder = Xorder_std[0].size();

                // // cout << y_sum << " " << y_sum2 << endl;
                // for (size_t q = 0; q < N_Xorder; q++)
                // {
                //     // cout << ind << " " << Ncutpoints << endl;
                //     if (q <= candidate_index[ind])
                //     {
                //         y_cumsum[ind] = y_cumsum[ind] + y_std[Xorder_std[i][q]];
                //     }
                //     else
                //     {
                //         if (ind < Ncutpoints - 1)
                //         {
                //             // y_cumsum_inv[ind] = y_sum - y_cumsum[ind];
                //             ind++;
                //             y_cumsum[ind] = y_cumsum[ind - 1] + y_std[Xorder_std[i][q]];
                //         }
                //         else
                //         {
                //             // have done cumulative sum, do no care about elements after index of last entry of candiate_index
                //             break;
                //         }
                //     }
                // }
                // end = system_clock::now();
                // // cout << "------------------------" << endl;
                // auto duration = duration_cast<std::chrono::nanoseconds>(end - start);
                // // cout << "old " << duration.count() << endl;
                // old_time = old_time + duration.count() / 1000 ;

                start = system_clock::now();
                current_index = Xorder_firstline[i];
                temp_index = 0;
                // count = 0;
                y_cumsum[0] = 0;
                for (size_t q = 0; q < N_Xorder; q++)
                {

                    if (q <= candidate_index[temp_index])
                    {
                        y_cumsum[temp_index] = y_cumsum[temp_index] + y_std[Xorder_full[i][current_index]];
                    }
                    else
                    {
                        possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][current_index]);

                        if (temp_index < Ncutpoints - 1)
                        {
                            temp_index++;
                            y_cumsum[temp_index] = y_cumsum[temp_index - 1] + y_std[Xorder_full[i][current_index]];
                            // count++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    current_index = Xorder_next_index[i][current_index];
                }

                end = system_clock::now();
                auto duration2 = duration_cast<std::chrono::nanoseconds>(end - start);
                old_time = old_time + duration2.count() / 1000;

                start = system_clock::now();
                current_index = Xorder_firstline[i];
                temp_index = 0;
                y_cumsum2[0] = 0;
                // current_index = Xorder_next_index[i][current_index];
                // for(size_t m = 0; m < Ncutpoints; m ++ ){
                //     // loop for index between candidate_index[m] and candidate_index[m - 1]
                //     for(size_t q = candidate_index2[m] + 1, q <= candidate_index2[m + 1]; q ++ ){
                //         y_cumsum2[m] = y_cumsum2[m] + y_std[Xorder_full[i][current_index]];
                //         current_index = Xorder_next_index[i][current_index];
                //     }
                // }

                // first element
                for (size_t m = 0; m <= candidate_index[0]; m++)
                {
                    y_cumsum2[0] = y_cumsum2[0] + y_std[Xorder_full[i][current_index]];
                    previous_index = current_index;
                    current_index = Xorder_next_index[i][current_index];
                }
                possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][previous_index]);
                y_cumsum2[1] = y_cumsum2[0];

                temp_index = 1;
                for (size_t m = 0; m < Ncutpoints - 1; m++)
                {
                    for (size_t q = candidate_index[m] + 1; q <= candidate_index[m + 1]; q++)
                    {
                        y_cumsum2[temp_index] = y_cumsum2[temp_index] + y_std[Xorder_full[i][current_index]];
                        previous_index = current_index;
                        current_index = Xorder_next_index[i][current_index];
                    }
                    possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][previous_index]);
                    temp_index++;
                    y_cumsum2[temp_index] = y_cumsum2[temp_index - 1];
                }

                // cout << "new " << duration2.count() << endl;
                new_time = new_time + duration2.count() / 1000;
                // cout << "y_sum diff " << sq_vec_diff(y_cumsum, y_cumsum2) << endl;

                // cout << y_cumsum - y_cumsum2 << endl;

                for (size_t j = 0; j < Ncutpoints; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;                   // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints)*i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    if (loglike[(Ncutpoints)*i + j] > loglike_max)
                    {
                        loglike_max = loglike[(Ncutpoints)*i + j];
                    }
                }
            }
        }
        else
        {

            // parallel computing

            likelihood_subset_std like_parallel(y_std, Xorder_std, N_Xorder, Ncutpoints, subset_vars, tau, sigma2, candidate_index, loglike);
            parallelFor(0, subset_vars.size(), like_parallel);
        }

        // no split option
        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        if (loglike[loglike.size() - 1] > loglike_max)
        {
            loglike_max = loglike[loglike.size() - 1];
        }

        // normalize loglike
        // double loglike_max = *std::max_element(loglike.begin(), loglike.end());

        // if(loglike_max != *std::max_element(loglike.begin(), loglike.end())){
        //     cout << "loglike_max " << loglike_max << " " <<  *std::max_element(loglike.begin(), loglike.end()) << endl;
        // }

        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if (ind == (Ncutpoints)*p)
        {
            no_split = true;
        }
        else
        {
            cutvalue = possible_cutpoints[split_var][ind % Ncutpoints];
            // cout << "cut value  " << cutvalue << "  " << *(X_std + N_y * split_var + Xorder_std[split_var][split_point]) << " variable " << split_var << endl;
        }
    }
    // cout << " fine fine "<<endl;
    return;
}

void BART_likelihood_adaptive_std_mtry_newXorder(std::vector<double> &y_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, size_t N_Xorder, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, xinfo_sizet &Xorder_full, xinfo_sizet &Xorder_next_index, std::vector<size_t> &Xorder_firstline, size_t &N_y, double &cutvalue, double &y_sum)
{
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = N_Xorder;
    size_t p = Xorder_full.size();
    size_t ind;
    // size_t N_Xorder = N;

    // double y_sum;
    double sigma2 = pow(sigma, 2);

    // std::vector<double> Y_sort2(N_Xorder);

    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double *ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // initialize log likelihood at -INFINITY
        std::vector<double> loglike((N_Xorder - 1) * p + 1, -INFINITY);
        std::vector<double> y_cumsum(N_Xorder);
        std::vector<double> y_cumsum_inv(N_Xorder);

        // std::vector<double> y_cumsum2(N_Xorder);
        // std::vector<double> loglike_2(loglike.size(), -INFINITY);

        xinfo possible_cutpoints;
        // xinfo possible_cutpoints2;
        // ini_xinfo(possible_cutpoints2, N_Xorder, p);
        ini_xinfo(possible_cutpoints, N_Xorder, p);

        if (parallel == false)
        {

            // for(size_t i = 0; i < p; i++){
            for (auto &&i : subset_vars)
            {
                // loop over variables
                // for(size_t q = 0;  q < N_Xorder; q++ ){
                // Y_sort[q] = y_std[Xorder_std[i][q]];
                // }

                // create_y_sort(Y_sort, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i);

                create_y_sort_2(Y_sort, possible_cutpoints[i], X_std, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i, N_y);

                compute_partial_sum_newXorder(y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i, N_y, y_cumsum, possible_cutpoints[i], X_std);
                // cout << possible_cutpoints[i] << endl;

                ypointer = &Y_sort[0];

                std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum.begin());

                // cout << possible_cutpoints[i] - possible_cutpoints2[i] << endl;

                y_sum = y_cumsum[y_cumsum.size() - 1]; // last one

                for (size_t k = 0; k < N_Xorder; k++)
                {
                    y_cumsum_inv[k] = y_sum - y_cumsum[k];
                }

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder - 1) * i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

                    // loglike[(N_Xorder-1) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2)) + 2 * 0.5 * log(sigma2);
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_fullset_std like_parallel_full(y_std, Xorder_std, N_Xorder, subset_vars, tau, Ntau, sigma2, loglike);
            // parallelFor(0, subset_vars.size(), like_parallel_full);
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth) + 0.5 * log(sigma2);

        // normalize loglike, take exp to likelihood
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            // if a variable is not selected, take exp will becomes 0
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        if ((N - 1) > 2 * Nmin)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
            }
        }
        else
        {
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

        if (ind == (N - 1) * p)
        {
            no_split = true;
        }
        else
        {
            cutvalue = possible_cutpoints[split_var][split_point];
        }

        if ((N - 1) <= 2 * Nmin)
        {
            no_split = true;
        }
    }
    else
    {

        // initialize loglikelihood at -INFINITY
        std::vector<double> loglike(Ncutpoints * p + 1, -INFINITY);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);
        std::vector<double> y_cumsum_inv(Ncutpoints);

        std::vector<double> y_cumsum2(Ncutpoints);

        // std::vector<double> loglike_2(loglike.size(), -INFINITY);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        double Ntau = N_Xorder * tau;

        xinfo possible_cutpoints;

        ini_xinfo(possible_cutpoints, Ncutpoints, p);

        std::vector<double> temp(Ncutpoints);

        if (parallel == false)
        {

            bool firstrun = true; // flag of the first loop
            std::vector<double> Y_sort(N_Xorder);
            double *ypointer;
            double n1tau;
            double n2tau;
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {

                // for(size_t q = 0;  q < N_Xorder; q++ ){
                // Y_sort[q] = y_std[Xorder_std[i][q]];
                // }

                // create_y_sort(Y_sort, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i);

                create_y_sort_3(Y_sort, possible_cutpoints[i], X_std, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i, N_y, candidate_index);

                ypointer = &Y_sort[0];

                if (firstrun)
                {
                    y_sum = sum_vec(Y_sort);
                    firstrun = false;
                }

                calculate_y_cumsum_std(ypointer, Y_sort.size(), y_sum, candidate_index, y_cumsum, y_cumsum_inv);

                // std::fill(y_cumsum2.begin(), y_cumsum2.end(), 0.0);
                // compute_partial_sum_adaptive_newXorder(y_std, candidate_index, y_cumsum2, Xorder_full, i, Xorder_next_index, Xorder_firstline, N_Xorder, temp, N_y, X_std);

                // y_sum = y_cumsum[y_cumsum.size() - 1]; // last one

                // for(size_t k = 0; k < N_Xorder; k ++ ){
                // y_cumsum_inv[k] = y_sum - y_cumsum[k];
                // }
                // cout << possible_cutpoints[i] - temp << endl;

                for (size_t j = 0; j < Ncutpoints; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;                   // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints)*i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

                    // loglike[(Ncutpoints) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2)) + 2 * 0.5 * log(sigma2);
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_subset_std like_parallel(y_std, Xorder_std, N_Xorder, Ncutpoints, subset_vars, tau, sigma2, candidate_index, loglike);
            // parallelFor(0, subset_vars.size(), like_parallel);
        }

        // no split option
        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth) + 0.5 * log(sigma2);

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        // double loglike_2_max = *std::max_element(loglike_2.begin(), loglike_2.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if (ind == (Ncutpoints)*p)
        {
            no_split = true;
        }
        else
        {
            cutvalue = possible_cutpoints[split_var][ind % Ncutpoints];
        }
    }

    return;
}

void cumulative_sum_std(std::vector<double> &y_cumsum, std::vector<double> &y_cumsum_inv, double &y_sum, double *y, xinfo_sizet &Xorder, size_t &i, size_t &N)
{
    // y_cumsum is the output cumulative sum
    // y is the original data
    // Xorder is sorted index matrix
    // i means take the i-th column of Xorder
    // N is length of y and y_cumsum
    if (N > 1)
    {
        y_cumsum[0] = y[Xorder[i][0]];
        for (size_t j = 1; j < N; j++)
        {
            y_cumsum[j] = y_cumsum[j - 1] + y[Xorder[i][j]];
        }
    }
    else
    {
        y_cumsum[0] = y[Xorder[i][0]];
    }
    y_sum = y_cumsum[N - 1];

    for (size_t j = 1; j < N; j++)
    {
        y_cumsum_inv[j] = y_sum - y_cumsum[j];
    }
    return;
}

arma::uvec range(size_t start, size_t end)
{
    // generate integers from start to end
    size_t N = end - start;
    arma::uvec output(N);
    for (size_t i = 0; i < N; i++)
    {
        output(i) = start + i;
    }
    return output;
}

void tree::grow_tree_adaptive_std_newXorder_old(double y_mean, double y_sum, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, size_t N_Xorder, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_next_index, std::vector<size_t> &Xorder_firstline, const double *X_std, double *split_var_count_pointer, size_t &mtry, const std::vector<size_t> &subset_vars, xinfo_sizet &Xorder_full)
{

    // grow a tree, users can control number of split points

    // size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_full.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if (N_Xorder <= Nmin)
    {
        return;
    }

    if (depth >= max_depth - 1)
    {
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    if (draw_mu == true)
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        this->theta_noise = this->theta;
    }
    else
    {

        this->theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        this->theta_noise = this->theta; // identical to theta
    }

    if (draw_sigma == true)
    {

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta

        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std(*top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    }

    this->sig = sigma;
    bool no_split = false;

    // cout << " ok 1" << endl;
    double cutvalue = 0.0;
    BART_likelihood_adaptive_std_mtry_newXorder_old(y_sum, y_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, N_Xorder, alpha, beta, no_split, split_var, split_point, parallel, subset_vars, Xorder_full, Xorder_next_index, Xorder_firstline, N_y, cutvalue);

    // cout << "ok 2" << endl;

    if (no_split == true)
    {
        return;
    }

    this->v = split_var;
    // this -> c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    // this -> c = *(X_std + N_y * split_var + Xorder_full[split_var][current_index]);

    this->c = cutvalue;
    // cout << "real split point " << this-> c <<"   " << cutvalue <<  endl;

    split_var_count_pointer[split_var]++;

    size_t N_Xorder_left = split_point + 1;
    size_t N_Xorder_right = N_Xorder - split_point - 1;

    // xinfo_sizet Xorder_left_std;
    // xinfo_sizet Xorder_right_std;
    // ini_xinfo_sizet(Xorder_left_std, N_Xorder_left, p);
    // ini_xinfo_sizet(Xorder_right_std, N_Xorder_right, p);

    // std::vector<size_t> Xorder_firstline(p);
    std::vector<size_t> Xorder_right_firstline(p, 0);
    std::vector<size_t> Xorder_left_firstline(p, 0);

    // std::fill(Xorder_right_firstline.begin(), Xorder_right_firstline.end(), 0);
    // std::fill(Xorder_left_firstline.begin(), Xorder_left_firstline.end(), 0);

    // cout << "OK 1" << endl;

    double yleft_mean_std = 0.0;
    double yright_mean_std = 0.0;

    // cout << "ok 1" << endl;
    split_xorder_std_newXorder_old(this->c, split_var, split_point, Xorder_next_index, X_std, N_y, p, N_Xorder, N_Xorder_left, N_Xorder_right, Xorder_firstline, Xorder_left_firstline, Xorder_right_firstline, Xorder_full, yleft_mean_std, yright_mean_std, y_mean, y_std);

    // cout << "OK 2" << endl;

    // free(Xorder_std)

    // double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    // double yleft_mean_std = subnode_mean_newXorder(y_std, Xorder_full, Xorder_next_index, split_var, Xorder_left_firstline, N_Xorder_left);
    // double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);

    // double yright_mean_std = subnode_mean_newXorder(y_std, Xorder_full, Xorder_next_index, split_var, Xorder_right_firstline, N_Xorder_right);

    double yleft_sum = yleft_mean_std * N_Xorder_left;
    double yright_sum = yright_mean_std * N_Xorder_right;

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_std_newXorder_old(yleft_mean_std, yleft_sum, depth, max_depth, Nmin, Ncutpoints, N_Xorder_left, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_next_index, Xorder_left_firstline, X_std, split_var_count_pointer, mtry, subset_vars, Xorder_full);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_std_newXorder_old(yright_mean_std, yright_sum, depth, max_depth, Nmin, Ncutpoints, N_Xorder_right, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_next_index, Xorder_right_firstline, X_std, split_var_count_pointer, mtry, subset_vars, Xorder_full);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    return;
}

void split_xorder_std_newXorder_old(const double &cutvalue, const size_t &split_var, const size_t &split_point, xinfo_sizet &Xorder_next_index, const double *X_std, size_t N_y, size_t p, size_t &N_Xorder, size_t &N_Xorder_left, size_t &N_Xorder_right, std::vector<size_t> &Xorder_firstline, std::vector<size_t> &Xorder_left_firstline, std::vector<size_t> &Xorder_right_firstline, xinfo_sizet &Xorder_std_full, double &y_left_mean, double &y_right_mean, const double &y_mean, std::vector<double> &y_std)
{

    y_left_mean = 0.0;
    y_right_mean = 0.0;

    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes
    size_t left_ix = 0;
    size_t right_ix = 0;

    const double *temp_pointer = X_std + N_y * split_var;

    size_t current_index = 0;
    size_t left_previous_index = 0;
    size_t right_previous_index = 0;
    size_t next_index = 0;

    size_t count = 0;

    left_ix = 0;
    right_ix = 0;

    // cout << "inside ok 1" << endl;

    for (size_t i = 0; i < p; i++)
    {
        left_ix = 0;
        right_ix = 0;
        current_index = Xorder_firstline[i];
        left_previous_index = current_index;
        right_previous_index = current_index;
        next_index = current_index;

        // cout << "inside ok 2" << endl;

        while (next_index < UINT_MAX)
        {

            next_index = Xorder_next_index[i][current_index];

            // cout << "inside ok 3" << " " << next_index <<   endl;

            if (*(temp_pointer + Xorder_std_full[i][current_index]) <= cutvalue)
            {
                // cout << "left " << endl;

                if (i == split_var && compute_left_side)
                {
                    y_left_mean = y_left_mean + y_std[Xorder_std_full[split_var][current_index]];
                }

                if (left_ix == 0)
                {
                    // cout << "left first " << endl;
                    Xorder_left_firstline[i] = current_index;
                    left_previous_index = current_index;
                    current_index = next_index;
                    left_ix++;
                }
                else
                {
                    Xorder_next_index[i][left_previous_index] = current_index;
                    left_previous_index = current_index;
                    current_index = next_index;
                    left_ix++;
                }
            }
            else
            {

                if (i == split_var && (!compute_left_side))
                {
                    y_right_mean = y_right_mean + y_std[Xorder_std_full[split_var][current_index]];
                }
                // cout << "right " << endl;
                if (right_ix == 0)
                {
                    // cout << "right first " << endl;
                    Xorder_right_firstline[i] = current_index;
                    right_previous_index = current_index;
                    current_index = next_index;
                    right_ix++;
                }
                else
                {
                    Xorder_next_index[i][right_previous_index] = current_index;
                    right_previous_index = current_index;
                    current_index = next_index;
                    right_ix++;
                }
            }
        }

        if (left_ix >= N_Xorder_left)
        {
            Xorder_next_index[i][left_previous_index] = UINT_MAX;
        }
        if (right_ix >= N_Xorder_right)
        {
            Xorder_next_index[i][right_previous_index] = UINT_MAX;
        }
    }

    if (compute_left_side)
    {
        y_right_mean = (y_mean * N_Xorder - y_left_mean) / N_Xorder_right;
        y_left_mean = y_left_mean / N_Xorder_left;
    }
    else
    {
        y_left_mean = (y_mean * N_Xorder - y_right_mean) / N_Xorder_left;
        y_right_mean = y_right_mean / N_Xorder_right;
    }

    // for(size_t i = 0; i < p; i ++ ){
    //     // cout << Xorder_left_std[i][0] << "   " << Xorder_std[i][Xorder_left_firstline[i]]<< endl;
    //     cout << Xorder_next_index[i] << endl;
    // }

    // // check that Xorder_left_firstline is correct
    // for(size_t i = 0; i < p; i ++ ){
    //     cout << Xorder_right_std[i][0] << "   " << Xorder_std[i][Xorder_right_firstline[i]]<< endl;
    //     // cout << Xorder_next_index[i] << endl;
    // }

    // cout << "first " << Xorder_right_firstline << endl;

    // cout << Xorder_firstline << endl;

    // cout << "---------------------- RIGHT " << endl;

    // for(size_t j = 0; j < p; j ++ ){
    //     cout << "variable " << j << endl;
    //     cout << Xorder_right_std[j] << endl;

    //     current_index = Xorder_right_firstline[j];
    //     while(current_index < 10000){
    //         cout << Xorder_std_full[j][current_index] << ", ";
    //         current_index = Xorder_next_index[j][current_index];
    //     }

    //     cout << endl;

    // }

    // cout << "---------------------- LEFT " << endl;

    // for(size_t j = 0; j < p; j ++ ){
    //     cout << "variable " << j << endl;
    //     cout << Xorder_left_std[j] << endl;

    //     current_index = Xorder_left_firstline[j];
    //     while(current_index < 10000){
    //         cout << Xorder_std_full[j][current_index] << "    ";
    //         current_index = Xorder_next_index[j][current_index];
    //     }

    //     cout << endl;

    // }

    // cout << "---------------------- Xorder NEXT " << endl;
    // for(size_t j = 0; j < p ; j ++ ){
    //     cout << Xorder_next_index[j] << endl;
    // }

    return;
}

void BART_likelihood_adaptive_std_mtry_newXorder_old(double &y_sum, std::vector<double> &y_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, size_t N_Xorder, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, xinfo_sizet &Xorder_full, xinfo_sizet &Xorder_next_index, std::vector<size_t> &Xorder_firstline, size_t &N_y, double &cutvalue)
{

    // cout << " begin " << endl;
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = N_Xorder;
    size_t p = Xorder_full.size();
    size_t ind;
    // size_t N_Xorder = N;

    // double y_sum;
    double sigma2 = pow(sigma, 2);

    // std::vector<double> Y_sort2(N_Xorder);

    system_clock::time_point start;
    system_clock::time_point end;

    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        std::vector<double> Y_sort(N_Xorder); // a container for sorted y
        double *ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // initialize log likelihood at -INFINITY
        std::vector<double> loglike((N_Xorder - 1) * p + 1, -INFINITY);
        std::vector<double> y_cumsum(N_Xorder);
        // std::vector<double> y_cumsum_inv(N_Xorder);
        // std::vector<double> y_cumsum2(N_Xorder);

        // std::vector<double> loglike_2(loglike.size(), -INFINITY);

        xinfo possible_cutpoints;

        ini_xinfo(possible_cutpoints, N_Xorder, p);

        // xinfo possible_cutpoints2;

        // ini_xinfo(possible_cutpoints2, N_Xorder, p);

        size_t current_index;
        size_t temp_index;

        if (parallel == false)
        {

            // for(size_t i = 0; i < p; i++){
            for (auto &&i : subset_vars)
            {
                // loop over variables
                // for(size_t q = 0;  q < N_Xorder; q++ ){
                //     Y_sort[q] = y_std[Xorder_std[i][q]];
                // }

                // create_y_sort(Y_sort, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i);

                // start = system_clock::now();
                // create_y_sort_2(Y_sort, possible_cutpoints[i], X_std, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i, N_y);
                // // cout << possible_cutpoints[i] << endl;
                // ypointer = &Y_sort[0];
                // std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum.begin());
                // end = system_clock::now();
                // auto duration = duration_cast<std::chrono::nanoseconds>(end - start);
                // cout << "function " << duration.count() << endl;

                // set initial valu
                // start = system_clock::now();
                // cout << " dddd d  " << endl;

                current_index = Xorder_firstline[i];
                y_cumsum[0] = y_std[Xorder_full[i][current_index]];
                possible_cutpoints[i][0] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                current_index = Xorder_next_index[i][current_index];
                temp_index = 1;
                while (current_index < UINT_MAX)
                {
                    possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                    y_cumsum[temp_index] = y_cumsum[temp_index - 1] + y_std[Xorder_full[i][current_index]];
                    current_index = Xorder_next_index[i][current_index];
                    temp_index++;
                }

                // cout << " deeefe " << endl;
                // end = system_clock::now();
                // duration = duration_cast<std::chrono::nanoseconds>(end - start);
                // cout << "loop " << duration.count() << endl;

                // y_sum = y_cumsum[y_cumsum.size() - 1]; // last one

                // for (size_t k = 0; k < N_Xorder; k++)
                // {
                //     y_cumsum_inv[k] = y_sum - y_cumsum[k];
                // }

                // cout << possible_cutpoints[i][1] << " " << possible_cutpoints2[i][1] << endl;
                // cout << sq_vec_diff(possible_cutpoints[i], possible_cutpoints2[i]) << endl;
                // cout << "------------------" << endl;
                // cout << y_sum2 << "  fdfdd " << y_sum << endl;

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder - 1) * i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    // loglike[(N_Xorder-1) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2)) + 2 * 0.5 * log(sigma2);
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_fullset_std like_parallel_full(y_std, Xorder_std, N_Xorder, subset_vars, tau, Ntau, sigma2, loglike);
            // parallelFor(0, subset_vars.size(), like_parallel_full);
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth) + 0.5 * log(sigma2);

        // normalize loglike, take exp to likelihood
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            // if a variable is not selected, take exp will becomes 0
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        if ((N - 1) > 2 * Nmin)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
            }
        }
        else
        {
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

        if (ind == (N - 1) * p)
        {
            no_split = true;
        }
        else
        {
            cutvalue = possible_cutpoints[split_var][split_point];
        }

        if ((N - 1) <= 2 * Nmin)
        {
            no_split = true;
        }
    }
    else
    {

        // initialize loglikelihood at -INFINITY
        std::vector<double> loglike(Ncutpoints * p + 1, -INFINITY);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);
        // std::vector<double> y_cumsum_inv(Ncutpoints);
        // std::vector<double> y_cumsum2(Ncutpoints);
        // std::vector<double> loglike_2(loglike.size(), -INFINITY);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        double Ntau = N_Xorder * tau;

        xinfo possible_cutpoints;

        ini_xinfo(possible_cutpoints, Ncutpoints, p);

        // xinfo possible_cutpoints2;
        // ini_xinfo(possible_cutpoints2, Ncutpoints, p);

        size_t current_index;
        size_t temp_index;
        size_t count;

        // double y_sum2 = 0.0;

        if (parallel == false)
        {

            bool firstrun = true; // flag of the first loop
            // std::vector<double> Y_sort(N_Xorder);
            // double *ypointer;
            double n1tau;
            double n2tau;
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {

                // // create_y_sort(Y_sort, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i);
                // start = system_clock::now();
                // create_y_sort_3(Y_sort, possible_cutpoints[i], X_std, y_std, Xorder_full, Xorder_next_index, Xorder_firstline, i, N_y, candidate_index);

                // // cout << Y_sort[Y_sort.size() - 1] << endl;

                // ypointer = &Y_sort[0];

                // if (firstrun)
                // {
                //     y_sum2 = sum_vec(Y_sort);
                //     firstrun = false;
                // }

                // calculate_y_cumsum_std(ypointer, Y_sort.size(), y_sum2, candidate_index, y_cumsum, y_cumsum_inv);

                // end = system_clock::now();
                // auto duration = duration_cast<std::chrono::nanoseconds>(end - start);
                // cout << "function " << duration.count() << endl;

                // // cout << y_sum << "  " << y_sum2 << endl;

                //             current_index = Xorder_firstline[i];
                //             y_cumsum[0] = y_std[Xorder_full[i][current_index]];
                //             possible_cutpoints2[i][0] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                //             current_index = Xorder_next_index[i][current_index];
                //             temp_index = 1;
                //             while(current_index < UINT_MAX){
                //                 // possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                //                 y_cumsum[temp_index] = y_cumsum[temp_index - 1] + y_std[Xorder_full[i][current_index]];
                //                 current_index = Xorder_next_index[i][current_index];
                //                 temp_index ++ ;
                //             }

                //     cout << " ppppp " << endl;

                current_index = Xorder_firstline[i];
                temp_index = 0;
                count = 0;
                y_cumsum[0] = 0;
                for (size_t q = 0; q < N_Xorder; q++)
                {

                    if (q <= candidate_index[temp_index])
                    {
                        y_cumsum[temp_index] = y_cumsum[temp_index] + y_std[Xorder_full[i][current_index]];
                    }
                    else
                    {
                        possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][current_index]);

                        if (temp_index < Ncutpoints - 1)
                        {
                            temp_index++;
                            y_cumsum[temp_index] = y_cumsum[temp_index - 1] + y_std[Xorder_full[i][current_index]];
                            // count++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    current_index = Xorder_next_index[i][current_index];
                }

                for (size_t j = 0; j < Ncutpoints; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;                   // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints)*i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    // loglike[(Ncutpoints) * i + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2)) + 2 * 0.5 * log(sigma2);
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_subset_std like_parallel(y_std, Xorder_std, N_Xorder, Ncutpoints, subset_vars, tau, sigma2, candidate_index, loglike);
            // parallelFor(0, subset_vars.size(), like_parallel);
        }

        // no split option
        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        // loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, - 1.0 * beta)) - log(alpha) + beta * log(1.0 + depth) + 0.5 * log(sigma2);

        // normalize loglike
        double loglike_max = *std::max_element(loglike.begin(), loglike.end());
        // double loglike_2_max = *std::max_element(loglike_2.begin(), loglike_2.end());
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if (ind == (Ncutpoints)*p)
        {
            no_split = true;
        }
        else
        {
            // cout << "ok1 " << endl;
            cutvalue = possible_cutpoints[split_var][ind % Ncutpoints];
            // cout << "ok2 " << endl;
        }
    }

    // cout << " end " << endl;

    return;
}

// core functions of linkedlist
void tree::grow_tree_adaptive_linkedlist(double y_mean, double y_sum, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, const double *X_std, double *split_var_count_pointer, size_t &mtry, const std::vector<size_t> &subset_vars, xinfo_sizet &Xorder_next_index, xinfo_sizet &Xorder_full, std::vector<size_t> &Xorder_firstline, size_t &N_Xorder)
{

    // grow a tree, users can control number of split points

    // size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_full.size();
    size_t N_y = y_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    if (N_Xorder <= Nmin)
    {
        return;
    }

    if (depth >= max_depth - 1)
    {
        return;
    }

    // tau is prior VARIANCE, do not take squares
    // set up random device

    std::default_random_engine generator;
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    if (draw_mu == true)
    {

        theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        theta_noise = theta;
    }
    else
    {

        theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        theta_noise = theta; // identical to theta
    }

    if (draw_sigma == true)
    {

        tree::tree_p top_p = this->gettop();
        // draw sigma use residual of noisy theta

        std::vector<double> reshat_std(N_y);
        fit_new_theta_noise_std(*top_p, X_std, p, N_y, reshat_std);
        reshat_std = y_std - reshat_std;

        std::gamma_distribution<double> gamma_samp((N_y + 16) / 2.0, 2.0 / (sum_squared(reshat_std) + 4.0));
        sigma = 1.0 / gamma_samp(generator);
    }

    sig = sigma;
    bool no_split = false;

    // double cutvalue = 0.0;
    // double cutvalue2 ;

    // cout << "ok1 " << endl;
    BART_likelihood_adaptive_linkedlist(y_mean * N_Xorder, y_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, v, split_point, parallel, subset_vars, Xorder_full, Xorder_next_index, Xorder_firstline, N_y, c, N_Xorder);
    // cout << "ok2 " << endl;
    if (no_split == true)
    {
        return;
    }

    // this->v = split_var;
    // v = split_var;
    split_var = v;
    // c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    // cutvalue = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    // c = cutvalue;
    // cutvalue2 = cutvalue;
    // this->c = cutvalue2;
    // if(cutvalue != this->c){
    // cout << "cut value  " << cutvalue << "  " << c << " variable " << split_var << endl;
    // }
    split_var_count_pointer[split_var]++;

    size_t N_Xorder_left = split_point + 1;
    size_t N_Xorder_right = N_Xorder - split_point - 1;

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    // system_clock::time_point start;
    // system_clock::time_point end;
    // start = system_clock::now();
    // split_xorder_std_old(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p);
    // double yleft_mean_std = subnode_mean(y_std, Xorder_left_std, split_var);
    // double yright_mean_std = subnode_mean(y_std, Xorder_right_std, split_var);
    // end = system_clock::now();
    // auto duration = duration_cast<microseconds>(end - start);
    // double running_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    //     cout << " ----- ---- " << endl;
    //     cout << "running time 1 " << duration.count() << endl;

    std::vector<size_t> Xorder_right_firstline(p, 0);
    std::vector<size_t> Xorder_left_firstline(p, 0);

    // std::fill(Xorder_right_firstline.begin(), Xorder_right_firstline.end(), 0);
    // std::fill(Xorder_left_firstline.begin(), Xorder_left_firstline.end(), 0);

    // auto start = system_clock::now();
    double yleft_mean_std = 0.0;
    double yright_mean_std = 0.0;
    // split_xorder_std(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p, yleft_mean_std, yright_mean_std, y_mean, y_std);
    // auto end = system_clock::now();
    // auto duration = duration_cast<std::chrono::nanoseconds>(end - start);
    // old_time = old_time + duration.count() / 1000;

    // start = system_clock::now();
    split_xorder_std_linkedlist(this->c, split_var, split_point, Xorder_next_index, X_std, N_y, p, N_Xorder, N_Xorder_left, N_Xorder_right, Xorder_firstline, Xorder_left_firstline, Xorder_right_firstline, Xorder_full, yleft_mean_std, yright_mean_std, y_mean, y_std);
    // end = system_clock::now();
    // duration = duration_cast<std::chrono::nanoseconds>(end - start);
    // new_time = new_time + duration.count() / 1000;

    // duration = duration_cast<microseconds>(end - start);
    // cout << "running time 2 " << duration.count() << endl;
    // free(Xorder_std);
    // cout<< "left " << yleft_mean_std << " " << yleft_mean2 << endl;
    // cout<< "right "<< yright_mean_std << " " << yright_mean2 << endl;
    double yleft_sum = yleft_mean_std * N_Xorder_left;
    double yright_sum = yright_mean_std * N_Xorder_right;

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_linkedlist(yleft_mean_std, yleft_sum, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, X_std, split_var_count_pointer, mtry, subset_vars, Xorder_next_index, Xorder_full, Xorder_left_firstline, N_Xorder_left);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_linkedlist(yright_mean_std, yright_sum, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, X_std, split_var_count_pointer, mtry, subset_vars, Xorder_next_index, Xorder_full, Xorder_right_firstline, N_Xorder_right);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    return;
}

void BART_likelihood_adaptive_linkedlist(double y_sum, std::vector<double> &y_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, const xinfo_sizet &Xorder_full, xinfo_sizet &Xorder_next_index, std::vector<size_t> Xorder_firstline, size_t &N_y, double &cutvalue, size_t &N_Xorder)
{
    // cout << "-------------------------------" << endl;
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = N_Xorder;
    // size_t N = Xorder_std[0].size();
    size_t p = Xorder_full.size();
    size_t ind;
    // size_t N_Xorder = N;

    double y_sum2;
    double sigma2 = pow(sigma, 2);

    double loglike_max = -INFINITY;

    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {

        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // initialize log likelihood at -INFINITY
        std::vector<double> loglike((N_Xorder - 1) * p + 1, -INFINITY);
        std::vector<double> y_cumsum(N_Xorder);

        xinfo possible_cutpoints;

        ini_xinfo(possible_cutpoints, N_Xorder, p);

        size_t current_index;
        size_t temp_index;
        if (parallel == false)
        {

            // for(size_t i = 0; i < p; i++){
            for (auto &&i : subset_vars)
            {

                current_index = Xorder_firstline[i];
                y_cumsum[0] = y_std[Xorder_full[i][current_index]];
                possible_cutpoints[i][0] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                current_index = Xorder_next_index[i][current_index];
                temp_index = 1;
                while (current_index < UINT_MAX)
                {
                    possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][current_index]);
                    y_cumsum[temp_index] = y_cumsum[temp_index - 1] + y_std[Xorder_full[i][current_index]];
                    current_index = Xorder_next_index[i][current_index];
                    temp_index++;
                }

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    loglike[(N_Xorder - 1) * i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    if (loglike[(N_Xorder - 1) * i + j] > loglike_max)
                    {
                        loglike_max = loglike[(N_Xorder - 1) * i + j];
                    }
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_fullset_std like_parallel_full(y_std, Xorder_std, N_Xorder, subset_vars, tau, Ntau, sigma2, loglike);
            // parallelFor(0, subset_vars.size(), like_parallel_full);
        }

        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        if (loglike[loglike.size() - 1] > loglike_max)
        {
            loglike_max = loglike[loglike.size() - 1];
        }

        // if(loglike_max != *std::max_element(loglike.begin(), loglike.end())){
        //     cout << "loglike_max " <<  loglike_max << " " <<  *std::max_element(loglike.begin(), loglike.end()) << endl;
        // }
        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            // if a variable is not selected, take exp will becomes 0
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        if ((N - 1) > 2 * Nmin)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                // delete some candidates, otherwise size of the new node can be smaller than Nmin
                std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
            }
        }
        else
        {
            no_split = true;
            cutvalue = 0;
            // cout << "no split " << endl;
            return;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(loglike.begin(), loglike.end());
        // sample one index of split point

        ind = d(gen);

        split_var = ind / (N - 1);
        split_point = ind % (N - 1);

        if (ind == (N - 1) * p)
        {
            no_split = true;
        }
        else
        {
            cutvalue = possible_cutpoints[split_var][split_point];
        }

        if ((N - 1) <= 2 * Nmin)
        {
            no_split = true;
        }
        // cout << " fine " << endl;

        // cout << "cutvalue " << cutvalue << " " << *(X_std + N_y * split_var + Xorder_std[split_var][split_point]) << endl;
    }
    else
    {

        // initialize loglikelihood at -INFINITY
        std::vector<double> loglike(Ncutpoints * p + 1, -INFINITY);
        std::vector<size_t> candidate_index(Ncutpoints);
        std::vector<double> y_cumsum(Ncutpoints);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        double Ntau = N_Xorder * tau;

        // double y_sum2;
        xinfo possible_cutpoints;

        ini_xinfo(possible_cutpoints, Ncutpoints, p);

        size_t current_index;
        size_t previous_index;
        size_t temp_index;
        size_t count;

        if (parallel == false)
        {

            bool firstrun = true; // flag of the first loop
            std::vector<double> Y_sort(N_Xorder);
            double *ypointer;
            double n1tau;
            double n2tau;
            for (auto &&i : subset_vars)
            {

                current_index = Xorder_firstline[i];
                temp_index = 0;
                y_cumsum[0] = 0;
                for (size_t m = 0; m <= candidate_index[0]; m++)
                {
                    y_cumsum[0] = y_cumsum[0] + y_std[Xorder_full[i][current_index]];
                    previous_index = current_index;
                    current_index = Xorder_next_index[i][current_index];
                }
                possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][previous_index]);
                y_cumsum[1] = y_cumsum[0];

                temp_index = 1;
                for (size_t m = 0; m < Ncutpoints - 1; m++)
                {
                    for (size_t q = candidate_index[m] + 1; q <= candidate_index[m + 1]; q++)
                    {
                        y_cumsum[temp_index] = y_cumsum[temp_index] + y_std[Xorder_full[i][current_index]];
                        previous_index = current_index;
                        current_index = Xorder_next_index[i][current_index];
                    }
                    possible_cutpoints[i][temp_index] = *(X_std + N_y * i + Xorder_full[i][previous_index]);
                    temp_index++;
                    y_cumsum[temp_index] = y_cumsum[temp_index - 1];
                }

                for (size_t j = 0; j < Ncutpoints; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;                   // number of points on right side (x > cutpoint)
                    loglike[(Ncutpoints)*i + j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum[j], 2) / (sigma2 * (n2tau + sigma2));

                    if (loglike[(Ncutpoints)*i + j] > loglike_max)
                    {
                        loglike_max = loglike[(Ncutpoints)*i + j];
                    }
                }
            }
        }
        else
        {

            // parallel computing

            // likelihood_subset_std like_parallel(y_std, Xorder_std, N_Xorder, Ncutpoints, subset_vars, tau, sigma2, candidate_index, loglike);
            // parallelFor(0, subset_vars.size(), like_parallel);
        }

        // no split option
        loglike[loglike.size() - 1] = log(N_Xorder) + log(p) - 0.5 * log(N_Xorder * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N_Xorder * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

        if (loglike[loglike.size() - 1] > loglike_max)
        {
            loglike_max = loglike[loglike.size() - 1];
        }

        // normalize loglike
        // double loglike_max = *std::max_element(loglike.begin(), loglike.end());

        // if(loglike_max != *std::max_element(loglike.begin(), loglike.end())){
        //     cout << "loglike_max " << loglike_max << " " <<  *std::max_element(loglike.begin(), loglike.end()) << endl;
        // }

        for (size_t ii = 0; ii < loglike.size(); ii++)
        {
            loglike[ii] = exp(loglike[ii] - loglike_max);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);

        split_var = ind / Ncutpoints;

        split_point = candidate_index[ind % Ncutpoints];

        if (ind == (Ncutpoints)*p)
        {
            no_split = true;
        }
        else
        {
            cutvalue = possible_cutpoints[split_var][ind % Ncutpoints];
        }
    }
    return;
}

void split_xorder_std_linkedlist(const double &cutvalue, const size_t &split_var, const size_t &split_point, xinfo_sizet &Xorder_next_index, const double *X_std, size_t N_y, size_t p, size_t &N_Xorder, size_t &N_Xorder_left, size_t &N_Xorder_right, std::vector<size_t> &Xorder_firstline, std::vector<size_t> &Xorder_left_firstline, std::vector<size_t> &Xorder_right_firstline, xinfo_sizet &Xorder_std_full, double &y_left_mean, double &y_right_mean, const double &y_mean, std::vector<double> &y_std)
{

    y_left_mean = 0.0;
    y_right_mean = 0.0;

    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes
    size_t left_ix = 0;
    size_t right_ix = 0;

    const double *temp_pointer = X_std + N_y * split_var;

    size_t current_index = 0;
    size_t left_previous_index = 0;
    size_t right_previous_index = 0;
    size_t next_index = 0;

    size_t count = 0;

    left_ix = 0;
    right_ix = 0;

    // cout << "inside ok 1" << endl;

    for (size_t i = 0; i < p; i++)
    {
        left_ix = 0;
        right_ix = 0;
        current_index = Xorder_firstline[i];
        left_previous_index = current_index;
        right_previous_index = current_index;
        next_index = current_index;

        // cout << "inside ok 2" << endl;

        if (i == split_var)
        {

            if (compute_left_side)
            {
                while (next_index < UINT_MAX)
                {

                    next_index = Xorder_next_index[i][current_index];

                    // cout << "inside ok 3" << " " << next_index <<   endl;

                    if (*(temp_pointer + Xorder_std_full[i][current_index]) <= cutvalue)
                    {
                        // cout << "left " << endl;

                        y_left_mean = y_left_mean + y_std[Xorder_std_full[split_var][current_index]];

                        if (left_ix == 0)
                        {
                            // cout << "left first " << endl;
                            Xorder_left_firstline[i] = current_index;
                            left_previous_index = current_index;
                            current_index = next_index;
                            left_ix++;
                        }
                        else
                        {
                            Xorder_next_index[i][left_previous_index] = current_index;
                            left_previous_index = current_index;
                            current_index = next_index;
                            left_ix++;
                        }
                    }
                    else
                    {

                        // cout << "right " << endl;
                        if (right_ix == 0)
                        {
                            // cout << "right first " << endl;
                            Xorder_right_firstline[i] = current_index;
                            right_previous_index = current_index;
                            current_index = next_index;
                            right_ix++;
                        }
                        else
                        {
                            Xorder_next_index[i][right_previous_index] = current_index;
                            right_previous_index = current_index;
                            current_index = next_index;
                            right_ix++;
                        }
                    }
                }
            }
            else
            {
                while (next_index < UINT_MAX)
                {

                    next_index = Xorder_next_index[i][current_index];

                    // cout << "inside ok 3" << " " << next_index <<   endl;

                    if (*(temp_pointer + Xorder_std_full[i][current_index]) <= cutvalue)
                    {
                        // cout << "left " << endl;

                        if (left_ix == 0)
                        {
                            // cout << "left first " << endl;
                            Xorder_left_firstline[i] = current_index;
                            left_previous_index = current_index;
                            current_index = next_index;
                            left_ix++;
                        }
                        else
                        {
                            Xorder_next_index[i][left_previous_index] = current_index;
                            left_previous_index = current_index;
                            current_index = next_index;
                            left_ix++;
                        }
                    }
                    else
                    {

                            y_right_mean = y_right_mean + y_std[Xorder_std_full[split_var][current_index]];
                        // cout << "right " << endl;
                        if (right_ix == 0)
                        {
                            // cout << "right first " << endl;
                            Xorder_right_firstline[i] = current_index;
                            right_previous_index = current_index;
                            current_index = next_index;
                            right_ix++;
                        }
                        else
                        {
                            Xorder_next_index[i][right_previous_index] = current_index;
                            right_previous_index = current_index;
                            current_index = next_index;
                            right_ix++;
                        }
                    }
                }
            }
        }
        else
        {
            while (next_index < UINT_MAX)
            {
                next_index = Xorder_next_index[i][current_index];

                // cout << "inside ok 3" << " " << next_index <<   endl;

                if (*(temp_pointer + Xorder_std_full[i][current_index]) <= cutvalue)
                {
                    // cout << "left " << endl;

                    if (left_ix == 0)
                    {
                        // cout << "left first " << endl;
                        Xorder_left_firstline[i] = current_index;
                        left_previous_index = current_index;
                        current_index = next_index;
                        left_ix++;
                    }
                    else
                    {
                        Xorder_next_index[i][left_previous_index] = current_index;
                        left_previous_index = current_index;
                        current_index = next_index;
                        left_ix++;
                    }
                }
                else
                {

                    // cout << "right " << endl;
                    if (right_ix == 0)
                    {
                        // cout << "right first " << endl;
                        Xorder_right_firstline[i] = current_index;
                        right_previous_index = current_index;
                        current_index = next_index;
                        right_ix++;
                    }
                    else
                    {
                        Xorder_next_index[i][right_previous_index] = current_index;
                        right_previous_index = current_index;
                        current_index = next_index;
                        right_ix++;
                    }
                }
            }
        }

        // while (next_index < UINT_MAX)
        // {

        //     next_index = Xorder_next_index[i][current_index];

        //     // cout << "inside ok 3" << " " << next_index <<   endl;

        //     if (*(temp_pointer + Xorder_std_full[i][current_index]) <= cutvalue)
        //     {
        //         // cout << "left " << endl;

        //         if (i == split_var && compute_left_side)
        //         {
        //             y_left_mean = y_left_mean + y_std[Xorder_std_full[split_var][current_index]];
        //         }

        //         if (left_ix == 0)
        //         {
        //             // cout << "left first " << endl;
        //             Xorder_left_firstline[i] = current_index;
        //             left_previous_index = current_index;
        //             current_index = next_index;
        //             left_ix++;
        //         }
        //         else
        //         {
        //             Xorder_next_index[i][left_previous_index] = current_index;
        //             left_previous_index = current_index;
        //             current_index = next_index;
        //             left_ix++;
        //         }
        //     }
        //     else
        //     {

        //         if (i == split_var && (!compute_left_side))
        //         {
        //             y_right_mean = y_right_mean + y_std[Xorder_std_full[split_var][current_index]];
        //         }
        //         // cout << "right " << endl;
        //         if (right_ix == 0)
        //         {
        //             // cout << "right first " << endl;
        //             Xorder_right_firstline[i] = current_index;
        //             right_previous_index = current_index;
        //             current_index = next_index;
        //             right_ix++;
        //         }
        //         else
        //         {
        //             Xorder_next_index[i][right_previous_index] = current_index;
        //             right_previous_index = current_index;
        //             current_index = next_index;
        //             right_ix++;
        //         }
        //     }
        // }

        if (left_ix >= N_Xorder_left)
        {
            Xorder_next_index[i][left_previous_index] = UINT_MAX;
        }
        if (right_ix >= N_Xorder_right)
        {
            Xorder_next_index[i][right_previous_index] = UINT_MAX;
        }
    }

    if (compute_left_side)
    {
        y_right_mean = (y_mean * N_Xorder - y_left_mean) / N_Xorder_right;
        y_left_mean = y_left_mean / N_Xorder_left;
    }
    else
    {
        y_left_mean = (y_mean * N_Xorder - y_right_mean) / N_Xorder_left;
        y_right_mean = y_right_mean / N_Xorder_right;
    }

    return;
}

#ifndef NoRcpp
// instead of returning y.test, let's return trees
// this conveniently avoids the need for x.test
// loosely based on pr()
// create an efficient list from a single tree
// tree2list calls itself recursively
Rcpp::List tree::tree2list(xinfo &xi, double center, double scale)
{
    Rcpp::List res;

    // five possible scenarios
    if (l)
    { // tree has branches
        //double cut=xi[v][c];
        size_t var = v, cut = c;

        var++;
        cut++; // increment from 0-based (C) to 1-based (R) array index

        if (l->l && r->l) // two sub-trees
            res = Rcpp::List::create(Rcpp::Named("var") = (size_t)var,
                                     //Rcpp::Named("cut")=cut,
                                     Rcpp::Named("cut") = (size_t)cut,
                                     Rcpp::Named("type") = 1,
                                     Rcpp::Named("left") = l->tree2list(xi, center, scale),
                                     Rcpp::Named("right") = r->tree2list(xi, center, scale));
        else if (l->l && !(r->l)) // left sub-tree and right terminal
            res = Rcpp::List::create(Rcpp::Named("var") = (size_t)var,
                                     //Rcpp::Named("cut")=cut,
                                     Rcpp::Named("cut") = (size_t)cut,
                                     Rcpp::Named("type") = 2,
                                     Rcpp::Named("left") = l->tree2list(xi, center, scale),
                                     Rcpp::Named("right") = r->gettheta() * scale + center);
        else if (!(l->l) && r->l) // left terminal and right sub-tree
            res = Rcpp::List::create(Rcpp::Named("var") = (size_t)var,
                                     //Rcpp::Named("cut")=cut,
                                     Rcpp::Named("cut") = (size_t)cut,
                                     Rcpp::Named("type") = 3,
                                     Rcpp::Named("left") = l->gettheta() * scale + center,
                                     Rcpp::Named("right") = r->tree2list(xi, center, scale));
        else // no sub-trees
            res = Rcpp::List::create(Rcpp::Named("var") = (size_t)var,
                                     //Rcpp::Named("cut")=cut,
                                     Rcpp::Named("cut") = (size_t)cut,
                                     Rcpp::Named("type") = 0,
                                     Rcpp::Named("left") = l->gettheta() * scale + center,
                                     Rcpp::Named("right") = r->gettheta() * scale + center);
    }
    else                                                 // no branches
        res = Rcpp::List::create(Rcpp::Named("var") = 0, // var=0 means root
                                 //Rcpp::Named("cut")=0.,
                                 Rcpp::Named("cut") = 0,
                                 Rcpp::Named("type") = 0,
                                 Rcpp::Named("left") = theta * scale + center,
                                 Rcpp::Named("right") = theta * scale + center);

    return res;
}

// for one tree, count the number of branches for each variable
Rcpp::IntegerVector tree::tree2count(size_t nvar)
{
    Rcpp::IntegerVector res(nvar);

    if (l)
    { // tree branches
        res[v]++;

        if (l->l)
            res += l->tree2count(nvar); // if left sub-tree
        if (r->l)
            res += r->tree2count(nvar); // if right sub-tree
    }                                   // else no branches and nothing to do

    return res;
}
#endif
