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

void tree::grow_tree_adaptive_abarth_train(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, double &run_time, Rcpp::IntegerVector &var_index_candidate, bool &use_all, std::vector<double> &mtry_weight_current_tree, std::vector<double> &split_count_current_tree)
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

    if (use_all)
    {
        subset_vars.resize(p);
        std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);
    }
    else
    {
        // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));

        // TEMP ADDED
        Rcpp::NumericVector mtry_rcpp(mtry_weight_current_tree.size(),0.0);
        for(int i=0;i<mtry_weight_current_tree.size();i++){
            mtry_rcpp[i] = mtry_weight_current_tree[i];
        }
        
        subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, mtry_rcpp));
    }
    cout << "ok1" << endl;
    BART_likelihood_adaptive_std_mtry_old(y_mean * N_Xorder, y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars);

    if (no_split == true)
    {
        cout << "no split" << endl;
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

    cout << "size of two sides " <<split_point + 1 << " " << N_Xorder - split_point - 1 << endl;

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

    cout << "ok2" << endl;

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
    lchild->grow_tree_adaptive_abarth_train(yleft_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_left_std, X_std, mtry, running_time_left, var_index_candidate, use_all,  mtry_weight_current_tree, split_count_current_tree);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_abarth_train(yright_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_right_std, X_std, mtry, running_time_right, var_index_candidate, use_all, mtry_weight_current_tree, split_count_current_tree);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    run_time = run_time + running_time + running_time_left + running_time_right;

    return;
}

void tree::grow_tree_adaptive_std_mtrywithinnode(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, double &run_time, Rcpp::IntegerVector &var_index_candidate, bool &use_all, Rcpp::NumericMatrix &split_count_all_tree, Rcpp::NumericVector &mtry_weight_current_tree, Rcpp::NumericVector &split_count_current_tree)
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

    if (use_all)
    {
        subset_vars.resize(p);
        std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);
    }
    else
    {
        // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));
        subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, mtry_weight_current_tree));
    }
    cout << "ok1" << endl;
    BART_likelihood_adaptive_std_mtry_old(y_mean * N_Xorder, y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars);

    if (no_split == true)
    {
        cout << "no split" << endl;
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

    cout << "size of two sides " <<split_point + 1 << " " << N_Xorder - split_point - 1 << endl;

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

    cout << "ok2" << endl;

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


void tree::grow_tree_adaptive_std_mtrywithinnode_ordinal(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_sigma, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, const int *X_recodepointer, xinfo_sizet &X_unique_counts, xinfo &X_unique_values, xinfo_sizet &index_changepoint, size_t &mtry, double &run_time, Rcpp::IntegerVector &var_index_candidate, bool &use_all, Rcpp::NumericMatrix &split_count_all_tree, Rcpp::NumericVector &mtry_weight_current_tree, Rcpp::NumericVector &split_count_current_tree, std::vector<size_t> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique)
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

    if (use_all)
    {
        subset_vars.resize(p);
        std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);
    }
    else
    {
        // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));
        // TEMP ADDED
        Rcpp::NumericVector mtry_rcpp(mtry_weight_current_tree.size(),0.0);
        for(int i=0;i<mtry_weight_current_tree.size();i++){
            mtry_rcpp[i] = mtry_weight_current_tree[i];
        }
        subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, mtry_rcpp));
    }
    BART_likelihood_adaptive_std_mtry_old_ordinal(y_mean * N_Xorder, y_std, Xorder_std, X_std, X_recodepointer, X_unique_counts, X_unique_values, index_changepoint, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars, X_values, X_counts, variable_ind, X_num_unique);

    cout <<"ok of BART_likelihood_adaptive_std_mtry_old_ordinal" << endl;

    if (no_split == true)
    {
        cout << "no split" << endl;
        return;
    }

    cout << "split var " << split_var << endl;
    cout << "split point " << split_point << " " << no_split << endl;

    this->v = split_var;
    this->c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    // this->c = X_unique_values[split_var][split_point];

    // split_var_count_pointer[split_var]++;

    split_count_current_tree[split_var] = split_count_current_tree[split_var] + 1;

        cout << "ok of all init" << endl;


    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

        cout << "ok of all init2" << endl;


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

    std::vector<size_t> X_counts_left(X_counts.size());
    std::vector<size_t> X_counts_right(X_counts.size());

    cout << X_counts << endl;
    cout << X_counts_left << endl;
    cout << X_counts_right << endl;

    cout << "split start " << endl;
    split_xorder_std_ordinal(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p, yleft_mean_std, yright_mean_std, y_mean, y_std, X_counts_left, X_counts_right, X_counts, X_values, variable_ind);

    cout << "ok of split" << endl;

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
    double running_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    xinfo_sizet X_unique_counts_left(p);
    xinfo_sizet X_unique_counts_right(p);
    xinfo X_unique_values_left(p);
    xinfo X_unique_values_right(p);
    xinfo_sizet index_changepoint_left(p);
    xinfo_sizet index_changepoint_right(p);

    // unique_value_count(X_std, X_unique_counts_left, X_unique_values_left, index_changepoint_left, Xorder_left_std);
    // unique_value_count(X_std, X_unique_counts_right, X_unique_values_right, index_changepoint_right, Xorder_right_std);

    cout << "ok5" << endl;

    // duration = duration_cast<microseconds>(end - start);
    // cout << "running time 2 " << duration.count() << endl;
    // free(Xorder_std);
    // cout<< "left " << yleft_mean_std << " " << yleft_mean2 << endl;
    // cout<< "right "<< yright_mean_std << " " << yright_mean2 << endl;

    double running_time_left = 0.0;
    double running_time_right = 0.0;

    depth = depth + 1;
    tree::tree_p lchild = new tree();
    lchild->grow_tree_adaptive_std_mtrywithinnode_ordinal(yleft_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_left_std, X_std, X_recodepointer, X_unique_counts_left, X_unique_values_left, index_changepoint_left, mtry, running_time_left, var_index_candidate, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, X_values, X_counts_left, variable_ind, X_num_unique);
    tree::tree_p rchild = new tree();
    rchild->grow_tree_adaptive_std_mtrywithinnode_ordinal(yright_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, y_std, Xorder_right_std, X_std, X_recodepointer, X_unique_counts_right, X_unique_values_right, index_changepoint_right, mtry, running_time_right, var_index_candidate, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, X_values, X_counts_right, variable_ind, X_num_unique);

    lchild->p = this;
    rchild->p = this;
    this->l = lchild;
    this->r = rchild;

    run_time = run_time + running_time + running_time_left + running_time_right;

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

void split_xorder_std_ordinal(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, const double *X_std, size_t N_y, size_t p, double &yleft_mean, double &yright_mean, const double &y_mean, std::vector<double> &y_std, std::vector<size_t> &X_counts_left, std::vector<size_t> &X_counts_right, std::vector<size_t> &X_counts, std::vector<size_t> &X_values, std::vector<size_t> &variable_ind)
{

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes

    // preserve order of other variables
    size_t N_Xorder = Xorder_std[0].size();
    size_t left_ix = 0;
    size_t right_ix = 0;
    size_t N_Xorder_left = Xorder_left_std[0].size();
    size_t N_Xorder_right = Xorder_right_std[0].size();

    size_t X_counts_index = 0;

    // if the left side is smaller, we only compute sum of it
    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    yleft_mean = 0.0;
    yright_mean = 0.0;

    size_t start;
    size_t end;

    

    double cutvalue = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);
    for (size_t i = 0; i < p; i++)
    {
        // loop over variables
        left_ix = 0;
        right_ix = 0;
        const double *temp_pointer = X_std + N_y * split_var;


        // index range of X_counts, X_values that are corresponding to current variable
        // start <= i <= end;
        start = variable_ind[i];
        end = variable_ind[i+1] - 1;



        if (i == split_var)
        {   
            // split the split_variable, only need to find row of cutvalue

            // I think this part can be optimizied, we know location of cutvalue (split_value variable)

            if (compute_left_side)
            {
                for (size_t j = 0; j < N_Xorder; j++)
                {

                    if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                    {   
                        // go to left side
                        yleft_mean = yleft_mean + y_std[Xorder_std[split_var][j]];
                        Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                        left_ix = left_ix + 1;
                    }
                    else
                    {
                        // go to right side
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

            // for the cut variable, it's easy to counts X_counts_left and X_counts_right, simply cut X_counts to two pieces.

            for(size_t k = start; k <= end; k ++){
                // loop from start to end!

                if(X_values[k] <= cutvalue){
                    // smaller than cutvalue, go left
                    X_counts_left[k] = X_counts[k];
                }else{
                    // otherwise go right
                    X_counts_right[k] = X_counts[k];
                }
            }



        }
        else
        {


            X_counts_index = start;


            // split other variables, need to compare each row
            for (size_t j = 0; j < N_Xorder; j++)
            {


                // while(*(temp_pointer + Xorder_std[i][j])!= X_counts[X_counts_index]){
                //     // for the current observation, find location of corresponding unique values
                //     X_counts_index ++ ;
                // }


                if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                {
                    // go to left side
                    Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                    left_ix = left_ix + 1;
                    // X_counts_left[X_counts_index] ++;
                }
                else
                {
                    // go to right side
                    Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                    right_ix = right_ix + 1;
                    // X_counts_right[X_counts_index] ++;
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

        // std::vector<double> Y_sort(N_Xorder); // a container for sorted y
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
            // std::vector<double> Y_sort(N_Xorder);
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
    }

    return;
}

void BART_likelihood_adaptive_std_mtry_old_ordinal(double y_sum, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, const int *X_recodepointer, xinfo_sizet &X_unique_count, xinfo &X_unique_values, xinfo_sizet &index_changepoint, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, std::vector<size_t> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique)
{
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    // for ordinal variables, always use all cutpoints

    cout << "---------------------------------------------------" << endl;
    cout << " begin of likelihood computation, print out inputs " << endl;
    cout << "X_counts " << X_counts << endl;
    cout << "X_values " << X_values << endl;
    cout << "variable_ind" << variable_ind << endl;
    cout << "Xnumunique" << X_num_unique << endl;
    

    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t ind;
    size_t N_Xorder = N;
    double loglike_max = -INFINITY;
    double sigma2 = pow(sigma, 2);
    size_t start;
    size_t end;
    size_t end2;
    double y_cumsum = 0.0;
    size_t n1;
    size_t n2;
    double n1tau;
    double n2tau;
    double ntau = (double)N_Xorder * tau;
    std::vector<double> loglike(X_values.size() + 1, -INFINITY);

    // cout << "variables to consider " << subset_vars << endl;
    size_t temp;


    // tau = 0.5;
    // sigma2 = 1.0;
    // alpha = 1.0;
    // beta = 2.0;
    // depth = 1;

    // ntau = (double) N_Xorder * tau;

    for (auto &&i : subset_vars)
    {
        // cout << "variable " << i << endl;
        if (X_num_unique[i] > 1)
        {
            // more than one unique values
            start = variable_ind[i];
            end = variable_ind[i + 1] - 1; // minus one for indexing starting at 0
            end2 = end;

            // cout << end2 << endl;

            // cout << "start " << start << endl;
            // cout << "end " << end << endl;

            while (X_counts[end2] == 0)
            {
                // move backward if the last unique value has zero counts
                end2 = end2 - 1;
                // cout << end2 << endl;
            }
            // move backward again, do not consider the last unique value as cutpoint
            end2 = end2 - 1;

            y_cumsum = 0.0;
            n1 = 0;
            for (size_t j = start; j <= end2; j++)
            {
                                // cout << "count " << X_counts[j] << endl;

                if (X_counts[j] != 0)
                {

                    temp = n1 + X_counts[j] - 1;

                    // cout << "----------------------------------" << endl;
                    // cout << "n1 = " << n1 << " temp = " << temp << endl;

                    partial_sum_y(y_std, Xorder_std, n1, temp, y_cumsum, i);

                    // cout << "y_cumsum " << y_cumsum << endl;

                    n1 = n1 + X_counts[j];
                    n1tau = (double)n1 * tau;
                    n2tau = ntau - n1tau;

                    // cout << "n1tau " << n1tau << " n2 tau " << n2tau << " sigma " << sigma2 << " N " << N_Xorder << endl;

                    loglike[j] = -0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum, 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_sum - y_cumsum, 2) / (sigma2 * (n2tau + sigma2));

                    // cout << "lll " << loglike[j] << endl;

                    if (loglike[j] > loglike_max)
                    {
                        loglike_max = loglike[j];
                    }
                }
            }
        }
    }
    loglike[loglike.size() - 1] = log(N) + log(p) - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

    // cout << "likelihood before exp " << loglike << endl;


    // find maximum
    if (loglike[loglike.size() - 1] > loglike_max)
    {
        loglike_max = loglike[loglike.size() - 1];
    }

    // take exp
    for (size_t ii = 0; ii < loglike.size(); ii++)
    {
        loglike[ii] = exp(loglike[ii] - loglike_max);
    }
    // cout << " ok of loglikelihood " << loglike << endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
    // // sample one index of split point
    ind = d(gen);

 
    for(size_t i = 0; i < (variable_ind.size() - 1); i ++){
        if(variable_ind[i] <= ind && variable_ind[i + 1] > ind){
            split_var = i;
        }
    }


    cout << X_values << endl;

    if (ind == (loglike.size() - 1))
    {   
        // cout << "do not split " << endl;
        no_split = true;
        split_var = 0;
        split_point = 0;
    }
    else
    {
        start = variable_ind[split_var];
        // count how many
        split_point = std::accumulate(X_counts.begin() + start, X_counts.begin() + ind + 1, 0);
        // minus one for correct index (start from 0)
        split_point = split_point - 1;
    }


    return;
}

void unique_value_count(const double *Xpointer, int *X_recodepointer, xinfo_sizet &X_unique_count, xinfo &X_unique_values, xinfo_sizet &index_changepoint, xinfo_sizet &Xorder_std, std::vector<size_t> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique)
{
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    double current_value = 0.0;
    size_t count_unique = 0;
    size_t N_unique;
    variable_ind[0] = 0;

    total_points = 0;
    for (size_t i = 0; i < p; i++)
    {
        X_counts.push_back(1);
        current_value = *(Xpointer + i * N + Xorder_std[i][0]);
        X_values.push_back(current_value);
        count_unique = 1;

        for (size_t j = 1; j < N; j++)
        {
            if (*(Xpointer + i * N + Xorder_std[i][j]) == current_value)
            {
                X_counts[total_points]++;
            }
            else
            {
                current_value = *(Xpointer + i * N + Xorder_std[i][j]);
                X_values.push_back(current_value);
                X_counts.push_back(1);
                count_unique++;
                total_points++;
            }
        }
        variable_ind[i + 1] = count_unique + variable_ind[i];
        X_num_unique[i] = count_unique;
        total_points++;
    }

    for (size_t i = 0; i < p; i++)
    {
        current_value = *(Xpointer + i * N + Xorder_std[i][0]);

        // for a new variable, push the first observation automatically
        X_unique_values[i].push_back(current_value);
        X_unique_count[i].push_back(1);
        count_unique = 0;
        *(X_recodepointer + i * N + Xorder_std[i][0]) = count_unique;

        for (size_t j = 1; j < N; j++)
        {
            if (*(Xpointer + i * N + Xorder_std[i][j]) == current_value)
            {
                X_unique_count[i][count_unique]++;
            }
            else
            {
                current_value = *(Xpointer + i * N + Xorder_std[i][j]);
                X_unique_values[i].push_back(current_value);
                X_unique_count[i].push_back(1);
                count_unique++;
            }

            *(X_recodepointer + i * N + Xorder_std[i][j]) = count_unique;
        }
        N_unique = X_unique_values[i].size();
        index_changepoint[i].resize(N_unique - 2);
        index_changepoint[i][0] = X_unique_count[i][0] + X_unique_count[i][1] - 1; // minus 1 for index, starts from 0

        // index_changepoint[i][0] = X_unique_count[i][0] - 1;
        for (size_t j = 1; j < N_unique - 2; j++)
        {
            index_changepoint[i][j] = index_changepoint[i][j - 1] + X_unique_count[i][j + 1];
        }
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
