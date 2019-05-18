
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
// #include <RcppArmadilloExtensions/sample.h>
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
        COUT << "tree size: " << treesize() << std::endl;
    COUT << pad << "(id,parent): " << id << sp << pid;
    COUT << sp << "(v,c): " << v << sp << c;
    COUT << sp << "theta: " << theta_vector;
    COUT << sp << "type: " << ntype();
    COUT << sp << "depth: " << depth();
    COUT << sp << "pointer: " << this << std::endl;

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
        COUT << "cp:error node has children\n";
        return;
    }

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

json tree::to_json()
{
    json j;
    if (l == 0)
    {
        j = this->theta_vector;
    }
    else
    {
        j["variable"] = this->v;
        j["cutpoint"] = this->c;
        j["left"] = this->l->to_json();
        j["right"] = this->r->to_json();
    }
    return j;
}

void tree::from_json(json &j3, size_t num_classes)
{
    if (j3.is_array())
    {
        std::vector<double> temp;
        j3.get_to(temp);
        if (temp.size() > 1)
        {
            this->theta_vector = temp;
        }
        else
        {
            this->theta_vector[0] = temp[0];
        }
    }
    else
    {
        j3.at("variable").get_to(this->v);
        j3.at("cutpoint").get_to(this->c);

        tree *lchild = new tree(num_classes);
        lchild->from_json(j3["left"], num_classes);
        tree *rchild = new tree(num_classes);
        rchild->from_json(j3["right"], num_classes);

        lchild->p = this;
        rchild->p = this;
        this->l = lchild;
        this->r = rchild;
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
        return is;
    }

    // The idea is to dump string to a lot of node_info structure first, then link them as a tree, by nid

    //read in vector of node information----------
    std::vector<node_info> nv(nn);
    for (size_t i = 0; i != nn; i++)
    {
        is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta_vector[0]; // Only works on first theta for now, fix latex if needed
        if (!is)
        {
            return is;
        }
    }

    //first node has to be the top one
    pts[1] = &t; //be careful! this is not the first pts, it is pointer of id 1.
    t.setv(nv[0].v);
    t.setc(nv[0].c);
    t.settheta(nv[0].theta_vector);
    t.p = 0;

    //now loop through the rest of the nodes knowing parent is already there.
    for (size_t i = 1; i != nv.size(); i++)
    {
        tree::tree_p np = new tree;
        np->v = nv[i].v;
        np->c = nv[i].c;
        np->theta_vector = nv[i].theta_vector;
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

void tree::grow_tree_adaptive_std_all(double y_mean, size_t depth, size_t max_depth, size_t Nmin, size_t Ncutpoints, double tau, double sigma, double alpha, double beta, bool draw_mu, bool parallel, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, size_t &mtry, bool &use_all, xinfo &split_count_all_tree, std::vector<double> &mtry_weight_current_tree, std::vector<double> &split_count_current_tree, bool &categorical_variables, size_t &p_categorical, size_t &p_continuous, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, Model *model, matrix<std::vector<double>*> &data_pointers, const size_t &tree_ind, std::mt19937 &gen, bool sample_weights_flag)

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

    model->samplePars(draw_mu, y_mean, N_Xorder, sigma, tau, gen, this->theta_vector, y_std, Xorder_std);

    this->sig = sigma;
    bool no_split = false;

    std::vector<size_t> subset_vars(p);

    if (use_all)
    {
        std::iota(subset_vars.begin(), subset_vars.end(), 0);

    }
    else
    {
        if (sample_weights_flag){
            std::vector<double> weight_samp(p);
            double weight_sum;

            // Sample Weights Dirchelet
            for (size_t i=0; i < p; i++)
            {
                std::gamma_distribution<double> temp_dist(mtry_weight_current_tree[i], 1.0);
                weight_samp[i] = temp_dist(gen);
            }
            weight_sum =  accumulate(weight_samp.begin(), weight_samp.end(), 0.0);
            for (size_t i=0; i < p; i++)
            {
                weight_samp[i] = weight_samp[i] / weight_sum;

            }

            subset_vars = sample_int_ccrank(p, mtry, weight_samp, gen);
        }else{
            subset_vars = sample_int_ccrank(p, mtry, mtry_weight_current_tree, gen);
        }
        
    }

    BART_likelihood_all(y_mean * N_Xorder, y_std, Xorder_std, X_std, tau, sigma, depth, Nmin, Ncutpoints, alpha, beta, no_split, split_var, split_point, parallel, subset_vars, p_categorical, p_continuous, X_values, X_counts, variable_ind, X_num_unique, model, gen, mtry, this->prob_split);

    if (no_split == true)
    {
         for (size_t i = 0; i < N_Xorder; i++)
        {
            data_pointers[tree_ind][Xorder_std[0][i]] = &this->theta_vector;
        }
        model->samplePars(draw_mu, y_mean, N_Xorder, sigma, tau, gen, this->theta_vector, y_std, Xorder_std);
        this->l = 0;
        this->r = 0;
        return;
    }

    this->v = split_var;
    this->c = *(X_std + N_y * split_var + Xorder_std[split_var][split_point]);

    // Update Cutpoint to be a true seperating point
    // Increase split_point (index) until it is no longer equal to cutpoint value
    while ((split_point < N_Xorder - 1) && (*(X_std + N_y * split_var + Xorder_std[split_var][split_point + 1]) == this->c))
    {
        split_point = split_point + 1;
    }
    
    // If our current split is same as parent, exit
    if ( (this->p) && (this->v == (this->p)->v) && (this->c == (this->p)->c) ) {
        return;
    }


    split_count_current_tree[split_var] = split_count_current_tree[split_var] + 1;

    //COUT << split_count_current_tree << endl;

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    double yleft_mean_std = 0.0;
    double yright_mean_std = 0.0;

    std::vector<size_t> X_num_unique_left(X_num_unique.size());
    std::vector<size_t> X_num_unique_right(X_num_unique.size());

    std::vector<size_t> X_counts_left(X_counts.size());
    std::vector<size_t> X_counts_right(X_counts.size());

    if (p_categorical > 0)
    {
        split_xorder_std_categorical(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p, p_continuous, p_categorical, yleft_mean_std, yright_mean_std, y_mean, y_std, X_counts_left, X_counts_right, X_num_unique_left, X_num_unique_right, X_counts, X_values, variable_ind, model);
    }

    if (p_continuous > 0)
    {
        split_xorder_std_continuous(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_std, N_y, p, p_continuous, p_categorical, yleft_mean_std, yright_mean_std, y_mean, y_std, model);
    }

    depth++;

    tree::tree_p lchild = new tree(model->getNumClasses(),this);
    lchild->grow_tree_adaptive_std_all(yleft_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta,
                                       draw_mu, parallel, y_std, Xorder_left_std, X_std, mtry, use_all, split_count_all_tree,
                                       mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous,
                                       X_values, X_counts_left, variable_ind, X_num_unique_left, model, data_pointers, tree_ind, gen,sample_weights_flag);

    tree::tree_p rchild = new tree(model->getNumClasses(),this);
    rchild->grow_tree_adaptive_std_all(yright_mean_std, depth, max_depth, Nmin, Ncutpoints, tau, sigma, alpha, beta,
                                       draw_mu, parallel, y_std, Xorder_right_std, X_std, mtry, use_all, split_count_all_tree,
                                       mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous,
                                       X_values, X_counts_right, variable_ind, X_num_unique_right, model, data_pointers, tree_ind, gen,sample_weights_flag);

    this->l = lchild;
    this->r = rchild;

    return;
}

void split_xorder_std_continuous(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, const double *X_std, size_t N_y, size_t p, size_t p_continuous, size_t p_categorical, double &yleft_mean, double &yright_mean, const double &y_mean, std::vector<double> &y_std, Model *model)
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

    const double *temp_pointer = X_std + N_y * split_var;

    for (size_t j = 0; j < N_Xorder; j++)
    {
        if (compute_left_side)
        {
            if (*(temp_pointer + Xorder_std[split_var][j]) <= cutvalue)
            {
                yleft_mean = yleft_mean + y_std[Xorder_std[split_var][j]];

                // // temp
                // std::vector<double> temp_vector = { y_std[Xorder_std[split_var][j]] };
                // model->updateTotalSuffStat(temp_vector ,N_Xorder_left);
            }
        }
        else
        {
            if (*(temp_pointer + Xorder_std[split_var][j]) > cutvalue)
            {
                yright_mean = yright_mean + y_std[Xorder_std[split_var][j]];
            }
        }
    }

    const double *split_var_x_pointer = X_std + N_y * split_var;

    for (size_t i = 0; i < p_continuous; i++) // loop over variables
    {
        // lambda callback for multithreading
        auto split_i = [&, i]() {
            size_t left_ix = 0;
            size_t right_ix = 0;

            std::vector<size_t> &xo = Xorder_std[i];
            std::vector<size_t> &xo_left = Xorder_left_std[i];
            std::vector<size_t> &xo_right = Xorder_right_std[i];

            for (size_t j = 0; j < N_Xorder; j++)
            {
                if (*(split_var_x_pointer + xo[j]) <= cutvalue)
                {
                    xo_left[left_ix] = xo[j];
                    left_ix = left_ix + 1;
                }
                else
                {
                    xo_right[right_ix] = xo[j];
                    right_ix = right_ix + 1;
                }
            }
        };
        if (thread_pool.is_active())
            thread_pool.add_task(split_i);
        else
            split_i();
    }
    if (thread_pool.is_active())
        thread_pool.wait();

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

    return;
}

void split_xorder_std_categorical(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, const double *X_std, size_t N_y, size_t p, size_t p_continuous, size_t p_categorical, double &yleft_mean, double &yright_mean, const double &y_mean, std::vector<double> &y_std, std::vector<size_t> &X_counts_left, std::vector<size_t> &X_counts_right, std::vector<size_t> &X_num_unique_left, std::vector<size_t> &X_num_unique_right, std::vector<size_t> &X_counts, std::vector<double> &X_values, std::vector<size_t> &variable_ind, Model *model)
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

    for (size_t i = p_continuous; i < p; i++)
    {
        // loop over variables
        left_ix = 0;
        right_ix = 0;
        const double *temp_pointer = X_std + N_y * split_var;

        // index range of X_counts, X_values that are corresponding to current variable
        // start <= i <= end;
        start = variable_ind[i - p_continuous];
        // COUT << "start " << start << endl;
        end = variable_ind[i + 1 - p_continuous];

        if (i == split_var)
        {
            // split the split_variable, only need to find row of cutvalue

            // I think this part can be optimizied, we know location of cutvalue (split_value variable)

            // COUT << "compute left side " << compute_left_side << endl;

            ///////////////////////////////////////////////////////////
            //
            // We should be able to run this part in parallel
            //
            //  just like split_xorder_std_continuous
            //
            ///////////////////////////////////////////////////////////

            if (compute_left_side)
            {
                for (size_t j = 0; j < N_Xorder; j++)
                {

                    if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                    {
                        // go to left side
                        yleft_mean = yleft_mean + y_std[Xorder_std[split_var][j]];
                        // model->
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

            for (size_t k = start; k < end; k++)
            {
                // loop from start to end!

                if (X_values[k] <= cutvalue)
                {
                    // smaller than cutvalue, go left
                    X_counts_left[k] = X_counts[k];
                }
                else
                {
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

                while (*(X_std + N_y * i + Xorder_std[i][j]) != X_values[X_counts_index])
                {
                    //     // for the current observation, find location of corresponding unique values
                    X_counts_index++;
                }

                if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                {
                    // go to left side
                    Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                    left_ix = left_ix + 1;

                    X_counts_left[X_counts_index]++;
                }
                else
                {
                    // go to right side

                    Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                    right_ix = right_ix + 1;

                    X_counts_right[X_counts_index]++;
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

    // update X_num_unique

    std::fill(X_num_unique_left.begin(), X_num_unique_left.end(), 0.0);
    std::fill(X_num_unique_right.begin(), X_num_unique_right.end(), 0.0);

    for (size_t i = p_continuous; i < p; i++)
    {
        start = variable_ind[i - p_continuous];
        end = variable_ind[i + 1 - p_continuous];

        // COUT << "start " << start << " end " << end << " size " << X_counts_left.size() << endl;
        for (size_t j = start; j < end; j++)
        {
            if (X_counts_left[j] > 0)
            {
                X_num_unique_left[i - p_continuous] = X_num_unique_left[i - p_continuous] + 1;
            }
            if (X_counts_right[j] > 0)
            {
                X_num_unique_right[i - p_continuous] = X_num_unique_right[i - p_continuous] + 1;
            }
        }
    }

    return;
}

void BART_likelihood_all(double y_sum, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double *X_std, double tau, double sigma, size_t depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, bool &no_split, size_t &split_var, size_t &split_point, bool parallel, const std::vector<size_t> &subset_vars, size_t &p_categorical, size_t &p_continuous, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, Model *model, std::mt19937 &gen, size_t &mtry, double &prob_split)

{
    // compute BART posterior (loglikelihood + logprior penalty)

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t ind;
    size_t N_Xorder = N;
    size_t total_categorical_split_candidates = 0;

    double y_sum2;
    double sigma2 = pow(sigma, 2);

    double loglike_max = -INFINITY;

    std::vector<double> loglike;

    size_t loglike_start;

    // decide lenght of loglike vector
    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {
        loglike.resize((N_Xorder - 1) * p_continuous + X_values.size() + 1, -INFINITY);
        loglike_start = (N_Xorder - 1) * p_continuous;
    }
    else
    {
        loglike.resize(Ncutpoints * p_continuous + X_values.size() + 1, -INFINITY);
        loglike_start = Ncutpoints * p_continuous;
    }

    // calculate for each cases
    if (p_continuous > 0)
    {
        calculate_loglikelihood_continuous(loglike, subset_vars, N_Xorder, Nmin, y_std, Xorder_std, y_sum, beta, alpha, depth, p, p_continuous, Ncutpoints, tau, sigma2, loglike_max, model, mtry);
    }

    if (p_categorical > 0)
    {
        calculate_loglikelihood_categorical(loglike, loglike_start, subset_vars, N_Xorder, Nmin, y_std, Xorder_std, y_sum, beta, alpha, depth, p, p_continuous, p_categorical, Ncutpoints, tau, sigma2, loglike_max, X_values, X_counts, variable_ind, X_num_unique, model, mtry, total_categorical_split_candidates);
    }

    // calculate likelihood of no-split option
    calculate_likelihood_no_split(loglike, N_Xorder, Nmin, y_sum, beta, alpha, depth, p, p_continuous, Ncutpoints, tau, sigma2, loglike_max, model, mtry, total_categorical_split_candidates);

    // transfer loglikelihood to likelihood
    for (size_t ii = 0; ii < loglike.size(); ii++)
    {
        // if a variable is not selected, take exp will becomes 0
        loglike[ii] = exp(loglike[ii] - loglike_max);
    }

    // sampling cutpoints
    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        if ((N - 1) > 2 * Nmin)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                if (i < p_continuous)
                {
                    // delete some candidates, otherwise size of the new node can be smaller than Nmin
                    std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + Nmin + 1, 0.0);
                    std::fill(loglike.begin() + i * (N - 1) + N - 2 - Nmin, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
                }
            }
        }
        else
        {
            // do not use all continuous variables
            std::fill(loglike.begin(), loglike.begin() + (N_Xorder - 1) * p_continuous - 1, 0.0);
        }

        std::discrete_distribution<> d(loglike.begin(), loglike.end());
        // sample one index of split point

        ind = d(gen);


        // save the posterior of the chosen split point
        vec_sum(loglike, prob_split);
        prob_split = loglike[ind] / prob_split;

        if (ind == loglike.size() - 1)
        {
            // no split
            no_split = true;
            split_var = 0;
            split_point = 0;
        }
        else if ((N - 1) <= 2 * Nmin)
        {
            // np split

            /////////////////////////////////
            //
            // Need optimization, move before calculating likelihood
            //
            /////////////////////////////////

            no_split = true;
            split_var = 0;
            split_point = 0;
        }
        else if (ind < loglike_start)
        {
            // split at continuous variable
            split_var = ind / (N - 1);
            split_point = ind % (N - 1);
        }
        else
        {
            // split at categorical variable
            size_t start;
            ind = ind - loglike_start;
            for (size_t i = 0; i < (variable_ind.size() - 1); i++)
            {
                if (variable_ind[i] <= ind && variable_ind[i + 1] > ind)
                {
                    split_var = i;
                }
            }
            start = variable_ind[split_var];
            // count how many
            split_point = std::accumulate(X_counts.begin() + start, X_counts.begin() + ind + 1, 0);
            // minus one for correct index (start from 0)
            split_point = split_point - 1;
            split_var = split_var + p_continuous;
        }
    }
    else
    {
        // use adaptive number of cutpoints

        std::vector<size_t> candidate_index(Ncutpoints);

        seq_gen_std(Nmin, N - Nmin, Ncutpoints, candidate_index);

        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());
        // // sample one index of split point
        ind = d(gen);


        // save the posterior of the chosen split point
        vec_sum(loglike, prob_split);
        prob_split = loglike[ind] / prob_split;

        
        if (ind == loglike.size() - 1)
        {
            // no split
            no_split = true;
            split_var = 0;
            split_point = 0;
        }
        else if (ind < loglike_start)
        {
            // split at continuous variable
            split_var = ind / Ncutpoints;
            split_point = candidate_index[ind % Ncutpoints];
        }
        else
        {
            // split at categorical variable
            size_t start;
            ind = ind - loglike_start;
            for (size_t i = 0; i < (variable_ind.size() - 1); i++)
            {
                if (variable_ind[i] <= ind && variable_ind[i + 1] > ind)
                {
                    split_var = i;
                }
            }
            start = variable_ind[split_var];
            // count how many
            split_point = std::accumulate(X_counts.begin() + start, X_counts.begin() + ind + 1, 0);
            // minus one for correct index (start from 0)
            split_point = split_point - 1;
            split_var = split_var + p_continuous;
        }
    }

    return;
}

void unique_value_count(const double *Xpointer, xinfo_sizet &Xorder_std, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique)
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

    return;
}



void calculate_loglikelihood_continuous(std::vector<double> &loglike, const std::vector<size_t> &subset_vars, size_t &N_Xorder, size_t &Nmin, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double &y_sum, const double &beta, const double &alpha, size_t &depth, const size_t &p, size_t &p_continuous, size_t &Ncutpoints, double &tau, double &sigma2, double &loglike_max, Model *model, size_t &mtry)
{

    size_t N = N_Xorder;
    size_t var_index;

    if (N <= Ncutpoints + 1 + 2 * Nmin)
    {
        // if we only have a few data observations in current node
        // use all of them as cutpoint candidates

        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;

        // to have a generalized function, have to pass an empty candidate_index object for this case
        // is there any smarter way to do it?
        std::vector<size_t> candidate_index(1);

        for (auto &&i : subset_vars)
        {
            if (i < p_continuous)
            {
                std::vector<size_t> &xorder = Xorder_std[i];

                // initialize sufficient statistics
                model->suff_stat_fill_zero();
                //model->suff_stat_fill(0.0);

                ////////////////////////////////////////////////////////////////
                //
                //  This part can be run in parallel, just like continuous case below, Ncutpoint case
                //
                //  If run in parallel, need to redefine model class for each thread
                //
                ////////////////////////////////////////////////////////////////

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    // loop over all possible cutpoints
                    n1tau = (j + 1) * tau; // number of points on left side (x <= cutpoint)
                    n2tau = Ntau - n1tau;  // number of points on right side (x > cutpoint)

                    model->calcSuffStat_continuous(xorder, y_std, candidate_index, j, false);

                    loglike[(N_Xorder - 1) * i + j] = model->likelihood(tau, n1tau, sigma2, y_sum, true) + model->likelihood(tau, n2tau, sigma2, y_sum, false);

                    if (loglike[(N_Xorder - 1) * i + j] > loglike_max)
                    {
                        loglike_max = loglike[(N_Xorder - 1) * i + j];
                    }
                }
            }
        }
    }
    else
    {

        // otherwise, adaptive number of cutpoints
        // use Ncutpoints

        std::vector<size_t> candidate_index2(Ncutpoints + 1);
        seq_gen_std2(Nmin, N - Nmin, Ncutpoints, candidate_index2);

        double Ntau = N_Xorder * tau;

        std::mutex llmax_mutex;

        for (auto &&i : subset_vars)
        {
            if (i < p_continuous)
            {

                // Lambda callback to perform the calculation
                auto calcllc_i = [i, &loglike, &loglike_max, &Xorder_std, &y_std, &candidate_index2, &model, &llmax_mutex, Ncutpoints, N_Xorder, Ntau, tau, sigma2, y_sum]() {
                    std::vector<size_t> &xorder = Xorder_std[i];
                    double llmax = -INFINITY;

                    // std::vector<double> y_cumsum(Ncutpoints);
                    Model *clone = model->clone();
                    //model -> suff_stat_init();
                    // clone->suff_stat_fill(y_std, xorder);
                    clone->suff_stat_fill_zero();
                    
                    for (size_t j = 0; j < Ncutpoints; j++)
                    {
                        clone->calcSuffStat_continuous(xorder, y_std, candidate_index2, j, true);

                        // loop over all possible cutpoints
                        double n1tau = (candidate_index2[j + 1] + 1) * tau; // number of points on left side (x <= cutpoint)
                        double n2tau = Ntau - n1tau;                        // number of points on right side (x > cutpoint)

                        loglike[(Ncutpoints)*i + j] = clone->likelihood(tau, n1tau, sigma2, y_sum, true) + clone->likelihood(tau, n2tau, sigma2, y_sum, false);

                        if (loglike[(Ncutpoints)*i + j] > llmax)
                        {
                            llmax = loglike[(Ncutpoints)*i + j];
                        }
                    }
                    delete clone;
                    llmax_mutex.lock();
                    if (llmax > loglike_max)
                        loglike_max = llmax;
                    llmax_mutex.unlock();
                };

                if (thread_pool.is_active())
                    thread_pool.add_task(calcllc_i);
                else
                    calcllc_i();
            }
        }
        if (thread_pool.is_active())
            thread_pool.wait();
    }
}

void calculate_loglikelihood_categorical(std::vector<double> &loglike, size_t &loglike_start, const std::vector<size_t> &subset_vars, size_t &N_Xorder, size_t &N_min, std::vector<double> &y_std, xinfo_sizet &Xorder_std, const double &y_sum, const double &beta, const double &alpha, size_t &depth, const size_t &p, const size_t &p_continuous, size_t &p_categorical, size_t &Ncutpoints, double &tau, double &sigma2, double &loglike_max, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, std::vector<size_t> &X_num_unique, Model *model, size_t &mtry, size_t &total_categorical_split_candidates)
{

    // loglike_start is an index to offset
    // consider loglikelihood start from loglike_start

    size_t start;
    size_t end;
    size_t end2;
    double y_cumsum = 0.0;
    size_t n1;
    size_t n2;
    double n1tau;
    double n2tau;
    double ntau = (double)N_Xorder * tau;
    size_t temp;
    size_t N = N_Xorder;

    size_t effective_cutpoints = 0;

    for (auto &&i : subset_vars)
    {

        // COUT << "variable " << i << endl;
        if ((i >= p_continuous) && (X_num_unique[i - p_continuous] > 1))
        {
            // more than one unique values
            start = variable_ind[i - p_continuous];
            end = variable_ind[i + 1 - p_continuous] - 1; // minus one for indexing starting at 0
            end2 = end;

            while (X_counts[end2] == 0)
            {
                // move backward if the last unique value has zero counts
                end2 = end2 - 1;
                // COUT << end2 << endl;
            }
            // move backward again, do not consider the last unique value as cutpoint
            end2 = end2 - 1;

            y_cumsum = 0.0;
            model->suff_stat_fill_zero();
            //model -> suff_stat_fill(0.0); // initialize sufficient statistics

            ////////////////////////////////////////////////////////////////
            //
            //  This part can be run in parallel, just like continuous case
            //
            //  If run in parallel, need to redefine model class for each thread
            //
            ////////////////////////////////////////////////////////////////

            n1 = 0;

            for (size_t j = start; j <= end2; j++)
            {

                if (X_counts[j] != 0)
                {

                    temp = n1 + X_counts[j] - 1;

                    // modify sufficient statistics vector directly inside model class
                    model->calcSuffStat_categorical(y_std, Xorder_std, n1, temp, i);

                    n1 = n1 + X_counts[j];
                    n1tau = (double)n1 * tau;
                    n2tau = ntau - n1tau;

                    loglike[loglike_start + j] = model->likelihood(tau, n1tau, sigma2, y_sum, true) + model->likelihood(tau, n2tau, sigma2, y_sum, false);

                    // count total number of cutpoint candidates
                    effective_cutpoints++;

                    if (loglike[loglike_start + j] > loglike_max)
                    {
                        loglike_max = loglike[loglike_start + j];
                    }
                }
            }
        }
    }
}

void calculate_likelihood_no_split(std::vector<double> &loglike, size_t &N_Xorder, size_t &Nmin, const double &y_sum, const double &beta, const double &alpha, size_t &depth, const size_t &p, size_t &p_continuous, size_t &Ncutpoints, double &tau, double &sigma2, double &loglike_max, Model *model, size_t &mtry, size_t &total_categorical_split_candidates)
{

    loglike[loglike.size() - 1] = model->likelihood_no_split(y_sum, tau, N_Xorder * tau, sigma2) + log(1.0 - alpha * pow(1.0 + depth, -1.0 * beta)) - log(alpha) + beta * log(1.0 + depth);

    // then adjust according to number of variables and split points

    ////////////////////////////////////////////////////////////////
    //
    //  For now, I didn't test much weights, but set it as p * Ncutpoints for all cases
    //
    //  BE CAREFUL, p is total number of variables, p = p_continuous + p_categorical
    //
    //  We might want to scale by mtry, the actual number of variables used in the current fit
    //
    //  WARNING, you need to consider weighting for both continuous and categorical variables here
    //
    //  This is the only function calculating no-split likelihood
    //
    ////////////////////////////////////////////////////////////////

    loglike[loglike.size() - 1] += log(p) + log(2.0) + model->getNoSplitPenality();

    ////////////////////////////////////////////////////////////////
    // The loop below might be useful when test different weights

    // if (p_continuous > 0)
    // {
    //     // if using continuous variable
    //     if (N_Xorder <= Ncutpoints + 1 + 2 * Nmin)
    //     {
    //         loglike[loglike.size() - 1] += log(p) + log(Ncutpoints);
    //     }
    //     else
    //     {
    //         loglike[loglike.size() - 1] += log(p) + log(Ncutpoints);
    //     }
    // }

    // if (p > p_continuous)
    // {
    //     COUT << "total_categorical_split_candidates  " << total_categorical_split_candidates << endl;
    //     // if using categorical variables
    //     // loglike[loglike.size() - 1] += log(total_categorical_split_candidates);
    // }

    // loglike[loglike.size() - 1] += log(p - p_continuous) + log(Ncutpoints);

    // this is important, update maximum of loglike vector
    if (loglike[loglike.size() - 1] > loglike_max)
    {
        loglike_max = loglike[loglike.size() - 1];
    }
}

void fit_new_std(tree &tree, const double *X_std, size_t N, size_t p, std::vector<double> &output)
{
    tree::tree_p bn;
    for (size_t i = 0; i < N; i++)
    {
        bn = tree.search_bottom_std(X_std, i, p, N);
        output[i] = bn->theta_vector[0];
    }
    return;
}

void fit_new_std_datapointers(const double *X_std, size_t N, size_t M, std::vector<double> &output, matrix<std::vector<double>*> &data_pointers)
{
    // tree search, but read from the matrix of pointers to end node directly
    // easier to get fitted value of training set
    for (size_t i = 0; i < N; i++)
    {
        output[i] = (*data_pointers[M][i])[0];
    }
    return;
}

#ifndef NoRcpp
#endif
