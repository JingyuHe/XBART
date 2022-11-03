//////////////////////////////////////////////////////////////////////////////////////
// tree operating functions
//////////////////////////////////////////////////////////////////////////////////////

#include "tree.h"
#include <chrono>
#include <ctime>
using namespace std;
using namespace chrono;

//--------------------
// node id
size_t tree::nid() const
{
    if (!p)
        return 1; // if you don't have a parent, you are the top
    if (this == p->l)
        return 2 * (p->nid()); // if you are a left child
    else
        return 2 * (p->nid()) + 1; // else you are a right child
}
//--------------------
tree::tree_p tree::getptr(size_t nid)
{
    if (this->nid() == nid)
        return this; // found it
    if (l == 0)
        return 0; // no children, did not find it
    tree_p lp = l->getptr(nid);
    if (lp)
        return lp; // found on left
    tree_p rp = r->getptr(nid);
    if (rp)
        return rp; // found on right
    return 0;      // never found it
}
//--------------------
// depth of node
// size_t tree::depth()
// {
//     if (!p)
//         return 0; //no parents
//     else
//         return (1 + p->depth());
// }
//--------------------
// tree size
size_t tree::treesize()
{
    if (l == 0)
        return 1; // if bottom node, tree size is 1
    else
        return (1 + l->treesize() + r->treesize());
}
//--------------------
// node type
char tree::ntype()
{
    // t:top, b:bottom, n:no grandchildren, i:internal
    if (!p)
        return 't';
    if (!l)
        return 'b';
    if (!(l->l) && !(r->l))
        return 'n';
    return 'i';
}
//--------------------
// print out tree(pc=true) or node(pc=false) information
void tree::pr(bool pc)
{
    size_t d = this->depth;
    size_t id = nid();

    size_t pid;
    if (!p)
        pid = 0; // parent of top node
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
    COUT << sp << "depth: " << this->depth;
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
// is the node a nog node
bool tree::isnog()
{
    bool isnog = true;
    if (l)
    {
        if (l->l || r->l)
            isnog = false; // one of the children has children.
    }
    else
    {
        isnog = false; // no children
    }
    return isnog;
}
//--------------------
size_t tree::nnogs()
{
    if (!l)
        return 0; // bottom node
    if (l->l || r->l)
    { // not a nog
        return (l->nnogs() + r->nnogs());
    }
    else
    { // is a nog
        return 1;
    }
}
//--------------------
size_t tree::nbots()
{
    if (l == 0)
    { // if a bottom node
        return 1;
    }
    else
    {
        return l->nbots() + r->nbots();
    }
}
//--------------------
// get bottom nodes
void tree::getbots(npv &bv)
{
    if (l)
    { // have children
        l->getbots(bv);
        r->getbots(bv);
    }
    else
    {
        bv.push_back(this);
    }
}
//--------------------
// get nog nodes
void tree::getnogs(npv &nv)
{
    if (l)
    { // have children
        if ((l->l) || (r->l))
        { // have grandchildren
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
// get pointer to the top tree
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
// get all nodes
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
tree::tree_p tree::bn(double *x, matrix<double> &xi)
{

    // original BART function, v and c are index of split point in matrix<double>& xi

    if (l == 0)
        return this; // no children
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
    // not index in matrix<double>, so compare x[v] with c directly

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
// find region for a given variable
void tree::rg(size_t v, size_t *L, size_t *U)
{
    if (this->p == 0)
    {
        return;
    }
    if ((this->p)->v == v)
    { // does my parent use v?
        if (this == p->l)
        { // am I left or right child
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
// cut back to one node
void tree::tonull()
{
    size_t ts = treesize();
    // loop invariant: ts>=1
    while (ts > 1)
    { // if false ts=1
        npv nv;
        getnogs(nv);
        for (size_t i = 0; i < nv.size(); i++)
        {
            delete nv[i]->l;
            delete nv[i]->r;
            nv[i]->l = 0;
            nv[i]->r = 0;
        }
        ts = treesize(); // make invariant true
    }
    v = 0;
    c = 0;
    p = 0;
    l = 0;
    r = 0;
}
//--------------------
// copy tree tree o to tree n
void tree::cp(tree_p n, tree_cp o)
// assume n has no children (so we don't have to kill them)
// recursion down
//  create a new copy of tree in NEW memory space
{
    if (n->l)
    {
        COUT << "cp:error node has children\n";
        return;
    }

    n->v = o->v;
    n->c = o->c;
    n->prob_split = o->prob_split;
    n->prob_leaf = o->prob_leaf;
    n->drawn_ind = o->drawn_ind;
    n->loglike_node = o->loglike_node;
    n->tree_like = o->tree_like;
    n->theta_vector = o->theta_vector;

    if (o->l)
    { // if o has children
        n->l = new tree;
        (n->l)->p = n;
        cp(n->l, o->l);
        n->r = new tree;
        (n->r)->p = n;
        cp(n->r, o->r);
    }
}

void tree::copy_only_root(tree_p o)
// assume n has no children (so we don't have to kill them)
// NOT LIKE cp() function
// this function pointer new root to the OLD structure
{
    this->v = o->v;
    this->c = o->c;
    this->prob_split = o->prob_split;
    this->prob_leaf = o->prob_leaf;
    this->drawn_ind = o->drawn_ind;
    this->loglike_node = o->loglike_node;
    this->tree_like = o->tree_like;
    this->theta_vector = o->theta_vector;

    if (o->l)
    {
        // keep the following structure, rather than create a new tree in memory
        this->l = o->l;
        this->r = o->r;
        // also update pointers to parents
        this->l->p = this;
        this->r->p = this;
    }
    else
    {
        this->l = 0;
        this->r = 0;
    }
}

json tree::to_json()
{
    json j;
    if (l == 0)
    {
        j["left"] = 0;
        j["right"] = 0;
        j["theta"] = this->theta_vector;
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

size_t tree::get_max_depth()
{
    size_t output = 0;
    std::vector<tree *> leaf_nodes;
    this->getbots(leaf_nodes);
    for (size_t i = 0; i < leaf_nodes.size();i++)
    {
        if(leaf_nodes[i]->getdepth() > output)
        {
            output = leaf_nodes[i]->getdepth();
        }
    }
    return output;
}

void tree::from_json(json &j3, size_t dim_theta)
{
    if (j3["left"].is_number())
    {
        std::vector<double> temp;
        j3.at("theta").get_to(temp);
        if (temp.size() > 1)
        {
            this->theta_vector = temp;
        }
        else
        {
            this->theta_vector[0] = temp[0];
        }
        this->l = 0;
        this->r = 0;
    }
    else
    {
        j3.at("variable").get_to(this->v);
        j3.at("cutpoint").get_to(this->c);
        tree *lchild = new tree(dim_theta);
        lchild->from_json(j3["left"], dim_theta);
        tree *rchild = new tree(dim_theta);
        rchild->from_json(j3["right"], dim_theta);

        lchild->p = this;
        rchild->p = this;
        this->l = lchild;
        this->r = rchild;
    }
}

//--------------------------------------------------
// operators
tree &tree::operator=(const tree &rhs)
{
    if (&rhs != this)
    {
        tonull();       // kill left hand side (this)
        cp(this, &rhs); // copy right hand side to left hand side
    }
    return *this;
}
//--------------------------------------------------
std::ostream &operator<<(std::ostream &os, const tree &t)
{
    tree::cnpv nds;
    t.getnodes(nds);
    // print size of theta first
    os << nds[0]->theta_vector.size() << std::endl;
    // next print number of nodes
    os << nds.size() << std::endl;
    for (size_t i = 0; i < nds.size(); i++)
    {
        os << nds[i]->nid() << " ";
        os << nds[i]->getv() << " ";
        os << nds[i]->getc();
        //   os << nds[i]->theta_vector[0] << std::endl;
        for (size_t kk = 0; kk < nds[i]->theta_vector.size(); kk++)
        {
            // be caucious, one space between c and theta
            os << " " << nds[i]->theta_vector[kk];
        }
        os << endl;
    }
    return os;
}

std::istream &operator>>(std::istream &is, tree &t)
{
    size_t tid, pid;                    // tid: id of current node, pid: parent's id
    std::map<size_t, tree::tree_p> pts; // pointers to nodes indexed by node id
    size_t nn;                          // number of nodes

    t.tonull(); // obliterate old tree (if there)

    // read size of theta first
    size_t theta_size;
    is >> theta_size;

    // read number of nodes----------
    is >> nn;
    if (!is)
    {
        return is;
    }

    // The idea is to dump string to a lot of node_info structure first, then link them as a tree, by nid

    // read in vector of node information----------
    std::vector<node_info> nv(nn);
    for (size_t i = 0; i != nn; i++)
    {
        is >> nv[i].v >> nv[i].c; // >> nv[i].theta_vector[0]; // Only works on first theta for now, fix latex if needed
        for (size_t kk = 0; kk < theta_size; kk++)
        {
            is >> nv[i].theta_vector[kk];
        }
        if (!is)
        {
            return is;
        }
    }

    // first node has to be the top one
    pts[1] = &t; // be careful! this is not the first pts, it is pointer of id 1.
    t.setv(nv[0].v);
    t.setc(nv[0].c);
    t.settheta(nv[0].theta_vector);
    t.p = 0;

    // now loop through the rest of the nodes knowing parent is already there.
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
        { // left child has even id
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

// main function to grow the tree recursively
void tree::grow_from_root(State &state, matrix<size_t> &Xorder_std, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, X_struct &x_struct, const size_t &sweeps, const size_t &tree_ind)
{
    // grow a tree, users can control number of split points
    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t split_var;
    size_t split_point;

    this->N = N_Xorder;

    bool no_split = false;
    // tau is prior VARIANCE, do not take squares

    // draw leaf parameter theta
    model->samplePars(state, this->suff_stat, this->theta_vector, this->prob_leaf);

    if (N_Xorder <= state.n_min)
        no_split = true;

    if (this->depth >= state.max_depth - 1)
        no_split = true;

    std::vector<size_t> subset_vars(p);

    if (state.use_all)
    {
        // use all variables
        std::iota(subset_vars.begin(), subset_vars.end(), 0);
    }
    else
    {
        // sample variables to use, like random forest
        if (state.sample_weights)
        {
            std::vector<double> weight_samp(p);
            double weight_sum;

            // Sample Weights Dirchelet
            for (size_t i = 0; i < p; i++)
            {
                std::gamma_distribution<double> temp_dist((*state.mtry_weight_current_tree)[i], 1.0);
                weight_samp[i] = temp_dist(state.gen);
            }
            weight_sum = accumulate(weight_samp.begin(), weight_samp.end(), 0.0);
            for (size_t i = 0; i < p; i++)
            {
                weight_samp[i] = weight_samp[i] / weight_sum;
            }
            // sample index of variables
            subset_vars = sample_int_ccrank(p, state.mtry, weight_samp, state.gen);
        }
        else
        {
            // sample index of variables
            subset_vars = sample_int_ccrank(p, state.mtry, (*state.mtry_weight_current_tree), state.gen);
        }
    }
    if (!no_split)
    {
        BART_likelihood_all(Xorder_std, no_split, split_var, split_point, subset_vars, X_counts, X_num_unique, model, x_struct, state, this);
    }

    this->loglike_node = model->likelihood(this->suff_stat, this->suff_stat, 1, false, true, state);

    // If our current split is same as parent, exit
    if (!no_split)
    {
        // only check this when split_var and split_point is not void
        double cutpoint = *(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point]);
        if ((this->p) && (this->v == (this->p)->v) && (cutpoint == (this->p)->c))
        {
            no_split = true;
        }

        // Update Cutpoint to be a true seperating point
        // Increase split_point (index) until it is no longer equal to cutpoint value
        while ((split_point < N_Xorder - 1) && (*(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point + 1]) == cutpoint))
        {
            split_point = split_point + 1;
        }

        if (split_point + 1 == N_Xorder)
        {
            no_split = true;
        }
    }

    if (no_split == true)
    {
        for (size_t i = 0; i < N_Xorder; i++)
        {
            x_struct.data_pointers[tree_ind][Xorder_std[0][i]] = &this->theta_vector;
        }

        this->l = 0;
        this->r = 0;
        return;
    }

    this->v = split_var;
    this->c = *(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point]);

    size_t index_in_full = 0;
    while ((*state.Xorder_std)[split_var][index_in_full] != Xorder_std[split_var][split_point])
    {
        index_in_full++;
    }
    this->c_index = (size_t)round((double)index_in_full / (double)state.n_y * (double)state.n_cutpoints);

    (*state.split_count_current_tree)[split_var] += 1;

    tree::tree_p lchild = new tree(model->getNumClasses(), this, model->dim_suffstat);
    tree::tree_p rchild = new tree(model->getNumClasses(), this, model->dim_suffstat);

    this->l = lchild;
    this->r = rchild;

    lchild->depth = this->depth + 1;
    rchild->depth = this->depth + 1;

    // inherit tau
    lchild->tau_prior = this->tau_prior;
    rchild->tau_prior = this->tau_prior;
    lchild->tau_post = this->tau_post;
    rchild->tau_post = this->tau_post;

    this->l->ini_suff_stat();
    this->r->ini_suff_stat();

    matrix<size_t> Xorder_left_std;
    matrix<size_t> Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    std::vector<size_t> X_num_unique_left(X_num_unique.size());
    std::vector<size_t> X_num_unique_right(X_num_unique.size());

    std::vector<size_t> X_counts_left(X_counts.size());
    std::vector<size_t> X_counts_right(X_counts.size());

    if (state.p_categorical > 0)
    {
        split_xorder_std_categorical(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_counts_left, X_counts_right, X_num_unique_left, X_num_unique_right, X_counts, model, x_struct, state, this);
    }

    if (state.p_continuous > 0)
    {
        split_xorder_std_continuous(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, model, x_struct, state, this);
    }

    this->l->grow_from_root(state, Xorder_left_std, X_counts_left, X_num_unique_left, model, x_struct, sweeps, tree_ind);

    this->r->grow_from_root(state, Xorder_right_std, X_counts_right, X_num_unique_right, model, x_struct, sweeps, tree_ind);

    return;
}

void calculate_entropy(matrix<size_t> &Xorder_std, State &state, std::vector<double> &theta_vector, double &entropy)
{
    size_t N_Xorder = Xorder_std[0].size();
    size_t dim_residual = (*state.residual_std).size();
    size_t next_obs;
    double sum_fits;
    entropy = 0.0;
    std::vector<double> fits(dim_residual, 0.0);

    for (size_t i = 0; i < N_Xorder; i++)
    {
        sum_fits = 0;
        next_obs = Xorder_std[0][i];
        for (size_t j = 0; j < dim_residual; ++j)
        {
            fits[j] = exp((*state.residual_std)[j][next_obs]) * theta_vector[j]; // f_j(x_i) = \prod lambdas
        }
        sum_fits = accumulate(fits.begin(), fits.end(), 0.0);
        for (size_t j = 0; j < dim_residual; ++j)
        {
            entropy += -fits[j] / sum_fits * log(fits[j] / sum_fits); // entropy = sum(- p_j * log(p_j))
        }
    }

    return;
}

void tree::grow_from_root_entropy(State &state, matrix<size_t> &Xorder_std, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, X_struct &x_struct, const size_t &sweeps, const size_t &tree_ind)
{
    // grow a tree, users can control number of split points
    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    // size_t ind;
    size_t split_var;
    size_t split_point;

    this->N = N_Xorder;

    bool no_split = false;

    // tau is prior VARIANCE, do not take squares

    model->samplePars(state, this->suff_stat, this->theta_vector, this->prob_leaf);

    if (N_Xorder <= state.n_min)
    {
        no_split = true;
        // return;
    }

    if (this->depth >= state.max_depth - 1)
    {
        no_split = true;
        // return;
    }

    std::vector<size_t> subset_vars(p);

    if (state.use_all | (state.mtry == state.p))
    {
        std::iota(subset_vars.begin(), subset_vars.end(), 0);
    }
    else
    {
        if (state.sample_weights)
        {
            std::vector<double> weight_samp(p);
            dirichlet_distribution(weight_samp, (*state.mtry_weight_current_tree), state.gen);

            subset_vars = sample_int_ccrank(p, state.mtry, weight_samp, state.gen);
        }
        else
        {
            subset_vars = sample_int_ccrank(p, state.mtry, (*state.mtry_weight_current_tree), state.gen);
        }
    }

    if (!no_split)
    {
        BART_likelihood_all(Xorder_std, no_split, split_var, split_point, subset_vars, X_counts, X_num_unique, model, x_struct, state, this);
    }

    this->loglike_node = model->likelihood(this->suff_stat, this->suff_stat, 1, false, true, state);

    if (!no_split)
    {
        double cutpoint = *(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point]);
        if ((this->p) && (this->v == (this->p)->v) && (cutpoint == (this->p)->c))
        {
            no_split = true;
        }

        // Update Cutpoint to be a true seperating point
        // Increase split_point (index) until it is no longer equal to cutpoint value
        while ((split_point < N_Xorder - 1) && (*(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point + 1]) == cutpoint))
        {
            split_point = split_point + 1;
        }

        if (split_point + 1 == N_Xorder)
        {
            no_split = true;
        }
    }

    if (no_split == true)
    {
        for (size_t i = 0; i < N_Xorder; i++)
        {
            x_struct.data_pointers[tree_ind][Xorder_std[0][i]] = &this->theta_vector;
        }
        // update lambdas in state
        (*state.lambdas)[tree_ind].push_back(this->theta_vector);

        // cout << "suff stat " << this->suff_stat << endl;
        // cout << "theta " << this->theta_vector << endl;

        for (auto i : this->theta_vector)
        {
            if (isinf(i) | isnan(i))
            {
                cout << "suff stat " << this->suff_stat << endl;
                cout << "theta " << this->theta_vector << endl;
                exit(1);
            }
        }

        // if (update_theta)
        // {
        //     model->samplePars(state, this->suff_stat, this->theta_vector, this->prob_leaf);
        // }

        this->l = 0;
        this->r = 0;

        // update leaf prob, for MH update useage
        // this->loglike_node = model->likelihood_no_split(this->suff_stat, state);

        return;
    }

    // If GROW FROM ROOT MODE
    this->v = split_var;
    this->c = *(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point]);

    // cout << "cut varialbe " << this->v << " cutpoint " << this->c << endl;

    size_t index_in_full = 0;
    while ((*state.Xorder_std)[split_var][index_in_full] != Xorder_std[split_var][split_point])
    {
        index_in_full++;
    }
    this->c_index = (size_t)round((double)index_in_full / (double)state.n_y * (double)state.n_cutpoints);

    (*state.split_count_current_tree)[split_var] += 1;

    tree::tree_p lchild = new tree(model->getNumClasses(), this, model->dim_suffstat);
    tree::tree_p rchild = new tree(model->getNumClasses(), this, model->dim_suffstat);

    this->l = lchild;
    this->r = rchild;

    this->l->depth = this->depth + 1;
    this->r->depth = this->depth + 1;

    this->l->ini_suff_stat();
    this->r->ini_suff_stat();

    matrix<size_t> Xorder_left_std;
    matrix<size_t> Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    std::vector<size_t> X_num_unique_left(X_num_unique.size());
    std::vector<size_t> X_num_unique_right(X_num_unique.size());

    std::vector<size_t> X_counts_left(X_counts.size());
    std::vector<size_t> X_counts_right(X_counts.size());

    if (state.p_categorical > 0)
    {
        split_xorder_std_categorical(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_counts_left, X_counts_right, X_num_unique_left, X_num_unique_right, X_counts, model, x_struct, state, this);
    }

    if (state.p_continuous > 0)
    {
        split_xorder_std_continuous(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, model, x_struct, state, this);
    }

    // cout << "left suff " << this->l->suff_stat << endl;

    this->l->grow_from_root_entropy(state, Xorder_left_std, X_counts_left, X_num_unique_left, model, x_struct, sweeps, tree_ind);

    // cout << "right suff " << this->r->suff_stat << endl;

    this->r->grow_from_root_entropy(state, Xorder_right_std, X_counts_right, X_num_unique_right, model, x_struct, sweeps, tree_ind);

    return;
}

void tree::grow_from_root_separate_tree(State &state, matrix<size_t> &Xorder_std, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, X_struct &x_struct, const size_t &sweeps, const size_t &tree_ind)
{
    // grow a tree, users can control number of split points
    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    // size_t ind;
    size_t split_var;
    size_t split_point;

    this->N = N_Xorder;

    // tau is prior VARIANCE, do not take squares

    bool no_split = false;

    // tau is prior VARIANCE, do not take squares

    model->samplePars(state, this->suff_stat, this->theta_vector, this->prob_leaf);

    if (N_Xorder <= state.n_min)
    {
        no_split = true;
    }

    if (this->depth >= state.max_depth - 1)
    {
        no_split = true;
    }

    std::vector<size_t> subset_vars(p);

    if (state.use_all)
    {
        std::iota(subset_vars.begin(), subset_vars.end(), 0);
    }
    else
    {
        if (state.sample_weights)
        {
            std::vector<double> weight_samp(p);
            double weight_sum;

            // Sample Weights Dirchelet
            for (size_t i = 0; i < p; i++)
            {
                std::gamma_distribution<double> temp_dist((*state.mtry_weight_current_tree)[i], 1.0);
                weight_samp[i] = temp_dist(state.gen);
            }
            weight_sum = accumulate(weight_samp.begin(), weight_samp.end(), 0.0);
            for (size_t i = 0; i < p; i++)
            {
                weight_samp[i] = weight_samp[i] / weight_sum;
            }

            subset_vars = sample_int_ccrank(p, state.mtry, weight_samp, state.gen);
        }
        else
        {
            subset_vars = sample_int_ccrank(p, state.mtry, (*state.mtry_weight_current_tree), state.gen);
        }
    }

    if (!no_split)
    {
        BART_likelihood_all(Xorder_std, no_split, split_var, split_point, subset_vars, X_counts, X_num_unique, model, x_struct, state, this);
    }

    this->loglike_node = model->likelihood(this->suff_stat, this->suff_stat, 1, false, true, state);

    if (no_split == true)
    {
        size_t j = model->get_class_operating();
        for (size_t i = 0; i < N_Xorder; i++)
        {
            x_struct.data_pointers_multinomial[j][tree_ind][Xorder_std[0][i]] = &this->theta_vector;
            if ((*(x_struct.data_pointers_multinomial[j][tree_ind][Xorder_std[0][i]]))[j] == 0)
            {
                COUT << "theta_vector = " << this->theta_vector << endl;
                exit(1);
            }
        }

        (*state.lambdas_separate)[tree_ind][j].push_back(this->theta_vector[j]);

        this->l = 0;
        this->r = 0;

        return;
    }

    this->v = split_var;
    this->c = *(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point]);

    size_t index_in_full = 0;
    while ((*state.Xorder_std)[split_var][index_in_full] != Xorder_std[split_var][split_point])
    {
        index_in_full++;
    }
    this->c_index = (size_t)round((double)index_in_full / (double)state.n_y * (double)state.n_cutpoints);

    // Update Cutpoint to be a true seperating point
    // Increase split_point (index) until it is no longer equal to cutpoint value
    while ((split_point < N_Xorder - 1) && (*(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point + 1]) == this->c))
    {
        split_point = split_point + 1;
    }

    // If our current split is same as parent, exit
    if ((this->p) && (this->v == (this->p)->v) && (this->c == (this->p)->c))
    {
        size_t j = model->get_class_operating();
        for (size_t i = 0; i < N_Xorder; i++)
        {
            x_struct.data_pointers_multinomial[j][tree_ind][Xorder_std[0][i]] = &this->theta_vector;
        }

        (*state.lambdas_separate)[tree_ind][j].push_back(this->theta_vector[j]);

        return;
    }

    (*state.split_count_current_tree)[split_var] += 1;

    tree::tree_p lchild = new tree(model->getNumClasses(), this, model->dim_suffstat);
    tree::tree_p rchild = new tree(model->getNumClasses(), this, model->dim_suffstat);

    this->l = lchild;
    this->r = rchild;

    lchild->depth = this->depth + 1;
    rchild->depth = this->depth + 1;

    this->l->ini_suff_stat();
    this->r->ini_suff_stat();

    matrix<size_t> Xorder_left_std;
    matrix<size_t> Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    std::vector<size_t> X_num_unique_left(X_num_unique.size());
    std::vector<size_t> X_num_unique_right(X_num_unique.size());

    std::vector<size_t> X_counts_left(X_counts.size());
    std::vector<size_t> X_counts_right(X_counts.size());
    if (state.p_categorical > 0)
    {
        split_xorder_std_categorical(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_counts_left, X_counts_right, X_num_unique_left, X_num_unique_right, X_counts, model, x_struct, state, this);
    }

    if (state.p_continuous > 0)
    {
        split_xorder_std_continuous(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, model, x_struct, state, this);
    }

    this->l->grow_from_root_separate_tree(state, Xorder_left_std, X_counts_left, X_num_unique_left, model, x_struct, sweeps, tree_ind);

    this->r->grow_from_root_separate_tree(state, Xorder_right_std, X_counts_right, X_num_unique_right, model, x_struct, sweeps, tree_ind);

    return;
}

void split_xorder_std_continuous(matrix<size_t> &Xorder_left_std, matrix<size_t> &Xorder_right_std, size_t split_var, size_t split_point, matrix<size_t> &Xorder_std, Model *model, X_struct &x_struct, State &state, tree *current_node)
{

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes

    // preserve order of other variables
    size_t N_Xorder = Xorder_std[0].size();
    // size_t left_ix = 0;
    // size_t right_ix = 0;
    size_t N_Xorder_left = Xorder_left_std[0].size();
    size_t N_Xorder_right = Xorder_right_std[0].size();

    // if the left side is smaller, we only compute sum of it
    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    current_node->l->ini_suff_stat();
    current_node->r->ini_suff_stat();

    double cutvalue = *(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point]);

    const double *temp_pointer = state.X_std + state.n_y * split_var;

    // TODO: this version yield negative suffstat on the other side sometime.
    // for (size_t j = 0; j < N_Xorder; j++)
    // {
    //     if (compute_left_side)
    //     {
    //         if (*(temp_pointer + Xorder_std[split_var][j]) <= cutvalue)
    //         {
    //             model->updateNodeSuffStat(state, current_node->l->suff_stat, Xorder_std, split_var, j);
    //         }
    //     }
    //     else
    //     {
    //         if (*(temp_pointer + Xorder_std[split_var][j]) > cutvalue)
    //         {
    //             model->updateNodeSuffStat(state, current_node->r->suff_stat, Xorder_std, split_var, j);
    //         }
    //     }
    // }

    for (size_t j = 0; j < N_Xorder; j++)
    {
        if (*(temp_pointer + Xorder_std[split_var][j]) <= cutvalue)
        {
            model->updateNodeSuffStat(state, current_node->l->suff_stat, Xorder_std, split_var, j);
        }
        else
        {
            model->updateNodeSuffStat(state, current_node->r->suff_stat, Xorder_std, split_var, j);
        }
    }

    const double *split_var_x_pointer = state.X_std + state.n_y * split_var;

    for (size_t i = 0; i < state.p_continuous; i++) // loop over variables
    {
        // lambda callback for multithreading
        auto split_i = [&, i]()
        {
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

        if (thread_pool.is_active() && state.parallel)
        {
            thread_pool.add_task(split_i);
        }
        else
        {
            split_i();
        }
    }

    if (thread_pool.is_active())
        thread_pool.wait();

    // model->calculateOtherSideSuffStat(current_node->suff_stat, current_node->l->suff_stat, current_node->r->suff_stat, N_Xorder, N_Xorder_left, N_Xorder_right, compute_left_side);

    return;
}

void split_xorder_std_categorical(matrix<size_t> &Xorder_left_std, matrix<size_t> &Xorder_right_std, size_t split_var, size_t split_point, matrix<size_t> &Xorder_std, std::vector<size_t> &X_counts_left, std::vector<size_t> &X_counts_right, std::vector<size_t> &X_num_unique_left, std::vector<size_t> &X_num_unique_right, std::vector<size_t> &X_counts, Model *model, X_struct &x_struct, State &state, tree *current_node)
{

    // when find the split point, split Xorder matrix to two sub matrices for both subnodes

    // preserve order of other variables
    size_t N_Xorder = Xorder_std[0].size();
    size_t N_Xorder_left = Xorder_left_std[0].size();
    size_t N_Xorder_right = Xorder_right_std[0].size();
    const double *temp_pointer = state.X_std + state.n_y * split_var;

    // if the left side is smaller, we only compute sum of it
    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    current_node->l->ini_suff_stat();
    current_node->r->ini_suff_stat();

    double cutvalue = *(state.X_std + state.n_y * split_var + Xorder_std[split_var][split_point]);

    std::fill(X_num_unique_left.begin(), X_num_unique_left.end(), 0.0);
    std::fill(X_num_unique_right.begin(), X_num_unique_right.end(), 0.0);

    for (size_t i = state.p_continuous; i < state.p; i++)
    {
        // loop over variables
        size_t left_ix = 0;
        size_t right_ix = 0;

        // index range of X_counts, X_values that are corresponding to current variable
        // start <= i <= end;
        size_t start = x_struct.variable_ind[i - state.p_continuous];
        size_t end = x_struct.variable_ind[i + 1 - state.p_continuous];

        if (i == split_var)
        {
            if (compute_left_side)
            {
                for (size_t j = 0; j < N_Xorder; j++)
                {
                    if (*(temp_pointer + Xorder_std[i][j]) <= cutvalue)
                    {
                        model->updateNodeSuffStat(state, current_node->l->suff_stat, Xorder_std, split_var, j);
                        Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                        left_ix = left_ix + 1;
                    }
                    else
                    {
                        // go to right side
                        model->updateNodeSuffStat(state, current_node->r->suff_stat, Xorder_std, split_var, j);
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
                        model->updateNodeSuffStat(state, current_node->l->suff_stat, Xorder_std, split_var, j);
                        Xorder_left_std[i][left_ix] = Xorder_std[i][j];
                        left_ix = left_ix + 1;
                    }
                    else
                    {
                        model->updateNodeSuffStat(state, current_node->r->suff_stat, Xorder_std, split_var, j);
                        Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                        right_ix = right_ix + 1;
                    }
                }
            }

            // for the cut variable, it's easy to counts X_counts_left and X_counts_right, simply cut X_counts to two pieces.

            for (size_t k = start; k < end; k++)
            {
                // loop from start to end!

                if (x_struct.X_values[k] <= cutvalue)
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
            size_t X_counts_index = start;
            // split other variables, need to compare each row
            for (size_t j = 0; j < N_Xorder; j++)
            {
                while (*(state.X_std + state.n_y * i + Xorder_std[i][j]) != x_struct.X_values[X_counts_index])
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

        for (size_t j = start; j < end; j++)
        {
            if (X_counts_left[j] > 0)
            {
                X_num_unique_left[i - state.p_continuous] = X_num_unique_left[i - state.p_continuous] + 1;
            }
            if (X_counts_right[j] > 0)
            {
                X_num_unique_right[i - state.p_continuous] = X_num_unique_right[i - state.p_continuous] + 1;
            }
        }
    }

    // model->calculateOtherSideSuffStat(current_node->suff_stat, current_node->l->suff_stat, current_node->r->suff_stat, N_Xorder, N_Xorder_left, N_Xorder_right, compute_left_side);

    // update X_num_unique

    // std::fill(X_num_unique_left.begin(), X_num_unique_left.end(), 0.0);
    // std::fill(X_num_unique_right.begin(), X_num_unique_right.end(), 0.0);

    // for (size_t i = state.p_continuous; i < state.p; i++)
    // {
    //     start = x_struct.variable_ind[i - state.p_continuous];
    //     end = x_struct.variable_ind[i + 1 - state.p_continuous];

    //     // COUT << "start " << start << " end " << end << " size " << X_counts_left.size() << endl;
    //     for (size_t j = start; j < end; j++)
    //     {
    //         if (X_counts_left[j] > 0)
    //         {
    //             X_num_unique_left[i - state.p_continuous] = X_num_unique_left[i - state.p_continuous] + 1;
    //         }
    //         if (X_counts_right[j] > 0)
    //         {
    //             X_num_unique_right[i - state.p_continuous] = X_num_unique_right[i - state.p_continuous] + 1;
    //         }
    //     }
    // }

    return;
}

void BART_likelihood_all(matrix<size_t> &Xorder_std, bool &no_split, size_t &split_var, size_t &split_point, const std::vector<size_t> &subset_vars, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, X_struct &x_struct, State &state, tree *tree_pointer)
{
    // compute BART posterior (loglikelihood + logprior penalty)

    // subset_vars: a vector of indexes of varibles to consider (like random forest)

    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder_std[0].size();
    // size_t p = Xorder_std.size();
    size_t ind;
    size_t N_Xorder = N;
    size_t total_categorical_split_candidates = 0;

    // double sigma2 = pow(sigma, 2);

    double loglike_max = -INFINITY;

    std::vector<double> loglike;

    size_t loglike_start;

    // decide lenght of loglike vector
    if (N <= state.n_cutpoints + 1 + 2 * state.n_min)
    {
        loglike.resize((N_Xorder - 1) * state.p_continuous + x_struct.X_values.size() + 1, -INFINITY);
        loglike_start = (N_Xorder - 1) * state.p_continuous;
    }
    else
    {
        loglike.resize(state.n_cutpoints * state.p_continuous + x_struct.X_values.size() + 1, -INFINITY);
        loglike_start = state.n_cutpoints * state.p_continuous;
    }

    // calculate for each cases
    if (state.p_continuous > 0)
    {
        calculate_loglikelihood_continuous(loglike, subset_vars, N_Xorder, Xorder_std, loglike_max, model, x_struct, state, tree_pointer);
    }

    if (state.p_categorical > 0)
    {
        calculate_loglikelihood_categorical(loglike, loglike_start, subset_vars, N_Xorder, Xorder_std, loglike_max, X_counts, X_num_unique, model, x_struct, total_categorical_split_candidates, state, tree_pointer);
    }

    // calculate likelihood of no-split option
    calculate_likelihood_no_split(loglike, N_Xorder, loglike_max, model, x_struct, total_categorical_split_candidates, state, tree_pointer);

    loglike_max = -INFINITY;
    for (size_t ii = 0; ii < loglike.size(); ii++)
    {
        if (loglike[ii] > loglike_max)
        {
            loglike_max = loglike[ii];
        }
    }

    // transfer loglikelihood to likelihood
    for (size_t ii = 0; ii < loglike.size(); ii++)
    {
        // if a variable is not selected, take exp will becomes 0
        loglike[ii] = exp(loglike[ii] - loglike_max);
    }

    // cout << "loglike " << loglike << endl;

    // sampling cutpoints
    if (N <= state.n_cutpoints + 1 + 2 * state.n_min)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        if ((N - 1) > 2 * state.n_min)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                if (i < state.p_continuous)
                {
                    // delete some candidates, otherwise size of the new node can be smaller than Nmin
                    std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + state.n_min + 1, 0.0);
                    std::fill(loglike.begin() + i * (N - 1) + N - 2 - state.n_min, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
                }
            }
        }
        else
        {
            // do not use all continuous variables
            if (state.p_continuous > 0)
            {
                std::fill(loglike.begin(), loglike.begin() + (N_Xorder - 1) * state.p_continuous - 1, 0.0);
            }
        }

        std::discrete_distribution<> d(loglike.begin(), loglike.end());

        // for MH update usage only
        tree_pointer->num_cutpoint_candidates = count_non_zero(loglike);

        // sample one index of split point
        ind = d(state.gen);
        tree_pointer->drawn_ind = ind;

        // save the posterior of the chosen split point
        vec_sum(loglike, tree_pointer->prob_split);
        tree_pointer->prob_split = loglike[ind] / tree_pointer->prob_split;

        if (ind == loglike.size() - 1)
        {
            // no split
            no_split = true;
            split_var = 0;
            split_point = 0;
        }
        else if ((N - 1) <= 2 * state.n_min)
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
            for (size_t i = 0; i < (x_struct.variable_ind.size() - 1); i++)
            {
                if (x_struct.variable_ind[i] <= ind && x_struct.variable_ind[i + 1] > ind)
                {
                    split_var = i;
                }
            }
            start = x_struct.variable_ind[split_var];
            // count how many
            split_point = std::accumulate(X_counts.begin() + start, X_counts.begin() + ind + 1, 0);
            // minus one for correct index (start from 0)
            if (split_point > 0)
            {
                split_point = split_point - 1;
            }
            split_var = split_var + state.p_continuous;
        }
    }
    else
    {
        // use adaptive number of cutpoints

        std::vector<size_t> candidate_index(state.n_cutpoints);

        seq_gen_std(state.n_min, N - state.n_min, state.n_cutpoints, candidate_index);

        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());

        // For MH update usage only
        tree_pointer->num_cutpoint_candidates = count_non_zero(loglike);

        // sample one index of split point
        ind = d(state.gen);
        tree_pointer->drawn_ind = ind;

        // save the posterior of the chosen split point
        vec_sum(loglike, tree_pointer->prob_split);
        tree_pointer->prob_split = loglike[ind] / tree_pointer->prob_split;

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
            split_var = ind / state.n_cutpoints;
            split_point = candidate_index[ind % state.n_cutpoints];
        }
        else
        {
            // split at categorical variable
            size_t start;
            ind = ind - loglike_start;
            for (size_t i = 0; i < (x_struct.variable_ind.size() - 1); i++)
            {
                if (x_struct.variable_ind[i] <= ind && x_struct.variable_ind[i + 1] > ind)
                {
                    split_var = i;
                }
            }
            start = x_struct.variable_ind[split_var];
            // count how many
            split_point = std::accumulate(X_counts.begin() + start, X_counts.begin() + ind + 1, 0);
            // minus one for correct index (start from 0)
            if (split_point > 0)
            {
                split_point = split_point - 1;
            }
            split_var = split_var + state.p_continuous;
        }
    }

    return;
}

void calculate_loglikelihood_continuous(std::vector<double> &loglike, const std::vector<size_t> &subset_vars, size_t &N_Xorder, matrix<size_t> &Xorder_std, double &loglike_max, Model *model, X_struct &x_struct, State &state, tree *tree_pointer)
{
    size_t N = N_Xorder;

    if (N_Xorder <= state.n_cutpoints + 1 + 2 * state.n_min)
    {
        // if we only have a few data observations in current node
        // use all of them as cutpoint candidates

        // to have a generalized function, have to pass an empty candidate_index object for this case
        // is there any smarter way to do it?
        std::vector<size_t> candidate_index(1);

        // set up parallel during burnin
        for (auto i : subset_vars)
        {
            if (i < state.p_continuous)
            {
                std::vector<size_t> &xorder = Xorder_std[i];

                // initialize sufficient statistics
                std::vector<double> temp_suff_stat(model->dim_suffstat);
                std::fill(temp_suff_stat.begin(), temp_suff_stat.end(), 0.0);

                for (size_t j = 0; j < N_Xorder - 1; j++)
                {
                    calcSuffStat_continuous(state, temp_suff_stat, xorder, candidate_index, j, false, model, (*state.residual_std));

                    loglike[(N_Xorder - 1) * i + j] = model->likelihood(temp_suff_stat, tree_pointer->suff_stat, j, true, false, state) + model->likelihood(temp_suff_stat, tree_pointer->suff_stat, j, false, false, state);
                }
            }
        }
    }
    else
    {

        // otherwise, adaptive number of cutpoints
        // use Ncutpoints

        std::vector<size_t> candidate_index2(state.n_cutpoints + 1);
        seq_gen_std2(state.n_min, N - state.n_min, state.n_cutpoints, candidate_index2);

        for (auto &&i : subset_vars)
        {
            if (i < state.p_continuous)
            {

                // Lambda callback to perform the calculation
                auto calcllc_i = [&, i]()
                {
                    std::vector<size_t> &xorder = Xorder_std[i];
                    // double llmax = -INFINITY;

                    std::vector<double> temp_suff_stat(model->dim_suffstat);

                    std::fill(temp_suff_stat.begin(), temp_suff_stat.end(), 0.0);

                    std::vector<double> &temp = loglike;

                    for (size_t j = 0; j < state.n_cutpoints; j++)
                    {
                        calcSuffStat_continuous(state, temp_suff_stat, xorder, candidate_index2, j, true, model, (*state.residual_std));

                        temp[(state.n_cutpoints) * i + j] = model->likelihood(temp_suff_stat, tree_pointer->suff_stat, candidate_index2[j + 1], true, false, state) + model->likelihood(temp_suff_stat, tree_pointer->suff_stat, candidate_index2[j + 1], false, false, state);
                    }
                };

                if (thread_pool.is_active() && state.parallel)
                    thread_pool.add_task(calcllc_i);
                else
                    calcllc_i();
            }
        }
        if (thread_pool.is_active())
            thread_pool.wait();
    }
}

void calculate_loglikelihood_categorical(std::vector<double> &loglike, size_t &loglike_start, const std::vector<size_t> &subset_vars, size_t &N_Xorder, matrix<size_t> &Xorder_std, double &loglike_max, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, X_struct &x_struct, size_t &total_categorical_split_candidates, State &state, tree *tree_pointer)
{

    // loglike_start is an index to offset
    // consider loglikelihood start from loglike_start
    for (size_t var_i = 0; var_i < subset_vars.size(); var_i++)
    {

        size_t i = subset_vars[var_i]; // get subset varaible

        if ((i >= state.p_continuous) && (X_num_unique[i - state.p_continuous] > 1))
        {
            // if this is a categorical variable, and it still has cutpoints
            std::vector<double> temp_suff_stat(model->dim_suffstat);
            std::fill(temp_suff_stat.begin(), temp_suff_stat.end(), 0.0);
            size_t start, end, end2, n1, temp;

            start = x_struct.variable_ind[i - state.p_continuous];
            end = x_struct.variable_ind[i + 1 - state.p_continuous] - 1; // minus one for indexing starting at 0
            end2 = end;

            while (X_counts[end2] == 0)
            {
                // move backward if the last unique value has zero counts
                end2 = end2 - 1;
            }
            // move backward again, do not consider the last unique value as cutpoint
            end2 = end2 - 1;

            n1 = 0;

            for (size_t j = start; j <= end2; j++)
            {

                if (X_counts[j] != 0)
                {
                    temp = n1 + X_counts[j] - 1;
                    // modify sufficient statistics vector directly inside model class
                    calcSuffStat_categorical(state, temp_suff_stat, Xorder_std[i], n1, temp, model);

                    n1 = n1 + X_counts[j];

                    loglike[loglike_start + j] = model->likelihood(temp_suff_stat, tree_pointer->suff_stat, n1 - 1, true, false, state) + model->likelihood(temp_suff_stat, tree_pointer->suff_stat, n1 - 1, false, false, state);

                    // adjust for the difference of number of cutpoints between continuous variable and categorical variables
                    loglike[loglike_start + j] += -log(x_struct.X_num_unique[i - state.p_continuous]);

                    if (state.p_continuous > 0)
                    {
                        loglike[loglike_start + j] += log(state.n_cutpoints);
                    }
                }
            }
        }
    }
}

void calculate_likelihood_no_split(std::vector<double> &loglike, size_t &N_Xorder, double &loglike_max, Model *model, X_struct &x_struct, size_t &total_categorical_split_candidates, State &state, tree *tree_pointer)
{
    size_t loglike_size = 0;
    for (size_t i = 0; i < loglike.size(); i++)
    {
        if (loglike[i] > -INFINITY)
        {
            loglike_size += 1;
        }
    }

    if (loglike_size > 0)
    {
        loglike[loglike.size() - 1] = model->likelihood(tree_pointer->suff_stat, tree_pointer->suff_stat, loglike.size() - 1, false, true, state) + log(pow(1.0 + tree_pointer->getdepth(), model->beta) / model->alpha - 1.0) + log((double)loglike_size) + model->getNoSplitPenalty();
        // !!Note loglike_size shouldn't get minus 1 when it count non zero of loglike.
    }
    else
    {
        loglike[loglike.size() - 1] = 1;
    }

    // loglike[loglike.size() - 1] = model->likelihood(tree_pointer->suff_stat, tree_pointer->suff_stat, loglike.size() - 1, false, true, state) + log(pow(1.0 + tree_pointer->getdepth(), model->beta) / model->alpha - 1.0) + log((double)loglike.size() - 1.0) + log(model->getNoSplitPenalty());

    //  then adjust according to number of variables and split points

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

    // loglike[loglike.size() - 1] += log(state.p) + log(2.0) + model->getNoSplitPenalty();

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
    // if (loglike[loglike.size() - 1] > loglike_max)
    // {
    // loglike_max = loglike[loglike.size() - 1];
    // }
}

void calcSuffStat_categorical(State &state, std::vector<double> &temp_suff_stat, std::vector<size_t> &xorder, size_t &start, size_t &end, Model *model)
{
    // calculate sufficient statistics for categorical variables

    // compute sum of y[Xorder[start:end, var]]
    for (size_t i = start; i <= end; i++)
    {
        model->incSuffStat(state, xorder[i], temp_suff_stat);
    }
    return;
}

void calcSuffStat_continuous(State &state, std::vector<double> &temp_suff_stat, std::vector<size_t> &xorder, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint, Model *model, matrix<double> &residual_std)
{
    // calculate sufficient statistics for continuous variables
    if (adaptive_cutpoint)
    {

        if (index == 0)
        {
            // initialize, only for the first cutpoint candidate, thus index == 0
            model->incSuffStat(state, xorder[0], temp_suff_stat);
        }

        // if use adaptive number of cutpoints, calculated based on vector candidate_index
        for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
        {
            model->incSuffStat(state, xorder[q], temp_suff_stat);
        }
    }
    else
    {
        // use all data points as candidates
        model->incSuffStat(state, xorder[index], temp_suff_stat);
    }
    return;
}

size_t get_split_point(const double *Xpointer, matrix<size_t> &Xorder_std, size_t n_y, size_t v, double c)
{
    // get split point
    // use bisection

    size_t left_ind = 0;
    size_t right_ind = Xorder_std[0].size() - 1;
    size_t split_point = (left_ind + right_ind) / 2;
    double split_val = *(Xpointer + n_y * v + Xorder_std[v][split_point]);

    if (c < *(Xpointer + n_y * v + Xorder_std[v][0]))
    {
        COUT << "Warning: cut point less than the smallest value" << endl;
        split_point = 0;
    }
    else if (c > *(Xpointer + n_y * v + Xorder_std[v][right_ind]))
    {
        COUT << "Warning: cut point greater than the smallest value" << endl;
        split_point = right_ind;
    }
    else
    {

        while ((c != split_val) & (left_ind <= right_ind))
        {
            if (split_val > c)
            {
                right_ind = split_point - 1;
            }
            else
            {
                left_ind = split_point + 1;
            }
            split_point = (left_ind + right_ind) / 2;
            split_val = *(Xpointer + n_y * v + Xorder_std[v][split_point]);
        }

        while ((split_point < Xorder_std[0].size() - 1) && (*(Xpointer + n_y * v + Xorder_std[v][split_point + 1]) == c))
        {
            split_point = split_point + 1;
        }
    }
    if (Xorder_std[0].size() == split_point + 1)
    {
        COUT << "split_point = N = " << split_point + 1 << endl;
        COUT << "v = " << v << ", c = " << c << ", min = " << *(Xpointer + n_y * v + Xorder_std[v][0]) << ", max = " << *(Xpointer + n_y * v + Xorder_std[v][right_ind]) << ", N = " << Xorder_std[0].size() << endl;
        throw;
    }

    return split_point;
}

void split_xorder_std_categorical_simplified(gp_struct &x_struct, matrix<size_t> &Xorder_left_std, matrix<size_t> &Xorder_right_std, size_t split_var, size_t split_point, matrix<size_t> &Xorder_std, std::vector<size_t> &X_counts_left, std::vector<size_t> &X_counts_right, std::vector<size_t> &X_num_unique_left, std::vector<size_t> &X_num_unique_right, std::vector<size_t> &X_counts, size_t p_categorical)
{
    // without model, state, don't update suff stats

    // preserve order of other variables
    size_t N_Xorder = Xorder_std[0].size();
    size_t N_Xorder_left = Xorder_left_std[0].size();
    size_t N_Xorder_right = Xorder_right_std[0].size();
    size_t p = Xorder_std.size();
    size_t p_continuous = p - p_categorical;
    const double *temp_pointer = x_struct.X_std + x_struct.n_y * split_var;

    // if the left side is smaller, we only compute sum of it
    bool compute_left_side = N_Xorder_left < N_Xorder_right;

    double cutvalue = *(x_struct.X_std + x_struct.n_y * split_var + Xorder_std[split_var][split_point]);

    std::fill(X_num_unique_left.begin(), X_num_unique_left.end(), 0.0);
    std::fill(X_num_unique_right.begin(), X_num_unique_right.end(), 0.0);

    for (size_t i = p_continuous; i < p; i++)
    {
        // loop over variables
        size_t left_ix = 0;
        size_t right_ix = 0;

        // index range of X_counts, X_values that are corresponding to current variable
        // start <= i <= end;
        size_t start = x_struct.variable_ind[i - p_continuous];
        size_t end = x_struct.variable_ind[i + 1 - p_continuous];

        if (i == split_var)
        {
            if (compute_left_side)
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
                        Xorder_right_std[i][right_ix] = Xorder_std[i][j];
                        right_ix = right_ix + 1;
                    }
                }
            }

            // for the cut variable, it's easy to counts X_counts_left and X_counts_right, simply cut X_counts to two pieces.

            for (size_t k = start; k < end; k++)
            {
                // loop from start to end!

                if (x_struct.X_values[k] <= cutvalue)
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
            size_t X_counts_index = start;
            // split other variables, need to compare each row
            for (size_t j = 0; j < N_Xorder; j++)
            {
                while (*(x_struct.X_std + x_struct.n_y * i + Xorder_std[i][j]) != x_struct.X_values[X_counts_index])
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

void split_xorder_std_continuous_simplified(gp_struct &x_struct, matrix<size_t> &Xorder_left_std, matrix<size_t> &Xorder_right_std, size_t split_var, size_t split_point, matrix<size_t> &Xorder_std, size_t p_continuous)
{
    // without model, state, don't update suff stats

    size_t N_Xorder = Xorder_std[0].size();

    double cutvalue = *(x_struct.X_std + x_struct.n_y * split_var + Xorder_std[split_var][split_point]);

    const double *split_var_x_pointer = x_struct.X_std + x_struct.n_y * split_var;

    for (size_t i = 0; i < p_continuous; i++) // loop over variables
    {
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
    }
    return;
}

void tree::gp_predict_from_root(matrix<size_t> &Xorder_std, gp_struct &x_struct, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, matrix<size_t> &Xtestorder_std, gp_struct &xtest_struct, std::vector<size_t> &Xtest_counts, std::vector<size_t> &Xtest_num_unique, matrix<double> &yhats_test_xinfo, std::vector<bool> active_var, const size_t &p_categorical, const size_t &sweeps, const size_t &tree_ind, const double &theta, const double &tau)
{
    // gaussian process prediction from root
    size_t N = Xorder_std[0].size();
    size_t Ntest = Xtestorder_std[0].size();
    size_t p = active_var.size();
    size_t p_continuous = p - p_categorical;

    if (Ntest == 0)
    { // no need to split if Ntest = 0
        return;
    }

    if (this->l)
    {
        active_var[v] = true;
        std::vector<bool> active_var_left(active_var.size());
        std::vector<bool> active_var_right(active_var.size());
        std::copy(active_var.begin(), active_var.end(), active_var_left.begin());
        std::copy(active_var.begin(), active_var.end(), active_var_right.begin());

        matrix<size_t> Xorder_left_std;
        matrix<size_t> Xorder_right_std;
        matrix<size_t> Xtestorder_left_std;
        matrix<size_t> Xtestorder_right_std;

        std::vector<size_t> X_num_unique_left(X_num_unique.size());
        std::vector<size_t> X_num_unique_right(X_num_unique.size());

        std::vector<size_t> X_counts_left(X_counts.size());
        std::vector<size_t> X_counts_right(X_counts.size());

        std::vector<size_t> Xtest_num_unique_left(Xtest_num_unique.size());
        std::vector<size_t> Xtest_num_unique_right(Xtest_num_unique.size());

        std::vector<size_t> Xtest_counts_left(Xtest_counts.size());
        std::vector<size_t> Xtest_counts_right(Xtest_counts.size());

        if (N > 0)
        {
            // get split point
            size_t split_point = get_split_point(x_struct.X_std, Xorder_std, x_struct.n_y, v, c);
            ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
            ini_xinfo_sizet(Xorder_right_std, N - split_point - 1, p);

            if (p_categorical > 0)
            {
                split_xorder_std_categorical_simplified(x_struct, Xorder_left_std, Xorder_right_std, this->v, split_point, Xorder_std, X_counts_left, X_counts_right,
                                                        X_num_unique_left, X_num_unique_right, X_counts, p_categorical);
            }

            if (p_continuous > 0)
            {
                split_xorder_std_continuous_simplified(x_struct, Xorder_left_std, Xorder_right_std, v, split_point, Xorder_std, p_continuous);
            }
        }

        if (Ntest > 0)
        {

            if (c < *(xtest_struct.X_std + xtest_struct.n_y * v + Xtestorder_std[v][0]))
            {
                // all test data goes to the right node
                this->r->gp_predict_from_root(Xorder_right_std, x_struct, X_counts_right, X_num_unique_right,
                                              Xtestorder_std, xtest_struct, Xtest_counts, Xtest_num_unique, yhats_test_xinfo, active_var_right, p_categorical, sweeps, tree_ind, theta, tau);
                return;
            }
            if (c >= *(xtest_struct.X_std + xtest_struct.n_y * v + Xtestorder_std[v][Ntest - 1]))
            {
                // all test data goes to the left node
                this->l->gp_predict_from_root(Xorder_left_std, x_struct, X_counts_left, X_num_unique_left,
                                              Xtestorder_std, xtest_struct, Xtest_counts, Xtest_num_unique, yhats_test_xinfo, active_var_left, p_categorical, sweeps, tree_ind, theta, tau);
                return;
            }

            size_t test_split_point = get_split_point(xtest_struct.X_std, Xtestorder_std, xtest_struct.n_y, v, c);

            ini_xinfo_sizet(Xtestorder_left_std, test_split_point + 1, p);
            ini_xinfo_sizet(Xtestorder_right_std, Ntest - test_split_point - 1, p);

            if (p_categorical > 0)
            {
                split_xorder_std_categorical_simplified(xtest_struct, Xtestorder_left_std, Xtestorder_right_std, v, test_split_point, Xtestorder_std, Xtest_counts_left, Xtest_counts_right, Xtest_num_unique_left, Xtest_num_unique_right, Xtest_counts, p_categorical);
            }
            if (p_continuous > 0)
            {
                split_xorder_std_continuous_simplified(xtest_struct, Xtestorder_left_std, Xtestorder_right_std, v, test_split_point, Xtestorder_std, p_continuous);
            }
        }

        this->l->gp_predict_from_root(Xorder_left_std, x_struct, X_counts_left, X_num_unique_left,
                                      Xtestorder_left_std, xtest_struct, Xtest_counts_left, Xtest_num_unique_left, yhats_test_xinfo, active_var_left, p_categorical, sweeps, tree_ind, theta, tau);

        this->r->gp_predict_from_root(Xorder_right_std, x_struct, X_counts_right, X_num_unique_right,
                                      Xtestorder_right_std, xtest_struct, Xtest_counts_right, Xtest_num_unique_right, yhats_test_xinfo, active_var_right, p_categorical, sweeps, tree_ind, theta, tau);
    }
    else
    {
        if (N == 0)
        {
            COUT << "0 training data in the leaf node" << endl;
            throw;
        }
        // assign mu
        for (size_t i = 0; i < Ntest; i++)
        {
            yhats_test_xinfo[sweeps][Xtestorder_std[0][i]] += this->theta_vector[0];
        }

        // get local range
        matrix<double> local_X_range;
        get_X_range(x_struct.X_std, Xorder_std, local_X_range, x_struct.n_y);

        // check out of range test sets
        // exterior points may not be out-of-range on all active variables
        // do we want to use all acitve_var info or just those having out-of-range points?
        // FOR NOW: just those having out-of-range points
        std::vector<size_t> test_ind;
        std::vector<bool> active_var_out_range(p_continuous, false);
        for (size_t i = 0; i < Ntest; i++)
        {
            for (size_t j = 0; j < p_continuous; j++)
            {
                if (active_var[j])
                {
                    if (*(xtest_struct.X_std + xtest_struct.n_y * j + Xtestorder_std[j][i]) > local_X_range[j][1])
                    {
                        test_ind.push_back(Xtestorder_std[j][i]);
                        active_var_out_range[j] = true;
                        break;
                    }
                    else if (*(xtest_struct.X_std + xtest_struct.n_y * j + Xtestorder_std[j][i]) < local_X_range[j][0])
                    {
                        test_ind.push_back(Xtestorder_std[j][i]);
                        active_var_out_range[j] = true;
                        break;
                    }
                }
            }
        }

        // construct covariance matrix
        // TODO: consider categorical active variables
        size_t p_active = std::accumulate(active_var_out_range.begin(), active_var_out_range.end(), 0);
        if (p_active == 0)
        {
            return;
        }

        Ntest = test_ind.size();
        if (Ntest == 0)
        {
            return;
        }

        // get training set
        std::vector<size_t> train_ind;
        if (N < 100)
        {
            train_ind.resize(N);
            std::copy(Xorder_std[0].begin(), Xorder_std[0].end(), train_ind.begin());
        }
        else
        {
            N = 100;
            train_ind.resize(100);
            std::sample(Xorder_std[0].begin(), Xorder_std[0].end(), train_ind.begin(), 100, x_struct.gen);
        }

        mat X(N + Ntest, p_active);
        std::vector<double> x_range(p_active);
        const double *split_var_x_pointer;
        size_t j_count = 0;
        for (size_t j = 0; j < p_continuous; j++)
        {
            if (active_var_out_range[j])
            {
                split_var_x_pointer = x_struct.X_std + x_struct.n_y * j;
                for (size_t i = 0; i < N; i++)
                {
                    X(i, j_count) = *(split_var_x_pointer + train_ind[i]);
                }

                // if (local_X_range[j][1] > local_X_range[j][0]){
                //     x_range[j_count] = sqrt(local_X_range[j][1] - local_X_range[j][0]);
                // }else{
                //     x_range[j_count] =  sqrt(*(split_var_x_pointer + Xorder_std[j][Xorder_std[j].size()-1]) - *(split_var_x_pointer + Xorder_std[j][0]));
                // }
                if (local_X_range[j][1] > local_X_range[j][0])
                {
                    x_range[j_count] = (local_X_range[j][1] - local_X_range[j][0]);
                }
                else
                {
                    x_range[j_count] = (*(split_var_x_pointer + Xorder_std[j][Xorder_std[j].size() - 1]) - *(split_var_x_pointer + Xorder_std[j][0]));
                }

                split_var_x_pointer = xtest_struct.X_std + xtest_struct.n_y * j;
                for (size_t i = 0; i < Ntest; i++)
                {
                    X(i + N, j_count) = *(split_var_x_pointer + test_ind[i]);
                }

                if (x_range[j_count] == 0)
                {
                    COUT << "x_range = 0"
                         << ", j = " << j << endl;
                    throw;
                }
                j_count += 1;
            }
        }

        mat resid(N, 1);
        for (size_t i = 0; i < N; i++)
        {
            resid(i, 0) = x_struct.resid[sweeps][tree_ind][train_ind[i]] - this->theta_vector[0];
        }

        mat cov(N + Ntest, N + Ntest);
        get_rel_covariance(cov, X, x_range, theta, tau);
        for (size_t i = 0; i < N; i++)
        {
            cov(i, i) += pow(x_struct.sigma[tree_ind], 2) / x_struct.num_trees;
        }

        mat mu(Ntest, 1);
        mat Sig(Ntest, Ntest);
        if (N > 0)
        {
            mat k = cov.submat(N, 0, N + Ntest - 1, N - 1);
            mat Kinv = pinv(cov.submat(0, 0, N - 1, N - 1));
            mu = k * Kinv * resid;
            Sig = cov.submat(N, N, N + Ntest - 1, N + Ntest - 1) - k * Kinv * trans(k);
        }
        else
        {
            // prior
            mu.zeros(Ntest, 1);
            Sig = cov.submat(0, 0, Ntest - 1, Ntest - 1);
        }
        mat U;
        vec S;
        mat V;
        svd(U, S, V, Sig);

        std::normal_distribution<double> normal_samp(0.0, 1.0);
        mat samp(Ntest, 1);
        for (size_t i = 0; i < Ntest; i++)
            samp(i, 0) = normal_samp(x_struct.gen);
        mat draws = mu + U * diagmat(sqrt(S)) * samp;
        for (size_t i = 0; i < Ntest; i++)
            yhats_test_xinfo[sweeps][test_ind[i]] += draws(i, 0);
    }
    return;
}

void getTheta_Insample(matrix<double> &output, size_t tree_ind, State &state, X_struct &x_struct)
{
    // get theta of ALL observations of ONE tree, in sample fit
    // input is x_struct because it is in sample

    // output should have dimension (dim_theta, num_obs)

    for (size_t i = 0; i < state.n_y; i++)
    {
        output[i] = *(x_struct.data_pointers[tree_ind][i]);
    }
    return;
}

void getTheta_Outsample(matrix<double> &output, tree &tree, const double *Xtest, size_t N_Xtest, size_t p)
{
    // get theta of ALL observations of ONE tree, out sample fit
    // input is a pointer to testing set matrix because it is out of sample
    // tree is a single tree to look at

    // output should have dimension (dim_theta, num_obs)

    tree::tree_p bn; // pointer to bottom node
    for (size_t i = 0; i < N_Xtest; i++)
    {
        // loop over observations
        // tree search
        bn = tree.search_bottom_std(Xtest, i, p, N_Xtest);
        output[i] = bn->theta_vector;
    }

    return;
}

void getThetaForObs_Insample(matrix<double> &output, size_t x_index, State &state, X_struct &x_struct)
{
    // get theta of ONE observation of ALL trees, in sample fit
    // input is x_struct because it is in sample

    // output should have dimension (dim_theta, num_trees)

    for (size_t i = 0; i < state.num_trees; i++)
    {
        output[i] = *(x_struct.data_pointers[i][x_index]);
    }

    return;
}

void getThetaForObs_Outsample(matrix<double> &output, std::vector<tree> &tree, size_t x_index, const double *Xtest, size_t N_Xtest, size_t p)
{
    // get theta of ONE observation of ALL trees, out sample fit
    // input is a pointer to testing set matrix because it is out of sample
    // tree is a vector of all trees

    // output should have dimension (dim_theta, num_trees)

    tree::tree_p bn; // pointer to bottom node

    for (size_t i = 0; i < tree.size(); i++)
    {
        // loop over trees
        // tree search
        // d = 0; // max distance of outliers
        bn = tree[i].search_bottom_std(Xtest, x_index, p, N_Xtest);
        output[i] = bn->theta_vector;
    }
    return;
}

void getThetaForObs_Outsample_ave(matrix<double> &output, std::vector<tree> &tree, size_t x_index, const double *Xtest, size_t N_Xtest, size_t p)
{
    // This function takes AVERAGE of ALL thetas on the PATH to leaf node

    // get theta of ONE observation of ALL trees, out sample fit
    // input is a pointer to testing set matrix because it is out of sample
    // tree is a vector of all trees

    // output should have dimension (dim_theta, num_trees)

    tree::tree_p bn; // pointer to bottom node
    size_t count = 1;

    for (size_t i = 0; i < tree.size(); i++)
    {

        // loop over trees
        // tree search
        bn = &tree[i]; // start from root node

        std::fill(output[i].begin(), output[i].end(), 0.0);
        count = 0;

        while (bn->getl())
        {
            // while bn has child (not bottom node)

            output[i] = output[i] + bn->theta_vector;
            count++;

            // move to the next level
            if (*(Xtest + N_Xtest * bn->getv() + x_index) <= bn->getc())
            {
                bn = bn->getl();
            }
            else
            {
                bn = bn->getr();
            }
        }

        // bn is the bottom node

        output[i] = output[i] + bn->theta_vector;
        count++;

        // take average of the path
        for (size_t j = 0; j < output[i].size(); j++)
        {
            output[i][j] = output[i][j] / (double)count;
        }
    }

    return;
}

#ifndef NoRcpp
#endif
