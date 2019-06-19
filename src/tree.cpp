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
// size_t tree::depth()
// {
//     if (!p)
//         return 0; //no parents
//     else
//         return (1 + p->depth());
// }
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
    size_t d = this->depth;
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
// create a new copy of tree in NEW memory space
{
    if (n->l)
    {
        COUT << "cp:error node has children\n";
        return;
    }

    n->v = o->v;
    n->c = o->c;
    n->sig = o->sig;
    n->prob_split = o->prob_split;
    n->prob_leaf = o->prob_leaf;
    n->drawn_ind = o->drawn_ind;
    n->N_Xorder = o->N_Xorder;
    n->y_mean = o->y_mean;
    n->loglike_leaf = o->loglike_leaf;
    n->tree_like = o->tree_like;
    n->theta_vector = o->theta_vector;
    // n->split_var = o->split_var;
    // n->split_point = o->split_point;
    // n->no_split = o->no_split;

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

void tree::copy_only_root(tree_p o)
//assume n has no children (so we don't have to kill them)
//NOT LIKE cp() function
//this function pointer new root to the OLD structure
{
    this->v = o->v;
    this->c = o->c;
    this->sig = o->sig;
    this->prob_split = o->prob_split;
    this->prob_leaf = o->prob_leaf;
    this->drawn_ind = o->drawn_ind;
    this->N_Xorder = o->N_Xorder;
    this->y_mean = o->y_mean;
    this->loglike_leaf = o->loglike_leaf;
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

// double tree::tree_likelihood(size_t N, double sigma, size_t tree_ind, Model *model, std::unique_ptr<State>& state, const double *Xpointer, vector<double>& y, bool proposal)
// {
//     /*
//         This function calculate the log of
//         the likelihood of all leaf parameters of given tree
//     */
//     double output = 0.0;
//     std::vector<double> pred(N);
//     if(proposal){
//         // calculate likelihood of proposal
//         predict_from_datapointers(Xpointer, N, tree_ind, pred, state->data_pointers, model);
//     }else{
//         // calculate likelihood of previous accpeted tree
//         predict_from_datapointers(Xpointer, N, tree_ind, pred, state->data_pointers_copy, model);
//     }

//     double sigma2 = pow(sigma, 2);

//     for(size_t i = 0; i < N; i ++ ){
//         output = output + normal_density(y[i], pred[i], sigma2, true);
//     }

//     return output;
// }

double tree::tree_likelihood(size_t N, double sigma, vector<double> y)
{
    /*
        This function calculate the log of 
        the likelihood of all leaf parameters of given tree
    */
    npv tree_vec;
    this->getbots(tree_vec);
    // cout << "bottom size " << tree_vec.size() << endl;
    double output = 0.0;
    for (size_t i = 0; i < tree_vec.size(); i++)
    {
        output += tree_vec[i]->loglike_leaf;
    }

    // cout << "output of tree_likelihood " << output << endl;

    // add constant
    // output = output - N * log(2 * 3.14159265359) / 2.0 - N * log(sigma) - std::inner_product(y.begin(), y.end(), y.begin(), 0.0) / pow(sigma, 2) / 2.0;

    tree_like = output;
    return output;
}

double tree::prior_prob(NormalModel *model)
{
    /*
        This function calculate the log of 
        the prior probability of drawing the given tree
    */
    double output = 0.0;
    double log_split_prob = 0.0;
    double log_leaf_prob = 0.0;
    npv tree_vec;

    // get a vector of all nodess
    this->getnodes(tree_vec);

    for (size_t i = 0; i < tree_vec.size(); i++)
    {
        if (tree_vec[i]->getl() == 0)
        {
            // if no children, it is end node, count leaf parameter probability

            // leaf prob, normal center at ZERO
            log_leaf_prob += normal_density(tree_vec[i]->theta_vector[0], 0.0, model->tau, true);

            // log_split_prob += log(1 - alpha * pow((1 + tree_vec[i]->depth()), -beta));
            log_split_prob += log(1.0 - model->alpha * pow(1 + tree_vec[i]->depth, -1.0 * model->beta));

            // add prior of split point
            log_split_prob = log_split_prob - log(tree_vec[i]->getnum_cutpoint_candidates());
        }
        else
        {
            // otherwise count cutpoint probability
            // log_split_prob += log(alpha * pow((1.0 + tree_vec[i]->depth()), -beta));

            log_split_prob += log(model->alpha) - model->beta * log(1.0 + tree_vec[i]->depth);
        }
    }
    output = log_split_prob + log_leaf_prob;
    // output = log_split_prob;
    return output;
}

void tree::grow_from_root(std::unique_ptr<State> &state, size_t max_depth, xinfo_sizet &Xorder_std, std::vector<double> &mtry_weight_current_tree, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, const size_t &tree_ind, NodeData &node_data, bool update_theta, bool update_split_prob, bool grow_new_tree)
{
    // grow a tree, users can control number of split points
    size_t N_Xorder = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    size_t ind;
    size_t split_var;
    size_t split_point;

    this->setN_Xorder(N_Xorder);

    if (N_Xorder <= state->n_min)
    {
        return;
    }

    if (this->depth >= max_depth - 1)
    {
        return;
    }

    // tau is prior VARIANCE, do not take squares

    if (update_theta)
    {
        model->samplePars(state->draw_mu, this->suff_stat[0], N_Xorder, node_data.sigma, state->gen, this->theta_vector, state->residual_std, Xorder_std, this->prob_leaf);

        this->sig = node_data.sigma;

        // node_data.sigma = sigma;
    }

    bool no_split = false;

    std::vector<size_t> subset_vars(p);

    if (state->use_all)
    {
        std::iota(subset_vars.begin(), subset_vars.end(), 0);
    }
    else
    {
        if (state->sample_weights_flag)
        {
            std::vector<double> weight_samp(p);
            double weight_sum;

            // Sample Weights Dirchelet
            for (size_t i = 0; i < p; i++)
            {
                std::gamma_distribution<double> temp_dist(mtry_weight_current_tree[i], 1.0);
                weight_samp[i] = temp_dist(state->gen);
            }
            weight_sum = accumulate(weight_samp.begin(), weight_samp.end(), 0.0);
            for (size_t i = 0; i < p; i++)
            {
                weight_samp[i] = weight_samp[i] / weight_sum;
            }

            subset_vars = sample_int_ccrank(p, state->mtry, weight_samp, state->gen);
        }
        else
        {
            subset_vars = sample_int_ccrank(p, state->mtry, mtry_weight_current_tree, state->gen);
        }
    }

    BART_likelihood_all(Xorder_std, node_data.sigma, no_split, split_var, split_point, subset_vars, X_counts, X_num_unique, model, state, this, node_data, update_split_prob);

    if (no_split == true)
    {
        if (!update_split_prob)
        {
            for (size_t i = 0; i < N_Xorder; i++)
            {
                state->data_pointers[tree_ind][Xorder_std[0][i]] = &this->theta_vector;
            }
        }

        if(update_theta){
            model->samplePars(state->draw_mu, this->suff_stat[0], N_Xorder, node_data.sigma, state->gen, this->theta_vector, state->residual_std, Xorder_std, this->prob_leaf);
        }

        this->l = 0;
        this->r = 0;

        // update leaf prob, for MH update useage
        // this->loglike_leaf = model->likelihood_no_split(this->suff_stat[0] * N_Xorder, model->tau, N_Xorder * model->tau, pow(node_data.sigma, 2));
        this->loglike_leaf = model->likelihood_no_split(node_data, this->suff_stat);

        // this->prob_leaf = normal_density(this->theta_vector[0], this->suff_stat[0] * node_data.N_Xorder / pow(node_data.sigma, 2) / (1.0 / model->tau + node_data.N_Xorder / pow(node_data.sigma, 2)), 1.0 / (1.0 / model->tau + node_data.N_Xorder / pow(node_data.sigma, 2)), true);

        return;
    }

    if(grow_new_tree){ 
        // If GROW FROM ROOT MODE
        this->v = split_var;
        this->c = *(state->X_std + state->n_y * split_var + Xorder_std[split_var][split_point]);
    }

    // Update Cutpoint to be a true seperating point
    // Increase split_point (index) until it is no longer equal to cutpoint value
    while ((split_point < node_data.N_Xorder - 1) && (*(state->X_std + state->n_y * split_var + Xorder_std[split_var][split_point + 1]) == this->c))
    {
        split_point = split_point + 1;
    }

    // If our current split is same as parent, exit
    if ((this->p) && (this->v == (this->p)->v) && (this->c == (this->p)->c))
    {
        return;
    }

    if (grow_new_tree)
    {
        // If do not update split prob ONLY
        // grow from root, initialize new nodes

        state->split_count_current_tree[split_var] += 1;

        tree::tree_p lchild = new tree(model->getNumClasses(), this);
        tree::tree_p rchild = new tree(model->getNumClasses(), this);

        this->l = lchild;
        this->r = rchild;

        lchild->depth = this->depth + 1;
        rchild->depth = this->depth + 1;
    }
    else
    {
        // For MH update usage, update probability of cutpoints given new data
        // Do not need to initialize new nodes
    }

    this->l->ini_suff_stat();
    this->r->ini_suff_stat();

    xinfo_sizet Xorder_left_std;
    xinfo_sizet Xorder_right_std;
    ini_xinfo_sizet(Xorder_left_std, split_point + 1, p);
    ini_xinfo_sizet(Xorder_right_std, N_Xorder - split_point - 1, p);

    std::vector<size_t> X_num_unique_left(X_num_unique.size());
    std::vector<size_t> X_num_unique_right(X_num_unique.size());

    std::vector<size_t> X_counts_left(X_counts.size());
    std::vector<size_t> X_counts_right(X_counts.size());

    if (state->p_categorical > 0)
    {
        split_xorder_std_categorical(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, X_counts_left, X_counts_right, X_num_unique_left, X_num_unique_right, X_counts, model, state, this);
    }

    if (state->p_continuous > 0)
    {
        split_xorder_std_continuous(Xorder_left_std, Xorder_right_std, split_var, split_point, Xorder_std, model, state, this);
    }

    NodeData left_node_data(node_data.sigma, split_point + 1);
    NodeData right_node_data(node_data.sigma, node_data.N_Xorder - split_point - 1);

    this->l->grow_from_root(state, max_depth, Xorder_left_std, mtry_weight_current_tree, X_counts_left, X_num_unique_left, model, tree_ind, left_node_data, update_theta, update_split_prob, grow_new_tree);

    this->r->grow_from_root(state, max_depth, Xorder_right_std, mtry_weight_current_tree, X_counts_right, X_num_unique_right, model, tree_ind, right_node_data, update_theta, update_split_prob, grow_new_tree);

    return;
}

double tree::transition_prob()
{
    /*
        This function calculate probability of given tree
        log P(all cutpoints) + log P(leaf parameters)
        Used in M-H ratio calculation
    */

    double output = 0.0;
    double log_p_cutpoints = 0.0;
    double log_p_leaf = 0.0;
    npv tree_vec;

    // get a vector of all nodess
    this->getnodes(tree_vec);

    for (size_t i = 0; i < tree_vec.size(); i++)
    {
        if (tree_vec[i]->getl() == 0)
        {
            // if no children, it is end node, count leaf parameter probability

            // prob_leaf is already in log scale
            log_p_leaf += tree_vec[i]->getprob_leaf();

            // prob_split is in original scale, need to take log
            log_p_cutpoints += log(tree_vec[i]->getprob_split());
        }
        else
        {
            // otherwise count cutpoint probability
            log_p_cutpoints += log(tree_vec[i]->getprob_split());
        }
    }
    // cout << "log_p_cutpoints " << log_p_cutpoints << endl;
    // cout << "log_p_leaf " << log_p_leaf << endl;
    output = log_p_cutpoints + log_p_leaf;

    return output;
};

double tree::log_like_tree(double sigma2, double tau)
{
    double output = 0.0;

    npv tree_vec;

    // get a vector of bottom nodes
    this->getbots(tree_vec);

    // calculate loglikelihood

    // be careful of y^ty term, second order
    // it is changed with new residual
    for (size_t i = 0; i < tree_vec.size(); i++)
    {
        output += 0.5 * (log(sigma2 / (sigma2 + tau * tree_vec[i]->getN_Xorder())) + tau / sigma2 / (sigma2 + tau * tree_vec[i]->getN_Xorder()) * pow(tree_vec[i]->getN_Xorder() * tree_vec[i]->gety_mean(), 2));
    }

    // cout << "output of log_like_tree  " << output << endl;

    return output;
}

void split_xorder_std_continuous(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, Model *model, std::unique_ptr<State> &state, tree *current_node)
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

    current_node->l->ini_suff_stat();
    current_node->r->ini_suff_stat();

    double cutvalue = *(state->X_std + state->n_y * split_var + Xorder_std[split_var][split_point]);

    const double *temp_pointer = state->X_std + state->n_y * split_var;

    for (size_t j = 0; j < N_Xorder; j++)
    {
        if (compute_left_side)
        {
            if (*(temp_pointer + Xorder_std[split_var][j]) <= cutvalue)
            {
                // current_node->l->suff_stat[0] += state->residual_std[Xorder_std[split_var][j]];

                // current_node->l->suff_stat[1] += pow(state->residual_std[Xorder_std[split_var][j]], 2);

                model->updateNodeSuffStat(current_node->l->suff_stat, state->residual_std, Xorder_std, split_var, j);
            }
        }
        else
        {
            if (*(temp_pointer + Xorder_std[split_var][j]) > cutvalue)
            {
                // current_node->r->suff_stat[0] += state->residual_std[Xorder_std[split_var][j]];

                // current_node->r->suff_stat[1] += pow(state->residual_std[Xorder_std[split_var][j]], 2);

                model->updateNodeSuffStat(current_node->r->suff_stat, state->residual_std, Xorder_std, split_var, j);
            }
        }
    }

    const double *split_var_x_pointer = state->X_std + state->n_y * split_var;

    for (size_t i = 0; i < state->p_continuous; i++) // loop over variables
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
        current_node->r->suff_stat[0] = (current_node->suff_stat[0] * N_Xorder - current_node->l->suff_stat[0]) / N_Xorder_right;

        current_node->r->suff_stat[1] = current_node->suff_stat[1] - current_node->l->suff_stat[1];

        current_node->l->suff_stat[0] = current_node->l->suff_stat[0] / N_Xorder_left;
    }
    else
    {
        current_node->l->suff_stat[0] = (current_node->suff_stat[0] * N_Xorder - current_node->r->suff_stat[0]) / N_Xorder_left;

        current_node->l->suff_stat[1] = current_node->suff_stat[1] - current_node->r->suff_stat[1];

        current_node->r->suff_stat[0] = current_node->r->suff_stat[0] / N_Xorder_right;
    }

    return;
}

void split_xorder_std_categorical(xinfo_sizet &Xorder_left_std, xinfo_sizet &Xorder_right_std, size_t split_var, size_t split_point, xinfo_sizet &Xorder_std, std::vector<size_t> &X_counts_left, std::vector<size_t> &X_counts_right, std::vector<size_t> &X_num_unique_left, std::vector<size_t> &X_num_unique_right, std::vector<size_t> &X_counts, Model *model, std::unique_ptr<State> &state, tree *current_node)
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

    current_node->l->ini_suff_stat();
    current_node->r->ini_suff_stat();

    size_t start;
    size_t end;

    double cutvalue = *(state->X_std + state->n_y * split_var + Xorder_std[split_var][split_point]);

    for (size_t i = state->p_continuous; i < state->p; i++)
    {
        // loop over variables
        left_ix = 0;
        right_ix = 0;
        const double *temp_pointer = state->X_std + state->n_y * split_var;

        // index range of X_counts, X_values that are corresponding to current variable
        // start <= i <= end;
        start = state->variable_ind[i - state->p_continuous];
        // COUT << "start " << start << endl;
        end = state->variable_ind[i + 1 - state->p_continuous];

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
                        // current_node->l->suff_stat[0] += state->residual_std[Xorder_std[split_var][j]];

                        // current_node->l->suff_stat[1] += pow(state->residual_std[Xorder_std[split_var][j]], 2);

                        model->updateNodeSuffStat(current_node->l->suff_stat, state->residual_std, Xorder_std, split_var, j);

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
                        // current_node->r->suff_stat[0] += state->residual_std[Xorder_std[split_var][j]];

                        // current_node->r->suff_stat[1] += pow(state->residual_std[Xorder_std[split_var][j]], 2);

                        model->updateNodeSuffStat(current_node->r->suff_stat, state->residual_std, Xorder_std, split_var, j);

                        Xorder_right_std[i][right_ix] = Xorder_std[i][j];

                        right_ix = right_ix + 1;
                    }
                }
            }

            // for the cut variable, it's easy to counts X_counts_left and X_counts_right, simply cut X_counts to two pieces.

            for (size_t k = start; k < end; k++)
            {
                // loop from start to end!

                if (state->X_values[k] <= cutvalue)
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

                while (*(state->X_std + state->n_y * i + Xorder_std[i][j]) != state->X_values[X_counts_index])
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

    // if (compute_left_side)
    // {
    //     current_node->r->suff_stat[0] = (current_node->suff_stat[0] * N_Xorder - current_node->l->suff_stat[0]) / N_Xorder_right;

    //     current_node->r->suff_stat[1] = current_node->suff_stat[1] - current_node->l->suff_stat[1];

    //     current_node->l->suff_stat[0] = current_node->l->suff_stat[0] / N_Xorder_left;
    // }
    // else
    // {
    //     current_node->l->suff_stat[0] = (current_node->suff_stat[0] * N_Xorder - current_node->r->suff_stat[0]) / N_Xorder_left;

    //     current_node->l->suff_stat[1] = current_node->suff_stat[1] - current_node->r->suff_stat[1];

    //     current_node->r->suff_stat[0] = current_node->r->suff_stat[0] / N_Xorder_right;
    // }

    model->calculateOtherSideSuffStat(current_node->suff_stat, current_node->l->suff_stat, current_node->r->suff_stat, N_Xorder, N_Xorder_left, N_Xorder_right, compute_left_side);

    // update X_num_unique

    std::fill(X_num_unique_left.begin(), X_num_unique_left.end(), 0.0);
    std::fill(X_num_unique_right.begin(), X_num_unique_right.end(), 0.0);

    for (size_t i = state->p_continuous; i < state->p; i++)
    {
        start = state->variable_ind[i - state->p_continuous];
        end = state->variable_ind[i + 1 - state->p_continuous];

        // COUT << "start " << start << " end " << end << " size " << X_counts_left.size() << endl;
        for (size_t j = start; j < end; j++)
        {
            if (X_counts_left[j] > 0)
            {
                X_num_unique_left[i - state->p_continuous] = X_num_unique_left[i - state->p_continuous] + 1;
            }
            if (X_counts_right[j] > 0)
            {
                X_num_unique_right[i - state->p_continuous] = X_num_unique_right[i - state->p_continuous] + 1;
            }
        }
    }

    return;
}

void BART_likelihood_all(xinfo_sizet &Xorder_std, double sigma, bool &no_split, size_t &split_var, size_t &split_point, const std::vector<size_t> &subset_vars, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, std::unique_ptr<State> &state, tree *tree_pointer, NodeData &node_data, bool update_split_prob)
{

    // if update_split_prob == true, only update split prob based on given split point, for MH update usage

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

    double sigma2 = pow(sigma, 2);

    double loglike_max = -INFINITY;

    std::vector<double> loglike;

    size_t loglike_start;

    // decide lenght of loglike vector
    if (N <= state->n_cutpoints + 1 + 2 * state->n_min)
    {
        // cout << "small set " << endl;
        loglike.resize((N_Xorder - 1) * state->p_continuous + state->X_values.size() + 1, -INFINITY);
        loglike_start = (N_Xorder - 1) * state->p_continuous;
    }
    else
    {
        // cout << "bigger set " << endl;
        loglike.resize(state->n_cutpoints * state->p_continuous + state->X_values.size() + 1, -INFINITY);
        loglike_start = state->n_cutpoints * state->p_continuous;
    }

    // calculate for each cases
    if (state->p_continuous > 0)
    {
        calculate_loglikelihood_continuous(loglike, subset_vars, N_Xorder, Xorder_std, sigma2, loglike_max, model, state, tree_pointer, node_data);
    }

    if (state->p_categorical > 0)
    {
        calculate_loglikelihood_categorical(loglike, loglike_start, subset_vars, N_Xorder, Xorder_std, sigma2, loglike_max, X_counts, X_num_unique, model, total_categorical_split_candidates, state, tree_pointer, node_data);
    }

    // calculate likelihood of no-split option
    calculate_likelihood_no_split(loglike, N_Xorder, sigma2, loglike_max, model, total_categorical_split_candidates, state, tree_pointer, node_data);

    // transfer loglikelihood to likelihood
    for (size_t ii = 0; ii < loglike.size(); ii++)
    {
        // if a variable is not selected, take exp will becomes 0
        loglike[ii] = exp(loglike[ii] - loglike_max);
    }
    // cout << "loglike " << loglike << endl;
    // cout << " ok " << endl;

    // sampling cutpoints
    if (N <= state->n_cutpoints + 1 + 2 * state->n_min)
    {

        // N - 1 - 2 * Nmin <= Ncutpoints, consider all data points

        // if number of observations is smaller than Ncutpoints, all data are splitpoint candidates
        // note that the first Nmin and last Nmin cannot be splitpoint candidate

        if ((N - 1) > 2 * state->n_min)
        {
            // for(size_t i = 0; i < p; i ++ ){
            for (auto &&i : subset_vars)
            {
                if (i < state->p_continuous)
                {
                    // delete some candidates, otherwise size of the new node can be smaller than Nmin
                    std::fill(loglike.begin() + i * (N - 1), loglike.begin() + i * (N - 1) + state->n_min + 1, 0.0);
                    std::fill(loglike.begin() + i * (N - 1) + N - 2 - state->n_min, loglike.begin() + i * (N - 1) + N - 2 + 1, 0.0);
                }
            }
        }
        else
        {
            // do not use all continuous variables
            std::fill(loglike.begin(), loglike.begin() + (N_Xorder - 1) * state->p_continuous - 1, 0.0);
        }

        std::discrete_distribution<> d(loglike.begin(), loglike.end());

        // for MH update usage only
        tree_pointer->num_cutpoint_candidates = count_non_zero(loglike);

        if (update_split_prob)
        {
            ind = tree_pointer->drawn_ind;
        }
        else
        {
            // sample one index of split point
            ind = d(state->gen);
            tree_pointer->drawn_ind = ind;
        }

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
        else if ((N - 1) <= 2 * state->n_min)
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
            for (size_t i = 0; i < (state->variable_ind.size() - 1); i++)
            {
                if (state->variable_ind[i] <= ind && state->variable_ind[i + 1] > ind)
                {
                    split_var = i;
                }
            }
            start = state->variable_ind[split_var];
            // count how many
            split_point = std::accumulate(X_counts.begin() + start, X_counts.begin() + ind + 1, 0);
            // minus one for correct index (start from 0)
            split_point = split_point - 1;
            split_var = split_var + state->p_continuous;
        }
    }
    else
    {
        // use adaptive number of cutpoints

        std::vector<size_t> candidate_index(state->n_cutpoints);

        seq_gen_std(state->n_min, N - state->n_min, state->n_cutpoints, candidate_index);

        std::discrete_distribution<size_t> d(loglike.begin(), loglike.end());

        // For MH update usage only
        tree_pointer->num_cutpoint_candidates = count_non_zero(loglike);

        if (update_split_prob)
        {
            ind = tree_pointer->drawn_ind;
        }
        else
        {
            // // sample one index of split point
            ind = d(state->gen);
            tree_pointer->drawn_ind = ind;
        }

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
            split_var = ind / state->n_cutpoints;
            split_point = candidate_index[ind % state->n_cutpoints];
        }
        else
        {
            // split at categorical variable
            size_t start;
            ind = ind - loglike_start;
            for (size_t i = 0; i < (state->variable_ind.size() - 1); i++)
            {
                if (state->variable_ind[i] <= ind && state->variable_ind[i + 1] > ind)
                {
                    split_var = i;
                }
            }
            start = state->variable_ind[split_var];
            // count how many
            split_point = std::accumulate(X_counts.begin() + start, X_counts.begin() + ind + 1, 0);
            // minus one for correct index (start from 0)
            split_point = split_point - 1;
            split_var = split_var + state->p_continuous;
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

void calculate_loglikelihood_continuous(std::vector<double> &loglike, const std::vector<size_t> &subset_vars, size_t &N_Xorder, xinfo_sizet &Xorder_std, double &sigma2, double &loglike_max, Model *model, std::unique_ptr<State> &state, tree *tree_pointer, NodeData &node_data)
{

    size_t N = N_Xorder;
    size_t var_index;
    double y_sum = N_Xorder * tree_pointer->suff_stat[0];

    if (N <= state->n_cutpoints + 1 + 2 * state->n_min)
    {
        // if we only have a few data observations in current node
        // use all of them as cutpoint candidates

        double n1tau;
        double n2tau;
        // double Ntau = N_Xorder * model->tau;

        // to have a generalized function, have to pass an empty candidate_index object for this case
        // is there any smarter way to do it?
        std::vector<size_t> candidate_index(1);

        for (auto &&i : subset_vars)
        {
            if (i < state->p_continuous)
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
                    // n1tau = (j + 1) * model->tau; // number of points on left side (x <= cutpoint)
                    // n2tau = Ntau - n1tau;        // number of points on right side (x > cutpoint)

                    model->calcSuffStat_continuous(xorder, state->residual_std, candidate_index, j, false);

                    // loglike[(N_Xorder - 1) * i + j] = model->likelihood(model->tau, n1tau, sigma2, y_sum, true) + model->likelihood(model->tau, n2tau, sigma2, y_sum, false);
                    loglike[(N_Xorder - 1) * i + j] = model->likelihood(node_data, tree_pointer->suff_stat, j, true) + model->likelihood(node_data, tree_pointer->suff_stat, j, false);


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

        std::vector<size_t> candidate_index2(state->n_cutpoints + 1);
        seq_gen_std2(state->n_min, N - state->n_min, state->n_cutpoints, candidate_index2);

        // double Ntau = N_Xorder * model->tau;

        std::mutex llmax_mutex;

        for (auto &&i : subset_vars)
        {
            if (i < state->p_continuous)
            {

                // Lambda callback to perform the calculation
                auto calcllc_i = [i, &loglike, &loglike_max, &Xorder_std, &state, &candidate_index2, &model, &llmax_mutex, N_Xorder, sigma2, y_sum, &tree_pointer, &node_data]() {
                    std::vector<size_t> &xorder = Xorder_std[i];
                    double llmax = -INFINITY;

                    // std::vector<double> y_cumsum(Ncutpoints);
                    Model *clone = model->clone();
                    //model -> suff_stat_init();
                    // clone->suff_stat_fill(y_std, xorder);
                    clone->suff_stat_fill_zero();

                    for (size_t j = 0; j < state->n_cutpoints; j++)
                    {
                        clone->calcSuffStat_continuous(xorder, state->residual_std, candidate_index2, j, true);

                        // loop over all possible cutpoints
                        // double n1tau = (candidate_index2[j + 1] + 1) * model->tau; // number of points on left side (x <= cutpoint)
                        // double n2tau = Ntau - n1tau;                              // number of points on right side (x > cutpoint)

                        // loglike[(state->n_cutpoints) * i + j] = clone->likelihood(model->tau, n1tau, sigma2, y_sum, true) + clone->likelihood(model->tau, n2tau, sigma2, y_sum, false);
                        loglike[(state->n_cutpoints) * i + j] = clone->likelihood(node_data, tree_pointer->suff_stat, candidate_index2[j + 1], true) + clone->likelihood(node_data, tree_pointer->suff_stat, candidate_index2[j + 1], false);


                        if (loglike[(state->n_cutpoints) * i + j] > llmax)
                        {
                            llmax = loglike[(state->n_cutpoints) * i + j];
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

void calculate_loglikelihood_categorical(std::vector<double> &loglike, size_t &loglike_start, const std::vector<size_t> &subset_vars, size_t &N_Xorder, xinfo_sizet &Xorder_std, double &sigma2, double &loglike_max, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique, Model *model, size_t &total_categorical_split_candidates, std::unique_ptr<State> &state, tree *tree_pointer, NodeData &node_data)
{

    // loglike_start is an index to offset
    // consider loglikelihood start from loglike_start
    double y_sum = N_Xorder * tree_pointer->suff_stat[0];

    size_t start;
    size_t end;
    size_t end2;
    double y_cumsum = 0.0;
    size_t n1;
    size_t n2;
    size_t temp;
    size_t N = N_Xorder;

    size_t effective_cutpoints = 0;

    for (auto &&i : subset_vars)
    {

        // COUT << "variable " << i << endl;
        if ((i >= state->p_continuous) && (X_num_unique[i - state->p_continuous] > 1))
        {
            // more than one unique values
            start = state->variable_ind[i - state->p_continuous];
            end = state->variable_ind[i + 1 - state->p_continuous] - 1; // minus one for indexing starting at 0
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
                    model->calcSuffStat_categorical(state->residual_std, Xorder_std, n1, temp, i);

                    n1 = n1 + X_counts[j];
                    // n1tau = (double)n1 * model->tau;
                    // n2tau = ntau - n1tau;

                    // loglike[loglike_start + j] = model->likelihood(model->tau, n1tau, sigma2, y_sum, true) + model->likelihood(model->tau, n2tau, sigma2, y_sum, false);
                    loglike[loglike_start + j] = model->likelihood(node_data, tree_pointer->suff_stat, n1-1, true) + model->likelihood(node_data, tree_pointer->suff_stat, n1-1, false);

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

void calculate_likelihood_no_split(std::vector<double> &loglike, size_t &N_Xorder, double &sigma2, double &loglike_max, Model *model, size_t &total_categorical_split_candidates, std::unique_ptr<State> &state, tree *tree_pointer, NodeData &node_data)
{
    double y_sum = N_Xorder * tree_pointer->suff_stat[0];

  //  loglike[loglike.size() - 1] = model->likelihood_no_split(y_sum, model->tau, N_Xorder * model->tau, sigma2) + log(1.0 - model->alpha * pow(1.0 + tree_pointer->depth, -1.0 * model->beta)) - log(model->alpha) + model->beta * log(1.0 + tree_pointer->depth);
    // loglike[loglike.size() - 1] = model->likelihood_no_split(y_sum, model->tau, N_Xorder * model->tau, sigma2) + log(pow(1.0 + tree_pointer->depth, model->beta) / model->alpha - 1.0) + log((double)loglike.size() - 1.0);  
    loglike[loglike.size() - 1] = model->likelihood_no_split(node_data, tree_pointer->suff_stat) + log(pow(1.0 + tree_pointer->depth, model->beta) / model->alpha - 1.0) + log((double)loglike.size() - 1.0);  

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

   // loglike[loglike.size() - 1] += log(state->p) + log(2.0) + model->getNoSplitPenality();

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

void predict_from_tree(tree &tree, const double *X_std, size_t N, size_t p, std::vector<double> &output, Model *model)
{
    tree::tree_p bn;
    for (size_t i = 0; i < N; i++)
    {
        bn = tree.search_bottom_std(X_std, i, p, N);
        output[i] = model->predictFromTheta(bn->theta_vector);
    }
    return;
}

void predict_from_datapointers(const double *X_std, size_t N, size_t M, std::vector<double> &output, matrix<std::vector<double> *> &data_pointers, Model *model)
{
    // tree search, but read from the matrix of pointers to end node directly
    // easier to get fitted value of training set
    for (size_t i = 0; i < N; i++)
    {
        // cout << "point " << i << " is ok " << endl;
        output[i] = model->predictFromTheta(*data_pointers[M][i]);
    }
    return;
}

#ifndef NoRcpp
#endif
