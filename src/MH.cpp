#include "MH.h"



double tree_likelihood(tree &tree)
{
    /*
        This function loops over all bottom nodes, take cumulative prod (usm of log) of bottom nodes
    */
    tree::npv tree_vec;
    tree.getbots(tree_vec);
    // cout << "bottom size " << tree_vec.size() << endl;
    double output = 0.0;
    for (size_t i = 0; i < tree_vec.size(); i++)
    {
        output += tree_vec[i]->getloglike_node();
    }

    // cout << "output of tree_likelihood " << output << endl;

    // add constant
    // output = output - N * log(2 * 3.14159265359) / 2.0 - N * log(sigma) - std::inner_product(y.begin(), y.end(), y.begin(), 0.0) / pow(sigma, 2) / 2.0;

    return output;
}

double prior_prob(NormalModel *model, tree &tree)
{
    /*
        This function calculate the log of 
        the prior probability of drawing the given tree
    */
    double output = 0.0;
    double log_split_prob = 0.0;
    double log_leaf_prob = 0.0;
    tree::npv tree_vec;

    // get a vector of all nodess
    tree.getnodes(tree_vec);

    for (size_t i = 0; i < tree_vec.size(); i++)
    {
        if (tree_vec[i]->getl() == 0)
        {
            // if no children, it is end node, count leaf parameter probability

            // leaf prob, normal center at ZERO
            log_leaf_prob += normal_density(tree_vec[i]->theta_vector[0], 0.0, model->tau, true);

            // log_split_prob += log(1 - alpha * pow((1 + tree_vec[i]->depth()), -beta));
            log_split_prob += log(1.0 - model->alpha * pow(1 + tree_vec[i]->getdepth(), -1.0 * model->beta));

            // add prior of split point
            log_split_prob = log_split_prob - log(tree_vec[i]->getnum_cutpoint_candidates());
        }
        else
        {
            // otherwise count cutpoint probability
            // log_split_prob += log(alpha * pow((1.0 + tree_vec[i]->depth()), -beta));

            log_split_prob += log(model->alpha) - model->beta * log(1.0 + tree_vec[i]->getdepth());
        }
    }
    output = log_split_prob + log_leaf_prob;
    // output = log_split_prob;
    return output;
}


double transition_prob(tree &tree)
{
    /*
        This function calculate probability of given tree
        log P(all cutpoints) + log P(leaf parameters)
        Used in M-H ratio calculation
    */

    double output = 0.0;
    double log_p_cutpoints = 0.0;
    double log_p_leaf = 0.0;
    tree::npv tree_vec;

    // get a vector of all nodess
    tree.getnodes(tree_vec);

    for (size_t i = 0; i < tree_vec.size(); i++)
    {
        output += tree_vec[i]->getloglike_node();
    }
    
    return output;
};