// #include "MH.h"

// double tree_likelihood(tree &tree)
// {
//     /*
//         This function loops over all bottom nodes, take cumulative prod (usm of log) of bottom nodes
//     */
//     tree::npv tree_vec;
//     tree.getbots(tree_vec);
//     // cout << "bottom size " << tree_vec.size() << endl;
//     double output = 0.0;
//     for (size_t i = 0; i < tree_vec.size(); i++)
//     {
//         output += tree_vec[i]->getloglike_node();
//     }

//     // cout << "output of tree_likelihood " << output << endl;

//     // add constant
//     // output = output - N * log(2 * 3.14159265359) / 2.0 - N * log(sigma) - std::inner_product(y.begin(), y.end(), y.begin(), 0.0) / pow(sigma, 2) / 2.0;

//     return output;
// }

// double prior_prob(NormalModel *model, tree &tree)
// {
//     /*
//         This function calculate the log of 
//         the prior probability of drawing the given tree
//     */
//     double output = 0.0;
//     double log_split_prob = 0.0;
//     double log_leaf_prob = 0.0;
//     tree::npv tree_vec;

//     // get a vector of all nodess
//     tree.getnodes(tree_vec);

//     for (size_t i = 0; i < tree_vec.size(); i++)
//     {
//         if (tree_vec[i]->getl() == 0)
//         {
//             // if no children, it is end node, count leaf parameter probability

//             // leaf prob, normal center at ZERO
//             log_leaf_prob += normal_density(tree_vec[i]->theta_vector[0], 0.0, model->tau, true);

//             // log_split_prob += log(1 - alpha * pow((1 + tree_vec[i]->depth()), -beta));
//             log_split_prob += log(1.0 - model->alpha * pow(1 + tree_vec[i]->getdepth(), -1.0 * model->beta));

//             // add prior of split point
//             log_split_prob = log_split_prob - log(tree_vec[i]->getnum_cutpoint_candidates());
//         }
//         else
//         {
//             // otherwise count cutpoint probability
//             // log_split_prob += log(alpha * pow((1.0 + tree_vec[i]->depth()), -beta));

//             log_split_prob += log(model->alpha) - model->beta * log(1.0 + tree_vec[i]->getdepth());
//         }
//     }
//     output = log_split_prob + log_leaf_prob;
//     // output = log_split_prob;
//     return output;
// }

// double transition_prob(tree &tree)
// {
//     /*
//         This function calculate probability of given tree
//         log P(all cutpoints) + log P(leaf parameters)
//         Used in M-H ratio calculation
//     */

//     double output = 0.0;
//     double log_p_cutpoints = 0.0;
//     double log_p_leaf = 0.0;
//     tree::npv tree_vec;

//     // get a vector of all nodess
//     tree.getnodes(tree_vec);

//     for (size_t i = 0; i < tree_vec.size(); i++)
//     {
//         if(tree_vec[i]->getdepth() == 1){
//             output += tree_vec[i]->getloglike_node();
//         }
//     }

//     return output;
// };

// bool check_same_path(tree *node1, tree *node2)
// {
//     bool output;
//     // given any two nodes of a tree, check are they on same path or not
//     size_t depth1 = node1->getdepth();
//     size_t depth2 = node2->getdepth();

//     tree *temp1 = node1;
//     tree *temp2 = node2;

//     while (depth1 != depth2)
//     {
//         // if they are not in same level
//         if (depth1 > depth2)
//         {
//             temp1 = temp1->getp();
//             depth1 = depth1 - 1;
//         }
//         else if (depth1 < depth2)
//         {
//             temp2 = temp2->getp();
//             depth2 = depth2 - 1;
//         }
//     }

//     // after reaching the same level
//     if (temp1->getID() != temp2->getID())
//     {
//         output = false;
//     }
//     else
//     {
//         output = true;
//     }

//     return output;
// }

// double w_overlap(tree *node1, tree *node2)
// {
//     double output = 0.0;
//     if (check_same_path(node1, node2))
//     {
//         // if they are on same path
//         tree *temp;

//         // pick the one at lower level
//         if (node1->getdepth() >= node2->getdepth())
//         {
//             temp = node1;
//         }
//         else
//         {
//             temp = node2;
//         }

//         // find all LEAF nodes of the lower level node
//         std::vector<tree *> leaf_nodes;
//         temp->getbots(leaf_nodes);

//         for (size_t i = 0; i < leaf_nodes.size(); i++)
//         {
//             // loop over all leaf nodes
//             output = output + (double)leaf_nodes[i]->getN() / (double)leaf_nodes[i]->getdepth();
//         }
//     }
//     else
//     {
//         output = 0.0;
//     }

//     return output;
// }

// void determinant_precision(tree &root, double t, double s2, std::unique_ptr<State> &state, double& val, double& sign, double& logdetA)
// {
//     // this function calculate determinat of precision matrix given a tree
//     // use for adjusted prior of MH ratio

//     /////////////////////////////////////////////////
//     //
//     // Warning! The tree depth start from 0, 
//     // but in the calculation below, it should start from 1
//     //
//     /////////////////////////////////////////////////     

//     // first find a list of all nodes
//     std::vector<tree *> all_nodes;
//     root.getnodes(all_nodes);

//     size_t tree_size = all_nodes.size();

//     arma::mat M(tree_size, tree_size);

//     logdetA = 0.0;

//     // M is symmetric matrix, only need to loop lower triangle
//     for (size_t i = 0; i < tree_size; i++)
//     {
//         // also checkeck if all_nodes[i] leaf nodes or not, to compute detA
//         if (!(all_nodes[i]->getl()))
//         {
//             // if no left child, it is leaf node
//             logdetA = logdetA + (double)all_nodes[i]->getN() * log((double)all_nodes[i]->getdepth() + 1);

//         // cout << "aaaa " << all_nodes[i]->getN()  << "  " << log((double)all_nodes[i]->getdepth()) << "  " << (double)all_nodes[i]->getN() * log((double)all_nodes[i]->getdepth() + 1) << endl;
//         }
//         for (size_t j = 0; j <= i; j++)
//         {
//             M(i, j) = w_overlap(all_nodes[i], all_nodes[j]) * t / (sqrt(s2 + t * all_nodes[i]->getN()) * sqrt(s2 + t * all_nodes[j]->getN()));

//             // fill the other side of M matrix
//             M(j, i) = M(i, j);
//         }
//     }

//     // adjust detA with constant
//     logdetA = logdetA - state->n_y * log(s2);

//     arma::log_det(val, sign, arma::eye<arma::mat>(tree_size, tree_size) - M);

//     return;
// }