#ifndef GUARD_MH_h
#define GUARD_MH_h

#include "common.h"
#include "tree.h"
#include "model.h"

double tree_likelihood(tree &tree);

double prior_prob(NormalModel *model, tree &tree);

double transition_prob(tree &tree);



#endif