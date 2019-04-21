#ifndef GUARD_json_io_
#define GUARD_json_io_

#include "tree.h"

json get_forest_json(std::vector<std::vector<tree>> &trees);

std::vector<std::vector<tree>>* from_json_to_forest(std::string &json_string);

#endif