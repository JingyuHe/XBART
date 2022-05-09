#ifndef GUARD_json_io_
#define GUARD_json_io_

#include "tree.h"

json get_forest_json(std::vector<std::vector<tree>> &trees, double y_mean);

void from_json_to_forest(std::string &json_string, vector<vector<tree>> &trees, double &y_mean);

json get_forest_json_3D(std::vector<std::vector<std::vector<tree>>> &trees);

void from_json_to_forest_3D(std::string &json_string, vector<vector<vector<tree>>> &trees);
#endif