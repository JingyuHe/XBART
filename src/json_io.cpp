#include "json_io.h"
// JSON 

json get_forest_json(std::vector<std::vector<tree>> &trees){
    json result;
    result["num_sweeps"] = trees.size();
    result["num_trees"] = trees[0].size();
    result["num_classes"] = trees[0][0].theta_vector.size();
    json trees_j;

    result[std::to_string(0)] = trees[0][0].to_json();
    for(size_t i; i< trees.size();i++){
        std::vector<tree> &tree_vec = trees[i];
        for(size_t j; j < tree_vec.size(); j++){
            trees_j.push_back(tree_vec[j].to_json());
        }
    }
    result["trees"] = trees_j;
    return result;
}

vector<vector<tree>>* from_json_to_forest(std::string &json_string){
    auto j3 = json::parse(json_string);
    
    size_t num_sweeps;
    j3.at("num_sweeps").get_to(num_sweeps);

    size_t num_trees;
    j3.at("num_trees").get_to(num_trees);

    size_t num_classes;
    j3.at("num_classes").get_to(num_classes);


    // // Create trees
    vector<vector<tree>>* result = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*result)[i] = vector<tree>(num_trees);
    }

    for(size_t i = 0; i< num_sweeps; i++){
        for(size_t j = 0;j<num_trees;j++){
            (*result)[i][j].from_json(j3[std::to_string(i)][std::to_string(j)],num_classes);
        }
    }

    return result;


}
