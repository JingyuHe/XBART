//////////////////////////////////////////////////////////////////////////////////////
// save tree to json objects
//////////////////////////////////////////////////////////////////////////////////////


#include "json_io.h"
// JSON

json get_forest_json(std::vector<std::vector<tree>> &trees, double y_mean)
{
    json result;
    result["xbart_version"] = "beta";
    result["xbart_serialization_version"] = 0;
    result["num_sweeps"] = trees.size();
    result["num_trees"] = trees[0].size();
    result["dim_theta"] = trees[0][0].theta_vector.size();
    result["y_mean"] = y_mean;

    json trees_j;
    // auto jsonObjects = json::array();

    // result[std::to_string(0)] = trees[0][0].to_json();
    for (size_t i = 0; i < trees.size(); i++)
    {
        std::vector<tree> &tree_vec = trees[i];
        for (size_t j = 0; j < tree_vec.size(); j++)
        {
            trees_j[std::to_string(i)][std::to_string(j)] = tree_vec[j].to_json();
            // jsonObjects.push_back(tree_vec[j].to_json());
        }
    }
    result["trees"] = trees_j;
    return result;
}

void from_json_to_forest(std::string &json_string, vector<vector<tree>> &trees, double &y_mean)
{
    auto j3 = json::parse(json_string);

    size_t num_sweeps;
    j3.at("num_sweeps").get_to(num_sweeps);

    size_t num_trees;
    j3.at("num_trees").get_to(num_trees);

    size_t dim_theta;
    j3.at("dim_theta").get_to(dim_theta);

    j3.at("y_mean").get_to(y_mean);

    // // Create trees
    trees.resize(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        trees[i] = vector<tree>(num_trees);
    }

    for (size_t i = 0; i < num_sweeps; i++)
    {
        for (size_t j = 0; j < num_trees; j++)
        {
            trees[i][j].from_json(j3["trees"][std::to_string(i)][std::to_string(j)], dim_theta);
        }
    }
    // tree::npv bv;
    // trees[0][0].getnodes(bv); //get bottom nodes
    // for (size_t i = 0; i < bv.size(); i++){
    //     cout << bv[i]->getv() << " " << bv[i]->getID() << endl;
    // }
    return;
}

json get_forest_json_3D(std::vector<std::vector<std::vector<tree>>> &trees)
{
    // push 3 dimensional matrix of trees to json string trees[class][sweeps][tree index]
    json result;
    result["xbart_version"] = "beta";
    result["xbart_serialization_version"] = 0;
    result["num_classes"] = trees.size();
    result["num_sweeps"] = trees[0].size();
    result["num_trees"] = trees[0][0].size();
    result["dim_theta"] = trees[0][0][0].theta_vector.size();

    json trees_j;
    for (size_t k = 0; k < trees.size(); k++)
    {
        for (size_t i = 0; i < trees[0].size(); i++)
        {
            for (size_t j = 0; j < trees[0][0].size(); j++)
            {
                trees_j[std::to_string(k)][std::to_string(i)][std::to_string(j)] = trees[k][i][j].to_json();
                // jsonObjects.push_back(tree_vec[j].to_json());
            }
        }
    }
    result["trees"] = trees_j;
    return result;
}

void from_json_to_forest_3D(std::string &json_string, vector<vector<vector<tree>>> &trees)
{
    // push json string to 3 dimensional matrix of trees

    auto j3 = json::parse(json_string);

    size_t num_classes;
    j3.at("num_classes").get_to(num_classes);

    size_t num_sweeps;
    j3.at("num_sweeps").get_to(num_sweeps);

    size_t num_trees;
    j3.at("num_trees").get_to(num_trees);

    size_t dim_theta;
    j3.at("dim_theta").get_to(dim_theta);

    // // Create trees
    trees.resize(num_classes);
    for (size_t k = 0; k < num_classes; k++)
    {
        trees[k] = vector<vector<tree>>(num_sweeps);
        for (size_t i = 0; i < num_sweeps; i++)
        {
            trees[k][i] = vector<tree>(num_trees);
        }
    }

    for (size_t k = 0; k < num_classes; k++)
    {
        for (size_t i = 0; i < num_sweeps; i++)
        {
            for (size_t j = 0; j < num_trees; j++)
            {
                trees[k][i][j].from_json(j3["trees"][std::to_string(k)][std::to_string(i)][std::to_string(j)], dim_theta);
            }
        }
    }
    return;
}
