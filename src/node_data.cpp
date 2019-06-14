#include "node_data.h"

void NodeData::unique_value_count(std::unique_ptr<FitInfo> &fit_info)
{
    double current_value = 0.0;
    size_t count_unique = 0;
    size_t N_unique;
    fit_info->variable_ind[0] = 0;
    size_t total_points = 0;
    for (size_t i = fit_info->p_continuous; i < fit_info->p; i++)
    {
        // only loop over categorical variables
        // suppose p = (p_continuous, p_categorical)
        // index starts from p_continuous
        this->X_counts.push_back(1);
        current_value = *(fit_info->X_std + i * this->N_Xorder + this->Xorder_std[i][0]);
        fit_info->X_values.push_back(current_value);
        count_unique = 1;

        for (size_t j = 1; j < this->N_Xorder; j++)
        {
            if (*(fit_info->X_std  + i * this->N_Xorder + Xorder_std[i][j]) == current_value)
            {
                X_counts[total_points]++;
            }
            else
            {
                current_value = *(fit_info->X_std  + i * this->N_Xorder + Xorder_std[i][j]);
                fit_info->X_values.push_back(current_value);
                this->X_counts.push_back(1);
                count_unique++;
                total_points++;
            }
        }
        fit_info->variable_ind[i + 1 - fit_info->p_continuous] = count_unique + fit_info->variable_ind[i - fit_info->p_continuous];
        this->X_num_unique[i - fit_info->p_continuous] = count_unique;
        total_points++;
    }

    return;
}
