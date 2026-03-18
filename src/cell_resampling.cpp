#include "./third_party/csv.h"
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <ranges>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using DistanceAndWeight = std::pair<double, std::reference_wrapper<double>>;

std::tuple<std::unique_ptr<arma::dmat>, std::unique_ptr<arma::dmat>,
           std::vector<double>, std::vector<double>, std::vector<int>>
prepare()
{
    rapidcsv::Document f{"./datasets/combined.csv"};

    auto p_t = f.GetColumn<double>("pt");
    auto y = f.GetColumn<double>("y");
    auto w = f.GetColumn<double>("weight");
    std::vector<double> negative_w{};
    std::vector<int> negative_indices{};

    for (auto [index, weight] : std::ranges::views::enumerate(w))
    {
        if (weight < 0)
        {
            negative_w.push_back(weight);
            negative_indices.push_back(index);
        }
    }

    auto pt_y = std::make_unique<arma::dmat>(2, 4500, arma::fill::zeros);
    auto negative_pt_y =
        std::make_unique<arma::dmat>(2, negative_w.size(), arma::fill::zeros);

    auto i{0};
    auto j{0};

    for (std::tuple<double&, double&, double&> k : std::views::zip(p_t, y, w))
    {
        (*pt_y)[0, i] = std::get<0>(k);
        (*pt_y)[1, i] = std::get<1>(k);

        if (w.at(i) < 0)
        {
            (*negative_pt_y)[0, j] = std::get<0>(k);
            (*negative_pt_y)[1, j] = std::get<1>(k);
            j += 1;
        }

        i += 1;
    }

    return {std::move(pt_y), std::move(negative_pt_y), w, negative_w,
            negative_indices};
}

std::vector<std::vector<DistanceAndWeight>> compute_distance(
    const std::unique_ptr<arma::dmat>& all,
    const std::unique_ptr<arma::dmat>& neg, std::vector<double>& weights)
{

    std::vector<std::vector<DistanceAndWeight>> dnw{};
    arma::dvec scale{1, 10};

    for (int i = 0; i < neg->n_cols; i++)
    {
        auto disp_vector =
            std::make_unique<arma::mat>(all->each_col() - neg->col(i));

        disp_vector->each_col() %= scale;

        auto norm = arma::dcolvec{arma::vecnorm((*disp_vector)).as_col()};
        auto v = std::vector<DistanceAndWeight>{};

        for (std::tuple<double, double&> i : std::views::zip(norm, weights))
        {
            v.push_back(DistanceAndWeight{std::get<0>(i), std::get<1>(i)});
        }

        std::sort(
            v.begin(), v.end(),
            [](const DistanceAndWeight& lhs, const DistanceAndWeight& rhs) {
                return lhs.first < rhs.first;
            });

        dnw.push_back(v);
    }

    return dnw;
}

void resample(int index, double weight,
              std::vector<DistanceAndWeight>& distances_and_weights)
{
    std::vector<double> added_weights{};
    std::vector<double> indices{};
    double abs_weight_sum{};

    int i = 0;
    while (weight < 0)
    {
        if (distances_and_weights.at(i).first == 0)
        {
            indices.push_back(i);
            added_weights.push_back(weight);
            abs_weight_sum += std::abs(weight);
            i += 1;
            continue;
        }

        weight += distances_and_weights.at(i).second;
        added_weights.push_back(distances_and_weights.at(i).second);
        i += 1;
    }

    std::for_each(indices.begin(), indices.end(),
                  [&weight, &distances_and_weights, &added_weights,
                   &abs_weight_sum](auto i) mutable {
                      distances_and_weights.at(i).second.get() =
                          std::abs(distances_and_weights.at(i).second.get()) /
                          abs_weight_sum * weight;
                  });
}

void resample_all(
    std::vector<std::vector<DistanceAndWeight>>& distances_and_weights,
    const std::vector<double>& negative_weights,
    const std::vector<int>& indices)
{
    for (int i = 0; i < negative_weights.size(); i++)
    {
        resample(indices.at(i), negative_weights.at(i),
                 distances_and_weights.at(i));
    }
}

int main()
{
    auto tuple{prepare()};
    auto pt_y{std::move(std::get<0>(tuple))};
    auto negative_pt_y{std::move(std::get<1>(tuple))};
    auto weights{std::get<2>(tuple)};
    auto neg_weights{std::get<3>(tuple)};
    auto neg_indices{std::get<4>(tuple)};

    auto distances_and_weights = compute_distance(pt_y, negative_pt_y, weights);

    resample_all(distances_and_weights, neg_weights, neg_indices);

    if (!std::filesystem::exists("./results"))
        std::filesystem::create_directory("./results");

    std::ofstream f{"./results/weights.csv", std::ios::trunc};
    f << "weight\n";

    std::for_each(weights.begin(), weights.end(),
                  [&f](auto i) mutable { f << i << "\n"; });
}
