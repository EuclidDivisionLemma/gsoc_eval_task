#include "./third_party/csv.h"
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <numeric>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

std::tuple<std::unique_ptr<arma::dmat>, std::unique_ptr<arma::dmat>,
           std::vector<double>, std::vector<double>, std::vector<double>>
prepare()
{
    rapidcsv::Document f{"./datasets/combined.csv"};

    auto p_t = f.GetColumn<double>("pt");
    auto y = f.GetColumn<double>("y");
    auto w = f.GetColumn<double>("weight");
    std::vector<double> negative_w{};
    std::vector<double> negative_indices{};

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

        if (w[i] < 0)
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

std::vector<std::pair<std::vector<double>, std::vector<double>>>
compute_distance(const std::unique_ptr<arma::dmat>& all,
                 const std::unique_ptr<arma::dmat>& neg,
                 const std::vector<double>& weights)
{

    std::vector<std::pair<std::vector<double>, std::vector<double>>> dnw{};
    arma::dvec scale{1, 10};

    for (int i = 0; i < neg->n_cols; i++)
    {
        auto weight{weights};
        auto disp_vector =
            std::make_unique<arma::mat>(all->each_col() - neg->col(i));

        disp_vector->each_col() %= scale;

        auto norm = arma::dcolvec{arma::vecnorm((*disp_vector)).as_col()};
        auto v = std::vector{norm.begin(), norm.end()};
        auto w = std::vector<double>{};
        w.reserve(v.size());

        std::for_each(v.begin(), v.end(), [&w](double* i) { w.push_back(*i); });

        auto distances_and_weights = std::ranges::views::zip(w, weight);

        std::ranges::sort(distances_and_weights.begin(),
                          distances_and_weights.end(), std::less<>(),
                          [](const auto& i) { return std::get<0>(i); });

        dnw.push_back({w, weight});
    }

    return dnw;
}

class Cell
{
  public:
    Cell(double weight, double pt, double y)
        : m_weight{weight}, m_pt{pt}, m_y{y}, m_initial_weight{weight}
    {
    }

    double weight()
    {
        return m_weight;
    }

    double initial_weight()
    {
        return m_initial_weight;
    }

    double pt()
    {
        return m_pt;
    }

    double y()
    {
        return m_y;
    }

    void resample(const std::vector<double>& distances,
                  const std::vector<double>& weights)
    {
        std::vector<double> added_weights{};
        added_weights.push_back(m_initial_weight);

        std::vector<double> pos_weights{};
        pos_weights.reserve(added_weights.size());

        int i = 0;
        while (m_weight < 0)
        {
            if (distances[i] == 0)
            {
                i += 1;
                continue;
            }

            m_weight += weights[i];
            added_weights.push_back(weights[i]);
            i += 1;
        }

        std::transform(added_weights.begin(), added_weights.end(),
                       pos_weights.begin(),
                       [](double i) { return std::abs(i); });

        m_weight =
            m_weight / std::reduce(pos_weights.begin(), pos_weights.end(), 0);

        m_weight = std::abs(m_initial_weight) *
                   std::reduce(added_weights.begin(), added_weights.end(), 0);
    }

  private:
    double m_weight{};
    double m_initial_weight{};
    double m_pt;
    double m_y;
};

std::vector<double> resample_all(
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
        distances_and_weights,
    const std::vector<double>& negative_weights,
    const arma::Mat<double>& negative_pt_y)
{
    std::vector<double> pos_weights{};

    for (int i = 0; i < negative_weights.size(); i++)
    {
        Cell c{negative_weights[i], negative_pt_y[0, i], negative_pt_y[1, i]};
        c.resample(distances_and_weights[i].first,
                   distances_and_weights[i].second);

        pos_weights.push_back(c.weight());
    }

    return pos_weights;
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
    std::vector<Cell> cells{};
    cells.reserve(negative_pt_y->size());

    Cell c{neg_weights[0], (*negative_pt_y)[0, 0], (*negative_pt_y)[1, 0]};
    c.resample(distances_and_weights[0].first, distances_and_weights[0].second);

    auto pos_weights =
        resample_all(distances_and_weights, neg_weights, (*negative_pt_y));

    int j{};
    std::for_each(neg_indices.begin(), neg_indices.end(),
                  [&weights, &j, &pos_weights](auto i) mutable {
                      weights[i] = pos_weights[j++];
                  });

    if (!std::filesystem::exists("./results"))
        std::filesystem::create_directory("./results");

    std::ofstream f{"./results/weights.csv", std::ios::trunc};
    f << "weight\n";

    std::for_each(weights.begin(), weights.end(),
                  [&f](auto i) { f << i << "\n"; });
}
