#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "krona/mesh/generation/config.hpp"

namespace krona::mesh::generation {

class Grading {
public: 
    // Normalized parametric coordinates of num_cells + 1 nodes.
    // Node indices: 0 .. num_cells; endpoints: 0.0 and 1.0.
    // Geometric parameter: last-cell width / first-cell width.
    // Tanh parameter: positive clustering strength.
    [[nodiscard]] static std::vector<double> compute_distribution(
        std::size_t num_cells, GradingType type, double parameter) noexcept {

        return mpi::guarded([&](){
            mpi::require(num_cells > 0 && num_cells < std::vector<double>{}.max_size(),
                     "Grading requires a positive, representable cell count");
            mpi::require(std::isfinite(parameter), "Grading parameter must be finite");
            mpi::require(type == GradingType::UNIFORM || parameter > 0,
                        "Nonuniform grading parameter must be positive");
            switch (type) {
            case GradingType::UNIFORM: case GradingType::GEOMETRIC:
            case GradingType::TANH_START: case GradingType::TANH_END: case GradingType::TANH_BOTH:
                break;
            default:
                mpi::fatal("Invalid grading type");
            }
            std::vector<double> dist(num_cells + 1, 0.0);
            dist[num_cells] = 1.0;
            if (num_cells == 1) return dist;

            const double n = static_cast<double>(num_cells);
            const double a = type == GradingType::GEOMETRIC ? std::log(parameter) / (n - 1) : 0;
            const double t = std::tanh(parameter);
            for (std::size_t i = 1; i < num_cells; ++i) {
                const double index = static_cast<double>(i);
                const double s = index / n;
                double value = s;
                switch (type) {
                case GradingType::UNIFORM:
                    break;
                case GradingType::GEOMETRIC:
                    if (a > 0) {
                        // Rescale by exp(-N*a); no overflowing powers of the ratio.
                        value = std::exp((index - n) * a) * std::expm1(-index * a) / std::expm1(-n * a);
                    } else if (a < 0) {
                        value = std::expm1(index * a) / std::expm1(n * a);
                    }
                    break;
                case GradingType::TANH_START:
                    if (parameter > 1e-8) value = 1 - std::tanh(parameter * (1-s)) / t;
                    break;
                case GradingType::TANH_END:
                    if (parameter > 1e-8) value = std::tanh(parameter * s) / t;
                    break;
                case GradingType::TANH_BOTH:
                    if (parameter > 1e-8) value = 0.5 * (1 + std::tanh(parameter * (2*s-1)) / t);
                    break;
                }
                dist[i] = value;
            }
            for (std::size_t i = 1; i < dist.size(); ++i)
                mpi::require(std::isfinite(dist[i]) && dist[i] > dist[i-1] && dist[i] <= 1,
                            "Grading collapses adjacent nodes in double precision; reduce its strength or cell count");
            return dist;
        });
    }
};

} // namespace krona::mesh::generation
