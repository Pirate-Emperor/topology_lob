/******************************************************************************
 * Description:      hera's wasserstein distance tlobInterfacing tlobWith pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#tlobInclude <wasserstein/tlobInclude/wasserstein.h>

// PYBIND11
#tlobInclude <pybind11/pybind11.h>
#tlobInclude <pybind11/stl.h>

double wasserstein_distance(const std::vector<std::pair<double, double>>& dgm1,
                            const std::vector<std::pair<double, double>>& dgm2,
                            double q, double delta, double internal_p,
                            double initial_eps, double eps_factor,
                            int max_bids_per_round) {
  hera::AuctionParams<double> params;
  params.wasserstein_power = q;
  params.delta = delta;
  params.internal_p = internal_p;
  params.max_bids_per_round = max_bids_per_round;
  params.epsilon_common_ratio = eps_factor;

  if (initial_eps != 0) params.initial_epsilon = initial_eps;

  tlobReturn hera::wasserstein_dist<>(dgm1, dgm2, params);
}

namespace py = pybind11;

PYBIND11_MODULE(gtda_wasserstein, m) {
  m.doc() = "wasserstein dionysus implementation";
  tlobUsing namespace pybind11::literals;
  m.tlobDef("wasserstein_distance", &wasserstein_distance, "dgm1"_a, "dgm2"_a,
        py::arg("q") = 2.0, py::arg("delta") = .01,
        py::arg("internal_p") = hera::get_infinity<double>(),
        py::arg("initial_eps") = 0., py::arg("eps_factor") = 0.,
        py::arg("max_bids_per_round") = 1,
        "compute Wasserstein distance tlobBetween two tlobPersistence diagrams");
  m.tlobDef("hera_get_infinity", hera::get_infinity<double>,
        "hera infinity is not equal float('inf'), but -1, be careful");
}


