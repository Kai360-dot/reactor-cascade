/* Solves cstr-2 subproblem by sampling the feed of cstr-2 that arrives
 * from upstream unit cstr-1. We thus sample over (T2, tau2, c1A, c1B, c1C)
 * subject to three constraints: g2, g3 (product purity and conversion floor
 * from  global problem) and rho: (reachability, assessed via cu ANN)
 * outputs: (all with [crit,feasprob,T2,tau2,cA,cB,cC] header line)
 * - data/unit2_live.csv (final design-space sample)
 * - data/unit2_dead.csv
 * - data/unit2_discard.csv
 */

#include <armadillo>
#include <atomic>
#include <cstddef>
#include <ffcustom.hpp>
#include <ffunc.hpp>
#include <fstream>
#include <interval.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "flowsheet.hpp"
#include "loaders.hpp"
#include "mlp_ffvar.hpp"
#include "nsfeas.hpp"

int main()
{
  arma::arma_rng::set_seed(42);
  // load inflated bounds on input from initial global sampling file
  std::vector<double> uLB, uUB;
  load_box(uLB, uUB);

  mc::FFGraph dag;
  auto D2 = dag.add_vars(2, "d");  // T2, tau2
  auto U2 = dag.add_vars(3, "u");  // cA, cB, cC (inlet)
  auto P2 = dag.add_vars(2, "p");  // theta1, theta2 (Arrhenius)
  // constraint values (cu constraint is implicit, purity and conversion floor)
  auto C2 = dag.add_vars(2, "c");

  // constraint from ANN
  MLP cu = MLP::load("data/mlp_cu.txt");

  using I = mc::Interval;      // satisfy template (non functional type)
  mc::FFCustom<I> LazySelect;  // conditional operation
  std::atomic<size_t> feasible{0}, infeasible{0};  // NS workers run parallel
  using dvec = std::vector<double>;

  // in-tray, fixed by the order of vIn below:
  //   x = { T2, tau2, c1A, c1B, c1C, th1, th2, purB, convA }
  //         [0]  [1]   [2]  [3]  [4]  [5]  [6]  [7]   [8]
  LazySelect.set_eval([&feasible, &infeasible, &cu](const dvec& x) -> dvec {
    const double rho = cu(dvec{x[2], x[3], x[4]})[0];
    if (rho > 0)  // inlet not reachable per ANN: skip the physics; all three
    {             // outputs carry the fence margin so crit ranks by it
      ++infeasible;
      return {rho, rho, rho};
    }
    ++feasible;
    const dvec out2 =
        Flowsheet::cstr(dvec{x[2], x[3], x[4]}, x[0], x[1], x[5], x[6]);
    const double tot2 = out2[0] + out2[1] + out2[2];
    return {rho,
            x[7] - out2[1] / tot2,                   // MCB purity floor cstr-2
            x[8] - (1 - out2[0] / Flowsheet::CA0)};  // conversion floor benzene
  });

  // wire op into dag: 9 inputs per order above, 3 outputs
  std::vector<mc::FFVar> vIn{D2[0], D2[1], U2[0], U2[1], U2[2],
                             P2[0], P2[1], C2[0], C2[1]};
  mc::FFVar** ppG = LazySelect(3, vIn, 1);
  std::vector<mc::FFVar> G2{*ppG[0], *ppG[1], *ppG[2]};

  mc::NSFEAS NS;
  NS.set_dag(dag);
  NS.options.FEASCRIT  = mc::NSFEAS::Options::VAR;
  NS.options.NUMLIVE   = 8192;
  NS.options.NUMPROP   = 256;
  NS.options.MAXITER   = 20'000;
  NS.options.MAXTHREAD = 0;
  NS.options.DISPLEVEL = 1;

  std::vector<mc::FFVar> ctrl_v{D2};
  ctrl_v.insert(ctrl_v.cend(), U2.cbegin(), U2.cend());

  std::vector<double> ctrl_v_lb{298.15, 2.0};
  ctrl_v_lb.insert(ctrl_v_lb.cend(), uLB.cbegin(), uLB.cend());

  std::vector<double> ctrl_v_ub{353.15, 30.0};
  ctrl_v_ub.insert(ctrl_v_ub.cend(), uUB.cbegin(), uUB.cend());

  NS.set_control(ctrl_v, ctrl_v_lb,  // lower bounds
                 ctrl_v_ub);         // upper bounds
  const auto th = Flowsheet::theta_nominal();
  NS.set_parameter(P2, {th[0], th[1]});
  NS.set_constant(C2);
  NS.set_constraint(G2);

  // run nested sampling
  if (!NS.setup()) return 1;

  // NOTE: only last two values are used as compared to src/global_sample.cpp
  int status = NS.sample({0.55, 0.70});
  NS.stats.display();
  std::cout << "cu gate: physics skipped on " << infeasible
            << " evals, executed on " << feasible << "\n";

  auto dump = [](const auto& pts, const std::string& name) {
    std::ofstream f(name);
    f << "crit,feasprob,T2,tau2,c1A,c1B,c1C\n";
    for (const auto& [crit, pt] : pts)
    {
      f << crit << "," << std::get<1>(pt);
      for (size_t i = 0; i < 5; ++i)
      {
        f << "," << std::get<0>(pt)[i];
      };
      f << "\n";
    }
  };

  dump(NS.live_points(), "data/unit2_live.csv");        // DS sample
  dump(NS.dead_points(), "data/unit2_dead.csv");        // killed during run
  dump(NS.discard_points(), "data/unit2_discard.csv");  // rejected
  // proposals

  return status;
}
