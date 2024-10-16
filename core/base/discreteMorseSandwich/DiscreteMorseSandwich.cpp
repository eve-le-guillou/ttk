#include <DiscreteMorseSandwich.h>
#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <unordered_map>

ttk::DiscreteMorseSandwich::DiscreteMorseSandwich() {
  this->setDebugMsgPrefix("DiscreteMorseSandwich");
}

void ttk::DiscreteMorseSandwich::tripletsToPersistencePairs(
  std::vector<PersistencePair> &pairs,
  std::vector<bool> &pairedExtrema,
  std::vector<bool> &pairedSaddles,
  std::vector<Rep> &reps,
  std::vector<tripletType> &triplets,
  const SimplexId *const saddlesOrder,
  const SimplexId *const extremaOrder,
  const SimplexId pairDim,
  const std::vector<std::array<ttk::SimplexId, 2>> &svToR,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema) const {
  // comparison functions
  const auto cmpSadMax
    = [=](const tripletType &t0, const tripletType &t1) -> bool {
    const auto s0 = t0[0];
    const auto s1 = t1[0];
    const auto m0 = t0[2];
    const auto m1 = t1[2];

#ifdef _LIBCPP_VERSION
    // libc++'s std::sort compares an entry to itself
    if(&t0 == &t1) {
      return true;
    }
#endif // _LIBCPP_VERSION

    if(s0 != s1)
      return saddlesOrder[s0] > saddlesOrder[s1];
    else
      return extremaOrder[m0] < extremaOrder[m1];
  };

  // ttk::Timer getRepTimer{};

  const auto cmpSadMin
    = [=](const tripletType &t0, const tripletType &t1) -> bool {
    const auto s0 = t0[0];
    const auto s1 = t1[0];
    const auto m0 = t0[2];
    const auto m1 = t1[2];
    if(s0 != s1)
      return saddlesOrder[s0] < saddlesOrder[s1];
    else
      return extremaOrder[m0] > extremaOrder[m1];
  };

  // sort triplets
  if(pairDim == 0) {
    TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(), cmpSadMin);
  } else {
    // saddle-saddle pairs from 1-saddles to 2-saddles
    TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(), cmpSadMax);
    // std::reverse(triplets.begin(), triplets.end());
  }

  // auto rng = std::default_random_engine{0};
  // std::shuffle(std::begin(triplets), std::end(triplets), rng);
  const bool increasing = (pairDim > 0);
  // Timer tm{};
  // std::vector<ttk::SimplexId> saddleToPairedExtrema(saddleNumber, -1);
  // saddleToPairedExtremaTime = tm.getElapsedTime();
  // get representative of current extremum
  const auto getRep
    = [this, &reps, &saddlesOrder, increasing /*, &getRepTimer, &getRepTime*/](
        SimplexId v, SimplexId sv) -> Rep {
    //    getRepTimer.reStart();
    auto rep = reps[v];
    ttk::SimplexId s = rep.saddleId_;
    while(rep.extremaId_ != v) {
      s = rep.saddleId_;
      if((s != -1) && (sv != s)
         && ((saddlesOrder[s] < saddlesOrder[sv]) == increasing)) {
        break;
      }
      v = rep.extremaId_;
      rep = reps[v];
    }
    // In case of the shadow triplet
    if(increasing && rep.extremaId_ == v && s != -1) {
      s = rep.saddleId_;
    }
    //    getRepTime += getRepTimer.getElapsedTime();
    return Rep{v, s};
  };

  const auto addPair
    = [this, &saddleToPairedExtrema, &pairedExtrema, &pairedSaddles](
        const SimplexId sad, const SimplexId extr) {
        saddleToPairedExtrema[sad] = extr;
        pairedSaddles[sad] = true;
        pairedExtrema[extr] = true;
      };

  const auto removePair = [this, &saddleToPairedExtrema, &pairedExtrema,
                           &pairedSaddles](const SimplexId sad) {
    pairedExtrema[saddleToPairedExtrema[sad]] = false;
    saddleToPairedExtrema[sad] = -1;
    pairedSaddles[sad] = false;
  };

  const std::function<int(tripletType)> processTriplet =
    [this, &increasing, &pairedExtrema, &pairedSaddles, &saddleToPairedExtrema,
     &extremaOrder, &reps, &getRep, &addPair, &removePair, &saddlesOrder,
     &processTriplet, &svToR](tripletType t) -> int {
    const auto sv = t[0];
    auto rep1 = getRep(t[1], sv);
    bool pairedR1 = pairedExtrema[rep1.extremaId_];
    bool isR1Invalid
      = ((rep1.saddleId_ != -1) && (rep1.saddleId_ != sv)
         && ((saddlesOrder[rep1.saddleId_] < saddlesOrder[sv]) == increasing));
    if(isR1Invalid)
      pairedR1 = false;
    isR1Invalid = isR1Invalid
                  && (saddleToPairedExtrema[rep1.saddleId_] == rep1.extremaId_);
    if(t[2] < 0) {
      // deal with "shadow" triplets (a 2-saddle with only one
      // ascending 1-separatrix leading to an unique maximum)
      if(!pairedR1 && !pairedSaddles[sv]) {
        // when considering the boundary, the "-1" of the triplets
        // indicate a virtual maximum of infinite persistence on the
        // boundary component. a pair is created with the other
        // maximum
        if(isR1Invalid) {
          removePair(rep1.saddleId_);
        }
        addPair(sv, rep1.extremaId_);
        reps[rep1.extremaId_] = Rep{rep1.extremaId_, sv};
        if(isR1Invalid) {
          return processTriplet(tripletType{rep1.saddleId_,
                                            svToR[rep1.saddleId_][0],
                                            svToR[rep1.saddleId_][1]});
        }
      }
      return 0;
    }
    auto rep2 = getRep(t[2], sv);
    bool pairedR2 = pairedExtrema[rep2.extremaId_];
    bool isR2Invalid
      = ((rep2.saddleId_ != -1) && (rep2.saddleId_ != sv)
         && ((saddlesOrder[rep2.saddleId_] < saddlesOrder[sv]) == increasing));
    if(isR2Invalid)
      pairedR2 = false;
    isR2Invalid = isR2Invalid
                  && (saddleToPairedExtrema[rep2.saddleId_] == rep2.extremaId_);
    if(rep1.extremaId_ != rep2.extremaId_) {
      if((((extremaOrder[rep1.extremaId_] > extremaOrder[rep2.extremaId_])
           == increasing)
          || pairedR1)
         && !pairedR2) {
        std::swap(rep1, rep2);
        std::swap(pairedR1, pairedR2);
        std::swap(isR1Invalid, isR2Invalid);
      }
      if(!pairedR1) {
        if(isR1Invalid) {
          removePair(rep1.saddleId_);
        }
        addPair(sv, rep1.extremaId_);
        reps[rep1.extremaId_] = Rep{rep2.extremaId_, sv};
        if(isR1Invalid) {
          return processTriplet(tripletType{rep1.saddleId_,
                                            svToR[rep1.saddleId_][0],
                                            svToR[rep1.saddleId_][1]});
        }
      }
    }

    return 0;
  };

  for(const auto &t : triplets) {
    processTriplet(t);
  }

  ttk::SimplexId saddleNumber = pairedSaddles.size();

//  ttk::Timer postTimer{};
#ifdef TTK_ENABLE_OPENMP
#pragma omp declare reduction (merge : std::vector<PersistencePair> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : pairs) schedule(static)
#endif
  for(int i = 0; i < saddleNumber; i++) {
    if(saddleToPairedExtrema[i] != -1) {
      if(increasing) {
        pairs.emplace_back(i, saddleToPairedExtrema[i], pairDim);
      } else {
        pairs.emplace_back(saddleToPairedExtrema[i], i, pairDim);
      }
    }
  }

  // postTreatmentTime = postTimer.getElapsedTime();
}

void ttk::DiscreteMorseSandwich::displayStats(
  const std::vector<PersistencePair> &pairs,
  const std::array<std::vector<SimplexId>, 4> &criticalCellsByDim,
  const std::vector<bool> &pairedMinima,
  const std::vector<bool> &paired1Saddles,
  const std::vector<bool> &paired2Saddles,
  const std::vector<bool> &pairedMaxima) const {

  const auto dim = this->dg_.getDimensionality();

  // display number of pairs per pair type
  std::vector<std::vector<std::string>> rows{
    {" #Min-saddle pairs",
     std::to_string(
       std::count_if(pairs.begin(), pairs.end(),
                     [](const PersistencePair &a) { return a.type == 0; }))},
    {" #Saddle-saddle pairs",
     std::to_string(dim == 3 ? std::count_if(
                      pairs.begin(), pairs.end(),
                      [](const PersistencePair &a) { return a.type == 1; })
                             : 0)},
    {" #Saddle-max pairs",
     std::to_string(std::count_if(
       pairs.begin(), pairs.end(),
       [dim](const PersistencePair &a) { return a.type == dim - 1; }))},
  };

  // display number of critical cells (paired and unpaired)
  std::vector<size_t> nCritCells(dim + 1);
  std::vector<size_t> nNonPairedCritCells(dim + 1);

  for(int i = 0; i < dim + 1; ++i) {
    nCritCells[i] = criticalCellsByDim[i].size();
    size_t nNonPaired{};
    for(size_t j = 0; j < criticalCellsByDim[i].size(); ++j) {
      const auto cell = criticalCellsByDim[i][j];
      if((i == 0 && !pairedMinima[cell]) || (i == 1 && !paired1Saddles[cell])
         || (i == 2 && dim == 3 && !paired2Saddles[cell])
         || (i == dim && !pairedMaxima[cell])) {
        nNonPaired++;
      }
    }
    nNonPairedCritCells[i] = nNonPaired;
  }

  std::vector<std::string> critCellsLabels{"Minima"};
  if(dim >= 2) {
    critCellsLabels.emplace_back("1-saddles");
  }
  if(dim >= 3) {
    critCellsLabels.emplace_back("2-saddles");
  }
  critCellsLabels.emplace_back("Maxima");

  for(int i = 0; i < dim + 1; ++i) {
    const std::string unpaired{nNonPairedCritCells[i] == 0
                                 ? " (all paired)"
                                 : " (" + std::to_string(nNonPairedCritCells[i])
                                     + " unpaired)"};

    rows.emplace_back(std::vector<std::string>{
      " #" + critCellsLabels[i], std::to_string(nCritCells[i]) + unpaired});
  }
  this->printMsg(rows, debug::Priority::DETAIL);
}