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
  std::vector<std::array<SimplexId, 2>> &reps,
  std::vector<tripletType> &triplets,
  const SimplexId *const saddlesOrder,
  const SimplexId *const extremaOrder,
  const SimplexId pairDim,
  int &rerunCounter,
  int &PCCounter) const {
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
    /*TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(),
    cmpSadMin);*/
  } else {
    // saddle-saddle pairs from 1-saddles to 2-saddles
    // TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(),
    // cmpSadMax);
    std::reverse(triplets.begin(), triplets.end());
  }

  // auto rng = std::default_random_engine{0};
  // std::shuffle(std::begin(triplets), std::end(triplets), rng);
  const bool increasing = (pairDim > 0);

  std::vector<bool> certain(reps.size(), true);
  std::vector<std::vector<ttk::SimplexId>> extremaToSaddle(
    reps.size(), std::vector<ttk::SimplexId>());
  for(int i = 0; i < triplets.size(); i++) {
    auto t = triplets[i];
    extremaToSaddle[t[1]].push_back(saddlesOrder[t[0]]);
    if(t[2] != -1) {
      extremaToSaddle[t[2]].push_back(saddlesOrder[t[0]]);
    } else {
      certain[t[1]] = false;
    }
  }
  /*int startingSizeMoy = 0;
  int totalNumber = 0;
  size_t max = 0;
  for(int i = 0; i < reps.size(); i++) {
    if (!extremaToSaddle[i].empty()){
      startingSizeMoy += extremaToSaddle[i].size();
      max = std::max(max, extremaToSaddle[i].size());
      totalNumber++;
    }
  }
  printErr("Mean number of extremaToSaddle:
  "+std::to_string(startingSizeMoy/totalNumber)); printErr("Max number of
  extremaToSaddle: "+std::to_string(max));
*/
  std::vector<ttk::SimplexId> saddleToPairedExtrema(pairedSaddles.size(), -1);
  // get representative of current extremum
  const auto getRep
    = [this, &reps, &saddlesOrder, increasing](
        SimplexId v, SimplexId sv) -> std::array<ttk::SimplexId, 2> {
    auto rep = reps[v];
    ttk::SimplexId s = rep[1];
    while(rep[0] != v) {
      s = rep[1];
      if(s != -1 && sv != s
         && ((saddlesOrder[s] < saddlesOrder[sv]) == increasing)) {
        break;
      }
      v = rep[0];
      rep = reps[v];
    }
    // In case of the shadow triplet
    if(increasing && rep[0] == v && s != -1) {
      s = rep[1];
    }
    return std::array<ttk::SimplexId, 2>{v, s};
  };

  const auto addPair = [this, &saddleToPairedExtrema, &pairedExtrema,
                        &pairedSaddles, &extremaOrder, increasing,
                        &saddlesOrder,
                        pairDim](const SimplexId sad, const SimplexId extr) {
    saddleToPairedExtrema[sad] = extr;
    pairedSaddles[sad] = true;
    pairedExtrema[extr] = true;
  };

  const auto removePair
    = [this, &saddleToPairedExtrema, &pairedExtrema, &pairedSaddles, increasing,
       &saddlesOrder](const SimplexId sad) {
        if(saddleToPairedExtrema[sad] != -1) {
          pairedExtrema[saddleToPairedExtrema[sad]] = false;
          saddleToPairedExtrema[sad] = -1;
        }
        pairedSaddles[sad] = false;
      };

  std::unordered_map<ttk::SimplexId, std::array<ttk::SimplexId, 2>> svToR;
  const std::function<int(tripletType)> processTriplet
    = [this, &increasing, &pairedExtrema, &pairedSaddles,
       &saddleToPairedExtrema, &extremaOrder, &reps, &getRep, &addPair,
       &removePair, &saddlesOrder, &processTriplet, &svToR, &rerunCounter,
       &certain, &extremaToSaddle, &PCCounter](tripletType t) -> int {
    const auto sv = t[0];
    auto rep1 = getRep(t[1], sv);
    auto r1 = rep1[0];
    auto s1 = rep1[1];
    bool pairedR1 = pairedExtrema[r1];
    if(s1 != -1 && sv != s1
       && ((saddlesOrder[s1] < saddlesOrder[sv]) == increasing))
      pairedR1 = false;
    bool pairedToR1 = false;
    if(s1 != -1)
      pairedToR1 = saddleToPairedExtrema[s1] == r1;
    if(t[2] < 0) {
      // deal with "shadow" triplets (a 2-saddle with only one
      // ascending 1-separatrix leading to an unique maximum)
      if(!pairedR1 && !pairedSaddles[sv]) {
        // when considering the boundary, the "-1" of the triplets
        // indicate a virtual maximum of infinite persistence on the
        // boundary component. a pair is created with the other
        // maximum
        if(s1 != -1 && pairedToR1
           && ((saddlesOrder[s1] < saddlesOrder[sv]) == increasing)) {
          removePair(s1);
        }
        addPair(sv, r1);
        svToR[sv] = std::array<ttk::SimplexId, 2>{t[1], t[2]};
        reps[r1][1] = sv;
        reps[r1][0] = r1;
        // certain[r1] = false;
        if(s1 != -1 && pairedToR1
           && ((saddlesOrder[s1] < saddlesOrder[sv]) == increasing)) {
          rerunCounter++;
          return processTriplet(tripletType{s1, svToR[s1][0], svToR[s1][1]});
        }
      }
      return 0;
    }
    auto rep2 = getRep(t[2], sv);
    auto r2 = rep2[0];
    auto s2 = rep2[1];
    bool pairedR2 = pairedExtrema[r2];
    if(s2 != -1 && sv != s2
       && ((saddlesOrder[s2] < saddlesOrder[sv]) == increasing))
      pairedR2 = false;
    bool pairedToR2 = false;
    if(s2 != -1)
      pairedToR2 = saddleToPairedExtrema[s2] == r2;
    if(r1 != r2) {
      if((((extremaOrder[r1] > extremaOrder[r2]) == increasing) || pairedR1)
         && !pairedR2) {
        std::swap(r1, r2);
        std::swap(pairedR1, pairedR2);
        std::swap(pairedToR1, pairedToR2);
        std::swap(s1, s2);
      }
      if(!pairedR1) {
        if(s1 != -1 && pairedToR1
           && ((saddlesOrder[s1] < saddlesOrder[sv]) == increasing)) {
          removePair(s1);
        }
        addPair(sv, r1);
        svToR[sv] = std::array<ttk::SimplexId, 2>{t[1], t[2]};
        reps[r1][0] = r2;
        reps[r1][1] = sv;
        if(certain[r1] && certain[r2] && certain[t[1]] && certain[t[2]]) {
          if(!extremaToSaddle[r1].empty()) {
            std::vector<ttk::SimplexId>::iterator it;
            if(increasing) {
              it = std::min_element(
                extremaToSaddle[r1].begin(), extremaToSaddle[r1].end());
            } else {
              it = std::max_element(
                extremaToSaddle[r1].begin(), extremaToSaddle[r1].end());
            }
            ttk::SimplexId extrSaddleOrder = (*it);
            /*std::string s ="";
            auto it = extremaToSaddle[r1].begin();
            while (it != extremaToSaddle[r1].end()){
              s += std::to_string((*it))+", ";
              it++;
            }
            //printMsg("extremaToSaddle: "+s+" with extrSaddleOrder:
            "+std::to_string(extrSaddleOrder));          */
            if(extrSaddleOrder == saddlesOrder[sv]) {
              PCCounter++;
              reps[t[1]][0] = r2;
              extremaToSaddle[r1].erase(std::remove(extremaToSaddle[r1].begin(),
                                                    extremaToSaddle[r1].end(),
                                                    extrSaddleOrder));
              extremaToSaddle[r2].erase(std::remove(extremaToSaddle[r2].begin(),
                                                    extremaToSaddle[r2].end(),
                                                    extrSaddleOrder));
              // extremaToSaddle[r1].erase(saddlesOrder[sv]);
              // extremaToSaddle[r2].erase(saddlesOrder[sv]);
              // std::vector<ttk::SimplexId> union_vec;
              size_t size = extremaToSaddle[r1].size();
              extremaToSaddle[r1].insert(extremaToSaddle[r1].end(),
                                         extremaToSaddle[r2].begin(),
                                         extremaToSaddle[r2].end());
              extremaToSaddle[r2].insert(extremaToSaddle[r2].end(),
                                         extremaToSaddle[r1].begin(),
                                         extremaToSaddle[r1].begin() + size);
              /*extremaToSaddle[r1].insert(
                extremaToSaddle[r2].begin(), extremaToSaddle[r2].end());
              extremaToSaddle[r2].insert(
                extremaToSaddle[r1].begin(), extremaToSaddle[r1].end());*/
            } else {
              certain[r1] = false;
              certain[r2] = false;
              certain[t[1]] = false;
              certain[t[2]] = false;
            }
          }
        } else {
          certain[r1] = false;
          certain[r2] = false;
          certain[t[1]] = false;
          certain[t[2]] = false;
        }
        if(s1 != -1 && pairedToR1
           && ((saddlesOrder[s1] < saddlesOrder[sv]) == increasing)) {
          rerunCounter++;
          return processTriplet(tripletType{s1, svToR[s1][0], svToR[s1][1]});
        }
      }
    }

    return 0;
  };

  for(const auto &t : triplets) {
    processTriplet(t);
  }
#ifdef TTK_ENABLE_OPENMP
#pragma omp declare reduction (merge : std::vector<PersistencePair> :  omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : pairs) schedule(static)
#endif
  for(int i = 0; i < saddleToPairedExtrema.size(); i++) {
    if(saddleToPairedExtrema[i] != -1) {
      if(increasing) {
        pairs.emplace_back(i, saddleToPairedExtrema[i], pairDim);
      } else {
        pairs.emplace_back(saddleToPairedExtrema[i], i, pairDim);
      }
    }
  }
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
