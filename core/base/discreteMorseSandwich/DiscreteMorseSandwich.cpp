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
  std::vector<std::vector<std::array<SimplexId, 2>>> &reps,
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

  /*const auto cmpSadMin
    = [=](const tripletType &t0, const tripletType &t1) -> bool {
    const auto s0 = t0[0];
    const auto s1 = t1[0];
    const auto m0 = t0[2];
    const auto m1 = t1[2];
    if(s0 != s1)
      return saddlesOrder[s0] < saddlesOrder[s1];
    else
      return extremaOrder[m0] > extremaOrder[m1];
  };*/

  // sort triplets
  if(pairDim == 0) {
    /*TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(),
    cmpSadMin);*/
  } else {
    // saddle-saddle pairs from 1-saddles to 2-saddles
    // TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(),
    // cmpSadMax);
    // std::reverse(triplets.begin(), triplets.end());
  }

  auto rng = std::default_random_engine{0};
  std::shuffle(std::begin(triplets), std::end(triplets), rng);

  const bool increasing = (pairDim > 0);

  std::vector<ttk::SimplexId> saddleToPairedExtrema(pairedSaddles.size(), -1);

  const auto getClosestRep
    = [this, increasing, &saddlesOrder, &reps](
        SimplexId v, SimplexId sv,
        std::vector<std::array<ttk::SimplexId, 2>> &repVect)
    -> std::array<ttk::SimplexId, 3> {
    /*if (sv == 412){
      std::string s = "";
      for (int i = 0; i < repVect.size(); i++){
        s += std::to_string(repVect[i][0])+", "+std::to_string(repVect[i][1])+"
    order("+std::to_string(saddlesOrder[repVect[i][1]])+"); ";
      }
      printMsg("reps for v: "+std::to_string(v)+": "+s);
    }  */

    if(repVect.size() == 1) {
      return std::array<ttk::SimplexId, 3>{repVect[0][0], repVect[0][1], 1};
    }
    ttk::SimplexId i = repVect.size() - 1;
    auto rep = repVect[i];
    auto precRep = repVect[i];
    if(!((saddlesOrder[rep[1]] < saddlesOrder[sv]) == increasing)) {
      return std::array<ttk::SimplexId, 3>{repVect[i][0], repVect[i][1], i};
    }
    i--;
    rep = repVect[i];
    while(
      rep[1] != sv && rep[1] != -1
      && ((saddlesOrder[rep[1]] < saddlesOrder[sv]) == increasing)
      && ((saddlesOrder[rep[1]] > saddlesOrder[precRep[1]]) == increasing)) {
      i--;
      precRep = rep;
      rep = repVect[i];
      // if (sv == 374)
      //  printMsg("in while: "+std::to_string(rep[0])+",
      //  "+std::to_string(rep[1]));
    }
    if(i == repVect.size() - 1) {
      printErr("NOT SUPPOSED TO HAPPEN");
      return std::array<ttk::SimplexId, 3>{repVect[i][0], repVect[i][1], i};
    }
    /*if (sv == 374)
      printMsg("leaving getClosestRep: "+std::to_string(rep[0])+",
      "+std::to_string(rep[1])
      +", " +std::to_string(repVect[i+1][0])+",
      "+std::to_string(repVect[i+1][1]));*/
    return std::array<ttk::SimplexId, 3>{
      repVect[i + 1][0], repVect[i + 1][1], i};
  };

  // get representative of current extremum
  const auto getRep
    = [this, &reps, &saddlesOrder, increasing, &getClosestRep](
        SimplexId v, SimplexId sv) -> std::array<ttk::SimplexId, 4> {
    auto rep = getClosestRep(v, sv, reps[v]);
    ttk::SimplexId s = rep[1];
    ttk::SimplexId index = rep[2];
    ttk::SimplexId vnext = rep[0];
    while(rep[0] != v) {
      if(sv == 374) {
        printMsg("In getRep: " + std::to_string(v)
                 + ", s: " + std::to_string(s));
      }
      s = rep[1];
      index = rep[2];
      vnext = rep[0];
      if(s != -1 && (sv != s)
         && ((saddlesOrder[s] < saddlesOrder[sv]) == increasing)) {
        break;
      }
      v = rep[0];
      rep = getClosestRep(v, sv, reps[v]);
    }
    // In case of the shadow triplet
    if(increasing && rep[0] == v && s != -1) {
      s = rep[1];
    }
    if(sv == 374) {
      printMsg("In getRep: " + std::to_string(v) + ", " + std::to_string(s)
               + ", index: " + std::to_string(index));
    }
    return std::array<ttk::SimplexId, 4>{v, s};
  };

  const auto addPair
    = [this, &saddleToPairedExtrema, &pairedExtrema, &pairedSaddles,
       &extremaOrder, increasing, &saddlesOrder,
       pairDim](const SimplexId sad, const SimplexId extr) {
        if(sad == 374 || sad == 412)
          printMsg("AddPair between " + std::to_string(sad) + ", and "
                   + std::to_string(extr));
        saddleToPairedExtrema[sad] = extr;
        pairedSaddles[sad] = true;
        pairedExtrema[extr] = true;
      };

  const auto removePair
    = [this, &saddleToPairedExtrema, &pairedExtrema, &pairedSaddles, increasing,
       &saddlesOrder](const SimplexId sad) {
        if(saddleToPairedExtrema[sad] != -1) {
          if(sad == 374 || sad == 412)
            printMsg("Remove pair of " + std::to_string(sad));
          pairedExtrema[saddleToPairedExtrema[sad]] = false;
          saddleToPairedExtrema[sad] = -1;
        }
        pairedSaddles[sad] = false;
      };

  std::unordered_map<ttk::SimplexId, std::array<ttk::SimplexId, 2>> svToR;
  const std::function<int(tripletType)> processTriplet =
    [this, &increasing, &pairedExtrema, &pairedSaddles, &saddleToPairedExtrema,
     &extremaOrder, &reps, &getRep, &addPair, &removePair, &saddlesOrder,
     &processTriplet, &svToR, &rerunCounter, &PCCounter](tripletType t) -> int {
    const auto sv = t[0];
    if(sv == 374) {
      printMsg("sv: " + std::to_string(sv)
               + ", order: " + std::to_string(saddlesOrder[sv]));
    }
    auto rep1 = getRep(t[1], sv);
    auto r1 = rep1[0];
    auto s1 = rep1[1];
    bool pairedR1 = pairedExtrema[r1];
    bool isR1Invalid
      = (s1 != -1 && (sv != s1)
         && ((saddlesOrder[s1] < saddlesOrder[sv]) == increasing));
    if(isR1Invalid)
      pairedR1 = false;
    isR1Invalid = isR1Invalid && (saddleToPairedExtrema[s1] == r1);
    /*if (isR1Invalid){
      printMsg("isR1Invalid for sv: "+std::to_string(sv)+",
    "+std::to_string(s1)+", "+std::to_string(sv)+", test:
    "+std::to_string(((saddlesOrder[s1] < saddlesOrder[sv]) == increasing))
      +", "+std::to_string((saddleToPairedExtrema[s1] == r1)));
    }*/
    if(t[2] < 0) {
      if(sv == 374 || sv == 412) {
        printMsg("sv: " + std::to_string(sv) + ", order("
                 + std::to_string(saddlesOrder[sv])
                 + "), t[1]: " + std::to_string(t[1])
                 + ", r1: " + std::to_string(r1) + ", s1: " + std::to_string(s1)
                 + ", order(" + std::to_string(saddlesOrder[s1]) + ")");
      }
      // deal with "shadow" triplets (a 2-saddle with only one
      // ascending 1-separatrix leading to an unique maximum)
      if(!pairedR1 && !pairedSaddles[sv]) {
        // when considering the boundary, the "-1" of the triplets
        // indicate a virtual maximum of infinite persistence on the
        // boundary component. a pair is created with the other
        // maximum
        if(isR1Invalid) {
          removePair(s1);
        }
        addPair(sv, r1);
        svToR[sv] = std::array<ttk::SimplexId, 2>{t[1], t[2]};
        reps[r1].push_back(std::array<ttk::SimplexId, 2>{r1, sv});
        if(isR1Invalid) {
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
    if(sv == 374 || sv == 412) {
      printMsg(
        "sv: " + std::to_string(sv) + ", order("
        + std::to_string(saddlesOrder[sv]) + "), t[1]: " + std::to_string(t[1])
        + ", r1: " + std::to_string(r1) + ", s1: " + std::to_string(s1)
        + ", t[2]: " + std::to_string(t[2]) + ", r2: " + std::to_string(r2))
        + ", s2: " + std::to_string(s2);
    }
    bool isR2Invalid
      = (s2 != -1 && (sv != s2)
         && ((saddlesOrder[s2] < saddlesOrder[sv]) == increasing));
    if(isR2Invalid)
      pairedR2 = false;
    isR2Invalid = isR2Invalid && (saddleToPairedExtrema[s2] == r2);
    if(r1 != r2) {
      if((((extremaOrder[r1] > extremaOrder[r2]) == increasing) || pairedR1)
         && !pairedR2) {
        if(sv == 374) {
          printMsg("Swaping for " + std::to_string(sv)
                   + ", because: " + std::to_string(pairedR1)
                   + ", or : " + std::to_string(!pairedR2) + ", and: "
                   + std::to_string((extremaOrder[r1] > extremaOrder[r2])
                                    == increasing));
          auto it = std::find(
            saddleToPairedExtrema.begin(), saddleToPairedExtrema.end(), r2);
          if(it != saddleToPairedExtrema.end()) {
            auto i = std::distance(saddleToPairedExtrema.begin(), it);
            printMsg("Swaping for " + std::to_string(sv) + ", because: r2 "
                     + std::to_string((saddleToPairedExtrema[i]))
                     + " paired to: " + std::to_string(i)
                     + ", of order: " + std::to_string(saddlesOrder[i]));
          }
        }
        std::swap(r1, r2);
        std::swap(pairedR1, pairedR2);
        std::swap(s1, s2);
        std::swap(isR1Invalid, isR2Invalid);
      } else {
        if(sv == 374) {
          printMsg("No swaping for " + std::to_string(sv)
                   + ", because: " + std::to_string(pairedR1)
                   + ", or : " + std::to_string(!pairedR2) + ", and: "
                   + std::to_string((extremaOrder[r1] > extremaOrder[r2])
                                    == increasing));
          auto it = std::find(
            saddleToPairedExtrema.begin(), saddleToPairedExtrema.end(), r2);
          if(it != saddleToPairedExtrema.end()) {
            auto i = std::distance(saddleToPairedExtrema.begin(), it);
            printMsg("No swaping for " + std::to_string(sv) + ", because: r2 "
                     + std::to_string((saddleToPairedExtrema[i]))
                     + " paired to: " + std::to_string(i)
                     + ", of order: " + std::to_string(saddlesOrder[i]));
          }
        }
      }

      if(!pairedR1) {
        if(isR1Invalid) {
          removePair(s1);
        }
        addPair(sv, r1);
        svToR[sv] = std::array<ttk::SimplexId, 2>{t[1], t[2]};
        reps[r1].push_back(std::array<ttk::SimplexId, 2>{r2, sv});
        if(r1 != t[1]) {
          /*if (t[1] == 637){
            std::string s = "";
            for (int i = 0; i < reps[t[1]].size(); i++){
              s += std::to_string(reps[t[1]][i][0])+",
          "+std::to_string(reps[t[1]][i][1])+"; ";
            }
            printMsg("reps for t[1]: "+std::to_string(t[1])+": "+s);
          }  */
          // reps[t[1]].push_back(std::array<ttk::SimplexId, 2>{r2, sv});
        }
        if(isR1Invalid) {
          /*std::string s = "";
          for (int i = 0; i < reps[r1].size(); i++){
            s += std::to_string(reps[r1][i][0])+",
          "+std::to_string(reps[r1][i][1])+"; ";
          }
          printMsg("R1 is invalid for sv: "+std::to_string(sv)+",
          "+std::to_string(index1)+", size: "+std::to_string(reps[r1].size() -
          1)+", "+s);*/
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
  //#ifdef TTK_ENABLE_OPENMP
  //#pragma omp declare reduction (merge : std::vector<PersistencePair> :
  //omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) #pragma omp
  //parallel for reduction(merge : pairs) schedule(static) #endif
  for(int i = 0; i < saddleToPairedExtrema.size(); i++) {
    if(saddleToPairedExtrema[i] != -1) {
      if(increasing) {
        pairs.emplace_back(i, saddleToPairedExtrema[i], pairDim);
      } else {
        pairs.emplace_back(saddleToPairedExtrema[i], i, pairDim);
      }
      if(pairs.size() - 1 == 80) {
        printErr("i: " + std::to_string(i) + ", "
                 + std::to_string(saddleToPairedExtrema[i]));
      }
    }
  }
}

void ttk::DiscreteMorseSandwich::tripletsToPersistencePairs_original(
  std::vector<PersistencePair> &pairs,
  std::vector<bool> &pairedExtrema,
  std::vector<bool> &pairedSaddles,
  std::vector<SimplexId> &reps,
  std::vector<tripletType> &triplets,
  const SimplexId *const saddlesOrder,
  const SimplexId *const extremaOrder,
  const SimplexId pairDim) const {

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
    TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(), cmpSadMin);
  } else {
    // saddle-saddle pairs from 1-saddles to 2-saddles
    TTK_PSORT(this->threadNumber_, triplets.begin(), triplets.end(), cmpSadMax);
  }

  // get representative of current extremum
  const auto getRep = [&reps](SimplexId v) -> SimplexId {
    auto r = reps[v];
    while(r != v) {
      v = r;
      r = reps[v];
    }
    return r;
  };

  const bool increasing = (pairDim > 0);

  const auto addPair = [&pairs, &pairedExtrema, &pairedSaddles, increasing,
                        pairDim](const SimplexId sad, const SimplexId extr) {
    if(increasing) {
      pairs.emplace_back(sad, extr, pairDim);
    } else {
      pairs.emplace_back(extr, sad, pairDim);
    }
    pairedSaddles[sad] = true;
    pairedExtrema[extr] = true;
  };

  for(const auto &t : triplets) {
    const auto sv = t[0];

    auto r1 = getRep(t[1]);

    if(t[2] < 0) {
      // deal with "shadow" triplets (a 2-saddle with only one
      // ascending 1-separatrix leading to an unique maximum)
      if(!pairedExtrema[r1] && !pairedSaddles[sv]) {
        // when considering the boundary, the "-1" of the triplets
        // indicate a virtual maximum of infinite persistence on the
        // boundary component. a pair is created with the other
        // maximum
        addPair(sv, r1);
        if(sv == 374)
          printMsg("AddPair between " + std::to_string(sv) + ", and "
                   + std::to_string(r1));
      }

      continue;
    }

    auto r2 = getRep(t[2]);
    if(sv == 374) {
      printMsg("sv: " + std::to_string(sv) + ", t[1]: " + std::to_string(t[1])
               + ", r1: " + std::to_string(r1) + ", t[2]: "
               + std::to_string(t[2]) + ", r2: " + std::to_string(r2));
    }
    if(r1 != r2) {
      if(((extremaOrder[r1] > extremaOrder[r2]) == increasing
          || pairedExtrema[r1])
         && !pairedExtrema[r2]) {
        std::swap(r1, r2);
        if(sv == 374) {
          printMsg("Swaping for " + std::to_string(sv)
                   + ", because: " + std::to_string(pairedExtrema[r1])
                   + ", or : " + std::to_string(!pairedExtrema[r1]) + ", and: "
                   + std::to_string((extremaOrder[r1] > extremaOrder[r2])
                                    == increasing));
        }
      }
      if(sv == 374) {
        printMsg("No swaping for " + std::to_string(sv)
                 + ", because: " + std::to_string(pairedExtrema[r1])
                 + ", or : " + std::to_string(!pairedExtrema[r2]) + ", and: "
                 + std::to_string(
                   ((extremaOrder[r1] > extremaOrder[r2]) == increasing)));
      }
      if(!pairedExtrema[r1]) {
        addPair(sv, r1);
        if(sv == 374)
          printMsg("AddPair between " + std::to_string(sv) + ", and "
                   + std::to_string(r1));
        reps[t[1]] = r2;
        reps[r1] = r2;
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
