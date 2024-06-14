#include <DiscreteMorseSandwichMPI.h>
#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <unordered_map>

ttk::DiscreteMorseSandwichMPI::DiscreteMorseSandwichMPI() {
  this->setDebugMsgPrefix("DiscreteMorseSandwichMPI");
}

/*template<typename datatype>
void ttk::DiscreteMorseSandwichMPI::storeMessageToSend(std::vector<char> &ghost,
char rank, std::vector<std::vector<messageType>> &sendBuffer,
GlobalLocalSimplexId sv, Rep rep1, Rep rep2){ messageType m {}; ttk::SimplexId
gid{}; if (rep1.extremaId_.isGlobal){ gid = rep1.extremaId_.id; } else { gid =
triangulation.getExtremaGlobalId(rep1.extremaId_.id) // TODO: add as argument of
function
  }
  m.m1 = gid;
  m.scalarM1 = ? //TODO:
  m.procM1 = rank;
  if (rep2.isGlobal){
    gid = rep2.extremaId_.id;
    m.procM2 = ghostMap.find(rep2.extremaId_.id)->second;
  } else {
    gid = triangulation.getExtremaGlobalId(rep2.extremaId_.id) // TODO: add as
argument of function m.procM2 = ttk::MPIrank_;
  }
  m.m2 = gid;
  m.scalarM2 = ? //TODO:
  if (sv.isGlobal){
    gid = s;
  } else {
    gid = triangulation.getSaddleGlobalId(s.id);
  }
  m.scalarS = ?

  if (rank != ttk::MPIrank_){
    sendBuffer.at(rank).emplace_back(m)
  } else {
    for (auto r: ghost){
      sendBuffer.at(r).emplace_back(m);
    }
  }
}*/

void ttk::DiscreteMorseSandwichMPI::tripletsToPersistencePairs(
  std::vector<PersistencePair> &pairs,
  const SimplexId pairDim,
  std::vector<extremaNode> &extremas,
  std::vector<saddleEdge> &saddles,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle
  /*std::vector<std::vector<ttk::SimplexId>> ghostPresence*/) const {
  /*template<typename datatype>
    struct messageType {
      ttk::SimplexId m1;
      datatype scalarM1;
      ttk::SimplexId m2;
      datatype scalarM2;
      ttk::SimplexId s;
      datatype scalarS:
      char procM1;
      char procM2;
      char procS;
      char HAS_BEEN_MODE{0};
    }*/
  // ttk::Timer getRepTimer{};
  // std::vector<std::vector<messageType>> sendBuffer(ttk::MPIrank_,
  // std::vector<messageType>());

  // auto rng = std::default_random_engine{0};
  // std::shuffle(std::begin(triplets), std::end(triplets), rng);
  const bool increasing = (pairDim > 0);
  // Timer tm{};
  // std::vector<ttk::SimplexId> saddleToPairedExtrema(saddleNumber, -1);
  // saddleToPairedExtremaTime = tm.getElapsedTime();
  // get representative of current extremum
  const auto getRep
    = [this, increasing, &extremas, &saddles /*, &getRepTimer, &getRepTime*/](
        extremaNode &extr, saddleEdge &sv) -> extremaNode & {
    //    getRepTimer.reStart();
    auto currentNode = extr;
    auto rep = extremas[extr.rep_.extremaId_];
    // printMsg("In getRep for "+std::to_string(extr.gid_)+" and
    // "+std::to_string(rep.gid_));
    saddleEdge s = saddleEdge{};
    if(currentNode.rep_.saddleId_ != -1) {
      s = saddles[currentNode.rep_.saddleId_];
    }
    while(rep != currentNode) {
      // Test if ghost
      if(currentNode.rank_ != ttk::MPIrank_) {
        return extremas[rep.lid_];
      }
      if(currentNode.rep_.saddleId_ != -1) {
        s = saddles[currentNode.rep_.saddleId_];
        if((s.gid_ != sv.gid_) && ((s < sv) == increasing)) {
          break;
        }
      }
      currentNode = rep;
      /*if (sv.gid_ == 506){
        printMsg("Correct in getRep: "+std::to_string(extremas[17].gid_)+",
      "+std::to_string(extremas[31].gid_));
      }*/
      rep = extremas[currentNode.rep_.extremaId_];
    }
    // In case of the shadow triplet TODO: ensure it is ok
    /*if(increasing && rep.saddleId_ != -1 && (extr == rep)) {
      s = saddles[extr.rep_.saddleId_];
    }*/
    return extremas[currentNode.lid_];
  };

  const auto addPair = [this, &saddleToPairedExtrema, &extremaToPairedSaddle](
                         const saddleEdge &sad, const extremaNode &extr) {
    printMsg("AddPair: " + std::to_string(sad.gid_) + ", "
             + std::to_string(extr.gid_));
    saddleToPairedExtrema[sad.lid_] = extr.lid_;
    extremaToPairedSaddle[extr.lid_] = sad.lid_;
  };

  const auto removePair
    = [this, &saddleToPairedExtrema, &extremaToPairedSaddle](
        const saddleEdge &sad, const extremaNode &extr) {
        printErr("removePair: " + std::to_string(sad.gid_) + ", "
                 + std::to_string(extr.gid_));
        saddleToPairedExtrema[sad.lid_] = -1;
        extremaToPairedSaddle[extr.lid_] = -1;
      };

  /*const auto compareSaddles = [this, &increasing](const GlobalLocalSimplexId
  sv, const GlobalLocalSimplexId s){ if (s.isGlobal){ ttk::SimplexId gid =
  triangulation.getSaddleGlobalId(sv); if (gid == s.id_) return false; auto
  scalarS = scalars.globalMap.find(s)->second; if (scalarS !=
  scalars.localVector[sv]){ return ((scalarS > scalars.localVector[sv]) ==
  increasing); } else { return (s.id < gid.id == increasing);
      }
    } else {
      return (s.id != sv.id) && ((saddlesOrder[s.id] < saddlesOrder[sv.id]) ==
  increasing);
    }
  };*/

  const std::function<int(saddleEdge)> processTriplet
    = [this, &increasing, &saddleToPairedExtrema, &extremaToPairedSaddle,
       &getRep, &addPair, &removePair, &processTriplet, &saddles,
       &extremas](saddleEdge sv) -> int {
    // printMsg("Start of processTriplet: "+std::to_string(extremas[17].gid_)+",
    // "+std::to_string(extremas[31].gid_)+", for "+std::to_string(sv.gid_));
    // rep1 is either last correct in local or a ghost
    auto &rep1 = getRep(extremas[sv.t_[0]], sv);
    bool pairedR1 = extremaToPairedSaddle[rep1.lid_] != -1;
    // TODO: comparison
    bool isR1Invalid = ((rep1.rep_.saddleId_ != -1)
                        && ((saddles[rep1.rep_.saddleId_] < sv) == increasing));
    ttk::SimplexId oldSaddle{-1};
    if(isR1Invalid)
      pairedR1 = false;
    // TODO: still necessary?
    // isR1Invalid = isR1Invalid
    //              && (saddleToPairedExtrema[rep1.saddleId_] ==
    //              rep1.extremaId_);
    if(sv.t_[1] < 0) {
      // deal with "shadow" triplets (a 2-saddle with only one
      // ascending 1-separatrix leading to an unique maximum)
      if(!pairedR1 && saddleToPairedExtrema[sv.lid_] == -1) {

        // when considering the boundary, the "-1" of the triplets
        // indicate a virtual maximum of infinite persistence on the
        // boundary component. a pair is created with the other
        // maximum
        if(isR1Invalid) {
          removePair(saddles[rep1.rep_.saddleId_], rep1);
          oldSaddle = rep1.rep_.saddleId_;
        }
        addPair(sv, rep1);
        // If extrema is has local id, then is present in local TODO: CAREFUL:
        // NOT TRUE extrema can be present in triangulation but not graph
        /*auto ghosts{ghostPresence[rep1.lid_]};
        if (rep1.rank != ttk::MPIrank_ || !ghosts.empty()){
          storeMessageToSend(ghost, rank, sendBuffer, sv, rep1, rep2)
        }*/
        // If ghost or on the border: send message, do what happens next?
        rep1.rep_.extremaId_ = rep1.lid_;
        rep1.rep_.saddleId_ = sv.lid_;
        if(isR1Invalid
           && rep1.rank_ == ttk::MPIrank_) { // TODO: Check if belongs to other
                                             // process, if so triggers message
          // TODO: if from another process
          printMsg("Recompute for " + std::to_string(saddles[oldSaddle].gid_));
          return processTriplet(saddles[oldSaddle]);
        }
      }
      return 0;
    }
    auto &rep2 = getRep(extremas[sv.t_[1]], sv);
    bool pairedR2 = extremaToPairedSaddle[rep2.lid_] != -1;
    bool isR2Invalid = ((rep2.rep_.saddleId_ != -1)
                        && ((saddles[rep2.rep_.saddleId_] < sv) == increasing));
    if(isR2Invalid)
      pairedR2 = false;
    /*isR2Invalid = isR2Invalid
                  && (saddleToPairedExtrema[rep2.saddleId_] ==
       rep2.extremaId_);*/
    if(rep1.gid_ != rep2.gid_) {
      if((((rep2 < rep1) == increasing) || pairedR1) && !pairedR2) {
        if(isR2Invalid) {
          removePair(saddles[rep2.rep_.saddleId_], rep2);
          oldSaddle = rep2.rep_.saddleId_;
        }
        addPair(sv, rep2);
        rep2.rep_.extremaId_ = rep1.lid_;
        rep2.rep_.saddleId_ = sv.lid_;
        if(isR2Invalid) {
          // TODO: if from another process
          printMsg("Recompute for " + std::to_string(saddles[oldSaddle].gid_));
          return processTriplet(saddles[oldSaddle]);
        }
      } else {
        if(!pairedR1) {
          if(isR1Invalid) {
            removePair(saddles[rep1.rep_.saddleId_], rep1);
            oldSaddle = rep1.rep_.saddleId_;
          }
          addPair(sv, rep1);
          rep1.rep_.extremaId_ = rep2.lid_;
          rep1.rep_.saddleId_ = sv.lid_;
          if(isR1Invalid) {
            // TODO: if from another process
            printMsg("Recompute for "
                     + std::to_string(saddles[oldSaddle].gid_));
            return processTriplet(saddles[oldSaddle]);
          }
        }
      }
    }
    return 0;
  };

  for(const auto &s : saddles) {
    processTriplet(s);
  }

  // Receive elements

  ttk::SimplexId saddleNumber = saddleToPairedExtrema.size();

//  ttk::Timer postTimer{};
#ifdef TTK_ENABLE_OPENMP
#pragma omp declare reduction (merge : std::vector<PersistencePair> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : pairs) schedule(static)
#endif
  for(int i = 0; i < saddleNumber; i++) {
    if(saddleToPairedExtrema[i] != -1) {
      if(increasing) {
        pairs.emplace_back(
          saddles[i].gid_, extremas[saddleToPairedExtrema[i]].gid_, pairDim);
      } else {
        pairs.emplace_back(
          extremas[saddleToPairedExtrema[i]].gid_, saddles[i].gid_, pairDim);
      }
    }
  }
  // postTreatmentTime = postTimer.getElapsedTime();
}

void ttk::DiscreteMorseSandwichMPI::displayStats(
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