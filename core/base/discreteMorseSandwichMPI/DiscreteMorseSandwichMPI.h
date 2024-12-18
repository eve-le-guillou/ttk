/// \ingroup baseCode
/// \class ttk::DiscreteMorseSandwichMPI
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \author Pierre Guillou <pierre.guillou@lip6.fr>
/// \date January 2021.
///
/// \brief TTK %DiscreteMorseSandwichMPI processing package.
///
/// %DiscreteMorseSandwichMPI computes a Persistence Diagram by using the
/// %Discrete Morse-Theory %DiscreteGradient algorithms.
///
/// \b Related \b publication \n
/// "Discrete Morse Sandwich: Fast Computation of Persistence Diagrams for
/// Scalar Data -- An Algorithm and A Benchmark" \n
/// Pierre Guillou, Jules Vidal, Julien Tierny \n
/// IEEE Transactions on Visualization and Computer Graphics, 2023.\n
/// arXiv:2206.13932, 2023.
///
///
/// \sa ttk::dcg::DiscreteGradient

#pragma once

#include <DiscreteGradient.h>

#include <algorithm>
#include <array>
#include <csignal>
//#include <execution>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>

namespace ttk {
  class DiscreteMorseSandwichMPI : virtual public Debug {
  public:
    DiscreteMorseSandwichMPI();

    /**
     * @brief Persistence pair struct as exported by DiscreteGradient
     */
    struct PersistencePair {
      /** first (lower/birth) simplex cell id */
      SimplexId birth;
      /** second (higher/death) simplex cell id */
      SimplexId death;
      /** pair type (min-saddle: 0, saddle-saddle: 1, saddle-max: 2) */
      int type;

      PersistencePair(SimplexId b, SimplexId d, int t)
        : birth{b}, death{d}, type{t} {
      }
    };

    struct Rep {
      ttk::SimplexId extremaId_{0};
      ttk::SimplexId saddleId_{-1};
    };

    template <typename triangulationType>
    void fillEdgeOrder(const ttk::SimplexId id,
                       const SimplexId *const offsets,
                       const triangulationType &triangulation,
                       ttk::SimplexId *vertsOrder) const {
      triangulation.getEdgeVertex(id, 0, vertsOrder[0]);
      triangulation.getEdgeVertex(id, 1, vertsOrder[1]);
      vertsOrder[0] = offsets[vertsOrder[0]];
      vertsOrder[1] = offsets[vertsOrder[1]];
      std::sort(vertsOrder, vertsOrder + 2, std::greater<ttk::SimplexId>());
    };

    template <typename triangulationType>
    void fillTriangleOrder(const ttk::SimplexId id,
                           const SimplexId *const offsets,
                           const triangulationType &triangulation,
                           ttk::SimplexId *vertsOrder) const {
      triangulation.getTriangleVertex(id, 0, vertsOrder[0]);
      triangulation.getTriangleVertex(id, 1, vertsOrder[1]);
      triangulation.getTriangleVertex(id, 2, vertsOrder[2]);
      vertsOrder[0] = offsets[vertsOrder[0]];
      vertsOrder[1] = offsets[vertsOrder[1]];
      vertsOrder[2] = offsets[vertsOrder[2]];
      // sort vertices in decreasing order
      std::sort(vertsOrder, vertsOrder + 3, std::greater<ttk::SimplexId>());
    };

    template <typename triangulationType>
    void fillTetraOrder(const ttk::SimplexId id,
                        const SimplexId *const offsets,
                        const triangulationType &triangulation,
                        ttk::SimplexId *vertsOrder) const {
      triangulation.getCellVertex(id, 0, vertsOrder[0]);
      triangulation.getCellVertex(id, 1, vertsOrder[1]);
      triangulation.getCellVertex(id, 2, vertsOrder[2]);
      triangulation.getCellVertex(id, 3, vertsOrder[3]);
      vertsOrder[0] = offsets[vertsOrder[0]];
      vertsOrder[1] = offsets[vertsOrder[1]];
      vertsOrder[2] = offsets[vertsOrder[2]];
      vertsOrder[3] = offsets[vertsOrder[3]];
      // sort vertices in decreasing order
      std::sort(vertsOrder, vertsOrder + 4, std::greater<ttk::SimplexId>());
    };

    struct vpathToSend {
      ttk::SimplexId saddleId_;
      ttk::SimplexId extremaId_;
      char saddleRank_;
    };

    template <int sizeExtr>
    struct vpathFinished {
      ttk::SimplexId saddleId_;
      ttk::SimplexId extremaId_;
      ttk::SimplexId vOrder_[sizeExtr];
      ttk::SimplexId ghostPresenceSize_;
      char extremaRank_;

      bool operator==(const vpathFinished<sizeExtr> &vp) {
        return this->saddleId_ == vp.saddleId_
               && this->extremaId_ == vp.extremaId_;
      }
    };

    void createVpathMPIType(MPI_Datatype &MPI_MessageType) const {
      ttk::SimplexId id = 0;
      MPI_Datatype MPI_SimplexId = getMPIType(id);
      MPI_Datatype types[] = {MPI_SimplexId, MPI_SimplexId, MPI_CHAR};
      int lengths[] = {1, 1, 1};
      const long int mpi_offsets[]
        = {offsetof(vpathToSend, saddleId_), offsetof(vpathToSend, extremaId_),
           offsetof(vpathToSend, saddleRank_)};
      MPI_Type_create_struct(3, lengths, mpi_offsets, types, &MPI_MessageType);
      MPI_Type_commit(&MPI_MessageType);
    };

    template <int sizeExtr>
    void createFinishedVpathMPIType(MPI_Datatype &MPI_MessageType) const {
      ttk::SimplexId id = 0;
      MPI_Datatype MPI_SimplexId = getMPIType(id);
      MPI_Datatype types[] = {
        MPI_SimplexId, MPI_SimplexId, MPI_SimplexId, MPI_SimplexId, MPI_CHAR};
      int lengths[] = {1, 1, sizeExtr, 1, 1};
      const long int mpi_offsets[]
        = {offsetof(vpathFinished<sizeExtr>, saddleId_),
           offsetof(vpathFinished<sizeExtr>, extremaId_),
           offsetof(vpathFinished<sizeExtr>, vOrder_),
           offsetof(vpathFinished<sizeExtr>, ghostPresenceSize_),
           offsetof(vpathFinished<sizeExtr>, extremaRank_)};
      MPI_Type_create_struct(5, lengths, mpi_offsets, types, &MPI_MessageType);
      MPI_Type_commit(&MPI_MessageType);
    };

    template <int sizeExtr, int sizeSad>
    struct messageType {
      ttk::SimplexId sOrder_[sizeSad];
      ttk::SimplexId s1Order_[sizeSad];
      ttk::SimplexId s2Order_[sizeSad];
      ttk::SimplexId t1Order_[sizeExtr];
      ttk::SimplexId t2Order_[sizeExtr];
      ttk::SimplexId t1_;
      ttk::SimplexId t2_;
      ttk::SimplexId s_;
      ttk::SimplexId s1_;
      ttk::SimplexId s2_;
      char t1Rank_;
      char t2Rank_;
      char sRank_;
      char s1Rank_;
      char s2Rank_;
      char hasBeenModified_{0};

      messageType(ttk::SimplexId t1,
                  ttk::SimplexId *t1Order,
                  ttk::SimplexId t2,
                  ttk::SimplexId *t2Order,
                  ttk::SimplexId s,
                  ttk::SimplexId *sOrder,
                  ttk::SimplexId s1,
                  ttk::SimplexId *s1Order,
                  ttk::SimplexId s2,
                  ttk::SimplexId *s2Order,
                  char t1Rank,
                  char t2Rank,
                  char sRank,
                  char s1Rank,
                  char s2Rank,
                  char mod) {
        this->t1_ = t1;
        this->t2_ = t2;
        this->s_ = s;
        this->s1_ = s1;
        this->s2_ = s2;
        this->t1Rank_ = t1Rank;
        this->t2Rank_ = t2Rank;
        this->sRank_ = sRank;
        this->s1Rank_ = s1Rank;
        this->s2Rank_ = s2Rank;
        this->hasBeenModified_ = mod;
        for(int i = 0; i < sizeExtr; i++) {
          t1Order_[i] = t1Order[i];
          t2Order_[i] = t2Order[i];
        }
        for(int i = 0; i < sizeSad; i++) {
          sOrder_[i] = sOrder[i];
          s1Order_[i] = s1Order[i];
          s2Order_[i] = s2Order[i];
        }
      }

      messageType() {
        this->t1_ = -1;
        this->t2_ = -1;
        this->s_ = -1;
        this->s1_ = -1;
        this->s2_ = -1;
        this->t1Rank_ = static_cast<char>(ttk::MPIrank_);
        this->t2Rank_ = static_cast<char>(ttk::MPIrank_);
        this->sRank_ = static_cast<char>(ttk::MPIrank_);
        this->s1Rank_ = static_cast<char>(ttk::MPIrank_);
        this->s2Rank_ = static_cast<char>(ttk::MPIrank_);
        this->hasBeenModified_ = 0;
        for(int i = 0; i < sizeExtr; i++) {
          t1Order_[i] = -1;
          t2Order_[i] = -1;
        }
        for(int i = 0; i < sizeSad; i++) {
          sOrder_[i] = -1;
          s1Order_[i] = -1;
          s2Order_[i] = -1;
        }
      }

      messageType(ttk::SimplexId s, ttk::SimplexId *sOrder, char sRank) {
        this->t1_ = -1;
        this->t2_ = -1;
        this->s_ = s;
        this->s1_ = -1;
        this->s2_ = -1;
        this->t1Rank_ = static_cast<char>(ttk::MPIrank_);
        this->t2Rank_ = static_cast<char>(ttk::MPIrank_);
        this->sRank_ = sRank;
        this->s1Rank_ = static_cast<char>(ttk::MPIrank_);
        this->s2Rank_ = static_cast<char>(ttk::MPIrank_);
        this->hasBeenModified_ = 0;
        for(int i = 0; i < sizeExtr; i++) {
          t1Order_[i] = -1;
          t2Order_[i] = -1;
        }
        for(int i = 0; i < sizeSad; i++) {
          sOrder_[i] = sOrder[i];
          s1Order_[i] = -1;
          s2Order_[i] = -1;
        }
      }
      messageType(ttk::SimplexId t1,
                  ttk::SimplexId *t1Order,
                  ttk::SimplexId s,
                  ttk::SimplexId *sOrder,
                  ttk::SimplexId s1,
                  ttk::SimplexId *s1Order,
                  char t1Rank,
                  char sRank,
                  char s1Rank,
                  char mod) {
        this->t1_ = t1;
        this->t2_ = -1;
        this->s2_ = -1;
        this->s_ = s;
        this->s1_ = s1;
        this->t1Rank_ = t1Rank;
        this->t2Rank_ = static_cast<char>(ttk::MPIrank_);
        this->sRank_ = sRank;
        this->s1Rank_ = s1Rank;
        this->s2Rank_ = static_cast<char>(ttk::MPIrank_);
        this->hasBeenModified_ = mod;
        for(int i = 0; i < sizeExtr; i++) {
          t1Order_[i] = t1Order[i];
          t2Order_[i] = -1;
        }
        for(int i = 0; i < sizeSad; i++) {
          sOrder_[i] = sOrder[i];
          s1Order_[i] = s1Order[i];
        }
      }
      ~messageType() = default;
    };

    bool compareArray(const ttk::SimplexId *arr1,
                      const ttk::SimplexId *arr2,
                      const int size) const {
      for(int i = 0; i < size; i++) {
        if(arr1[i] != arr2[i]) {
          return arr1[i] < arr2[i];
        }
      }
      return false;
    };
    template <int extrSize, int sadSize>
    void createMPIMessageType(MPI_Datatype &MPI_MessageType) const {
      ttk::SimplexId id = 0;
      MPI_Datatype MPI_SimplexId = getMPIType(id);
      MPI_Datatype types[]
        = {MPI_SimplexId, MPI_SimplexId, MPI_SimplexId, MPI_SimplexId,
           MPI_SimplexId, MPI_SimplexId, MPI_SimplexId, MPI_SimplexId,
           MPI_SimplexId, MPI_SimplexId, MPI_CHAR,      MPI_CHAR,
           MPI_CHAR,      MPI_CHAR,      MPI_CHAR,      MPI_CHAR};
      int lengths[] = {sadSize, sadSize, sadSize, extrSize, extrSize, 1, 1, 1,
                       1,       1,       1,       1,        1,        1, 1, 1};
      using simplexMessageType = messageType<extrSize, sadSize>;
      const long int mpi_offsets[]
        = {offsetof(simplexMessageType, sOrder_),
           offsetof(simplexMessageType, s1Order_),
           offsetof(simplexMessageType, s2Order_),
           offsetof(simplexMessageType, t1Order_),
           offsetof(simplexMessageType, t2Order_),
           offsetof(simplexMessageType, t1_),
           offsetof(simplexMessageType, t2_),
           offsetof(simplexMessageType, s_),
           offsetof(simplexMessageType, s1_),
           offsetof(simplexMessageType, s2_),
           offsetof(simplexMessageType, t1Rank_),
           offsetof(simplexMessageType, t2Rank_),
           offsetof(simplexMessageType, sRank_),
           offsetof(simplexMessageType, s1Rank_),
           offsetof(simplexMessageType, s2Rank_),
           offsetof(simplexMessageType, hasBeenModified_)};
      MPI_Type_create_struct(16, lengths, mpi_offsets, types, &MPI_MessageType);
      MPI_Type_commit(&MPI_MessageType);
    };

    template <int size>
    struct extremaNode {
      ttk::SimplexId gid_{-1};
      ttk::SimplexId lid_{-1};
      ttk::SimplexId order_{-1};
      Rep rep_;
      char rank_{static_cast<char>(ttk::MPIrank_)};
      ttk::SimplexId vOrder_[size];

      extremaNode() {
        rep_ = Rep{-1, -1};
      };

      extremaNode(ttk::SimplexId gid) : gid_{gid} {
        rep_ = Rep{-1, -1};
      };

      extremaNode(ttk::SimplexId gid,
                  ttk::SimplexId lid,
                  ttk::SimplexId order,
                  Rep rep,
                  char rank,
                  ttk::SimplexId *vOrder)
        : gid_{gid}, order_{order}, rank_{rank}, lid_{lid} {
        for(ttk::SimplexId i = 0; i < size; i++) {
          vOrder_[i] = vOrder[i];
        }
        rep_ = rep;
      };

      extremaNode(ttk::SimplexId gid,
                  ttk::SimplexId lid,
                  ttk::SimplexId order,
                  Rep rep,
                  char rank)
        : gid_{gid}, lid_{lid}, order_{order}, rank_{rank} {
        for(ttk::SimplexId i = 0; i < size; i++) {
          vOrder_[i] = 0;
        }
        rep_ = rep;
      };

      bool operator==(const extremaNode<size> &t1) {
        return this->gid_ == t1.gid_;
      }

      bool operator!=(const extremaNode<size> &t1) {
        return this->gid_ != t1.gid_;
      }
      bool operator<(const extremaNode<size> &t1) {
        if(this->gid_ == t1.gid_) {
          return false;
        }
        if(this->order_ != -1 && t1.order_ != -1) {
          return this->order_ < t1.order_;
        }
        for(size_t i = 0; i < size; i++) {
          if(this->vOrder_[i] != t1.vOrder_[i]) {
            return this->vOrder_[i] < t1.vOrder_[i];
          }
        }
        return this->gid_ < t1.gid_;
      }
    };

    struct saddleIdPerProcess {
      std::vector<ttk::SimplexId> saddleIds_;
      char rank_;
    };

    template <int size>
    struct saddleEdge {
      ttk::SimplexId gid_{-1};
      ttk::SimplexId lid_{-1};
      ttk::SimplexId order_{-1};
      ttk::SimplexId vOrder_[size];
      std::array<ttk::SimplexId, 2> t_{-1, -1};
      char rank_{static_cast<char>(ttk::MPIrank_)};

      saddleEdge() {
        for(ttk::SimplexId i = 0; i < size; i++) {
          vOrder_[i] = -1;
        }
      };

      saddleEdge(ttk::SimplexId gid,
                 ttk::SimplexId order,
                 ttk::SimplexId *vOrder,
                 char rank)

        : gid_{gid}, order_{order}, rank_{rank} {
        for(ttk::SimplexId i = 0; i < size; i++) {
          vOrder_[i] = vOrder[i];
        }
      };

      saddleEdge(ttk::SimplexId gid, ttk::SimplexId *vOrder, char rank)

        : gid_{gid}, rank_{rank} {
        for(ttk::SimplexId i = 0; i < size; i++) {
          vOrder_[i] = vOrder[i];
        }
      };

      bool operator==(const saddleEdge<size> &s1) {
        return this->gid_ == s1.gid_;
      }

      bool operator<(const saddleEdge<size> &s1) {
        if(this->gid_ == s1.gid_) {
          return false;
        }
        if(this->order_ != -1 && s1.order_ != -1) {
          return this->order_ < s1.order_;
        }
        for(int i = 0; i < size; i++) {
          if(this->vOrder_[i] != s1.vOrder_[i]) {
            return this->vOrder_[i] < s1.vOrder_[i];
          }
        }
        return this->gid_ < s1.gid_;
      }
    };

    template <int size>
    struct saddle {
      ttk::SimplexId gid_{-1};
      ttk::SimplexId lid_{-1};
      ttk::SimplexId order_{-1};
      ttk::SimplexId vOrder_[size];

      saddle() {
        for(ttk::SimplexId i = 0; i < size; i++) {
          vOrder_[i] = -1;
        }
      };

      bool operator==(const saddle<size> &s1) {
        return this->gid_ == s1.gid_;
      }

      bool operator<(const saddle<size> s1) const {
        if(this->gid_ == s1.gid_) {
          return false;
        }
        if(this->order_ != -1 && s1.order_ != -1) {
          return this->order_ < s1.order_;
        }
        for(int i = 0; i < size; i++) {
          if(this->vOrder_[i] != s1.vOrder_[i]) {
            return this->vOrder_[i] < s1.vOrder_[i];
          }
        }
        return this->gid_ < s1.gid_;
      };

      bool operator>(const saddle<size> s1) const {
        if(this->gid_ == s1.gid_) {
          return false;
        }
        if(this->order_ != -1 && s1.order_ != -1) {
          return this->order_ > s1.order_;
        }
        for(int i = 0; i < size; i++) {
          if(this->vOrder_[i] != s1.vOrder_[i]) {
            return this->vOrder_[i] > s1.vOrder_[i];
          }
        }
        return this->gid_ > s1.gid_;
      };
    };

    struct maxPerProcess {
      ttk::SimplexId proc_;
      ttk::SimplexId max_[2];

      maxPerProcess(ttk::SimplexId rank) : proc_{rank} {
        for(int i = 0; i < 2; i++) {
          max_[i] = -1;
        }
      };

      maxPerProcess(ttk::SimplexId rank, ttk::SimplexId *max) : proc_{rank} {
        for(int i = 0; i < 2; i++) {
          max_[i] = max[i];
        }
      };

      bool operator==(const maxPerProcess m1) const {
        return this->proc_ == m1.proc_;
      }

      bool operator<(const maxPerProcess b) const {
        if(this->proc_ == b.proc_) {
          return false;
        }
        for(int i = 0; i < 2; i++) {
          if(this->max_[i] != b.max_[i]) {
            return this->max_[i] < b.max_[i];
          }
        }
        return false;
      }
    };

    template <typename GlobalBoundary>
    void updateMax(maxPerProcess m, GlobalBoundary &maxBoundary) const;

    template <typename GlobalBoundary>
    void getMaxOfProc(ttk::SimplexId rank,
                      ttk::SimplexId *currentMax,
                      GlobalBoundary &maxBoundary) const;

    template <typename GlobalBoundary, typename LocalBoundary>
    bool isEmpty(LocalBoundary &localBoundary,
                 GlobalBoundary &globalBoundary) const;

    template <typename LocalBoundary>
    bool addBoundary(const SimplexId e,
                     bool isOnBoundary,
                     LocalBoundary &localBoundary) const;

    template <typename GlobalBoundary>
    void updateMaxBoundary(
      std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
      const SimplexId s2,
      const ttk::SimplexId *tauOrder,
      GlobalBoundary &boundary,
      ttk::SimplexId rank) const;

    template <typename GlobalBoundary>
    void updateLocalBoundary(
      std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
      const saddle<3> &s2,
      const SimplexId egid1,
      const SimplexId egid2,
      const ttk::SimplexId *tauOrder,
      GlobalBoundary &globalBoundary,
      ttk::SimplexId rank) const;

    template <typename LocalBoundary, typename GlobalBoundary>
    bool mergeGlobalBoundaries(std::vector<bool> &onBoundary,
                               LocalBoundary &s2LocalBoundary,
                               GlobalBoundary &s2GlobalBoundary,
                               LocalBoundary &pTauLocalBoundary,
                               GlobalBoundary &pTauGlobalBoundary) const;

    template <typename GlobalBoundary>
    void updateMergedBoundary(
      std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
      const saddle<3> &s2,
      const SimplexId pTau,
      const ttk::SimplexId *tauOrder,
      GlobalBoundary &globalBoundary) const;

    template <typename triangulationType,
              typename GlobalBoundary,
              typename LocalBoundary,
              typename compareEdges>
    void receiveBoundaryUpdate(std::vector<ttk::SimplexId> &recvBoundaryBuffer,
                               std::vector<int> &s2Locks,
                               std::vector<GlobalBoundary> &globalBoundaries,
                               std::vector<LocalBoundary> &localBoundaries,
                               std::vector<saddle<3>> &saddles2,
                               triangulationType &triangulation,
                               compareEdges &cmpEdges) const;

    template <typename triangulationType,
              typename GlobalBoundary,
              typename LocalBoundary>
    void addEdgeToBoundary(
      const ttk::SimplexId s2Gid,
      const ttk::SimplexId pTau,
      const ttk::SimplexId edgeId,
      std::vector<bool> &onBoundary,
      GlobalBoundary &globalBoundaryIds,
      LocalBoundary &localBoundaryIds,
      const triangulationType &triangulation,
      const SimplexId *const offsets,
      std::vector<std::pair<ttk::SimplexId, ttk::SimplexId>> &ghostEdges,
      std::vector<bool> &hasChangedMax) const;

    template <typename triangulationType, typename GlobalBoundary>
    void packageLocalBoundaryUpdate(
      const saddle<3> &s2,
      ttk::SimplexId *tauOrder,
      GlobalBoundary &globalBoundaryIds,
      const triangulationType &triangulation,
      std::vector<std::pair<ttk::SimplexId, ttk::SimplexId>> &ghostEdges,
      std::vector<bool> &hasChangedMax,
      std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer) const;

    inline void preconditionTriangulation(AbstractTriangulation *const data) {
      this->dg_.preconditionTriangulation(data);
    }

    inline void setInputOffsets(const SimplexId *const offsets) {
      this->dg_.setInputOffsets(offsets);
    }

    inline void setComputeMinSad(const bool data) {
      this->ComputeMinSad = data;
    }
    inline void setComputeSadSad(const bool data) {
      this->ComputeSadSad = data;
    }
    inline void setComputeSadMax(const bool data) {
      this->ComputeSadMax = data;
    }

    template <typename triangulationType>
    inline int buildGradient(const void *const scalars,
                             const size_t scalarsMTime,
                             const SimplexId *const offsets,
                             const triangulationType &triangulation) {
      this->dg_.setDebugLevel(this->debugLevel_);
      this->dg_.setThreadNumber(this->threadNumber_);
      this->dg_.setInputOffsets(offsets);
      this->dg_.setInputScalarField(scalars, scalarsMTime);
      return this->dg_.buildGradient(triangulation);
    }

    /**
     * @brief Ugly hack to avoid a call to buildGradient()
     *
     * An externally computed gradient can be retrofitted into this
     * class using move semantics with setGradient().
     * The internal gradient can be fetched back with getGradient()
     * once the persistence pairs are computed .
     * c.f. ttk::MorseSmaleComplex::returnSaddleConnectors
     *
     * @param[in] dg External gradient instance
     */
    inline void setGradient(ttk::dcg::DiscreteGradient &&dg) {
      this->dg_ = std::move(dg);
      // reset gradient pointer to local storage
      this->dg_.setLocalGradient();
    }
    inline ttk::dcg::DiscreteGradient &&getGradient() {
      return std::move(this->dg_);
    }

    void setUseTasks(bool useTasks) {
      this->UseTasks = useTasks;
    }

    template <typename triangulationType>
    inline SimplexId
      getCellGreaterVertex(const dcg::Cell &c,
                           const triangulationType &triangulation) {
      return this->dg_.getCellGreaterVertex(c, triangulation);
    }

    inline const std::vector<std::vector<SimplexId>> &
      get2SaddlesChildren() const {
      return this->s2Children_;
    }

    /**
     * @brief Compute the persistence pairs from the discrete gradient
     *
     * @pre @ref buildGradient and @ref preconditionTriangulation
     * should be called prior to this function
     *
     * @param[out] pairs Output persistence pairs
     * @param[in] offsets Order field
     * @param[in] triangulation Preconditionned triangulation
     * @param[in] ignoreBoundary Ignore the boundary component
     * @param[in] compute2SaddlesChildren Extract links between 2-saddles
     *
     * @return 0 when success
     */
    template <typename triangulationType>
    int computePersistencePairs(std::vector<PersistencePair> &pairs,
                                const SimplexId *const offsets,
                                const triangulationType &triangulation,
                                const bool ignoreBoundary,
                                const bool compute2SaddlesChildren = false);

    /**
     * @brief Type for exporting persistent generators
     *
     * A generator = a 2-saddle index + vector of edges with 1-saddle
     * at index 0.
     */
    struct GeneratorType {
      /** Generator edges beginning with the 1-saddle */
      std::vector<SimplexId> boundary;
      /** Critical triangle index (-1 if infinite) */
      SimplexId critTriangleId;
      /** Vertex indices for the critical triangle (or global max) and
          the critical edge */
      std::array<SimplexId, 2> critVertsIds;
    };

  protected:
    /**
     * @brief Follow the descending 1-separatrices to compute the saddles ->
     * minima association
     *
     * @param[in] criticalEdges Critical edges identifiers
     * @param[in] triangulation Triangulation
     *
     * @return a vector of minima per 1-saddle
     */
    template <typename triangulationType>
    int getSaddle1ToMinima(
      const std::vector<SimplexId> &criticalEdges,
      const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
        &localTriangToLocalVectExtrema,
      const triangulationType &triangulation,
      const SimplexId *const offsets,
      std::vector<std::array<extremaNode<1>, 2>> &res,
      std::vector<std::vector<char>> &ghostPresence,
      std::unordered_map<ttk::SimplexId, std::vector<char>>
        &localGhostPresenceMap,
      std::vector<std::vector<saddleIdPerProcess>> &ghostPresenceVector,
      MPI_Comm &MPIcomm,
      int localThreadNumber) const;

    inline void extractGhost(
      std::vector<std::vector<char>> &ghostPresence,
      std::vector<std::vector<saddleIdPerProcess>> &ghostPresenceVector,
      int localThreadNumber) const;

    template <int sizeExtr>
    void mergeThreadVectors(
      std::vector<std::vector<vpathFinished<sizeExtr>>> &finishedVPathToSend,
      std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
        &finishedVPathToSendThread,
      std::vector<std::vector<char>> &ghostPresenceToSend,
      std::vector<std::vector<std::vector<char>>> &ghostPresenceToSendThread,
      std::vector<std::vector<ttk::SimplexId>> &ghostCounterThread,
      int localThreadNumber) const;

    template <int sizeExtr>
    void exchangeFinalVPathAndGhosts(
      std::vector<std::vector<char>> &ghostPresenceToSend,
      std::vector<std::vector<vpathFinished<sizeExtr>>> &finishedVPathToSend,
      std::vector<std::vector<char>> &recvGhostPresence,
      std::vector<std::vector<vpathFinished<sizeExtr>>> &recvVPathFinished,
      MPI_Datatype &MPI_SimplexId,
      MPI_Comm &MPIcomm) const;

    template <int sizeExtr,
              int sizeRes,
              typename GLI,
              typename GSR>
    void unpackGhostPresence(
      std::vector<std::vector<char>> &recvGhostPresence,
      std::vector<Lock> &extremaLocks,
      std::vector<std::vector<char>> &ghostPresence,
      std::unordered_map<ttk::SimplexId, std::vector<char>>
        &localGhostPresenceMap,
      const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
        &localTriangToLocalVectExtrema,
      std::vector<std::vector<vpathFinished<sizeExtr>>> &recvVPathFinished,
      std::vector<char> &saddleAtomic,
      std::vector<std::array<extremaNode<sizeExtr>, sizeRes>> &res,
      const GLI getSimplexLocalId,
      const GSR getSimplexRank,
      int localThreadNumber) const;

    template <int sizeExtr, typename GLI>
    void packageGhost(
      std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
        &finishedVPathToSendThread,
      std::vector<std::vector<std::vector<char>>> &ghostPresenceToSendThread,
      std::vector<std::vector<ttk::SimplexId>> &ghostCounterThread,
      std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
        &sendFinishedVPathBufferThread,
      const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
        &localTriangToLocalVectExtrema,
      const GLI getSimplexLocalId,
      std::vector<std::vector<char>> &ghostPresence,
      int localThreadNumber) const;
    /**
     * @brief Follow the ascending 1-separatrices to compute the saddles ->
     * maxima association
     *
     * @param[in] criticalCells Critical cells identifiers
     * @param[in] getFaceStar Either getEdgeStar (in 2D) or getTriangleStar
     * (in 3D)
     * @param[in] getFaceStarNumber Either getEdgeStarNumber (in 2D) or
     * getTriangleStarNumber (in 3D)
     * @param[in] isOnBoundary Either isEdgeOnBoundary (in 2D) or
     * isTriangleOnBoundary (in 3D)
     * @param[in] triangulation Triangulation
     *
     * @return a vector of maxima per 2-saddle
     */
    template <int sizeExtr,
              int sizeSad,
              typename triangulationType,
              typename GFS,
              typename GFSN,
              typename OB,
              typename FEO>
    void getSaddle2ToMaxima(
      const std::vector<SimplexId> &criticalSaddles,
      const GFS &getFaceStar,
      const GFSN &getFaceStarNumber,
      const OB &isOnBoundary,
      const FEO &fillExtremaOrder,
      const triangulationType &triangulation,
      std::vector<std::array<extremaNode<sizeExtr>, sizeSad + 1>> &res,
      const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
        &localTriangToLocalVectExtrema,
      std::vector<std::vector<char>> &ghostPresence,
      std::unordered_map<ttk::SimplexId, std::vector<char>>
        &localGhostPresenceMap,
      std::vector<std::vector<saddleIdPerProcess>> &ghostPresenceVector,
      const std::vector<SimplexId> &critMaxsOrder,
      const SimplexId *const offset,
      MPI_Comm &MPIcomm,
      int localThreadNumber) const;

    /**
     * @brief Compute the pairs of dimension 0
     *
     * @param[out] pairs Output persistence pairs
     * @param[in] pairedMinima If minima are paired
     * @param[in] paired1Saddles If 1-saddles (or maxima in 1D) are paired
     * @param[in] criticalEdges List of 1-saddles (or maxima in 1D)
     * @param[in] critEdgesOrder Filtration order on critical edges
     * @param[in] offsets Vertex offset field
     * @param[in] triangulation Triangulation
     */
    template <typename triangulationType>
    void getMinSaddlePairs(std::vector<PersistencePair> &pairs,
                           const std::vector<ttk::SimplexId> &criticalEdges,
                           const std::vector<ttk::SimplexId> &critEdgesOrder,
                           const std::vector<ttk::SimplexId> &criticalExtremas,
                           const SimplexId *const offsets,
                           size_t &nConnComp,
                           const triangulationType &triangulation,
                           MPI_Comm &MPIcomm,
                           int localThreadNumber) const;

    /**
     * @brief Compute the pairs of dimension dim - 1
     *
     * @param[out] pairs Output persistence pairs
     * @param[in] pairedMaxima If maxima are paired
     * @param[in] pairedSaddles If 2-saddles (or 1-saddles in 2D) are paired
     * @param[in] criticalSaddles List of 2-saddles (or 1-saddles in 2D)
     * @param[in] critSaddlesOrder Filtration order on critical saddles
     * @param[in] critMaxsOrder Filtration order on maxima
     * @param[in] triangulation Triangulation
     */
    template <typename triangulationType>
    void getMaxSaddlePairs(std::vector<PersistencePair> &pairs,
                           const std::vector<SimplexId> &criticalSaddles,
                           const std::vector<SimplexId> &critSaddlesOrder,
                           const std::vector<ttk::SimplexId> &criticalExtremas,
                           const std::vector<SimplexId> &critMaxsOrder,
                           const triangulationType &triangulation,
                           const bool ignoreBoundary,
                           const SimplexId *const offsets,
                           MPI_Comm &MPIcomm,
                           int localThreadNumber);

    template <int sizeExtr,
              int sizeSad,
              typename triangulationType,
              typename GFS,
              typename GFSN,
              typename OB,
              typename FEO,
              typename GSGID,
              typename FSO>
    void computeMaxSaddlePairs(
      std::vector<PersistencePair> &pairs,
      const std::vector<SimplexId> &criticalSaddles,
      const std::vector<SimplexId> &critSaddlesOrder,
      const std::vector<ttk::SimplexId> &criticalExtremas,
      const std::vector<SimplexId> &critMaxsOrder,
      const triangulationType &triangulation,
      const SimplexId *const offsets,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      const GFS &getFaceStar,
      const GFSN &getFaceStarNumber,
      const OB &isOnBoundary,
      const FEO &fillExtremaOrder,
      const GSGID &getSaddleGlobalId,
      const FSO &fillSaddleOrder,
      MPI_Comm &MPIcomm,
      int localThreadNumber);
    /**
     * @brief Compute the saddle-saddle pairs (in 3D)
     *
     * @param[out] pairs Output persistence pairs
     * @param[in] paired1Saddles If 1-saddles are paired
     * @param[in] paired2Saddles If 2-saddles are paired
     * @param[in] exportBoundaries If 2-saddles boundaries must be exported
     * @param[out] boundaries Vector of 2-saddles boundaries
     * @param[in] critical1Saddles Full list of 1-saddles
     * @param[in] critical2Saddles Full list of 2-saddles
     * @param[in] crit1SaddlesOrder Filtration order on 1-saddles
     * @param[in] triangulation Triangulation
     */
    template <typename triangulationType>
    void getSaddleSaddlePairs(std::vector<PersistencePair> &pairs,
                              const bool exportBoundaries,
                              std::vector<GeneratorType> &boundaries,
                              const std::vector<SimplexId> &critical1Saddles,
                              const std::vector<SimplexId> &critical2Saddles,
                              const std::vector<SimplexId> &crit1SaddlesOrder,
                              const std::vector<SimplexId> &crit2SaddlesOrder,
                              const triangulationType &triangulation,
                              const SimplexId *const offsets) const;

    /**
     * @brief Extract & sort critical cell from the DiscreteGradient
     *
     * @param[out] criticalCellsByDim Store critical cells ids per dimension
     * @param[out] critCellsOrder Filtration order on critical cells
     * @param[in] offsets Vertex offset field
     * @param[in] triangulation Triangulation
     * @param[in] sortEdges Sort all edges vs. only 1-saddles
     */
    template <typename triangulationType>
    void extractCriticalCells(
      std::array<std::vector<SimplexId>, 4> &criticalCellsByDim,
      std::array<std::vector<SimplexId>, 4> &critCellsOrder,
      const SimplexId *const offsets,
      const triangulationType &triangulation,
      const bool sortEdges) const;

    /**
     * @brief Print number of pairs, critical cells per dimension & unpaired
     * cells
     *
     * @param[in] pairs Computed persistence pairs
     * @param[in] criticalCellsByDim Store critical cells ids per dimension
     * @param[in] pairedMinima If minima are paired
     * @param[in] paired1Saddles If 1-saddles are paired
     * @param[in] paired2Saddles If 2-saddles are paired
     * @param[in] pairedMaxima If maxima are paired
     */
    void displayStats(
      const std::vector<PersistencePair> &pairs,
      const std::array<std::vector<SimplexId>, 4> &criticalCellsByDim,
      const std::vector<bool> &pairedMinima,
      const std::vector<bool> &paired1Saddles,
      const std::vector<bool> &paired2Saddles,
      const std::vector<bool> &pairedMaxima) const;

    /**
     * @brief Triplet type for persistence pairs
     *
     * [0]: saddle cell id
     * [1]: extremum 1 cell id
     * [2]: extremum 2 cell id
     */
    using tripletType = std::array<ttk::SimplexId, 3>;

    template <int sizeExtr, int sizeSad>
    int processTriplet(
      saddleEdge<sizeSad> sv,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
      std::vector<ttk::SimplexId> &extremaToPairedSaddle,
      std::vector<saddleEdge<sizeSad>> &saddles,
      std::vector<extremaNode<sizeExtr>> &extremas,
      bool increasing,
      std::vector<std::vector<char>> &ghostPresence,
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      std::set<messageType<sizeExtr, sizeSad>,
               std::function<bool(const messageType<sizeExtr, sizeSad> &,
                                  const messageType<sizeExtr, sizeSad> &)>>
        &recomputations,
      const std::function<bool(const messageType<sizeExtr, sizeSad> &,
                               const messageType<sizeExtr, sizeSad> &)>
        &cmpMessages,
      std::vector<messageType<sizeExtr, sizeSad>> &recvBuffer,
      ttk::SimplexId beginVect) const;

    template <int sizeExtr, int sizeSad>
    void storeMessageToSend(
      std::vector<std::vector<char>> &ghostPresence,
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      saddleEdge<sizeSad> &sv,
      saddleEdge<sizeSad> &s1,
      saddleEdge<sizeSad> &s2,
      extremaNode<sizeExtr> &rep1,
      extremaNode<sizeExtr> &rep2,
      char sender = static_cast<char>(ttk::MPIrank_),
      char hasBeenModifed = 0) const;

    template <int sizeExtr, int sizeSad>
    void storeMessageToSend(
      std::vector<std::vector<char>> &ghostPresence,
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      saddleEdge<sizeSad> &sv,
      saddleEdge<sizeSad> &s1,
      extremaNode<sizeExtr> &rep1,
      char sender = static_cast<char>(ttk::MPIrank_),
      char hasBeenModifed = 0) const;

    template <int sizeExtr, int sizeSad>
    void storeMessageToSendToRepOwner(
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      saddleEdge<sizeSad> &sv,
      std::vector<saddleEdge<sizeSad>> &saddles,
      extremaNode<sizeExtr> &rep1,
      extremaNode<sizeExtr> &rep2) const;

    template <int sizeExtr, int sizeSad>
    void storeMessageToSendToRepOwner(
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      saddleEdge<sizeSad> &sv,
      std::vector<saddleEdge<sizeSad>> &saddles,
      extremaNode<sizeExtr> &rep1) const;

    template <int sizeExtr, int sizeSad>
    void storeRerunToSend(
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      saddleEdge<sizeSad> &sv) const;

    template <int sizeExtr, int sizeSad>
    void addPair(const saddleEdge<sizeSad> &sad,
                 const extremaNode<sizeExtr> &extr,
                 std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                 std::vector<ttk::SimplexId> &extremaToPairedSaddle) const;

    template <int sizeExtr, int sizeSad>
    void addToRecvBuffer(
      saddleEdge<sizeSad> &sad,
      std::set<messageType<sizeExtr, sizeSad>,
               std::function<bool(const messageType<sizeExtr, sizeSad> &,
                                  const messageType<sizeExtr, sizeSad> &)>>
        &recomputations,
      const std::function<bool(const messageType<sizeExtr, sizeSad> &,
                               const messageType<sizeExtr, sizeSad> &)>
        &cmpMessages,
      std::vector<messageType<sizeExtr, sizeSad>> &recvBuffer,
      ttk::SimplexId beginVect) const;

    template <int sizeExtr, int sizeSad>
    void removePair(const saddleEdge<sizeSad> &sad,
                    const extremaNode<sizeExtr> &extr,
                    std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                    std::vector<ttk::SimplexId> &extremaToPairedSaddle) const;

    template <int sizeExtr, int sizeSad>
    ttk::SimplexId getRep(extremaNode<sizeExtr> extr,
                          saddleEdge<sizeSad> sv,
                          bool increasing,
                          std::vector<extremaNode<sizeExtr>> &extremas,
                          std::vector<saddleEdge<sizeSad>> &saddles) const;

    template <int sizeSad>
    struct saddleEdge<sizeSad> addSaddle(
      saddleEdge<sizeSad> s,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
      std::vector<saddleEdge<sizeSad>> &saddles,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema) const;

    template <int sizeExtr, int sizeSad>
    ttk::SimplexId getUpdatedT1(
      const ttk::SimplexId extremaGid,
      messageType<sizeExtr, sizeSad> &elt,
      saddleEdge<sizeSad> s,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      std::vector<extremaNode<sizeExtr>> &extremas,
      std::vector<saddleEdge<sizeSad>> &saddles,
      bool increasing) const;

    template <int sizeExtr, int sizeSad>
    ttk::SimplexId getUpdatedT2(
      const ttk::SimplexId extremaGid,
      messageType<sizeExtr, sizeSad> &elt,
      saddleEdge<sizeSad> s,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      std::vector<extremaNode<sizeExtr>> &extremas,
      std::vector<saddleEdge<sizeSad>> &saddles,
      bool increasing) const;

    template <int sizeExtr, int sizeSad>
    void swapT1T2(messageType<sizeExtr, sizeSad> &elt,
                  ttk::SimplexId &t1Lid,
                  ttk::SimplexId &t2Lid,
                  bool increasing) const;

    template <int sizeExtr>
    void addLocalExtrema(
      ttk::SimplexId &lid,
      const ttk::SimplexId gid,
      char rank,
      ttk::SimplexId *vOrder,
      std::vector<extremaNode<sizeExtr>> &extremas,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      std::vector<ttk::SimplexId> &extremaToPairedSaddle) const;

    template <int sizeExtr, int sizeSad>
    void receiveElement(
      messageType<sizeExtr, sizeSad> element,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      std::vector<saddleEdge<sizeSad>> &saddles,
      std::vector<extremaNode<sizeExtr>> &extremas,
      std::vector<ttk::SimplexId> &extremaToPairedSaddle,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      std::vector<std::vector<char>> &ghostPresence,
      char sender,
      bool increasing,
      std::set<messageType<sizeExtr, sizeSad>,
               std::function<bool(const messageType<sizeExtr, sizeSad> &,
                                  const messageType<sizeExtr, sizeSad> &)>>
        &recomputations,
      const std::function<bool(const messageType<sizeExtr, sizeSad> &,
                               const messageType<sizeExtr, sizeSad> &)>
        &cmpMessages,
      std::vector<messageType<sizeExtr, sizeSad>> &recvBuffer,
      ttk::SimplexId beginVect) const;

    /**
     * @brief Compute persistence pairs from triplets
     *
     * @param[out] pairs Store generated persistence pairs
     * @param[in,out] pairedExtrema If critical extrema are paired
     * @param[in,out] pairedSaddles If critical saddles are paired
     * @param[in,out] reps Extrema representatives
     * @param[in] triplets Input triplets (saddle, extremum, extremum)
     * @param[in] saddlesOrder Order on saddles
     * @param[in] extremaOrder Order on extrema
     * @param[in] pairDim Pair birth simplex dimension
     */
    template <int sizeExtr, int sizeSad>
    void tripletsToPersistencePairs(
      const SimplexId pairDim,
      std::vector<extremaNode<sizeExtr>> &extremas,
      std::vector<saddleEdge<sizeSad>> &saddles,
      std::vector<ttk::SimplexId> &saddleIds,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
      std::vector<ttk::SimplexId> &extremaToPairedSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      std::vector<std::vector<char>> ghostPresence,
      MPI_Datatype &MPI_MessageType,
      bool isFirstTime,
      MPI_Comm &MPIcomm,
      int localThreadNumber) const;

    template <int sizeExtr, int sizeSad>
    void extractPairs(std::vector<PersistencePair> &pairs,
                      std::vector<extremaNode<sizeExtr>> &extremas,
                      std::vector<saddleEdge<sizeSad>> &saddles,
                      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                      bool increasing,
                      const int pairDim,
                      int localThreadNumber) const;

    template <int sizeExtr, int sizeSad>
    ttk::SimplexId
      computePairNumbers(std::vector<saddleEdge<sizeSad>> &saddles,
                         std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                         int localThreadNumber) const;
    /**
     * @brief Detect 1-saddles paired to a given 2-saddle
     *
     * Adapted version of ttk::PersistentSimplexPairs::eliminateBoundaries()
     *
     * @param[in] s2 Input 2-saddle (critical triangle)
     * @param[in,out] onBoundary Propagation mask
     * @param[in,out] s2Boundaries Boundaries storage (compact)
     * @param[in] s1Mapping From edge id to 1-saddle compact id in @p s1Locks
     * @param[in] s2Mapping From triangle id to compact id
     *   in @p s2Boundaries and @p s2Locks
     * @param[in] partners Get 2-saddles paired to 1-saddles on boundary
     * @param[in] s1Locks Vector of locks over 1-saddles
     * @param[in] s2Locks Vector of locks over 2-saddles
     * @param[in] triangulation Simplicial complex
     *
     * @return Identifier of paired 1-saddle or -1
     */
    template <typename triangulationType,
              typename GlobalBoundary,
              typename LocalBoundary>
    SimplexId eliminateBoundariesSandwich(
      const saddle<3> &s2,
      std::vector<bool> &onBoundary,
      std::vector<GlobalBoundary> &s2GlobalBoundaries,
      std::vector<LocalBoundary> &s2LocalBoundaries,
      std::vector<SimplexId> &partners,
      std::vector<int> &s1Locks,
      std::vector<int> &s2Locks,
      const std::vector<saddle<2>> &saddles1,
      const std::vector<saddle<3>> &saddles2,
      const triangulationType &triangulation,
      const SimplexId *const offsets,
      std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
      std::vector<std::vector<ttk::SimplexId>> &sendComputeBuffer) const;

    /**
     * @brief Ad-hoc struct for sorting simplices
     *
     * Adapted version of ttk::PersistentSimplexPairs::Simplex
     */
    template <size_t n>
    struct Simplex {
      /** Index in the triangulation */
      SimplexId id_{};
      /** Order field value of the simplex vertices, sorted in
          decreasing order */
      std::array<SimplexId, n> vertsOrder_{};
      /** To compare two vertices according to the filtration (lexicographic
       * order) */
      friend bool operator<(const Simplex<n> &lhs, const Simplex<n> &rhs) {
        return lhs.vertsOrder_ < rhs.vertsOrder_;
      }
    };

    /**
     * @brief \ref Simplex adaptation for edges
     */
    struct EdgeSimplex : Simplex<2> {
      template <typename triangulationType>
      void fillEdge(const SimplexId id,
                    const SimplexId *const offsets,
                    const triangulationType &triangulation) {
        this->id_ = id;
        triangulation.getEdgeVertex(id, 0, this->vertsOrder_[0]);
        triangulation.getEdgeVertex(id, 1, this->vertsOrder_[1]);
        this->vertsOrder_[0] = offsets[this->vertsOrder_[0]];
        this->vertsOrder_[1] = offsets[this->vertsOrder_[1]];
        // sort vertices in decreasing order
        std::sort(this->vertsOrder_.rbegin(), this->vertsOrder_.rend());
      }
    };

    /**
     * @brief \ref Simplex adaptation for triangles
     */
    struct TriangleSimplex : Simplex<3> {
      template <typename triangulationType>
      void fillTriangle(const SimplexId id,
                        const SimplexId *const offsets,
                        const triangulationType &triangulation) {
        this->id_ = id;
        triangulation.getTriangleVertex(id, 0, this->vertsOrder_[0]);
        triangulation.getTriangleVertex(id, 1, this->vertsOrder_[1]);
        triangulation.getTriangleVertex(id, 2, this->vertsOrder_[2]);
        this->vertsOrder_[0] = offsets[this->vertsOrder_[0]];
        this->vertsOrder_[1] = offsets[this->vertsOrder_[1]];
        this->vertsOrder_[2] = offsets[this->vertsOrder_[2]];
        // sort vertices in decreasing order
        std::sort(this->vertsOrder_.rbegin(), this->vertsOrder_.rend());
      }
    };

    /**
     * @brief \ref Simplex adaptation for tetrahedra
     */
    struct TetraSimplex : Simplex<4> {
      template <typename triangulationType>
      void fillTetra(const SimplexId id,
                     const SimplexId *const offsets,
                     const triangulationType &triangulation) {
        this->id_ = id;
        triangulation.getCellVertex(id, 0, this->vertsOrder_[0]);
        triangulation.getCellVertex(id, 1, this->vertsOrder_[1]);
        triangulation.getCellVertex(id, 2, this->vertsOrder_[2]);
        triangulation.getCellVertex(id, 3, this->vertsOrder_[3]);
        this->vertsOrder_[0] = offsets[this->vertsOrder_[0]];
        this->vertsOrder_[1] = offsets[this->vertsOrder_[1]];
        this->vertsOrder_[2] = offsets[this->vertsOrder_[2]];
        this->vertsOrder_[3] = offsets[this->vertsOrder_[3]];
        // sort vertices in decreasing order
        std::sort(this->vertsOrder_.rbegin(), this->vertsOrder_.rend());
      }
    };

    template <typename triangulationType>
    void alloc(const triangulationType &triangulation) {
#ifdef TTK_ENABLE_MPI_TIME
      ttk::Timer t_mpi;
      ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
#endif
      // Timer tm{};
      const auto dim{this->dg_.getDimensionality()};
      if(dim > 3 || dim < 1) {
        return;
      }
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel master num_threads(threadNumber_)
#endif
      {
        if(dim > 2) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->critEdges_.resize(triangulation.getNumberOfEdges());
/*#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->edgeTrianglePartner_.resize(
            triangulation.getNumberOfEdges(), -1);*/
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->onBoundary_.resize(triangulation.getNumberOfEdges(), false);
          /*#ifdef TTK_ENABLE_OPENMP
          #pragma omp task
          #endif
                    this->s2Mapping_.resize(triangulation.getNumberOfTriangles(),
          -1); #ifdef TTK_ENABLE_OPENMP #pragma omp task #endif
                    this->s1Mapping_.resize(triangulation.getNumberOfEdges(),
          -1);*/
        }
        /*for(int i = 0; i < dim + 1; ++i) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->pairedCritCells_[i].resize(
            this->dg_.getNumberOfCells(i, triangulation), false);
        }*/
        for(int i = 1; i < dim + 1; ++i) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->critCellsOrder_[i].resize(
            this->dg_.getNumberOfCells(i, triangulation), -1);
        }
      }
#ifdef TTK_ENABLE_MPI_TIME
      double elapsedTime
        = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
      if(ttk::MPIrank_ == 0) {
        printMsg("Memory allocations performed using "
                 + std::to_string(ttk::MPIsize_)
                 + " MPI processes lasted :" + std::to_string(elapsedTime));
      }
#endif
      /*this->printMsg("Memory allocations", 1.0, tm.getElapsedTime(), 1,
                     debug::LineMode::NEW);*/
    }

    void clear() {
#ifdef TTK_ENABLE_MPI_TIME
      ttk::Timer t_mpi;
      ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
#endif
      // Timer tm{};
      this->edgeTrianglePartner_ = {};
      this->s2Mapping_ = {};
      this->s1Mapping_ = {};
      this->critEdges_ = {};
      this->pairedCritCells_ = {};
      this->onBoundary_ = {};
      this->critCellsOrder_ = {};
      this->saddleToPairedMin_ = {};
      this->saddleToPairedMax_ = {};
      this->minToPairedSaddle_ = {};
      this->maxToPairedSaddle_ = {};
      this->globalToLocalSaddle1_ = {};
      this->globalToLocalSaddle2_ = {};
      /*this->printMsg(
        "Memory cleanup", 1.0, tm.getElapsedTime(), 1, debug::LineMode::NEW);*/
#ifdef TTK_ENABLE_MPI_TIME
      double elapsedTime
        = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
      if(ttk::MPIrank_ == 0) {
        printMsg("Memory cleanup performed using "
                 + std::to_string(ttk::MPIsize_)
                 + " MPI processes lasted :" + std::to_string(elapsedTime));
      }
#endif
    }

    dcg::DiscreteGradient dg_{};

    // factor memory allocations outside computation loops
    mutable std::vector<ttk::SimplexId> edgeTrianglePartner_{}, s2Mapping_{},
      s1Mapping_{};
    mutable std::vector<ttk::SimplexId> saddleToPairedMin_{},
      saddleToPairedMax_{}, minToPairedSaddle_{}, maxToPairedSaddle_{};
    mutable std::unordered_map<ttk::SimplexId, ttk::SimplexId>
      globalToLocalSaddle1_{}, globalToLocalSaddle2_{};
    mutable std::vector<EdgeSimplex> critEdges_{};
    mutable std::array<std::vector<bool>, 4> pairedCritCells_{};
    mutable std::vector<bool> onBoundary_{};
    mutable std::array<std::vector<SimplexId>, 4> critCellsOrder_{};
    mutable std::vector<std::vector<SimplexId>> s2Children_{};
    bool ComputeMinSad{true};
    bool ComputeSadSad{true};
    bool ComputeSadMax{true};
    bool Compute2SaddlesChildren{false};
    bool UseTasks{true};
  };
} // namespace ttk

inline void ttk::DiscreteMorseSandwichMPI::extractGhost(
  std::vector<std::vector<char>> &ghostPresence,
  std::vector<std::vector<saddleIdPerProcess>> &ghostPresenceVector,
  int localThreadNumber) const {
#pragma omp parallel for shared(ghostPresence) num_threads(localThreadNumber)
  for(size_t i = 0; i < ghostPresence.size(); i++) {
    auto &ghost{ghostPresenceVector[i]};
    auto &ranks{ghostPresence[i]};
    // Check if some saddleId appear twice for the same rank
    for(size_t j = 0; j < ghost.size(); j++) {
      ttk::SimplexId ghostSize = ghost[j].saddleIds_.size();
      // Only occurs when the number of saddles is even
      if(ghostSize > 0) {
        if(ghostSize % 2 == 0) {
          std::sort(ghost[j].saddleIds_.begin(), ghost[j].saddleIds_.end());
          const auto last = std::unique(
            ghost[j].saddleIds_.begin(), ghost[j].saddleIds_.end());
          ttk::SimplexId newSize
            = std::distance(ghost[j].saddleIds_.begin(), last);
          if(!(ghostSize % newSize == 0 && ghostSize / newSize == 2)) {
            ranks.emplace_back(ghost[j].rank_);
          }
        } else {
          ranks.emplace_back(ghost[j].rank_);
        }
      }
    }
    ghost.clear();
  }
}

template <int sizeExtr>
void ttk::DiscreteMorseSandwichMPI::mergeThreadVectors(
  std::vector<std::vector<vpathFinished<sizeExtr>>> &finishedVPathToSend,
  std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
    &finishedVPathToSendThread,
  std::vector<std::vector<char>> &ghostPresenceToSend,
  std::vector<std::vector<std::vector<char>>> &ghostPresenceToSendThread,
  std::vector<std::vector<ttk::SimplexId>> &ghostCounterThread,
  int localThreadNumber) const {
#pragma omp parallel for schedule(static, 1) num_threads(localThreadNumber)
  for(int j = 0; j < ttk::MPIsize_; j++) {
    ttk::SimplexId ghostCounter{0};
    for(int i = 0; i < this->threadNumber_; i++) {
      std::transform(finishedVPathToSendThread[i][j].begin(),
                     finishedVPathToSendThread[i][j].end(),
                     finishedVPathToSendThread[i][j].begin(),
                     [this, &ghostCounter](vpathFinished<sizeExtr> &vp) {
                       vp.ghostPresenceSize_ += ghostCounter;
                       return vp;
                     });
      ghostPresenceToSend[j].insert(ghostPresenceToSend[j].end(),
                                    ghostPresenceToSendThread[i][j].begin(),
                                    ghostPresenceToSendThread[i][j].end());
      finishedVPathToSend[j].insert(finishedVPathToSend[j].end(),
                                    finishedVPathToSendThread[i][j].begin(),
                                    finishedVPathToSendThread[i][j].end());
      ghostCounter += ghostCounterThread[i][j];
      ghostPresenceToSendThread[i][j].clear();
      finishedVPathToSendThread[i][j].clear();
    }
  }
}

template <int sizeExtr>
void ttk::DiscreteMorseSandwichMPI::exchangeFinalVPathAndGhosts(
  std::vector<std::vector<char>> &ghostPresenceToSend,
  std::vector<std::vector<vpathFinished<sizeExtr>>> &finishedVPathToSend,
  std::vector<std::vector<char>> &recvGhostPresence,
  std::vector<std::vector<vpathFinished<sizeExtr>>> &recvVPathFinished,
  MPI_Datatype &MPI_SimplexId,
  MPI_Comm &MPIcomm) const {

  MPI_Datatype MPI_FinishedVPathMPIType;
  createFinishedVpathMPIType<sizeExtr>(MPI_FinishedVPathMPIType);
  std::vector<ttk::SimplexId> recvMessageSize(2 * ttk::MPIsize_, 0);
  std::vector<ttk::SimplexId> sendMessageSize(2 * ttk::MPIsize_, 0);
  std::vector<MPI_Request> requests(4 * ttk::MPIsize_, MPI_REQUEST_NULL);
  for(int i = 0; i < ttk::MPIsize_; i++) {
    if(i != ttk::MPIrank_) {
      sendMessageSize[2 * i] = finishedVPathToSend[i].size();
      sendMessageSize[2 * i + 1] = ghostPresenceToSend[i].size();
    }
  }
  MPI_Alltoall(sendMessageSize.data(), 2, MPI_SimplexId, recvMessageSize.data(),
               2, MPI_SimplexId, MPIcomm);
  int count{0};
  // Exchange of the data
  for(int i = 0; i < ttk::MPIsize_; i++) {
    recvVPathFinished[i].resize(recvMessageSize[2 * i]);
    recvGhostPresence[i].resize(recvMessageSize[2 * i + 1]);
    if(recvMessageSize[2 * i] > 0) {
      MPI_Irecv(recvVPathFinished[i].data(), recvMessageSize[2 * i],
                MPI_FinishedVPathMPIType, i, 1, MPIcomm, &requests[count]);
      count++;
    }
    if(sendMessageSize[2 * i] > 0) {
      MPI_Isend(finishedVPathToSend[i].data(), sendMessageSize[2 * i],
                MPI_FinishedVPathMPIType, i, 1, MPIcomm, &requests[count]);
      count++;
    }
    if(recvMessageSize[2 * i + 1] > 0) {
      MPI_Irecv(recvGhostPresence[i].data(), recvMessageSize[2 * i + 1],
                MPI_CHAR, i, 2, MPIcomm, &requests[count]);
      count++;
    }
    if(sendMessageSize[2 * i + 1] > 0) {
      MPI_Isend(ghostPresenceToSend[i].data(), sendMessageSize[2 * i + 1],
                MPI_CHAR, i, 2, MPIcomm, &requests[count]);
      count++;
    }
  }
  MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
}
template <typename triangulationType>
int ttk::DiscreteMorseSandwichMPI::getSaddle1ToMinima(
  const std::vector<SimplexId> &criticalEdges,
  const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
    &localTriangToLocalVectExtrema,
  const triangulationType &triangulation,
  const SimplexId *const offsets,
  std::vector<std::array<extremaNode<1>, 2>> &res,
  std::vector<std::vector<char>> &ghostPresence,
  std::unordered_map<ttk::SimplexId, std::vector<char>> &localGhostPresenceMap,
  std::vector<std::vector<saddleIdPerProcess>> &ghostPresenceVector,
  MPI_Comm &MPIcomm,
  int localThreadNumber) const {

  Timer tm{};
  ttk::SimplexId criticalExtremasNumber = localTriangToLocalVectExtrema.size();
  const std::vector<int> neighbors = triangulation.getNeighborRanks();
  const std::map<int, int> neighborsToId = triangulation.getNeighborsToId();
  int neighborNumber = neighbors.size();
  std::vector<std::vector<std::vector<vpathToSend>>> sendBufferThread(
    threadNumber_);
  std::vector<std::vector<std::vector<vpathFinished<1>>>>
    sendFinishedVPathBufferThread(threadNumber_);
  std::vector<char> saddleAtomic(criticalEdges.size(), 0);
  std::vector<Lock> extremaLocks(criticalExtremasNumber);
  MPI_Datatype MPI_SimplexId = getMPIType(static_cast<ttk::SimplexId>(0));
  for(int i = 0; i < this->threadNumber_; i++) {
    sendBufferThread[i].resize(neighborNumber);
    sendFinishedVPathBufferThread[i].resize(ttk::MPIsize_);
  }
  ttk::SimplexId localElementNumber{0};
  ttk::SimplexId totalFinishedElement{2 * criticalEdges.size()};
  ttk::SimplexId totalElement{0};
  MPI_Allreduce(
    MPI_IN_PLACE, &totalFinishedElement, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
  localElementNumber = 0;
  const auto followVPath = [this, &triangulation, &neighborsToId, &extremaLocks,
                            &res, &saddleAtomic, &ghostPresenceVector,
                            &sendBufferThread, &sendFinishedVPathBufferThread,
                            offsets, &localTriangToLocalVectExtrema](
                             const SimplexId v, ttk::SimplexId saddleId,
                             char saddleRank, int threadNumber,
                             ttk::SimplexId &elementNumber) {
    std::vector<Cell> vpath{};
    this->dg_.getDescendingPath(Cell{0, v}, vpath, triangulation);
    const Cell &lastCell = vpath.back();
    if(lastCell.dim_ == 0) {
      ttk::SimplexId extremaId = triangulation.getVertexGlobalId(lastCell.id_);
      int rank = triangulation.getVertexRank(lastCell.id_);
      if(rank != ttk::MPIrank_) {
        sendBufferThread[threadNumber][neighborsToId.find(rank)->second]
          .emplace_back(vpathToSend{.saddleId_ = saddleId,
                                    .extremaId_ = extremaId,
                                    .saddleRank_ = saddleRank});
      } else {
        elementNumber++;
        if(this->dg_.isCellCritical(lastCell)) {
          ttk::SimplexId id
            = localTriangToLocalVectExtrema.find(lastCell.id_)->second;
          int r{0};
          extremaLocks[id].lock();
          auto &ghost{ghostPresenceVector[id]};
          while(r < ghost.size()) {
            if(ghost[r].rank_ == saddleRank) {
              ghost[r].saddleIds_.emplace_back(saddleId);
              break;
            }
            r++;
          }
          if(r == ghost.size()) {
            ghost.emplace_back(saddleIdPerProcess{
              std::vector<ttk::SimplexId>{saddleId}, saddleRank});
          }
          extremaLocks[id].unlock();
          if(saddleRank == ttk::MPIrank_) {
            ttk::SimplexId vOrd[] = {offsets[lastCell.id_]};
            extremaNode<1> n(extremaId, -1, offsets[lastCell.id_], Rep{-1, -1},
                             static_cast<char>(ttk::MPIrank_), vOrd);
            // We store it in the current rank
            // TODO: only locks for second one?
            char saddleLocalId;
#pragma omp atomic capture
            saddleLocalId = saddleAtomic[saddleId]++;
            res[saddleId][saddleLocalId] = n;
          } else {
            // We store it to send it back to whoever will own the extrema
            sendFinishedVPathBufferThread[threadNumber][saddleRank]
              .emplace_back(vpathFinished<1>{
                .saddleId_ = saddleId,
                .extremaId_ = extremaId,
                .vOrder_ = {offsets[lastCell.id_]},
                .ghostPresenceSize_ = 0,
                .extremaRank_ = static_cast<char>(ttk::MPIrank_)});
          }
        }
      }
    } else {
      elementNumber++;
    }
  };
  ttk::SimplexId elementNumber = 0;
  // follow vpaths from 1-saddles to minima
#pragma omp parallel shared(extremaLocks, saddleAtomic) reduction(+: elementNumber) \
  num_threads(localThreadNumber)
  {
    int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static)
    for(size_t i = 0; i < criticalEdges.size(); ++i) {
      // critical edge vertices
      SimplexId v0{}, v1{};
      triangulation.getEdgeVertex(criticalEdges[i], 0, v0);
      triangulation.getEdgeVertex(criticalEdges[i], 1, v1);

      // follow vpath from each vertex of the critical edge
      followVPath(v0, i, ttk::MPIrank_, threadNumber, elementNumber);
      followVPath(v1, i, ttk::MPIrank_, threadNumber, elementNumber);
    }
  }
  localElementNumber += elementNumber;
  elementNumber = 0;
  // Send receive elements
  MPI_Datatype MPI_MessageType;
  this->createVpathMPIType(MPI_MessageType);
  MPI_Allreduce(
    &localElementNumber, &totalElement, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
  std::vector<std::vector<vpathToSend>> sendBuffer(neighborNumber);
  std::vector<std::vector<vpathToSend>> recvBuffer(neighborNumber);
  bool keepWorking = (totalElement != totalFinishedElement);
  while(keepWorking) {
#pragma omp parallel for schedule(static, 1) num_threads(localThreadNumber)
    for(int j = 0; j < neighborNumber; j++) {
      sendBuffer[j].clear();
      for(int i = 0; i < this->threadNumber_; i++) {
        sendBuffer[j].insert(sendBuffer[j].end(),
                             sendBufferThread[i][j].begin(),
                             sendBufferThread[i][j].end());
        sendBufferThread[i][j].clear();
      }
    }
    // TODO: put that in function?
    std::vector<MPI_Request> sendRequests(neighborNumber);
    std::vector<MPI_Request> recvRequests(neighborNumber);
    std::vector<MPI_Status> sendStatus(neighborNumber);
    std::vector<MPI_Status> recvStatus(neighborNumber);
    std::vector<ttk::SimplexId> sendMessageSize(neighborNumber, 0);
    std::vector<ttk::SimplexId> recvMessageSize(neighborNumber, 0);
    std::vector<int> recvCompleted(neighborNumber, 0);
    std::vector<int> sendCompleted(neighborNumber, 0);
    int sendPerformedCount = 0;
    int recvPerformedCount = 0;
    int sendPerformedCountTotal = 0;
    int recvPerformedCountTotal = 0;
    for(int i = 0; i < neighborNumber; i++) {
      // Send size of sendbuffer
      sendMessageSize[i] = sendBuffer[i].size();
      MPI_Isend(&sendMessageSize[i], 1, MPI_SimplexId, neighbors[i], 0, MPIcomm,
                &sendRequests[i]);
      MPI_Irecv(&recvMessageSize[i], 1, MPI_SimplexId, neighbors[i], 0, MPIcomm,
                &recvRequests[i]);
    }
    std::vector<MPI_Request> sendRequestsData(neighborNumber);
    std::vector<MPI_Request> recvRequestsData(neighborNumber);
    std::vector<MPI_Status> recvStatusData(neighborNumber);
    int recvCount = 0;
    int sendCount = 0;
    int r;
    while((sendPerformedCountTotal < neighborNumber
           || recvPerformedCountTotal < neighborNumber)) {
      if(sendPerformedCountTotal < neighborNumber) {
        MPI_Waitsome(neighborNumber, sendRequests.data(), &sendPerformedCount,
                     sendCompleted.data(), sendStatus.data());
        if(sendPerformedCount > 0) {
          for(int i = 0; i < sendPerformedCount; i++) {
            int rankId = sendCompleted[i];
            r = neighbors[rankId];
            if((sendMessageSize[rankId] > 0)) {
              MPI_Isend(sendBuffer[rankId].data(), sendMessageSize[rankId],
                        MPI_MessageType, r, 1, MPIcomm,
                        &sendRequestsData[sendCount]);
              sendCount++;
            }
          }
          sendPerformedCountTotal += sendPerformedCount;
        }
      }
      if(recvPerformedCountTotal < neighborNumber) {
        MPI_Waitsome(neighborNumber, recvRequests.data(), &recvPerformedCount,
                     recvCompleted.data(), recvStatus.data());
        if(recvPerformedCount > 0) {
          for(int i = 0; i < recvPerformedCount; i++) {
            r = recvStatus[i].MPI_SOURCE;
            int rankId = neighborsToId.find(r)->second;
            if((recvMessageSize[rankId] > 0)) {
              recvBuffer[rankId].resize(recvMessageSize[rankId]);
              MPI_Irecv(recvBuffer[rankId].data(), recvMessageSize[rankId],
                        MPI_MessageType, r, 1, MPIcomm,
                        &recvRequestsData[recvCount]);

              recvCount++;
            }
          }
          recvPerformedCountTotal += recvPerformedCount;
        }
      }
    }
    recvPerformedCountTotal = 0;
    while(recvPerformedCountTotal < recvCount) {
      MPI_Waitsome(recvCount, recvRequestsData.data(), &recvPerformedCount,
                   recvCompleted.data(), recvStatusData.data());
      if(recvPerformedCount > 0) {
        for(int i = 0; i < recvPerformedCount; i++) {
          r = recvStatusData[i].MPI_SOURCE;
          int rankId = neighborsToId.find(r)->second;
#pragma omp parallel num_threads(localThreadNumber) \
  shared(extremaLocks, saddleAtomic)
          {
            int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static) reduction(+ : elementNumber)
            for(ttk::SimplexId j = 0; j < recvMessageSize[rankId]; j++) {
              struct vpathToSend element = recvBuffer[rankId][j];
              ttk::SimplexId v
                = triangulation.getVertexLocalId(element.extremaId_);
              followVPath(v, element.saddleId_, element.saddleRank_,
                          threadNumber, elementNumber);
            }
          }
          localElementNumber += elementNumber;
          elementNumber = 0;
        }
        recvPerformedCountTotal += recvPerformedCount;
      }
    }
    MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);
    // Stop condition computation
    MPI_Allreduce(
      &localElementNumber, &totalElement, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
    keepWorking = (totalElement != totalFinishedElement);
  }
  // Create ghostPresence and send finished VPath back
  std::vector<std::vector<char>> ghostPresenceToSend(ttk::MPIsize_);
  std::vector<std::vector<vpathFinished<1>>> finishedVPathToSend(ttk::MPIsize_);
  std::vector<std::vector<std::vector<char>>> ghostPresenceToSendThread(
    threadNumber_);
  std::vector<std::vector<std::vector<vpathFinished<1>>>>
    finishedVPathToSendThread(threadNumber_);
  std::vector<std::vector<ttk::SimplexId>> ghostCounterThread(threadNumber_);
  for(int i = 0; i < threadNumber_; i++) {
    ghostPresenceToSendThread[i].resize(ttk::MPIsize_);
    finishedVPathToSendThread[i].resize(ttk::MPIsize_);
    ghostCounterThread[i].resize(ttk::MPIsize_, 0);
  }
  // Transform the ghostPresenceVector in proper ghostPresence
  this->extractGhost(ghostPresence, ghostPresenceVector, localThreadNumber);

  // Package the ghostPresence to send it back
  this->packageGhost<1>(
    finishedVPathToSendThread, ghostPresenceToSendThread, ghostCounterThread,
    sendFinishedVPathBufferThread, localTriangToLocalVectExtrema,
    [&triangulation](const SimplexId a) {
      return triangulation.getVertexLocalId(a);
    },
    ghostPresence, localThreadNumber);

  this->mergeThreadVectors(finishedVPathToSend, finishedVPathToSendThread,
                           ghostPresenceToSend, ghostPresenceToSendThread,
                           ghostCounterThread, localThreadNumber);
  // Send/Recv them
  std::vector<std::vector<vpathFinished<1>>> recvVPathFinished(ttk::MPIsize_);
  std::vector<std::vector<char>> recvGhostPresence(ttk::MPIsize_);
  // Send back the ghostpresence
  this->exchangeFinalVPathAndGhosts(ghostPresenceToSend, finishedVPathToSend,
                                    recvGhostPresence, recvVPathFinished,
                                    MPI_SimplexId, MPIcomm);

  // Unpack the received ghostPresence
  this->unpackGhostPresence<1, 2>(
    recvGhostPresence, extremaLocks, ghostPresence, localGhostPresenceMap,
    localTriangToLocalVectExtrema, recvVPathFinished, saddleAtomic, res,
    [&triangulation](const SimplexId a) {
      return triangulation.getVertexLocalId(a);
    },
    [&triangulation](const SimplexId a) {
      return triangulation.getVertexRank(a);
    },
    localThreadNumber);

  /*this->printMsg("Computed the descending 1-separatrices", 1.0,
                 tm.getElapsedTime(), localThreadNumber,
                 debug::LineMode::NEW);*/
  return 0;
}

template <int sizeExtr,
          int sizeSad,
          typename triangulationType,
          typename GFS,
          typename GFSN,
          typename OB,
          typename FEO>
void ttk::DiscreteMorseSandwichMPI::getSaddle2ToMaxima(
  const std::vector<SimplexId> &criticalSaddles,
  const GFS &getFaceStar,
  const GFSN &getFaceStarNumber,
  const OB &isOnBoundary,
  const FEO &fillExtremaOrder,
  const triangulationType &triangulation,
  std::vector<std::array<extremaNode<sizeExtr>, sizeSad + 1>> &res,
  const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
    &localTriangToLocalVectExtrema,
  std::vector<std::vector<char>> &ghostPresence,
  std::unordered_map<ttk::SimplexId, std::vector<char>> &localGhostPresenceMap,
  std::vector<std::vector<saddleIdPerProcess>> &ghostPresenceVector,
  const std::vector<SimplexId> &critMaxsOrder,
  const SimplexId *const offsets,
  MPI_Comm &MPIcomm,
  int localThreadNumber) const {

  Timer tm{};
  const auto dim = this->dg_.getDimensionality();
  ttk::SimplexId criticalExtremasNumber = localTriangToLocalVectExtrema.size();
  const std::vector<int> neighbors = triangulation.getNeighborRanks();
  const std::map<int, int> neighborsToId = triangulation.getNeighborsToId();
  int neighborNumber = neighbors.size();
  std::vector<std::vector<std::vector<vpathToSend>>> sendBufferThread(
    threadNumber_);
  std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
    sendFinishedVPathBufferThread(threadNumber_);
  std::vector<char> saddleAtomic(criticalSaddles.size(), 0);
  std::vector<Lock> extremaLocks(criticalExtremasNumber);
  MPI_Datatype MPI_SimplexId = getMPIType(static_cast<ttk::SimplexId>(0));
  for(int i = 0; i < this->threadNumber_; i++) {
    sendBufferThread[i].resize(neighborNumber);
    sendFinishedVPathBufferThread[i].resize(ttk::MPIsize_);
  }
  ttk::SimplexId localElementNumber{0};
  ttk::SimplexId totalFinishedElement{0};
  ttk::SimplexId totalElement{0};
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(localThreadNumber) reduction(+:totalFinishedElement)
#endif
  for(size_t i = 0; i < criticalSaddles.size(); ++i) {
    totalFinishedElement += getFaceStarNumber(criticalSaddles[i]);
  }
  localElementNumber = 0;
  ttk::SimplexId elementNumber = 0;
  MPI_Allreduce(
    MPI_IN_PLACE, &totalFinishedElement, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
  // follow vpaths from 2-saddles to maxima
  const auto followVPath = [this, dim, &triangulation, &neighborsToId,
                            &extremaLocks, &res, &saddleAtomic,
                            &ghostPresenceVector, &sendBufferThread,
                            &sendFinishedVPathBufferThread, offsets,
                            &localTriangToLocalVectExtrema, &fillExtremaOrder,
                            &critMaxsOrder](const SimplexId v,
                                            ttk::SimplexId saddleId,
                                            char saddleRank, int threadNumber,
                                            ttk::SimplexId &eltNumber) {
    std::vector<Cell> vpath{};
    this->dg_.getAscendingPath(Cell{dim, v}, vpath, triangulation);
    const Cell &lastCell = vpath.back();
    ttk::SimplexId saddleLocalId;
    if(lastCell.dim_ == dim) {
      ttk::SimplexId extremaId = triangulation.getCellGlobalId(lastCell.id_);
      int rank = triangulation.getCellRank(lastCell.id_);
      if(rank != ttk::MPIrank_) {
        sendBufferThread[threadNumber][neighborsToId.find(rank)->second]
          .emplace_back(vpathToSend{.saddleId_ = saddleId,
                                    .extremaId_ = extremaId,
                                    .saddleRank_ = saddleRank});
      } else {
        eltNumber++;
        if(this->dg_.isCellCritical(lastCell)) {
          ttk::SimplexId id
            = localTriangToLocalVectExtrema.find(lastCell.id_)->second;
          size_t r{0};
          extremaLocks[id].lock();
          auto &ghost{ghostPresenceVector[id]};
          while(r < ghost.size()) {
            if(ghost[r].rank_ == saddleRank) {
              ghost[r].saddleIds_.emplace_back(saddleId);
              break;
            }
            r++;
          }
          if(r == ghost.size()) {
            ghost.emplace_back(saddleIdPerProcess{
              std::vector<ttk::SimplexId>{saddleId}, saddleRank});
          }
          extremaLocks[id].unlock();
          // TODO: do also for 2D
          if(saddleRank == ttk::MPIrank_) {
            extremaNode<sizeExtr> n(extremaId, -1, critMaxsOrder[lastCell.id_],
                                    Rep{-1, -1},
                                    static_cast<char>(ttk::MPIrank_));
            fillExtremaOrder(lastCell.id_, n.vOrder_);
            // We store it in the current rank
#pragma omp atomic capture
            saddleLocalId = saddleAtomic[saddleId]++;
            res[saddleId][saddleLocalId] = n;
          } else {
            auto vp{vpathFinished<sizeExtr>{
              .saddleId_ = saddleId,
              .extremaId_ = extremaId,
              .ghostPresenceSize_ = 0,
              .extremaRank_ = static_cast<char>(ttk::MPIrank_)}};
            fillExtremaOrder(lastCell.id_, vp.vOrder_);
            // We store it to send it back to whoever will own the extrema
            sendFinishedVPathBufferThread[threadNumber][saddleRank]
              .emplace_back(vp);
          }
        }
      }
    } else {
      if(lastCell.dim_ == dim - 1) {
        if(saddleRank == ttk::MPIrank_) {
          extremaNode<sizeExtr> n(
            -1, -1, -1, Rep{-1, -1}, static_cast<char>(ttk::MPIrank_));
          // We store it in the current rank
#pragma omp atomic capture
          saddleLocalId = saddleAtomic[saddleId]++;
          res[saddleId][saddleLocalId] = n;
        } else {
          // We store it to send it back to whoever will own the extrema
          sendFinishedVPathBufferThread[threadNumber][saddleRank].emplace_back(
            vpathFinished<sizeExtr>{
              // TODO: no vOrd: ok?
              .saddleId_ = saddleId,
              .extremaId_ = -1,
              .ghostPresenceSize_ = 0,
              .extremaRank_ = static_cast<char>(ttk::MPIrank_)});
        }
      }
      eltNumber++;
    }
  };
  // follow vpaths from 2-saddles to maxima
  ttk::SimplexId saddleLocalId;
#pragma omp parallel shared(extremaLocks, saddleAtomic) reduction(+: elementNumber) \
  num_threads(localThreadNumber)
  {
    int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static)
    for(size_t i = 0; i < criticalSaddles.size(); ++i) {
      const auto sid = criticalSaddles[i];
      for(int j = 0; j < sizeSad + 1; j++) {
        res[i][j].gid_ = -2;
      }
      const auto starNumber = getFaceStarNumber(sid);

      for(SimplexId j = 0; j < starNumber; ++j) {
        SimplexId cellId{};
        getFaceStar(sid, j, cellId);
        followVPath(cellId, i, ttk::MPIrank_, threadNumber, elementNumber);
      }

      if(isOnBoundary(sid)) {
        // critical saddle is on boundary
        extremaNode<sizeExtr> n(
          -1, -1, -1, Rep{-1, -1}, static_cast<char>(ttk::MPIrank_));
#pragma omp atomic capture
        saddleLocalId = saddleAtomic[i]++;
        res[i][saddleLocalId] = n;
      }
    }
  }
  localElementNumber += elementNumber;
  elementNumber = 0;
  // Send receive elements
  MPI_Datatype MPI_MessageType;
  this->createVpathMPIType(MPI_MessageType);
  MPI_Allreduce(
    &localElementNumber, &totalElement, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
  std::vector<std::vector<vpathToSend>> sendBuffer(neighborNumber);
  std::vector<std::vector<vpathToSend>> recvBuffer(neighborNumber);
  bool keepWorking = (totalElement != totalFinishedElement);
  while(keepWorking) {
#pragma omp parallel for schedule(static, 1) num_threads(localThreadNumber)
    for(int j = 0; j < neighborNumber; j++) {
      sendBuffer[j].clear();
      for(int i = 0; i < this->threadNumber_; i++) {
        sendBuffer[j].insert(sendBuffer[j].end(),
                             sendBufferThread[i][j].begin(),
                             sendBufferThread[i][j].end());
        sendBufferThread[i][j].clear();
      }
    }
    // TODO: put that in function?
    std::vector<MPI_Request> sendRequests(neighborNumber);
    std::vector<MPI_Request> recvRequests(neighborNumber);
    std::vector<MPI_Status> sendStatus(neighborNumber);
    std::vector<MPI_Status> recvStatus(neighborNumber);
    std::vector<ttk::SimplexId> sendMessageSize(neighborNumber, 0);
    std::vector<ttk::SimplexId> recvMessageSize(neighborNumber, 0);
    std::vector<int> recvCompleted(neighborNumber, 0);
    std::vector<int> sendCompleted(neighborNumber, 0);
    int sendPerformedCount = 0;
    int recvPerformedCount = 0;
    int sendPerformedCountTotal = 0;
    int recvPerformedCountTotal = 0;
    for(int i = 0; i < neighborNumber; i++) {
      // Send size of sendbuffer
      sendMessageSize[i] = sendBuffer[i].size();
      MPI_Isend(&sendMessageSize[i], 1, MPI_SimplexId, neighbors[i], 0, MPIcomm,
                &sendRequests[i]);
      MPI_Irecv(&recvMessageSize[i], 1, MPI_SimplexId, neighbors[i], 0, MPIcomm,
                &recvRequests[i]);
    }
    std::vector<MPI_Request> sendRequestsData(neighborNumber);
    std::vector<MPI_Request> recvRequestsData(neighborNumber);
    std::vector<MPI_Status> recvStatusData(neighborNumber);
    int recvCount = 0;
    int sendCount = 0;
    int r;
    while((sendPerformedCountTotal < neighborNumber
           || recvPerformedCountTotal < neighborNumber)) {
      if(sendPerformedCountTotal < neighborNumber) {
        MPI_Waitsome(neighborNumber, sendRequests.data(), &sendPerformedCount,
                     sendCompleted.data(), sendStatus.data());
        if(sendPerformedCount > 0) {
          for(int i = 0; i < sendPerformedCount; i++) {
            int rankId = sendCompleted[i];
            r = neighbors[rankId];
            if((sendMessageSize[rankId] > 0)) {
              MPI_Isend(sendBuffer[rankId].data(), sendMessageSize[rankId],
                        MPI_MessageType, r, 1, MPIcomm,
                        &sendRequestsData[sendCount]);
              sendCount++;
            }
          }
          sendPerformedCountTotal += sendPerformedCount;
        }
      }
      if(recvPerformedCountTotal < neighborNumber) {
        MPI_Waitsome(neighborNumber, recvRequests.data(), &recvPerformedCount,
                     recvCompleted.data(), recvStatus.data());
        if(recvPerformedCount > 0) {
          for(int i = 0; i < recvPerformedCount; i++) {
            r = recvStatus[i].MPI_SOURCE;
            int rankId = neighborsToId.find(r)->second;
            if((recvMessageSize[rankId] > 0)) {
              recvBuffer[rankId].resize(recvMessageSize[rankId]);
              MPI_Irecv(recvBuffer[rankId].data(), recvMessageSize[rankId],
                        MPI_MessageType, r, 1, MPIcomm,
                        &recvRequestsData[recvCount]);

              recvCount++;
            }
          }
          recvPerformedCountTotal += recvPerformedCount;
        }
      }
    }
    recvPerformedCountTotal = 0;
    while(recvPerformedCountTotal < recvCount) {
      MPI_Waitsome(recvCount, recvRequestsData.data(), &recvPerformedCount,
                   recvCompleted.data(), recvStatusData.data());
      if(recvPerformedCount > 0) {
        for(int i = 0; i < recvPerformedCount; i++) {
          r = recvStatusData[i].MPI_SOURCE;
          int rankId = neighborsToId.find(r)->second;
#pragma omp parallel num_threads(localThreadNumber) \
  shared(extremaLocks, saddleAtomic)
          {
            int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static) reduction(+ : elementNumber)
            for(ttk::SimplexId j = 0; j < recvMessageSize[rankId]; j++) {
              struct vpathToSend element = recvBuffer[rankId][j];
              ttk::SimplexId v
                = triangulation.getCellLocalId(element.extremaId_);
              followVPath(v, element.saddleId_, element.saddleRank_,
                          threadNumber, elementNumber);
            }
          }
          localElementNumber += elementNumber;
          elementNumber = 0;
        }
        recvPerformedCountTotal += recvPerformedCount;
      }
    }
    MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);
    // Stop condition computation
    MPI_Allreduce(
      &localElementNumber, &totalElement, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
    keepWorking = (totalElement != totalFinishedElement);
  }
  // Create ghostPresence and send finished VPath back
  std::vector<std::vector<char>> ghostPresenceToSend(ttk::MPIsize_);
  std::vector<std::vector<vpathFinished<sizeExtr>>> finishedVPathToSend(
    ttk::MPIsize_);
  std::vector<std::vector<std::vector<char>>> ghostPresenceToSendThread(
    threadNumber_);
  std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
    finishedVPathToSendThread(threadNumber_);
  std::vector<std::vector<ttk::SimplexId>> ghostCounterThread(threadNumber_);
  for(int i = 0; i < threadNumber_; i++) {
    ghostPresenceToSendThread[i].resize(ttk::MPIsize_);
    finishedVPathToSendThread[i].resize(ttk::MPIsize_);
    ghostCounterThread[i].resize(ttk::MPIsize_, 0);
  }
  // Transform the ghostPresenceVector in proper ghostPresence
  this->extractGhost(ghostPresence, ghostPresenceVector, localThreadNumber);

  // Package the ghostPresence to send it back
  this->packageGhost<sizeExtr>(
    finishedVPathToSendThread, ghostPresenceToSendThread, ghostCounterThread,
    sendFinishedVPathBufferThread, localTriangToLocalVectExtrema,
    [&triangulation](const SimplexId a) {
      return triangulation.getCellLocalId(a);
    },
    ghostPresence, localThreadNumber);

  // Merge the vectors
  this->mergeThreadVectors(finishedVPathToSend, finishedVPathToSendThread,
                           ghostPresenceToSend, ghostPresenceToSendThread,
                           ghostCounterThread, localThreadNumber);

  // Send/Recv them
  std::vector<std::vector<vpathFinished<sizeExtr>>> recvVPathFinished(
    ttk::MPIsize_);
  std::vector<std::vector<char>> recvGhostPresence(ttk::MPIsize_);

  // Send back the ghostpresence
  this->exchangeFinalVPathAndGhosts<sizeExtr>(
    ghostPresenceToSend, finishedVPathToSend, recvGhostPresence,
    recvVPathFinished, MPI_SimplexId, MPIcomm);

  // Unpack the received ghostPresence
  this->unpackGhostPresence<sizeExtr, sizeSad + 1>(
    recvGhostPresence, extremaLocks, ghostPresence, localGhostPresenceMap,
    localTriangToLocalVectExtrema, recvVPathFinished, saddleAtomic, res,
    [&triangulation](const SimplexId a) {
      return triangulation.getCellLocalId(a);
    },
    [&triangulation](const SimplexId a) {
      return triangulation.getCellRank(a);
    },
    localThreadNumber);

  /*if(ttk::MPIrank_ == 0)
    this->printMsg("Computed the ascending 1-separatrices", 1.0,
                   tm.getElapsedTime(), localThreadNumber,
                   debug::LineMode::NEW);*/
}

template <int sizeExtr,
          int sizeRes,
          typename GLI,
          typename GSR>
void ttk::DiscreteMorseSandwichMPI::unpackGhostPresence(
  std::vector<std::vector<char>> &recvGhostPresence,
  std::vector<Lock> &extremaLocks,
  std::vector<std::vector<char>> &ghostPresence,
  std::unordered_map<ttk::SimplexId, std::vector<char>> &localGhostPresenceMap,
  const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
    &localTriangToLocalVectExtrema,
  std::vector<std::vector<vpathFinished<sizeExtr>>> &recvVPathFinished,
  std::vector<char> &saddleAtomic,
  std::vector<std::array<extremaNode<sizeExtr>, sizeRes>> &res,
  const GLI getSimplexLocalId,
  const GSR getSimplexRank,
  int localThreadNumber) const {
  for(int i = 0; i < ttk::MPIsize_; i++) {
#pragma omp parallel for schedule(static) shared(localGhostPresenceMap) \
  num_threads(localThreadNumber)
    for(size_t j = 0; j < recvVPathFinished[i].size(); j++) {
      // Receive element: create VPath and add it to the list
      auto &vp{recvVPathFinished[i][j]};
      ttk::SimplexId beginGhost
        = (j == 0) ? 0 : recvVPathFinished[i][j - 1].ghostPresenceSize_;
      ttk::SimplexId order{-1};
      if(sizeExtr == 1) {
        order = vp.vOrder_[0];
      }
      extremaNode<sizeExtr> n(
        vp.extremaId_, -1, order, Rep{-1, -1}, vp.extremaRank_, vp.vOrder_);
      // We store it in the current rank
      char saddleLocalId;
#pragma omp atomic capture
      saddleLocalId = saddleAtomic[vp.saddleId_]++;
      res[vp.saddleId_][saddleLocalId] = n;
      if(vp.ghostPresenceSize_ != 0) {
        // Add the received ghostPresence to the local ghostPresence
        // If there is only one process, then the extrema won't be on the
        // boundary of the new graph, there is no need to record it
        if(vp.ghostPresenceSize_ - beginGhost > 1) {
          std::vector<char> ghost{};
          ghost.insert(ghost.end(), recvGhostPresence[i].begin() + beginGhost,
                       recvGhostPresence[i].begin()
                         + static_cast<ttk::SimplexId>(vp.ghostPresenceSize_));
          ttk::SimplexId lid = getSimplexLocalId(vp.extremaId_);
          // If the extrema is not locally present in the triangulation,
          // Add the entry to the map
          if(lid == -1 || getSimplexRank(lid) != ttk::MPIrank_) {
#pragma omp critical
            { localGhostPresenceMap[vp.extremaId_] = ghost; }
          } else {
            lid = localTriangToLocalVectExtrema.find(lid)->second;
            extremaLocks[lid].lock();
            ghostPresence[lid] = ghost;
            extremaLocks[lid].unlock();
          }
        }
      }
    }
  }
};

template <int sizeExtr, typename GLI>
void ttk::DiscreteMorseSandwichMPI::packageGhost(
  std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
    &finishedVPathToSendThread,
  std::vector<std::vector<std::vector<char>>> &ghostPresenceToSendThread,
  std::vector<std::vector<ttk::SimplexId>> &ghostCounterThread,
  std::vector<std::vector<std::vector<vpathFinished<sizeExtr>>>>
    &sendFinishedVPathBufferThread,
  const std::unordered_map<ttk::SimplexId, ttk::SimplexId>
    &localTriangToLocalVectExtrema,
  const GLI getSimplexLocalId,
  std::vector<std::vector<char>> &ghostPresence,
  int localThreadNumber) const {
#pragma omp parallel for schedule(static, 1) num_threads(localThreadNumber) \
  shared(ghostCounterThread)
  for(int j = 0; j < localThreadNumber; j++) {
    for(int i = 0; i < ttk::MPIsize_; i++) {
      for(size_t k = 0; k < sendFinishedVPathBufferThread[j][i].size(); k++) {
        // Find owner by applying the following rule:
        // if the current rank is in ghostPresence, then the current rank is
        // the owner if not, it is the rank with the lowest rank id that is
        // the owner
        auto vp = sendFinishedVPathBufferThread[j][i][k];
        if(vp.extremaId_ > -1) {
          ttk::SimplexId lid = localTriangToLocalVectExtrema
                                 .find(getSimplexLocalId(vp.extremaId_))
                                 ->second;
          auto &ghost{ghostPresence[lid]};
          auto it = std::find(
            ghost.begin(), ghost.end(), static_cast<char>(ttk::MPIrank_));
          if(it != ghost.end()) {
            // The rank of the extrema is the current rank
            // We store to send the finished vpath
            vp.extremaRank_ = ttk::MPIrank_;
            vp.ghostPresenceSize_ = ghostCounterThread[j][i];
          } else {
            // The rank of the extrema is NOT the current rank
            // We find the smallest rank
            auto minRank = std::min_element(ghost.begin(), ghost.end());
            vp.extremaRank_ = (*minRank);
            if(i == (*minRank) && ghost.size() > 1) {
              ghostCounterThread[j][i] += ghost.size();
            }
            vp.ghostPresenceSize_ = ghostCounterThread[j][i];
            // Send the ghostPresence to that rank
            if(ghost.size() > 1) {
              ghostPresenceToSendThread[j][i].insert(
                ghostPresenceToSendThread[j][i].end(), ghost.begin(),
                ghost.end());
            }
          }
        } else {
          vp.ghostPresenceSize_ = ghostCounterThread[j][i];
        }
        finishedVPathToSendThread[j][i].emplace_back(vp);
      }
      sendFinishedVPathBufferThread[j][i].clear();
    }
  }
}

template <typename triangulationType>
void ttk::DiscreteMorseSandwichMPI::getMinSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const std::vector<ttk::SimplexId> &criticalEdges,
  const std::vector<ttk::SimplexId> &critEdgesOrder,
  const std::vector<ttk::SimplexId> &criticalExtremas,
  const SimplexId *const offsets,
  size_t &nConnComp,
  const triangulationType &triangulation,
  MPI_Comm &MPIcomm,
  int localThreadNumber) const {
  ttk::SimplexId totalNumberOfVertices{-1};
  ttk::SimplexId criticalExtremasNumber = criticalExtremas.size();
  ttk::SimplexId criticalEdgesNumber = criticalEdges.size();
  MPI_Datatype MPI_SimplexId = getMPIType(totalNumberOfVertices);
  ttk::SimplexId localNumberOfVertices = triangulation.getNumberOfVertices();

  MPI_Allreduce(&localNumberOfVertices, &totalNumberOfVertices, 1,
                MPI_SimplexId, MPI_SUM, MPIcomm);

  ttk::SimplexId globalMinOffset{-1};
  std::pair<ttk::SimplexId, ttk::SimplexId> localMin(-1, totalNumberOfVertices);

  if(criticalExtremasNumber > 0) {
    // extracts the global min
#pragma omp declare reduction(get_min : std::pair<ttk::SimplexId, ttk::SimplexId> :omp_out = omp_out.second < omp_in.second ? omp_out : omp_in)
#pragma omp parallel for reduction(get_min \
                                   : localMin) num_threads(localThreadNumber)
    for(ttk::SimplexId i = 0; i < criticalExtremasNumber; i++) {
      if(offsets[criticalExtremas[i]] < localMin.second) {
        localMin.first = criticalExtremas[i];
        localMin.second = offsets[localMin.first];
      }
    }
  }

  MPI_Allreduce(
    &localMin.second, &globalMinOffset, 1, MPI_SimplexId, MPI_MIN, MPIcomm);
  if(this->ComputeMinSad) {
    // minima - saddle pairs
    Timer tm{};
    std::vector<std::array<extremaNode<1>, 2>> saddle1ToMinima;
    std::vector<std::vector<saddleIdPerProcess>> ghostPresenceVector;
    std::unordered_map<ttk::SimplexId, std::vector<char>> localGhostPresenceMap;
    std::vector<std::vector<char>> localGhostPresenceVector;
    std::unordered_map<ttk::SimplexId, ttk::SimplexId>
      localTriangToLocalVectExtrema;
    std::vector<ttk::SimplexId> extremasGid;
    std::vector<saddleEdge<2>> saddles;
    extremasGid.reserve(criticalEdgesNumber * 2);
#pragma omp parallel master num_threads(localThreadNumber)
    {
#pragma omp task
      for(ttk::SimplexId i = 0; i < criticalExtremasNumber; i++) {
        localTriangToLocalVectExtrema[criticalExtremas[i]] = i;
      }
#pragma omp task
      ghostPresenceVector.resize(
        criticalExtremasNumber, std::vector<saddleIdPerProcess>());
#pragma omp task
      saddle1ToMinima.resize(
        criticalEdges.size(), std::array<extremaNode<1>, 2>());
#pragma omp task
      localGhostPresenceVector.resize(
        criticalExtremasNumber, std::vector<char>());
#pragma omp task
      saddles.resize(criticalEdgesNumber);
    }

    this->getSaddle1ToMinima(criticalEdges, localTriangToLocalVectExtrema,
                             triangulation, offsets, saddle1ToMinima,
                             localGhostPresenceVector, localGhostPresenceMap,
                             ghostPresenceVector, MPIcomm, localThreadNumber);
    // Timer tmseq{};
    auto &saddleToPairedExtrema{this->saddleToPairedMin_};
    auto &extremaToPairedSaddle{this->minToPairedSaddle_};
    auto &globalToLocalSaddle{this->globalToLocalSaddle1_};
    std::vector<ttk::SimplexId> globalMinLid;
    ttk::SimplexId totalNumberOfPairs = criticalExtremasNumber;
    MPI_Allreduce(
      MPI_IN_PLACE, &totalNumberOfPairs, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
    globalToLocalSaddle.reserve(criticalEdgesNumber);

#pragma omp declare reduction (merge : std::vector<ttk::SimplexId>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge                           \
                                   : extremasGid) schedule(static) \
  shared(saddles) num_threads(localThreadNumber)
    for(ttk::SimplexId i = 0; i < criticalEdgesNumber; ++i) {
      auto &mins = saddle1ToMinima[i];
      const auto s1 = criticalEdges[i];
      // remove duplicates
      std::sort(mins.begin(), mins.end());
      const auto last = std::unique(mins.begin(), mins.end());
      if(last != mins.end()) {
        continue;
      }
      saddles[i].lid_ = i;
      ttk::SimplexId gid = triangulation.getEdgeGlobalId(s1);
      saddles[i].gid_ = gid;
      for(int j = 0; j < 2; j++) {
        extremasGid.emplace_back(mins[j].gid_);
      }
    }
    TTK_PSORT(localThreadNumber, extremasGid.begin(), extremasGid.end());
    const auto lastGid = std::unique(
      /*std::execution::par_unseq,*/ extremasGid.begin(), extremasGid.end());
    extremasGid.erase(lastGid, extremasGid.end());
    std::unordered_map<ttk::SimplexId, ttk::SimplexId> globalToLocalExtrema{};
    globalToLocalExtrema.reserve(extremasGid.size());
    std::vector<std::vector<char>> ghostPresence(
      extremasGid.size(), std::vector<char>());
    std::vector<int> extremaLocks(extremasGid.size(), 0);
    std::vector<extremaNode<1>> extremas(extremasGid.size(), extremaNode<1>());
#pragma omp parallel master shared(extremaLocks, extremas, globalMinLid,    \
                                   ghostPresence, saddles, globalMinOffset) \
  num_threads(localThreadNumber)
    {
#pragma omp task
      {
        for(ttk::SimplexId i = 0; i < extremasGid.size(); i++) {
          globalToLocalExtrema.emplace(extremasGid[i], i);
        }
      }
#pragma omp task
      {
        for(ttk::SimplexId i = 0; i < criticalEdgesNumber; i++) {
          if(saddles[i].gid_ != -1) {
            globalToLocalSaddle.emplace(saddles[i].gid_, i);
          }
        }
      }
      int numTask = std::max(localThreadNumber - 2, 1);
#pragma omp taskloop num_tasks(numTask)
      for(ttk::SimplexId i = 0; i < criticalEdgesNumber; ++i) {
        auto &mins = saddle1ToMinima[i];
        const auto s1 = criticalEdges[i];
        const auto last = std::unique(mins.begin(), mins.end());
        // mins.erase(last, mins.end());
        if(last != mins.end()) {
          continue;
        }
        saddleEdge<2> &e{saddles[i]};
        fillEdgeOrder(s1, offsets, triangulation, e.vOrder_);
        e.order_ = critEdgesOrder[s1];
        for(int j = 0; j < 2; j++) {
          int extremaExists;
          ttk::SimplexId lid = std::lower_bound(extremasGid.begin(),
                                                extremasGid.end(), mins[j].gid_)
                               - extremasGid.begin();
#pragma omp atomic capture
        extremaExists = extremaLocks[lid]++;
        if(extremaExists == 0) {
          std::vector<char> ghosts{};
          mins[j].lid_ = lid;
          mins[j].rep_.extremaId_ = lid;
          extremas[lid] = mins[j];
          if(globalMinOffset == mins[j].vOrder_[0]) {
#pragma omp critical
            globalMinLid.emplace_back(lid);
          }
          ttk::SimplexId triangLid
            = triangulation.getVertexLocalId(mins[j].gid_);
          if(triangLid == -1
             || triangulation.getVertexRank(triangLid) != ttk::MPIrank_) {
            auto it = localGhostPresenceMap.find(mins[j].gid_);
            if(it != localGhostPresenceMap.end()) {
              ghosts = localGhostPresenceMap[mins[j].gid_];
            } else {
              ghosts.resize(0);
            }
          } else {
            triangLid = localTriangToLocalVectExtrema.find(triangLid)->second;
            ghosts = localGhostPresenceVector[triangLid];
          }
          ghostPresence[lid] = ghosts;
        }
        e.t_[j] = lid;
      }
      }
    }
    const auto cmpSadMin
      = [=, &extremas, &saddles](
          const ttk::SimplexId &id1, const ttk::SimplexId &id2) -> bool {
      const saddleEdge<2> &s0{saddles[id1]};
      const saddleEdge<2> &s1{saddles[id2]};
      if(s0.gid_ != s1.gid_) {
        if(s0.order_ != -1 && s1.order_ != -1) {
          return s0.order_ < s1.order_;
        }
        for(int i = 0; i < 2; i++) {
          if(s0.vOrder_[i] != s1.vOrder_[i]) {
            return s0.vOrder_[i] < s1.vOrder_[i];
          }
        }
      }
      if(s0.gid_ == -1 && s1.gid_ == -1) {
        return false;
      }
      return extremas[s0.t_[0]].vOrder_ > extremas[s1.t_[0]].vOrder_;
    };

    // TRI des arcs
    std::vector<ttk::SimplexId> saddleIds(saddles.size());
    std::iota(saddleIds.begin(), saddleIds.end(), 0);
    TTK_PSORT(localThreadNumber, saddleIds.begin(), saddleIds.end(), cmpSadMin);
    // Mise en place des lid des arcs
    extremaToPairedSaddle.resize(globalToLocalExtrema.size(), -1);
    saddleToPairedExtrema.resize(saddles.size(), -1);
    MPI_Datatype MPI_MessageType;
    createMPIMessageType<1, 2>(MPI_MessageType);
    tripletsToPersistencePairs<1, 2>(
      0, extremas, saddles, saddleIds, saddleToPairedExtrema,
      extremaToPairedSaddle, globalToLocalSaddle, globalToLocalExtrema,
      ghostPresence, MPI_MessageType, true, MPIcomm, localThreadNumber);
    ttk::SimplexId nMinSadPairs = computePairNumbers<1, 2>(
      saddles, saddleToPairedExtrema, localThreadNumber);
    char rerunNeeded{0};
    for(const auto lid : globalMinLid) {
      if(extremaToPairedSaddle[lid] != -1
         && saddles[extremaToPairedSaddle[lid]].rank_ == ttk::MPIrank_) {
        saddleToPairedExtrema[extremaToPairedSaddle[lid]] = -2;
        rerunNeeded = 1;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &rerunNeeded, 1, MPI_CHAR, MPI_LOR, MPIcomm);
    MPI_Allreduce(
      MPI_IN_PLACE, &nMinSadPairs, 1, MPI_SimplexId, MPI_SUM, MPIcomm);

    while((nMinSadPairs != totalNumberOfPairs - 1) || rerunNeeded) {
      printMsg("Re-computation, rerun needed: " + std::to_string(rerunNeeded)
               + " nMinSadPairs: " + std::to_string(nMinSadPairs) + ", "
               + std::to_string(totalNumberOfPairs));
      tripletsToPersistencePairs<1, 2>(
        0, extremas, saddles, saddleIds, saddleToPairedExtrema,
        extremaToPairedSaddle, globalToLocalSaddle, globalToLocalExtrema,
        ghostPresence, MPI_MessageType, false, MPIcomm, localThreadNumber);
      nMinSadPairs = computePairNumbers<1, 2>(
        saddles, saddleToPairedExtrema, localThreadNumber);
      rerunNeeded = 0;
      for(const auto lid : globalMinLid) {
        if(extremaToPairedSaddle[lid] != -1
           && saddles[extremaToPairedSaddle[lid]].rank_ == ttk::MPIrank_) {
          saddleToPairedExtrema[extremaToPairedSaddle[lid]] = -2;
          rerunNeeded = 1;
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &rerunNeeded, 1, MPI_CHAR, MPI_LOR, MPIcomm);
      MPI_Allreduce(
        MPI_IN_PLACE, &nMinSadPairs, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
    }
    extractPairs<1, 2>(pairs, extremas, saddles, saddleToPairedExtrema, false,
                       0, localThreadNumber);
    /*std::ofstream myfile;
    myfile.open("/home/eveleguillou/experiment/DiscreteMorseSandwich/"
                + std::to_string(ttk::MPIsize_) + "_pairs_"
                + std::to_string(ttk::MPIrank_) + ".csv");
    myfile << "min,sad\n";
    for(ttk::SimplexId i = 0; i < pairs.size(); i++) {
      myfile << std::to_string(pairs[i].birth) + ","
                  + std::to_string(pairs[i].death) + "\n";
    }
    myfile.close();*/

    // non-paired minima
    if(totalNumberOfPairs > 1) {
#pragma omp parallel for shared(pairs, nConnComp) num_threads(localThreadNumber)
      for(const auto extr : extremas) {
        if(extr.rank_ == ttk::MPIrank_) {
          if(extremaToPairedSaddle[extr.lid_] < 0) {
#pragma omp critical
            {
              pairs.emplace_back(extr.gid_, -1, 0);
              nConnComp++;
            }
          }
        }
      }
    } else {
      if(globalMinOffset == localMin.second) {
        pairs.emplace_back(
          triangulation.getVertexGlobalId(localMin.first), -1, 0);
        nConnComp++;
      }
    }
    /*this->printMsg("min-saddle pairs sequential part", 1.0,
                   tmseq.getElapsedTime(), 1, debug::LineMode::NEW);
    */
    if(ttk::MPIrank_ == 0)
      this->printMsg(
        "Computed " + std::to_string(nMinSadPairs) + " min-saddle pairs", 1.0,
        tm.getElapsedTime(), localThreadNumber);
  } else {
    if(globalMinOffset == localMin.second) {
      pairs.emplace_back(
        triangulation.getVertexGlobalId(localMin.first), -1, 0);
      nConnComp++;
    }
  }
}

template <int sizeExtr,
          int sizeSad,
          typename triangulationType,
          typename GFS,
          typename GFSN,
          typename OB,
          typename FEO,
          typename GSGID,
          typename FSO>
void ttk::DiscreteMorseSandwichMPI::computeMaxSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const std::vector<SimplexId> &criticalSaddles,
  const std::vector<SimplexId> &critSaddlesOrder,
  const std::vector<ttk::SimplexId> &criticalExtremas,
  const std::vector<SimplexId> &critMaxsOrder,
  const triangulationType &triangulation,
  const SimplexId *const offsets,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  const GFS &getFaceStar,
  const GFSN &getFaceStarNumber,
  const OB &isOnBoundary,
  const FEO &fillExtremaOrder,
  const GSGID &getSaddleGlobalId,
  const FSO &fillSaddleOrder,
  MPI_Comm &MPIcomm,
  int localThreadNumber) {
  Timer tm{};
  const auto dim = this->dg_.getDimensionality();
  std::vector<std::array<extremaNode<sizeExtr>, sizeSad + 1>> saddle2ToMaxima;
  std::vector<std::vector<saddleIdPerProcess>> ghostPresenceVector;
  ttk::SimplexId criticalExtremasNumber = criticalExtremas.size();
  ttk::SimplexId criticalSaddlesNumber = criticalSaddles.size();
  std::unordered_map<ttk::SimplexId, std::vector<char>> localGhostPresenceMap;
  std::vector<std::vector<char>> localGhostPresenceVector;
  std::unordered_map<ttk::SimplexId, ttk::SimplexId>
    localTriangToLocalVectExtrema;
  std::vector<ttk::SimplexId> extremasGid;
  std::vector<saddleEdge<sizeSad>> saddles;
  MPI_Datatype MPI_SimplexId = getMPIType(criticalExtremasNumber);
  extremasGid.reserve(criticalSaddlesNumber * 2);
#pragma omp parallel master num_threads(localThreadNumber)
  {
#pragma omp task
    for(ttk::SimplexId i = 0; i < criticalExtremasNumber; i++) {
      localTriangToLocalVectExtrema[criticalExtremas[i]] = i;
    }
#pragma omp task
    ghostPresenceVector.resize(
      criticalExtremasNumber, std::vector<saddleIdPerProcess>());
#pragma omp task
    saddle2ToMaxima.resize(
      criticalSaddles.size(), std::array<extremaNode<sizeExtr>, sizeSad + 1>());
#pragma omp task
    localGhostPresenceVector.resize(
      criticalExtremasNumber, std::vector<char>());
#pragma omp task
    saddles.resize(criticalSaddlesNumber);
  }
  this->getSaddle2ToMaxima<sizeExtr, sizeSad>(
    criticalSaddles, getFaceStar, getFaceStarNumber, isOnBoundary,
    fillExtremaOrder, triangulation, saddle2ToMaxima,
    localTriangToLocalVectExtrema, localGhostPresenceVector,
    localGhostPresenceMap, ghostPresenceVector, critMaxsOrder, offsets, MPIcomm,
    localThreadNumber);

  Timer tmseq{};

  auto &saddleToPairedExtrema{this->saddleToPairedMax_};
  auto &extremaToPairedSaddle{this->maxToPairedSaddle_};

  globalToLocalExtrema.reserve(2 * saddle2ToMaxima.size());
  /*if(dim == 3) {
    globalToLocalSaddle.reserve(saddle2ToMaxima.size());
  } else {
    globalToLocalSaddle.reserve(globalToLocalSaddle.size()
                                + saddle2ToMaxima.size());
  }*/
  ttk::SimplexId totalNumberOfPairs = criticalExtremasNumber;
  MPI_Allreduce(
    MPI_IN_PLACE, &totalNumberOfPairs, 1, MPI_SimplexId, MPI_SUM, MPIcomm);
  globalToLocalSaddle.reserve(criticalSaddlesNumber);
#pragma omp declare reduction (merge : std::vector<ttk::SimplexId>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge                           \
                                   : extremasGid) schedule(static) \
  shared(saddles, saddle2ToMaxima) num_threads(localThreadNumber)
  for(ttk::SimplexId i = 0; i < criticalSaddlesNumber; ++i) {
    auto &maxs = saddle2ToMaxima[i];
    const auto s2 = criticalSaddles[i];
    std::sort(
      maxs.begin(), maxs.end(),
      [](const extremaNode<sizeExtr> &a, const extremaNode<sizeExtr> &b) {
        // positive values (actual maxima) before negative ones
        // (boundary component id)
        if(a.gid_ * b.gid_ >= 0) {
          return std::abs(a.gid_) < std::abs(b.gid_);
        } else {
          return a.gid_ > b.gid_;
        }
      });
    auto last = std::unique(maxs.begin(), maxs.end());

    if(last->gid_ == -2) {
      last--;
    }
    // store the size to reuse later
    maxs[sizeSad].gid_ = std::distance(maxs.begin(), last);

    // remove "doughnut" configurations: two ascending separatrices
    // leading to the same maximum/boundary component
    if(maxs[sizeSad].gid_ != 2) {
      continue;
    }
    // TODO: modify for 2D (exclude saddle already paired?)
    saddles[i].lid_ = i;
    ttk::SimplexId gid = getSaddleGlobalId(s2);
    saddles[i].gid_ = gid;
    for(int j = 0; j < 2; j++) {
      if(maxs[j].gid_ != -1) {
        extremasGid.emplace_back(maxs[j].gid_);
      }
    }
  }
  TTK_PSORT(localThreadNumber, extremasGid.begin(), extremasGid.end());
  const auto last = std::unique(
    /*std::execution::par_unseq,*/ extremasGid.begin(), extremasGid.end());
  extremasGid.erase(last, extremasGid.end());
  globalToLocalExtrema.reserve(extremasGid.size());
  std::vector<std::vector<char>> ghostPresence(
    extremasGid.size(), std::vector<char>());
  std::vector<int> extremaLocks(extremasGid.size(), 0);
  std::vector<extremaNode<sizeExtr>> extremas(
    extremasGid.size(), extremaNode<sizeExtr>());
#pragma omp parallel master shared(extremaLocks, extremas, ghostPresence, \
                                   saddles) num_threads(localThreadNumber)
  {
#pragma omp task
    {
      for(ttk::SimplexId i = 0; i < extremasGid.size(); i++) {
        globalToLocalExtrema.emplace(extremasGid[i], i);
      }
    }
#pragma omp task
    {
      for(ttk::SimplexId i = 0; i < criticalSaddlesNumber; i++) {
        if(saddles[i].gid_ != -1) {
          globalToLocalSaddle.emplace(saddles[i].gid_, i);
        }
      }
    }
    int numTask = std::max(localThreadNumber - 2, 1);
#pragma omp taskloop num_tasks(numTask)
    for(ttk::SimplexId i = 0; i < criticalSaddlesNumber; ++i) {
      auto &maxs = saddle2ToMaxima[i];
      const auto s2 = criticalSaddles[i];
      // const auto last = std::unique(maxs.begin(), maxs.end());
      // mins.erase(last, mins.end());
      if(maxs[sizeSad].gid_ != 2) {
        continue;
      }
      saddleEdge<sizeSad> &e{saddles[i]};
      fillSaddleOrder(s2, e.vOrder_);
      e.order_ = critSaddlesOrder[s2];
      for(int j = 0; j < 2; j++) {
        if(maxs[j].gid_ > -1) {
          int extremaExists;
          ttk::SimplexId lid = std::lower_bound(extremasGid.begin(),
                                                extremasGid.end(), maxs[j].gid_)
                               - extremasGid.begin();
#pragma omp atomic capture
          extremaExists = extremaLocks[lid]++;
          if(extremaExists == 0) {
            std::vector<char> ghosts{};
            maxs[j].lid_ = lid;
            maxs[j].rep_.extremaId_ = lid;
            extremas[lid] = maxs[j];
            ttk::SimplexId triangLid
              = triangulation.getCellLocalId(maxs[j].gid_);
            if(triangLid == -1
               || triangulation.getCellRank(triangLid) != ttk::MPIrank_) {
              auto it = localGhostPresenceMap.find(maxs[j].gid_);
              if(it != localGhostPresenceMap.end()) {
                ghosts = localGhostPresenceMap[maxs[j].gid_];
              } else {
                ghosts.resize(0);
              }
            } else {
              triangLid = localTriangToLocalVectExtrema.find(triangLid)->second;
              ghosts = localGhostPresenceVector[triangLid];
            }
            ghostPresence[lid] = ghosts;
          }
          e.t_[j] = lid;
        }
      }
    }
  }

  const auto cmpSadMax
    = [this, &extremas, &saddles](
        const ttk::SimplexId &id1, const ttk::SimplexId &id2) -> bool {
    const auto &s0{saddles[id1]};
    const auto &s1{saddles[id2]};
    if(&s0 != &s1) {
      if(s0.order_ != -1 && s1.order_ != -1) {
        return s0.order_ > s1.order_;
      }
      for(size_t i = 0; i < sizeSad; i++) {
        if(s0.vOrder_[i] != s1.vOrder_[i]) {
          return s0.vOrder_[i] > s1.vOrder_[i];
        }
      }
      return s0.gid_ > s1.gid_;
    }
    if(s0.gid_ == -1 && s1.gid_ == -1) {
      return false;
    }
    if(s0.t_[1] != s1.t_[1]) {
      if(s0.t_[1] == -1) {
        return false;
      }
      if(s1.t_[1] == -1) {
        return true;
      }
      auto &t0 = extremas[s0.t_[1]];
      auto &t1 = extremas[s1.t_[1]];
      if(t0.order_ != -1 && t1.order_ != -1) {
        return t0.order_ < t1.order_;
      }
      for(size_t i = 0; i < sizeExtr; i++) {
        if(t0.vOrder_[i] != t1.vOrder_[i]) {
          return t0.vOrder_[i] < t1.vOrder_[i];
        }
      }
    }
    return true;
  };
  // TRI des arcs
  std::vector<ttk::SimplexId> saddleIds(saddles.size());
  std::iota(saddleIds.begin(), saddleIds.end(), 0);
  TTK_PSORT(localThreadNumber, saddleIds.begin(), saddleIds.end(), cmpSadMax);
  // Mise en place des lid des arcs
  extremaToPairedSaddle.resize(globalToLocalExtrema.size(), -1);
  saddleToPairedExtrema.resize(saddles.size(), -1);

  MPI_Datatype MPI_MessageType;
  createMPIMessageType<sizeExtr, sizeSad>(MPI_MessageType);
  tripletsToPersistencePairs<sizeExtr, sizeSad>(
    dim - 1, extremas, saddles, saddleIds, saddleToPairedExtrema,
    extremaToPairedSaddle, globalToLocalSaddle, globalToLocalExtrema,
    ghostPresence, MPI_MessageType, true, MPIcomm, localThreadNumber);
  ttk::SimplexId nMinSadPairs = pairs.size();
  extractPairs<sizeExtr, sizeSad>(pairs, extremas, saddles,
                                  saddleToPairedExtrema, true, dim - 1,
                                  localThreadNumber);
  ttk::SimplexId nSadMaxPairs = pairs.size() - nMinSadPairs;
  /*std::ofstream myfile;
  myfile.open("/home/eveleguillou/experiment/DiscreteMorseSandwich/"
              + std::to_string(ttk::MPIsize_) + "_pairs_"
              + std::to_string(ttk::MPIrank_) + ".csv");
  myfile << "sad,max\n";
  for(ttk::SimplexId i = 0; i < pairs.size(); i++) {
    myfile << std::to_string(pairs[i].birth) + ","
                + std::to_string(pairs[i].death) + "\n";
  }
  myfile.close();*/
  MPI_Allreduce(
    MPI_IN_PLACE, &nSadMaxPairs, 1, MPI_SimplexId, MPI_SUM, MPIcomm);

  if(ttk::MPIrank_ == 0)
    this->printMsg(
      "Computed " + std::to_string(nSadMaxPairs) + " saddle-max pairs", 1.0,
      tm.getElapsedTime(), localThreadNumber);
  /*this->printMsg("saddle-max pairs sequential part", 1.0,
                 tmseq.getElapsedTime(), 1, debug::LineMode::NEW);*/
}
template <typename triangulationType>
void ttk::DiscreteMorseSandwichMPI::getMaxSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const std::vector<SimplexId> &criticalSaddles,
  const std::vector<SimplexId> &critSaddlesOrder,
  const std::vector<ttk::SimplexId> &criticalExtremas,
  const std::vector<SimplexId> &critMaxsOrder,
  const triangulationType &triangulation,
  const bool ignoreBoundary,
  const SimplexId *const offsets,
  MPI_Comm &MPIcomm,
  int localThreadNumber) {
  Timer t{};
  const auto dim = this->dg_.getDimensionality();
  auto &globalToLocalSaddle{dim == 3 ? this->globalToLocalSaddle2_
                                     : this->globalToLocalSaddle1_};
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> globalToLocalExtrema{};

  ttk::SimplexId criticalExtremasNumber = criticalExtremas.size();
  ttk::SimplexId vertexNumber = triangulation.getNumberOfVertices();
  MPI_Datatype MPI_SimplexId = getMPIType(criticalExtremasNumber);
  std::vector<ttk::SimplexId> localMaxId;
  std::vector<ttk::SimplexId> globalMaxId;
  ttk::SimplexId globalMaxOffset{0};
  if(criticalExtremasNumber > 0) {
    // extracts the global max
#pragma omp parallel for reduction(max                \
                                   : globalMaxOffset) \
  num_threads(localThreadNumber)
    for(ttk::SimplexId i = 0; i < vertexNumber; i++) {
      if(globalMaxOffset < offsets[i]) {
        globalMaxOffset = offsets[i];
      }
    }
  }

  MPI_Allreduce(
    MPI_IN_PLACE, &globalMaxOffset, 1, MPI_SimplexId, MPI_MAX, MPIcomm);

#pragma omp declare reduction (merge : std::vector<ttk::SimplexId> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge         \
                                   : localMaxId) \
  num_threads(localThreadNumber)
  for(ttk::SimplexId i = 0; i < criticalExtremasNumber; i++) {
    if(triangulation.getCellRank(criticalExtremas[i]) == ttk::MPIrank_) {
      const Cell cmax{dim, criticalExtremas[i]};
      const auto vmax{this->getCellGreaterVertex(cmax, triangulation)};
      if(offsets[vmax] == globalMaxOffset - 1) { // TODO: pq -1?
        localMaxId.emplace_back(
          triangulation.getCellGlobalId(criticalExtremas[i]));
      }
    }
  }
  int localMaxIdSize = localMaxId.size();
  std::vector<int> recvCount(ttk::MPIsize_, 0);
  std::vector<int> displs(ttk::MPIsize_, 0);
  MPI_Allgather(
    &localMaxIdSize, 1, MPI_INTEGER, recvCount.data(), 1, MPI_INTEGER, MPIcomm);
  std::partial_sum(
    recvCount.data(), recvCount.data() + ttk::MPIsize_ - 1, displs.data() + 1);
  globalMaxId.resize(std::accumulate(recvCount.begin(), recvCount.end(), 0));
  MPI_Allgatherv(localMaxId.data(), localMaxIdSize, MPI_SimplexId,
                 globalMaxId.data(), recvCount.data(), displs.data(),
                 MPI_SimplexId, MPIcomm);
  if(dim > 1 && this->ComputeSadMax) {
    if(dim == 3) {
      computeMaxSaddlePairs<4, 3>(
        pairs, criticalSaddles, critSaddlesOrder, criticalExtremas,
        critMaxsOrder, triangulation, offsets, globalToLocalSaddle,
        globalToLocalExtrema,
        [&triangulation](const SimplexId a, const SimplexId i, SimplexId &r) {
          return triangulation.getTriangleStar(a, i, r);
        },
        [&triangulation](const SimplexId a) {
          return triangulation.getTriangleStarNumber(a);
        },
        [&triangulation](const SimplexId a) {
          return triangulation.isTriangleOnBoundary(a);
        },
        [this, &triangulation, offsets](
          const ttk::SimplexId id, ttk::SimplexId *vertsOrder) {
          return fillTetraOrder(id, offsets, triangulation, vertsOrder);
        },
        [&triangulation](ttk::SimplexId lid) {
          return triangulation.getTriangleGlobalId(lid);
        },
        [this, &triangulation, offsets](
          const ttk::SimplexId id, ttk::SimplexId *vOrd) {
          return fillTriangleOrder(id, offsets, triangulation, vOrd);
        },
        MPIcomm, localThreadNumber);
    } else {
      computeMaxSaddlePairs<3, 2>(
        pairs, criticalSaddles, critSaddlesOrder, criticalExtremas,
        critMaxsOrder, triangulation, offsets, globalToLocalSaddle,
        globalToLocalExtrema,
        [&triangulation](const SimplexId a, const SimplexId i, SimplexId &r) {
          return triangulation.getEdgeStar(a, i, r);
        },
        [&triangulation](const SimplexId a) {
          return triangulation.getEdgeStarNumber(a);
        },
        [&triangulation](const SimplexId a) {
          return triangulation.isEdgeOnBoundary(a);
        },
        [this, &triangulation, offsets](
          const ttk::SimplexId id, ttk::SimplexId *vertsOrder) {
          return fillTriangleOrder(id, offsets, triangulation, vertsOrder);
        },
        [&triangulation](ttk::SimplexId lid) {
          return triangulation.getEdgeGlobalId(lid);
        },
        [this, &triangulation, offsets](
          const ttk::SimplexId id, ttk::SimplexId *vOrd) {
          return fillEdgeOrder(id, offsets, triangulation, vOrd);
        },
        MPIcomm, localThreadNumber);
    }
  }
  if(ignoreBoundary) {
    std::vector<ttk::SimplexId> locatedId;
    // post-process saddle-max pairs: remove the one with the global
    // maximum (if it exists) to be (more) compatible with FTM
#pragma omp parallel for shared(saddleToPairedMax_, maxToPairedSaddle_) \
  num_threads(localThreadNumber)
    for(ttk::SimplexId i = 0; i < pairs.size(); i++) {
      if(pairs[i].type < dim - 1) {
        continue;
      }
      const auto it
        = std::find(globalMaxId.begin(), globalMaxId.end(), pairs[i].death);
      if(it != globalMaxId.end()) {
#pragma omp critical
        locatedId.emplace_back(i);
      }
    }
    // TODO: doesn't work in distributed
    for(ttk::SimplexId i = 0; i < locatedId.size(); i++) {
      ttk::SimplexId id = locatedId[i] - i;
      this->saddleToPairedMax_[globalToLocalSaddle[pairs[id].death]] = -1;
      this->maxToPairedSaddle_[globalToLocalExtrema[pairs[id].birth]] = -1;
      pairs.erase(pairs.begin() + id);
    }
  }
}

// get representative of current extremum
template <int sizeExtr, int sizeSad>
ttk::SimplexId ttk::DiscreteMorseSandwichMPI::getRep(
  extremaNode<sizeExtr> extr,
  saddleEdge<sizeSad> sv,
  bool increasing,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<saddleEdge<sizeSad>> &saddles) const {
  auto currentNode = extr;
  if(currentNode.rep_.extremaId_ == -1) {
    return currentNode.lid_;
  }
  int count{0};
  auto rep = extremas[extr.rep_.extremaId_];
  saddleEdge<sizeSad> s;
  while(rep != currentNode && count < 1000) {
    count++;
    if(currentNode.rep_.extremaId_ == -1) {
      break;
    }
    if(currentNode.rep_.saddleId_ > -1) {
      s = saddles[currentNode.rep_.saddleId_];
      if((s.gid_ == sv.gid_) || ((s < sv) == increasing)) {
        break;
      }
    }
    if(count >= 1000) {
      printMsg("Overflow reached for " + std::to_string(extr.gid_) + " ("
               + std::to_string(extr.rank_) + ") and " + std::to_string(sv.gid_)
               + " (" + std::to_string(sv.rank_) + ")");
    }
    currentNode = rep;
    if(currentNode.rep_.extremaId_ == -1) {
      break;
    }
    rep = extremas[currentNode.rep_.extremaId_];
  }
  /*if (sv.gid_ ==  4900){
    printMsg("final, for saddle: "+std::to_string(sv.gid_)+", original extr:
  "+std::to_string(extr.gid_)+", rep:
  "+std::to_string(currentNode.gid_)+"("+std::to_string(currentNode.rank_)+")");
  }*/
  return currentNode.lid_;
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::addPair(
  const saddleEdge<sizeSad> &sad,
  const extremaNode<sizeExtr> &extr,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle) const {
  /*if(sad.gid_ == 4900) {
    printMsg("AddPair: " + std::to_string(sad.gid_) + ", "
             + std::to_string(extr.gid_));
    //kill(getpid(), SIGINT);
  }*/
  saddleToPairedExtrema[sad.lid_] = extr.lid_;
  extremaToPairedSaddle[extr.lid_] = sad.lid_;
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::addToRecvBuffer(
  saddleEdge<sizeSad> &sad,
  std::set<messageType<sizeExtr, sizeSad>,
           std::function<bool(const messageType<sizeExtr, sizeSad> &,
                              const messageType<sizeExtr, sizeSad> &)>>
    &recomputations,
  const std::function<bool(const messageType<sizeExtr, sizeSad> &,
                           const messageType<sizeExtr, sizeSad> &)>
    &cmpMessages,
  std::vector<messageType<sizeExtr, sizeSad>> &recvBuffer,
  ttk::SimplexId beginVect) const {
  messageType<sizeExtr, sizeSad> m
    = messageType<sizeExtr, sizeSad>(sad.gid_, sad.vOrder_, sad.rank_);
  auto it = std::lower_bound(
    recvBuffer.begin() + beginVect, recvBuffer.end(), m, cmpMessages);
  if(it == recvBuffer.end() || it->s_ != m.s_) {
    recomputations.insert(m);
  } else {
    if(it->t1_ != -1) {
      it->t1_ = -1;
      it->t2_ = -1;
      for(int i = 0; i < sizeExtr; i++) {
        it->t1Order_[i] = 0;
        it->t2Order_[i] = 0;
      }
    }
  }
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::removePair(
  const saddleEdge<sizeSad> &sad,
  const extremaNode<sizeExtr> &extr,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle) const {
  /*if(sad.gid_ == 4900) {
    printMsg("removePair: " + std::to_string(sad.gid_) + ", "
             + std::to_string(extr.gid_));
  }*/
  if(extremaToPairedSaddle[extr.lid_] == sad.lid_) {
    extremaToPairedSaddle[extr.lid_] = -1;
  } else {
    printMsg("HAPPENING HERE FOR " + std::to_string(sad.gid_) + " and "
             + std::to_string(extr.gid_) + " (true: "
             + std::to_string(extremaToPairedSaddle.at(extr.lid_)) + ")");
    // extremaToPairedSaddle.at(saddleToPairedExtrema.at(sad.lid_)] = -1;
  }
  saddleToPairedExtrema[sad.lid_] = -1;
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::tripletsToPersistencePairs(
  const SimplexId pairDim,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<ttk::SimplexId> &saddleIds,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  std::vector<std::vector<char>> ghostPresence,
  MPI_Datatype &MPI_MessageType,
  bool isFirstTime,
  MPI_Comm &MPIcomm,
  int localThreadNumber) const {
  std::array<std::vector<std::vector<messageType<sizeExtr, sizeSad>>>, 2>
    sendBuffer;
  // recomputations.reserve(static_cast<ttk::SimplexId>(saddleIds.size() *
  // 0.2));
  sendBuffer[0].resize(
    ttk::MPIsize_, std::vector<messageType<sizeExtr, sizeSad>>());
  sendBuffer[1].resize(
    ttk::MPIsize_, std::vector<messageType<sizeExtr, sizeSad>>());
  const bool increasing = (pairDim > 0);
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> recvBuffer(
    ttk::MPIsize_, std::vector<messageType<sizeExtr, sizeSad>>());
  std::function<bool(const messageType<sizeExtr, sizeSad> &,
                     const messageType<sizeExtr, sizeSad> &)>
    cmpMessages;
  if(increasing) {
    cmpMessages = [=](const messageType<sizeExtr, sizeSad> &elt0,
                      const messageType<sizeExtr, sizeSad> &elt1) -> bool {
      if(elt0.s_ != elt1.s_) {
        for(int i = 0; i < sizeSad; i++) {
          if(elt0.sOrder_[i] != elt1.sOrder_[i]) {
            return elt0.sOrder_[i] > elt1.sOrder_[i];
          }
        }
      }
      return elt0.t1Order_[0] < elt1.t1Order_[0];
    };
  } else {
    cmpMessages = [=](const messageType<sizeExtr, sizeSad> &elt0,
                      const messageType<sizeExtr, sizeSad> &elt1) -> bool {
      if(elt0.s_ != elt1.s_) {
        for(int i = 0; i < sizeSad; i++) {
          if(elt0.sOrder_[i] != elt1.sOrder_[i]) {
            return elt0.sOrder_[i] < elt1.sOrder_[i];
          }
        }
      }
      return elt0.t1Order_[0] > elt1.t1Order_[0];
    };
  }
  std::set<messageType<sizeExtr, sizeSad>,
           std::function<bool(const messageType<sizeExtr, sizeSad> &,
                              const messageType<sizeExtr, sizeSad> &)>>
    recomputations(cmpMessages);
  if(isFirstTime) {
    for(const auto &sid : saddleIds) {
      if(saddles[sid].gid_ != -1) {
        processTriplet<sizeExtr, sizeSad>(
          saddles[sid], saddleToPairedExtrema, extremaToPairedSaddle, saddles,
          extremas, increasing, ghostPresence, sendBuffer[0], recomputations,
          cmpMessages, recvBuffer[ttk::MPIrank_], 0);
      }
    }
  } else {
    for(const auto &sid : saddleIds) {
      if(saddleToPairedExtrema[sid] == -2 && saddles[sid].gid_ != -1) {
        processTriplet<sizeExtr, sizeSad>(
          saddles[sid], saddleToPairedExtrema, extremaToPairedSaddle, saddles,
          extremas, increasing, ghostPresence, sendBuffer[0], recomputations,
          cmpMessages, recvBuffer[ttk::MPIrank_], 0);
      }
    }
  }

  const auto equalSadMin
    = [=](const messageType<sizeExtr, sizeSad> &elt0,
          const messageType<sizeExtr, sizeSad> &elt1) -> bool {
    return (elt0.s_ == elt1.s_) && (elt0.t1_ == elt1.t1_)
           && (elt0.t2_ == elt1.t2_) && (elt0.s1_ == elt1.s1_)
           && (elt0.s2_ == elt1.s2_)
           && (elt0.hasBeenModified_ == elt1.hasBeenModified_);
  };
  // Receive elements
  ttk::SimplexId hasSentMessages{10};
  MPI_Datatype MPI_SimplexId = getMPIType(hasSentMessages);
  char currentSendBuffer{0};
  while(hasSentMessages > 0) {
    ttk::SimplexId localSentMessageNumber{0};
    std::vector<MPI_Request> sendRequests(ttk::MPIsize_ - 1);
    std::vector<MPI_Request> recvRequests(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> sendStatus(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> recvStatus(ttk::MPIsize_ - 1);
    std::vector<ttk::SimplexId> sendMessageSize(ttk::MPIsize_, 0);
    std::vector<ttk::SimplexId> recvMessageSize(ttk::MPIsize_, 0);
    std::vector<int> recvCompleted(ttk::MPIsize_ - 1, 0);
    std::vector<int> sendCompleted(ttk::MPIsize_ - 1, 0);
    int recvPerformedCount = 0;
    int recvPerformedCountTotal = 0;
    for(int i = 0; i < ttk::MPIsize_; i++) {
      // Send size of Sendbuffer
      if(i != ttk::MPIrank_) {
        sendMessageSize[i] = sendBuffer[currentSendBuffer][i].size();
        localSentMessageNumber += sendMessageSize[i];
      }
    }
    MPI_Alltoall(sendMessageSize.data(), 1, MPI_SimplexId,
                 recvMessageSize.data(), 1, MPI_SimplexId, MPIcomm);
    std::vector<MPI_Request> sendRequestsData(ttk::MPIsize_ - 1);
    std::vector<MPI_Request> recvRequestsData(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> recvStatusData(ttk::MPIsize_ - 1);
    int recvCount = 0;
    int sendCount = 0;
    int r = 0;
    for(int i = 0; i < ttk::MPIsize_; i++) {
      if((sendMessageSize[i] > 0)) {
        MPI_Isend(sendBuffer[currentSendBuffer][i].data(), sendMessageSize[i],
                  MPI_MessageType, i, 1, MPIcomm, &sendRequestsData[sendCount]);
        sendCount++;
      }
      if((recvMessageSize[i] > 0)) {
        recvBuffer[i].resize(recvMessageSize[i]);
        MPI_Irecv(recvBuffer[i].data(), recvMessageSize[i], MPI_MessageType, i,
                  1, MPIcomm, &recvRequestsData[recvCount]);
        recvCount++;
      }
    }
    recvPerformedCountTotal = 0;
    while(recvPerformedCountTotal < recvCount) {
      MPI_Waitsome(recvCount, recvRequestsData.data(), &recvPerformedCount,
                   recvCompleted.data(), recvStatusData.data());

      if(recvPerformedCount > 0) {
        for(int i = 0; i < recvPerformedCount; i++) {
          // ttk::SimplexId sid{-1};
          r = recvStatusData[i].MPI_SOURCE;
          // recomputations.clear();
          TTK_PSORT(localThreadNumber, recvBuffer.at(r).begin(),
                    recvBuffer.at(r).end(), cmpMessages);
        }
        recvPerformedCountTotal += recvPerformedCount;
      }
    }
    MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);
    // Stop condition computation
    for(int i = 0; i < ttk::MPIsize_; i++) {
      ttk::SimplexId sid{-1};
      size_t j{0};
      // ttk::SimplexId recomp{0};
      recomputations.clear();
      while(j < recvBuffer.at(i).size()) {
        if((j == 0
            || !equalSadMin(
              recvBuffer.at(i).at(j), recvBuffer.at(i).at(j - 1)))) {
          messageType<sizeExtr, sizeSad> elt;
          if(!recomputations.empty()) {
            elt = (*recomputations.begin());
            if(cmpMessages(elt, recvBuffer.at(i).at(j))) {
              recomputations.erase(recomputations.begin());
            } else {
              elt = recvBuffer.at(i).at(j);
              j++;
            }
          } else {
            elt = recvBuffer.at(i).at(j);
            j++;
          }
          // Condition sur le premier élément de la liste
          if(elt.s_ != sid) {
            receiveElement<sizeExtr, sizeSad>(
              elt, globalToLocalSaddle, globalToLocalExtrema, saddles, extremas,
              extremaToPairedSaddle, saddleToPairedExtrema,
              sendBuffer[1 - currentSendBuffer], ghostPresence,
              static_cast<char>(i), increasing, recomputations, cmpMessages,
              recvBuffer[i], j + 1);
            if(elt.t1_ == -1) {
              sid = elt.s_;
            }
          }
        } else {
          j++;
        }
      }
      auto it = recomputations.begin();
      while(it != recomputations.end()) {
        receiveElement<sizeExtr, sizeSad>(
          (*it), globalToLocalSaddle, globalToLocalExtrema, saddles, extremas,
          extremaToPairedSaddle, saddleToPairedExtrema,
          sendBuffer[1 - currentSendBuffer], ghostPresence,
          static_cast<char>(i), increasing, recomputations, cmpMessages,
          recvBuffer[i], recvBuffer.at(i).size());
        it++;
      }
      recomputations.clear();
    }

    MPI_Allreduce(&localSentMessageNumber, &hasSentMessages, 1, MPI_SimplexId,
                  MPI_SUM, MPIcomm);
    if(hasSentMessages) {
      for(int i = 0; i < ttk::MPIsize_; i++) {
        sendBuffer[currentSendBuffer][i].clear();
        recvBuffer[i].clear();
      }
      currentSendBuffer = 1 - currentSendBuffer;
    }
  }
}

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::extractPairs(
  std::vector<PersistencePair> &pairs,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  bool increasing,
  const int pairDim,
  int localThreadNumber) const {
  ttk::SimplexId saddleNumber = saddleToPairedExtrema.size();

#ifdef TTK_ENABLE_OPENMP
#pragma omp declare reduction (merge : std::vector<PersistencePair> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge                     \
                                   : pairs) schedule(static) \
  num_threads(localThreadNumber)
#endif
  for(ttk::SimplexId i = 0; i < saddleNumber; i++) {
    if(saddleToPairedExtrema[i] > -1 && saddles[i].rank_ == ttk::MPIrank_) {
      if(increasing) {
        if(saddles[i].rank_ == ttk::MPIrank_) {
          pairs.emplace_back(
            saddles[i].gid_, extremas[saddleToPairedExtrema[i]].gid_, pairDim);
        }
      } else {
        if(saddles[i].rank_ == ttk::MPIrank_) {
          pairs.emplace_back(
            extremas[saddleToPairedExtrema[i]].gid_, saddles[i].gid_, pairDim);
        }
      }
    }
  }
}

template <int sizeExtr, int sizeSad>
ttk::SimplexId ttk::DiscreteMorseSandwichMPI::computePairNumbers(
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  int localThreadNumber) const {
  ttk::SimplexId saddleNumber = saddleToPairedExtrema.size();
  ttk::SimplexId computedSaddleNumber{0};
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for reduction(+ : computedSaddleNumber) schedule(static) num_threads(localThreadNumber)
#endif
  for(ttk::SimplexId i = 0; i < saddleNumber; i++) {
    if(saddleToPairedExtrema[i] > -1 && saddles[i].rank_ == ttk::MPIrank_) {
      computedSaddleNumber++;
    }
  }
  return computedSaddleNumber;
}

template <int sizeSad>
struct ttk::DiscreteMorseSandwichMPI::saddleEdge<sizeSad>
  ttk::DiscreteMorseSandwichMPI::addSaddle(
    saddleEdge<sizeSad> s,
    std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
    std::vector<saddleEdge<sizeSad>> &saddles,
    std::vector<ttk::SimplexId> &saddleToPairedExtrema) const {
  if(s.lid_ == -1 && s.gid_ > -1) {
    globalToLocalSaddle[s.gid_] = saddles.size();
    s.lid_ = saddles.size();
    saddles.emplace_back(s);
    s = saddles.back();
    saddleToPairedExtrema.emplace_back(static_cast<ttk::SimplexId>(-1));
  }
  return s;
};

template <int sizeExtr, int sizeSad>
ttk::SimplexId ttk::DiscreteMorseSandwichMPI::getUpdatedT1(
  const ttk::SimplexId extremaGid,
  messageType<sizeExtr, sizeSad> &elt,
  saddleEdge<sizeSad> s,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<saddleEdge<sizeSad>> &saddles,
  bool increasing) const {
  ttk::SimplexId lid{-1};
  if(extremaGid > -1) {
    auto it = globalToLocalExtrema.find(extremaGid);
    if(it != globalToLocalExtrema.end()) {
      lid = it->second;
      auto e{extremas[lid]};
      ttk::SimplexId repLid
        = getRep(extremas[lid], s, increasing, extremas, saddles);
      if(extremas[repLid].gid_ != e.gid_) {
        lid = repLid;
        elt.t1_ = extremas[repLid].gid_;
        for(int i = 0; i < sizeExtr; i++) {
          elt.t1Order_[i] = extremas[repLid].vOrder_[i];
        }
        elt.t1Rank_ = extremas[repLid].rank_;
        elt.hasBeenModified_ = 1;
        saddleEdge<sizeSad> s1;
        if(extremas[repLid].rep_.saddleId_ > -1) {
          s1 = saddles[extremas[repLid].rep_.saddleId_];
        }
        elt.s1Rank_ = s1.rank_;
        elt.s1_ = s1.gid_;
        for(int i = 0; i < sizeSad; i++) {
          elt.s1Order_[i] = s1.vOrder_[i];
        }
      } else {
        // Update s1 anyway
        if(elt.t1_ != -1) {
          saddleEdge<sizeSad> s1Loc;
          if(e.rep_.saddleId_ > -1) {
            s1Loc = saddles[e.rep_.saddleId_];
            if((elt.s1_ == -1 && s1Loc.gid_ != -1)
               || (compareArray(elt.s1Order_, s1Loc.vOrder_, sizeSad)
                   == increasing)) {
              elt.s1_ = s1Loc.gid_;
              elt.s1Rank_ = s1Loc.rank_;
              for(int i = 0; i < sizeSad; i++) {
                elt.s1Order_[i] = s1Loc.vOrder_[i];
              }
            }
          }
        }
      }
    }
  }
  return lid;
};

template <int sizeExtr, int sizeSad>
ttk::SimplexId ttk::DiscreteMorseSandwichMPI::getUpdatedT2(
  const ttk::SimplexId extremaGid,
  messageType<sizeExtr, sizeSad> &elt,
  saddleEdge<sizeSad> s,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<saddleEdge<sizeSad>> &saddles,
  bool increasing) const {
  ttk::SimplexId lid{-1};
  if(extremaGid > -1) {
    auto it = globalToLocalExtrema.find(extremaGid);
    if(it != globalToLocalExtrema.end()) {
      lid = it->second;
      auto e{extremas[lid]};
      ttk::SimplexId repLid = getRep(e, s, increasing, extremas, saddles);
      if(extremas[repLid].gid_ != e.gid_) {
        lid = repLid;
        elt.t2_ = extremas[repLid].gid_;
        for(int i = 0; i < sizeExtr; i++) {
          elt.t2Order_[i] = extremas[repLid].vOrder_[i];
        }
        elt.t2Rank_ = extremas[repLid].rank_;
        elt.hasBeenModified_ = 1;
        saddleEdge<sizeSad> s2;
        if(extremas[repLid].rep_.saddleId_ > -1) {
          s2 = saddles[extremas[repLid].rep_.saddleId_];
        }
        elt.s2Rank_ = s2.rank_;
        elt.s2_ = s2.gid_;
        for(int i = 0; i < sizeSad; i++) {
          elt.s2Order_[i] = s2.vOrder_[i];
        }
      } else {
        // Update s2 anyway
        if(elt.t2_ != -1) {
          saddleEdge<sizeSad> s2Loc;
          if(e.rep_.saddleId_ > -1) {
            s2Loc = saddles[e.rep_.saddleId_];
            if((elt.s2_ == -1 && s2Loc.gid_ != -1)
               || compareArray(elt.s2Order_, s2Loc.vOrder_, sizeSad)
                    == increasing) {
              elt.s2_ = s2Loc.gid_;
              elt.s2Rank_ = s2Loc.rank_;
              for(int i = 0; i < sizeSad; i++) {
                elt.s2Order_[i] = s2Loc.vOrder_[i];
              }
              // elt.hasBeenModified_ = 1;
            }
          }
        }
      }
    }
  }
  return lid;
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::swapT1T2(
  messageType<sizeExtr, sizeSad> &elt,
  ttk::SimplexId &t1Lid,
  ttk::SimplexId &t2Lid,
  bool increasing) const {
  if(elt.t1_ != elt.t2_ && elt.t2_ != -1) {
    bool pairedR2 = elt.s2_ != -1 && elt.s2_ != elt.s_;
    bool isR2Invalid
      = ((elt.s2_ != -1)
         && (compareArray(elt.s2Order_, elt.sOrder_, sizeSad) == increasing));
    if(isR2Invalid)
      pairedR2 = false;
    bool pairedR1 = elt.s1_ != -1 && elt.s1_ != elt.s_;
    bool isR1Invalid
      = ((elt.s1_ != -1)
         && (compareArray(elt.s1Order_, elt.sOrder_, sizeSad) == increasing));
    if(isR1Invalid)
      pairedR1 = false;
    if((((compareArray(elt.t2Order_, elt.t1Order_, sizeExtr)) == increasing)
        || pairedR1)
       && !pairedR2) {
      std::swap(t1Lid, t2Lid);
      std::swap(elt.t1_, elt.t2_);
      std::swap(elt.t1Order_, elt.t2Order_);
      std::swap(elt.s1Order_, elt.s2Order_);
      std::swap(elt.t1Rank_, elt.t2Rank_);
      std::swap(elt.s2Rank_, elt.s1Rank_);
      std::swap(elt.s1_, elt.s2_);
      elt.hasBeenModified_ = 1;
    }
  }
};

template <int sizeExtr>
void ttk::DiscreteMorseSandwichMPI::addLocalExtrema(
  ttk::SimplexId &lid,
  const ttk::SimplexId gid,
  char rank,
  ttk::SimplexId *vOrder,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle) const {
  if(lid == -1) {
    lid = extremas.size();
    extremaNode n
      = extremaNode<sizeExtr>(gid, lid, -1, Rep{-1, -2}, rank, vOrder);
    extremas.emplace_back(n);
    extremaToPairedSaddle.emplace_back(-1);
    globalToLocalExtrema[gid] = lid;
  }
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::receiveElement(
  messageType<sizeExtr, sizeSad> element,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  std::vector<std::vector<char>> &ghostPresence,
  char sender,
  bool increasing,
  std::set<messageType<sizeExtr, sizeSad>,
           std::function<bool(const messageType<sizeExtr, sizeSad> &,
                              const messageType<sizeExtr, sizeSad> &)>>
    &recomputations,
  const std::function<bool(const messageType<sizeExtr, sizeSad> &,
                           const messageType<sizeExtr, sizeSad> &)>
    &cmpMessages,
  std::vector<messageType<sizeExtr, sizeSad>> &recvBuffer,
  ttk::SimplexId beginVect) const {
  /*if(element.s_ == 71029541) {
    printMsg("ReceiveElement: " + std::to_string(element.s_) + ", "
             + std::to_string(element.t1_) + ", " + std::to_string(element.t2_)
             + " from " + std::to_string(sender));
  }*/
  struct saddleEdge<sizeSad> s;
  auto it = globalToLocalSaddle.find(element.s_);
  if(it != globalToLocalSaddle.end()) {
    s = saddles[it->second];
  } else {
    s = saddleEdge<sizeSad>(element.s_, -1, element.sOrder_, element.sRank_);
  }
  if(s.rank_ == ttk::MPIrank_ && element.t1_ == -1 && element.t2_ == -1) {
    if(saddleToPairedExtrema[s.lid_] > -1) {
      if(extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_ != -1) {
        extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
          = extremas[saddleToPairedExtrema[s.lid_]].lid_;
        extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_ = -1;
      }
      removePair(saddles[s.lid_], extremas[saddleToPairedExtrema[s.lid_]],
                 saddleToPairedExtrema, extremaToPairedSaddle);
    }
    processTriplet<sizeExtr>(
      saddles[s.lid_], saddleToPairedExtrema, extremaToPairedSaddle, saddles,
      extremas, increasing, ghostPresence, sendBuffer, recomputations,
      cmpMessages, recvBuffer, beginVect);
    return;
  }
  ttk::SimplexId t1Lid
    = getUpdatedT1(element.t1_, element, s, globalToLocalExtrema, extremas,
                   saddles, increasing);
  ttk::SimplexId t2Lid
    = getUpdatedT2(element.t2_, element, s, globalToLocalExtrema, extremas,
                   saddles, increasing);

  swapT1T2(element, t1Lid, t2Lid, increasing);
  /*if (element.s_ == 94981358){
    printMsg("UpdatedElement: "+std::to_string(element.s_)+",
  "+std::to_string(element.t1_)+", "+std::to_string(element.t2_));
  }
  if ((t1Lid > -1 ) && ( extremas.at(t1Lid).rep_.saddleId_ > -1) &&
  (element.s1_
  != saddles[ extremas.at(t1Lid).rep_.saddleId_].gid_)){ printMsg("Error: local
  link: "+std::to_string(saddles[ extremas.at(t1Lid).rep_.saddleId_].gid_)+",
  "+std::to_string(element.s1_));
  }
  if ((t2Lid > -1 ) && ( extremas.at(t2Lid).rep_.saddleId_ > -1) && (element.s2_
  != saddles[ extremas.at(t2Lid).rep_.saddleId_].gid_)){ printMsg("Error: local
  link:
  "+std::to_string(saddles[ extremas.at(t2Lid).rep_.saddleId_].gid_)+",
  "+std::to_string(element.s2_));
  }  */
  if(element.sRank_ != ttk::MPIrank_) {
    // Send now if the extrema rep is -1
    if(t1Lid == -1 || extremas[t1Lid].rep_.extremaId_ == -1) {
      element.hasBeenModified_ = 1;
      sendBuffer.at(element.t1Rank_).emplace_back(element);
    } else {
      // Send back to owner of s if hasBeenModified
      if(element.hasBeenModified_) {
        element.hasBeenModified_ = 0;
        sendBuffer.at(s.rank_).emplace_back(element);
        element.hasBeenModified_ = 1;
      }
      // If t1 present locally
      // Who is t1 paired with?
      ttk::SimplexId ls1Id = extremaToPairedSaddle[t1Lid];
      if(element.t1_ == element.t2_) {
        if(s.lid_ > -1 && saddleToPairedExtrema[s.lid_] > -1) {
          extremaToPairedSaddle[saddleToPairedExtrema[s.lid_]] = -1;
          extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
            = extremas[saddleToPairedExtrema[s.lid_]].lid_;
          extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_ = -1;
          if(element.t1Rank_ != ttk::MPIrank_ && element.t1Rank_ != sender) {
            sendBuffer.at(element.t1Rank_).emplace_back(element);
            } else {
              if(extremas[saddleToPairedExtrema[s.lid_]].rank_ == ttk::MPIrank_
                 && ghostPresence[extremas[saddleToPairedExtrema[s.lid_]].lid_]
                        .size()
                      > 1) {
                for(const auto rank :
                    ghostPresence[extremas[saddleToPairedExtrema[s.lid_]]
                                    .lid_]) {
                  if((rank != ttk::MPIrank_)
                     && ((rank == sender && element.hasBeenModified_)
                         || rank != sender)) {
                    sendBuffer.at(rank).emplace_back(element);
                  }
                }
              }
            }
            saddleToPairedExtrema[s.lid_] = static_cast<ttk::SimplexId>(-2);
        }
      } else {
        if(ls1Id != -1 && ls1Id != s.lid_) {
          // To s1:
          auto ls1{saddles[ls1Id]};
          // s1 is owned by the process
          if((ls1.gid_ != s.gid_) && ((ls1 < s) == increasing)) {
            // s1 is incorrect
            ttk::SimplexId oldSaddleId = extremas[t1Lid].rep_.saddleId_;
            removePair(ls1, extremas[t1Lid], saddleToPairedExtrema,
                       extremaToPairedSaddle);
            s = addSaddle(
              s, globalToLocalSaddle, saddles, saddleToPairedExtrema);
            if(saddleToPairedExtrema[s.lid_] > -1) {
              if(extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_ != -1
                 && extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_
                      == s.lid_) {
                extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
                  = extremas[saddleToPairedExtrema[s.lid_]].lid_;
                extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_ = -1;
              }
              removePair(s, extremas[saddleToPairedExtrema[s.lid_]],
                         saddleToPairedExtrema, extremaToPairedSaddle);
            }
            if(element.t2_ > -1) {
              addLocalExtrema(t2Lid, element.t2_, element.t2Rank_,
                              element.t2Order_, extremas, globalToLocalExtrema,
                              extremaToPairedSaddle);
            }
            addPair(
              s, extremas[t1Lid], saddleToPairedExtrema, extremaToPairedSaddle);
            if(element.t2_ > -1) {
              extremas[t1Lid].rep_.extremaId_ = t2Lid;
            } else {
              extremas[t1Lid].rep_.extremaId_ = t1Lid;
            }
            extremas[t1Lid].rep_.saddleId_ = s.lid_;
            // If t1 owned but with !ghostPresence.empty() -> send to
            // ghostPresence
            saddleEdge<sizeSad> lst1;
            if(extremas[t1Lid].rep_.saddleId_ > -1) {
              lst1 = saddles[extremas[t1Lid].rep_.saddleId_];
            } else {
              if(element.s1_ != -1) {
                lst1 = saddleEdge<sizeSad>(
                  element.s1_, element.s1Order_, element.s1Rank_);
              }
            }
            if(element.t2_ > -1) {
              saddleEdge<sizeSad> lst2;
              if(extremas[t2Lid].rep_.saddleId_ > -1) {
                lst2 = saddles[extremas[t2Lid].rep_.saddleId_];
              } else {
                if(element.s2_ != -1) {
                  lst2 = saddleEdge<sizeSad>(
                    element.s2_, element.s2Order_, element.s2Rank_);
                }
              }
              storeMessageToSend<sizeExtr, sizeSad>(
                ghostPresence, sendBuffer, s, lst1, lst2, extremas[t1Lid],
                extremas[t2Lid], sender, element.hasBeenModified_);
            } else {
              storeMessageToSend<sizeExtr, sizeSad>(
                ghostPresence, sendBuffer, s, lst1, extremas[t1Lid], sender,
                element.hasBeenModified_);
            }
            // Re-compute for incorrect s1
            if(saddles[ls1.lid_].rank_ == ttk::MPIrank_) {
              addToRecvBuffer(
                ls1, recomputations, cmpMessages, recvBuffer, beginVect);
            } else {
              // Send s1 for re-computation
              storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddleId]);
            }
          }
        } else {
          bool changesApplied{true};
          // Update rep + send update to other processes
          s = addSaddle(s, globalToLocalSaddle, saddles, saddleToPairedExtrema);
          if(saddleToPairedExtrema[s.lid_] > -1) {
            if((t2Lid > -1)
               && (extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
                   == t2Lid)
               && (saddleToPairedExtrema[s.lid_] == t1Lid)) {
              changesApplied = false;
            }
            if(extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_ != -1
               && extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_
                    == s.lid_) {
              extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
                = extremas[saddleToPairedExtrema[s.lid_]].lid_;
              extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_ = -1;
            }
            removePair(s, extremas[saddleToPairedExtrema[s.lid_]],
                       saddleToPairedExtrema, extremaToPairedSaddle);
          }
          if(element.t2_ > -1) {
            addLocalExtrema(t2Lid, element.t2_, element.t2Rank_,
                            element.t2Order_, extremas, globalToLocalExtrema,
                            extremaToPairedSaddle);
          }
          addPair(
            s, extremas[t1Lid], saddleToPairedExtrema, extremaToPairedSaddle);
          if(extremas[t1Lid].rep_.extremaId_ != -1) {
            if(element.t2_ > -1) {
              extremas[t1Lid].rep_.extremaId_ = t2Lid;
            } else {
              extremas[t1Lid].rep_.extremaId_ = t1Lid;
            }
            extremas[t1Lid].rep_.saddleId_ = s.lid_;
          }
          if(changesApplied) {
            // If t1 owned but with !ghostPresence.empty() -> send to
            // ghostPresence
            saddleEdge<sizeSad> lst1;
            if(extremas[t1Lid].rep_.saddleId_ > -1) {
              lst1 = saddles[extremas[t1Lid].rep_.saddleId_];
            } else {
              if(element.s1_ != -1) {
                lst1 = saddleEdge<sizeSad>(
                  element.s1_, element.s1Order_, element.s1Rank_);
              }
            }
            if(element.t2_ > -1) {
              saddleEdge<sizeSad> lst2;
              if(extremas[t2Lid].rep_.saddleId_ > -1) {
                lst2 = saddles[extremas[t2Lid].rep_.saddleId_];
              } else {
                if(element.s2_ != -1) {
                  lst2 = saddleEdge<sizeSad>(
                    element.s2_, element.s2Order_, element.s2Rank_);
                }
              }
              storeMessageToSend<sizeExtr, sizeSad>(
                ghostPresence, sendBuffer, s, lst1, lst2, extremas[t1Lid],
                extremas[t2Lid], sender, element.hasBeenModified_);
            } else {
              storeMessageToSend<sizeExtr, sizeSad>(
                ghostPresence, sendBuffer, s, lst1, extremas[t1Lid], sender,
                element.hasBeenModified_);
            }
          }
        }
      }
    }
  } else {
    if(((t1Lid != -1 && extremas[t1Lid].rep_.extremaId_ == -1) || t1Lid == -1)
       && element.hasBeenModified_ == 1) {
      sendBuffer.at(element.t1Rank_).emplace_back(element);
    } else {
      if(element.t1_ == element.t2_) {
        if(saddleToPairedExtrema[s.lid_] > -1) {
          extremaToPairedSaddle[saddleToPairedExtrema[s.lid_]] = -1;
          if(extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_ != -1) {
            extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
              = extremas[saddleToPairedExtrema[s.lid_]].lid_;
            extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_ = -1;
            }
            if(extremas[saddleToPairedExtrema[s.lid_]].rank_ == ttk::MPIrank_
               && ghostPresence[extremas[saddleToPairedExtrema[s.lid_]].lid_]
                      .size()
                    > 1) {
              for(const auto rank :
                  ghostPresence[extremas[saddleToPairedExtrema[s.lid_]].lid_]) {
                if((rank != ttk::MPIrank_)
                   && ((rank == sender && element.hasBeenModified_)
                       || rank != sender)) {
                  sendBuffer.at(rank).emplace_back(element);
                }
              }
          }
        }
        saddleToPairedExtrema[s.lid_] = static_cast<ttk::SimplexId>(-2);
      } else {
        if((element.s1_ != -1) && (element.s1_ != element.s_)
           && (!(compareArray(element.s1Order_, element.sOrder_, sizeSad)
                 == increasing))) {
          if(saddleToPairedExtrema[s.lid_] > -1) {
            extremaToPairedSaddle[saddleToPairedExtrema[s.lid_]] = -1;
            if(extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_ != -1) {
              extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
                = extremas[saddleToPairedExtrema[s.lid_]].lid_;
              extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_ = -1;
            }
            saddleToPairedExtrema[s.lid_] = static_cast<ttk::SimplexId>(-2);
          }
        } else {
          addLocalExtrema(t1Lid, element.t1_, element.t1Rank_, element.t1Order_,
                          extremas, globalToLocalExtrema,
                          extremaToPairedSaddle);

          bool isPairedWithWrongExtrema
            = ((saddleToPairedExtrema[s.lid_] > -1)
               && (saddleToPairedExtrema[s.lid_] != t1Lid));
          // TODO: test this
          if(((saddleToPairedExtrema[s.lid_] > -1)
              && (saddleToPairedExtrema[s.lid_] == t1Lid))
             && ((extremaToPairedSaddle[t1Lid] > -1)
                 && (extremaToPairedSaddle[t1Lid] == s.lid_))) {
            if(extremas[t1Lid].rep_.extremaId_ != -1
               && extremas[t1Lid].rep_.extremaId_ != t2Lid) {
              if(element.t2_ > -1) {
                addLocalExtrema(t2Lid, element.t2_, element.t2Rank_,
                                element.t2Order_, extremas,
                                globalToLocalExtrema, extremaToPairedSaddle);
                extremas[t1Lid].rep_.extremaId_ = t2Lid;
              } else {
                extremas[t1Lid].rep_.extremaId_ = t1Lid;
              }
            }
          }
          if(isPairedWithWrongExtrema) {
            if(extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_ != -1) {
              extremas[saddleToPairedExtrema[s.lid_]].rep_.extremaId_
                = extremas[saddleToPairedExtrema[s.lid_]].lid_;
              extremas[saddleToPairedExtrema[s.lid_]].rep_.saddleId_ = -1;
            }
            auto extr{extremas[saddleToPairedExtrema[s.lid_]]};
            removePair(s, extremas[saddleToPairedExtrema[s.lid_]],
                       saddleToPairedExtrema, extremaToPairedSaddle);
            if(extr.rank_ == ttk::MPIrank_
               && ghostPresence[extr.lid_].size() > 1) {
              for(const auto rank : ghostPresence[extr.lid_]) {
                if((rank != ttk::MPIrank_)
                   && ((rank == sender && element.hasBeenModified_)
                       || rank != sender)) {
                  sendBuffer.at(rank).emplace_back(element);
                }
              }
            }
          }
          // If extrema is not paired at all
          if(extremaToPairedSaddle[t1Lid] < 0) {
            addPair(
              s, extremas[t1Lid], saddleToPairedExtrema, extremaToPairedSaddle);
            ttk::SimplexId oldSaddle = extremas[t1Lid].rep_.saddleId_;
            if(element.t2_ > -1) {
              addLocalExtrema(t2Lid, element.t2_, element.t2Rank_,
                              element.t2Order_, extremas, globalToLocalExtrema,
                              extremaToPairedSaddle);
            }
            if(extremas[t1Lid].rep_.extremaId_ != -1) {
              if(element.t2_ > -1) {
                extremas[t1Lid].rep_.extremaId_ = t2Lid;
              } else {
                extremas[t1Lid].rep_.extremaId_ = t1Lid;
              }
              extremas[t1Lid].rep_.saddleId_ = s.lid_;
            }
            if(extremas[t1Lid].rank_ != ttk::MPIrank_
               || (ghostPresence[extremas[t1Lid].lid_].size() > 1)) {
              saddleEdge<sizeSad> ls1;
              saddleEdge<sizeSad> ls2;
              if(oldSaddle > -1) {
                ls1 = saddles[oldSaddle];
              }
              if(element.t2_ > -1) {
                if(extremas[t2Lid].rep_.saddleId_ > -1) {
                  ls2 = saddles[extremas[t2Lid].rep_.saddleId_];
                } else {
                  if(element.s2_ != -1) {
                    ls2 = saddleEdge<sizeSad>(
                      element.s2_, element.s2Order_, element.s2Rank_);
                  }
                }
                storeMessageToSend<sizeExtr, sizeSad>(
                  ghostPresence, sendBuffer, s, ls1, ls2, extremas[t1Lid],
                  extremas[t2Lid]);
              } else {
                storeMessageToSend<sizeExtr, sizeSad>(
                  ghostPresence, sendBuffer, s, ls1, extremas[t1Lid]);
              }
            }
          }
          bool isPairedWithWrongSaddle
            = ((extremaToPairedSaddle[t1Lid] > -1)
               && (extremaToPairedSaddle[t1Lid] != s.lid_));
          struct saddleEdge<sizeSad> sCurrent;
          if(extremaToPairedSaddle[t1Lid] > -1) {
            sCurrent = saddles[extremaToPairedSaddle[t1Lid]];
          }
          // extrema paired to wrong saddle
          if(isPairedWithWrongSaddle && ((sCurrent < s) == increasing)) {
            // If the paired extrema is the global min, re-computation is
            // triggered The new saddle is right, the old one is false
            removePair(sCurrent, extremas[t1Lid], saddleToPairedExtrema,
                       extremaToPairedSaddle);
          }
          if(isPairedWithWrongSaddle) {
            if((sCurrent < s) == increasing) {
              addPair(s, extremas[t1Lid], saddleToPairedExtrema,
                      extremaToPairedSaddle);
              ttk::SimplexId oldSaddle = extremas[t1Lid].rep_.saddleId_;
              if(element.t2_ > -1) {
                addLocalExtrema(t2Lid, element.t2_, element.t2Rank_,
                                element.t2Order_, extremas,
                                globalToLocalExtrema, extremaToPairedSaddle);
              }
              if(extremas[t1Lid].rep_.extremaId_ != -1) {
                if(element.t2_ > -1) {
                  extremas[t1Lid].rep_.extremaId_ = t2Lid;
                } else {
                  extremas[t1Lid].rep_.extremaId_ = t1Lid;
                }
                extremas[t1Lid].rep_.saddleId_ = s.lid_;
              }
              if(extremas[t1Lid].rank_ != ttk::MPIrank_
                 || (ghostPresence[extremas[t1Lid].lid_].size() > 1)) {
                saddleEdge<sizeSad> ls1;
                saddleEdge<sizeSad> ls2;
                if(oldSaddle > -1) {
                  ls1 = saddles[oldSaddle];
                }
                if(element.t2_ > -1) {
                  if(extremas[t2Lid].rep_.saddleId_ > -1) {
                    ls2 = saddles[extremas[t2Lid].rep_.saddleId_];
                  } else {
                    if(element.s2_ != -1) {
                      ls2 = saddleEdge<sizeSad>(
                        element.s2_, element.s2Order_, element.s2Rank_);
                    }
                  }
                  storeMessageToSend<sizeExtr, sizeSad>(
                    ghostPresence, sendBuffer, s, ls1, ls2, extremas[t1Lid],
                    extremas[t2Lid]);
                } else {
                  storeMessageToSend<sizeExtr, sizeSad>(
                    ghostPresence, sendBuffer, s, ls1, extremas[t1Lid]);
                }
              }
              if(sCurrent.gid_ > -1 && sCurrent.gid_ != s.gid_) {
                if(sCurrent.rank_ == ttk::MPIrank_) {
                  addToRecvBuffer(sCurrent, recomputations, cmpMessages,
                                  recvBuffer, beginVect);
                } else {
                  storeRerunToSend<sizeExtr>(sendBuffer, sCurrent);
                }
              }
            } else {
              if(s.rank_ == ttk::MPIrank_) {
                addToRecvBuffer(
                  s, recomputations, cmpMessages, recvBuffer, beginVect);
              } else {
                storeRerunToSend<sizeExtr>(sendBuffer, s);
              }
            }
          }
        }
      }
    }
  }
}

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::storeMessageToSend(
  std::vector<std::vector<char>> &ghostPresence,
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  saddleEdge<sizeSad> &sv,
  saddleEdge<sizeSad> &s1,
  extremaNode<sizeExtr> &rep1,
  char sender,
  char hasBeenModified) const {
  struct messageType<sizeExtr, sizeSad> m = messageType<sizeExtr, sizeSad>(
    rep1.gid_, rep1.vOrder_, sv.gid_, sv.vOrder_, s1.gid_, s1.vOrder_,
    rep1.rank_, sv.rank_, s1.rank_, 0);
  if(rep1.rank_ != ttk::MPIrank_ && (rep1.rank_ != sender || hasBeenModified)) {
    sendBuffer[rep1.rank_].emplace_back(m);
  } else {
    for(auto r : ghostPresence[rep1.lid_]) {
      sendBuffer[r].emplace_back(m);
    }
  }
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::storeMessageToSend(
  std::vector<std::vector<char>> &ghostPresence,
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  saddleEdge<sizeSad> &sv,
  saddleEdge<sizeSad> &s1,
  saddleEdge<sizeSad> &s2,
  extremaNode<sizeExtr> &rep1,
  extremaNode<sizeExtr> &rep2,
  char sender,
  char hasBeenModified) const {
  messageType<sizeExtr, sizeSad> m = messageType<sizeExtr, sizeSad>(
    rep1.gid_, rep1.vOrder_, rep2.gid_, rep2.vOrder_, sv.gid_, sv.vOrder_,
    s1.gid_, s1.vOrder_, s2.gid_, s2.vOrder_, rep1.rank_, rep2.rank_, sv.rank_,
    s1.rank_, s2.rank_, 0);
  if(rep1.rank_ != ttk::MPIrank_ && (rep1.rank_ != sender || hasBeenModified)) {
    sendBuffer[rep1.rank_].emplace_back(m);
  } else {
    for(auto r : ghostPresence[rep1.lid_]) {
      sendBuffer[r].emplace_back(m);
    }
  }
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::storeMessageToSendToRepOwner(
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  saddleEdge<sizeSad> &sv,
  std::vector<saddleEdge<sizeSad>> &saddles,
  extremaNode<sizeExtr> &rep1,
  extremaNode<sizeExtr> &rep2) const {
  saddleEdge<sizeSad> s1;
  saddleEdge<sizeSad> s2;
  if(rep1.rep_.saddleId_ > -1) {
    s1 = saddles[rep1.rep_.saddleId_];
  }
  if(rep2.rep_.saddleId_ > -1) {
    s2 = saddles[rep2.rep_.saddleId_];
  }
  messageType<sizeExtr, sizeSad> m = messageType<sizeExtr, sizeSad>(
    rep1.gid_, rep1.vOrder_, rep2.gid_, rep2.vOrder_, sv.gid_, sv.vOrder_,
    s1.gid_, s1.vOrder_, s2.gid_, s2.vOrder_, rep1.rank_, rep2.rank_, sv.rank_,
    s1.rank_, s2.rank_, 1);
  sendBuffer[rep1.rank_].emplace_back(m);
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::storeMessageToSendToRepOwner(
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  saddleEdge<sizeSad> &sv,
  std::vector<saddleEdge<sizeSad>> &saddles,
  extremaNode<sizeExtr> &rep1) const {
  saddleEdge<sizeSad> s1;
  if(rep1.rep_.saddleId_ > -1) {
    s1 = saddles[rep1.rep_.saddleId_];
  }
  struct messageType<sizeExtr, sizeSad> m = messageType<sizeExtr, sizeSad>(
    rep1.gid_, rep1.vOrder_, sv.gid_, sv.vOrder_, s1.gid_, s1.vOrder_,
    rep1.rank_, sv.rank_, s1.rank_, 1);
  sendBuffer[rep1.rank_].emplace_back(m);
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::storeRerunToSend(
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  saddleEdge<sizeSad> &sv) const {
  messageType<sizeExtr, sizeSad> m
    = messageType<sizeExtr, sizeSad>(sv.gid_, sv.vOrder_, sv.rank_);
  sendBuffer[sv.rank_].emplace_back(m);
};

template <int sizeExtr, int sizeSad>
int ttk::DiscreteMorseSandwichMPI::processTriplet(
  saddleEdge<sizeSad> sv,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle,
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<extremaNode<sizeExtr>> &extremas,
  bool increasing,
  std::vector<std::vector<char>> &ghostPresence,
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  std::set<messageType<sizeExtr, sizeSad>,
           std::function<bool(const messageType<sizeExtr, sizeSad> &,
                              const messageType<sizeExtr, sizeSad> &)>>
    &recomputations,
  const std::function<bool(const messageType<sizeExtr, sizeSad> &,
                           const messageType<sizeExtr, sizeSad> &)>
    &cmpMessages,
  std::vector<messageType<sizeExtr, sizeSad>> &recvBuffer,
  ttk::SimplexId beginVect) const {
  // TODO: enlever les .at
  // rep1 is either last correct in local or a ghost
  /*if(sv.gid_ == 4900 || sv.gid_ == 4701) {
    printMsg("ProcessTriplet: " + std::to_string(sv.gid_)+",
  "+std::to_string(extremas[sv.t_[0]].gid_)+",
  "+std::to_string(extremas[sv.t_[1]].gid_));
    //kill(getpid(), SIGINT);
  }*/
  ttk::SimplexId r1Lid
    = getRep(extremas[sv.t_[0]], sv, increasing, extremas, saddles);

  bool pairedR1 = extremaToPairedSaddle[r1Lid] > -1;
  bool isR1Invalid
    = (pairedR1
       && ((saddles[extremaToPairedSaddle[r1Lid]] < sv) == increasing));
  ttk::SimplexId oldSaddle{-1};
  if(isR1Invalid)
    pairedR1 = false;
  if(sv.t_[1] < 0) {
    // deal with "shadow" triplets (a 2-saddle with only one
    // ascending 1-separatrix leading to an unique maximum)
    if(extremas[r1Lid].rep_.extremaId_ == -1) {
      // Send to owner to continue computation
      storeMessageToSendToRepOwner(sendBuffer, sv, saddles, extremas[r1Lid]);
      return 0;
    }
    if(!pairedR1) {

      // when considering the boundary, the "-1" of the triplets
      // indicate a virtual maximum of infinite persistence on the
      // boundary component. a pair is created with the other
      // maximum
      if(isR1Invalid) {
        oldSaddle = extremaToPairedSaddle[r1Lid];
        removePair(saddles[extremaToPairedSaddle[r1Lid]], extremas[r1Lid],
                   saddleToPairedExtrema, extremaToPairedSaddle);
      }
      addPair(
        sv, extremas[r1Lid], saddleToPairedExtrema, extremaToPairedSaddle);
      // If extrema is has local id, then is present in local TODO: CAREFUL:
      // NOT TRUE extrema can be present in triangulation but not graph
      if(extremas[r1Lid].rank_ != ttk::MPIrank_
         || (ghostPresence[r1Lid].size() > 1)) {
        saddleEdge<sizeSad> s1;
        if(oldSaddle > -1) {
          s1 = saddles[oldSaddle];
        }
        storeMessageToSend<sizeExtr, sizeSad>(
          ghostPresence, sendBuffer, sv, s1, extremas[r1Lid]);
      }
      // If ghost or on the border: send message, do what happens next?
      if(extremas[r1Lid].rep_.extremaId_ != -1) {
        extremas[r1Lid].rep_.extremaId_ = r1Lid;
        extremas[r1Lid].rep_.saddleId_ = sv.lid_;
      }
      if(isR1Invalid) {
        if(saddles[oldSaddle].rank_ == ttk::MPIrank_) {
          addToRecvBuffer(saddles[oldSaddle], recomputations, cmpMessages,
                          recvBuffer, beginVect);
        } else {
          storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddle]);
        }
      }
    }
    return 0;
  }
  ttk::SimplexId r2Lid
    = getRep(extremas[sv.t_[1]], sv, increasing, extremas, saddles);
  bool pairedR2 = extremaToPairedSaddle[r2Lid] > -1;
  bool isR2Invalid
    = (pairedR2
       && ((saddles[extremaToPairedSaddle[r2Lid]] < sv) == increasing));
  if(isR2Invalid)
    pairedR2 = false;
  if(extremas[r2Lid].rep_.extremaId_ == -1) {
    // Send to owner to continue computation
    storeMessageToSendToRepOwner(
      sendBuffer, sv, saddles, extremas[r2Lid], extremas[r1Lid]);
    return 0;
  } else {
    if(extremas[r1Lid].rep_.extremaId_ == -1) {
      // Send to owner to continue computation
      storeMessageToSendToRepOwner(
        sendBuffer, sv, saddles, extremas[r1Lid], extremas[r2Lid]);
      return 0;
    }
  }
  if(extremas[r1Lid].gid_ != extremas[r2Lid].gid_) {
    if((((extremas[r2Lid] < extremas[r1Lid]) == increasing) || pairedR1)
       && !pairedR2) {
      if(isR2Invalid) {
        oldSaddle = extremaToPairedSaddle[r2Lid];
        removePair(saddles[extremaToPairedSaddle[r2Lid]], extremas[r2Lid],
                   saddleToPairedExtrema, extremaToPairedSaddle);
      }
      addPair(
        sv, extremas[r2Lid], saddleToPairedExtrema, extremaToPairedSaddle);
      extremas[r2Lid].rep_.extremaId_ = r1Lid;
      extremas[r2Lid].rep_.saddleId_ = sv.lid_;
      // send to other processes if necessary
      if(extremas[r2Lid].rank_ != ttk::MPIrank_
         || (ghostPresence[r2Lid].size() > 1)) {
        saddleEdge<sizeSad> s1;
        saddleEdge<sizeSad> s2;
        if(extremaToPairedSaddle[r1Lid] > -1) {
          s1 = saddles[extremaToPairedSaddle[r1Lid]];
        }
        if(oldSaddle > -1) {
          s2 = saddles[oldSaddle];
        }
        storeMessageToSend<sizeExtr, sizeSad>(ghostPresence, sendBuffer, sv, s2,
                                              s1, extremas[r2Lid],
                                              extremas[r1Lid]);
      }
      if(isR2Invalid) {
        if(saddles[oldSaddle].rank_ == ttk::MPIrank_
           && saddles[oldSaddle].gid_ != sv.gid_) {
          addToRecvBuffer(saddles[oldSaddle], recomputations, cmpMessages,
                          recvBuffer, beginVect);
        } else {
          storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddle]);
        }
      }
    } else {
      if(!pairedR1) {
        if(isR1Invalid) {
          oldSaddle = extremaToPairedSaddle[r1Lid];
          removePair(saddles[extremaToPairedSaddle[r1Lid]], extremas[r1Lid],
                     saddleToPairedExtrema, extremaToPairedSaddle);
        }
        addPair(
          sv, extremas[r1Lid], saddleToPairedExtrema, extremaToPairedSaddle);
        extremas[r1Lid].rep_.extremaId_ = r2Lid;
        extremas[r1Lid].rep_.saddleId_ = sv.lid_;
        if(extremas[r1Lid].rank_ != ttk::MPIrank_
           || (ghostPresence[r1Lid].size() > 1)) {
          saddleEdge<sizeSad> s1;
          saddleEdge<sizeSad> s2;
          if(oldSaddle > -1) {
            s1 = saddles[oldSaddle];
          }
          if(extremaToPairedSaddle[r2Lid] > -1) {
            s2 = saddles[extremaToPairedSaddle[r2Lid]];
          }
          storeMessageToSend<sizeExtr, sizeSad>(ghostPresence, sendBuffer, sv,
                                                s1, s2, extremas[r1Lid],
                                                extremas[r2Lid]);
        }
        if(isR1Invalid) {
          if(saddles[oldSaddle].rank_ == ttk::MPIrank_
             && saddles[oldSaddle].gid_ != sv.gid_) {
            addToRecvBuffer(saddles[oldSaddle], recomputations, cmpMessages,
                            recvBuffer, beginVect);
          } else {
            storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddle]);
          }
        }
      }
    }
  } else {
    if(saddleToPairedExtrema[sv.lid_] > -1) {
      extremaToPairedSaddle[saddleToPairedExtrema[sv.lid_]] = -1;
      extremas[saddleToPairedExtrema[sv.lid_]].rep_.extremaId_
        = extremas[saddleToPairedExtrema[sv.lid_]].lid_;
      extremas[saddleToPairedExtrema[sv.lid_]].rep_.saddleId_ = -1;
      if(extremas[saddleToPairedExtrema[sv.lid_]].rank_ != ttk::MPIrank_
         || (ghostPresence[extremas[saddleToPairedExtrema[sv.lid_]].lid_].size()
             > 1)) {
        saddleEdge<sizeSad> s1;
        if(extremaToPairedSaddle[saddleToPairedExtrema[sv.lid_]] > -1) {
          s1 = saddles[extremaToPairedSaddle[saddleToPairedExtrema[sv.lid_]]];
        }
        storeMessageToSend<sizeExtr, sizeSad>(
          ghostPresence, sendBuffer, sv, s1, s1,
          extremas[saddleToPairedExtrema[sv.lid_]],
          extremas[saddleToPairedExtrema[sv.lid_]]);
      }
      saddleToPairedExtrema[sv.lid_] = static_cast<ttk::SimplexId>(-2);
    }
  }

  return 0;
};

template <typename GlobalBoundary>
void ttk::DiscreteMorseSandwichMPI::updateMax(
  maxPerProcess m, GlobalBoundary &maxBoundary) const {
  auto it = std::find(maxBoundary.begin(), maxBoundary.end(), m);
  if(it != maxBoundary.end()) {
    maxBoundary.erase(it);
  }
  maxBoundary.insert(m);
};

template <typename GlobalBoundary>
void ttk::DiscreteMorseSandwichMPI::getMaxOfProc(
  ttk::SimplexId rank,
  ttk::SimplexId *currentMax,
  GlobalBoundary &maxBoundary) const {
  auto it
    = std::find(maxBoundary.begin(), maxBoundary.end(), maxPerProcess(rank));
  if(it != maxBoundary.end()) {
    currentMax[0] = it->max_[0];
    currentMax[1] = it->max_[1];
  }
};

template <typename GlobalBoundary, typename LocalBoundary>
bool ttk::DiscreteMorseSandwichMPI::isEmpty(
  LocalBoundary &localBoundary, GlobalBoundary &globalBoundary) const {
  if(localBoundary.empty()) {
    for(const auto elt : globalBoundary) {
      if(elt.max_[0] != -1) {
        return false;
      }
    }
  } else {
    return false;
  }
  return true;
};

template <typename LocalBoundary>
bool ttk::DiscreteMorseSandwichMPI::addBoundary(
  const SimplexId e, bool isOnBoundary, LocalBoundary &localBoundary) const {
  // add edge e to boundaryIds/onBoundary modulo 2
  if(!isOnBoundary) {
    localBoundary.emplace(e);
    isOnBoundary = true;
  } else {
    const auto it = localBoundary.find(e);
    localBoundary.erase(it);
    isOnBoundary = false;
  }
  return isOnBoundary;
};

template <typename GlobalBoundary>
void ttk::DiscreteMorseSandwichMPI::updateMaxBoundary(
  std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
  const SimplexId s2,
  const ttk::SimplexId *tauOrder,
  GlobalBoundary &globalBoundary,
  ttk::SimplexId rank) const {
  std::vector<ttk::SimplexId> vect(5);
  if(s2 == 9244) {
    printMsg("Update max for " + std::to_string(s2));
    kill(getpid(), SIGINT);
  }
  vect[0] = -4;
  vect[1] = s2;
  vect[2] = rank;
  vect[3] = tauOrder[0];
  vect[4] = tauOrder[1];
  for(const auto max : globalBoundary) {
    if(max.proc_ != rank) {
      sendBoundaryBuffer.at(max.proc_).insert(
        sendBoundaryBuffer.at(max.proc_).end(), vect.begin(), vect.end());
    }
  }
}

template <typename GlobalBoundary>
void ttk::DiscreteMorseSandwichMPI::updateLocalBoundary(
  std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
  const saddle<3> &s2,
  const SimplexId egid1,
  const SimplexId egid2,
  const ttk::SimplexId *tauOrder,
  GlobalBoundary &globalBoundary,
  ttk::SimplexId rank) const {
  std::vector<ttk::SimplexId> vect(10);
  vect[1] = s2.gid_;
  for(int i = 0; i < 3; i++) {
    vect[2 + i] = s2.vOrder_[i];
  }
  vect[5] = egid1;
  vect[6] = egid2;
  vect[7] = ttk::MPIrank_;
  vect[8] = tauOrder[0];
  vect[9] = tauOrder[1];
  for(const auto max : globalBoundary) {
    vect.emplace_back(max.proc_);
    vect.emplace_back(max.max_[0]);
    vect.emplace_back(max.max_[1]);
  }
  vect[0] = -(vect.size() - 1);
  sendBoundaryBuffer.at(rank).insert(
    sendBoundaryBuffer.at(rank).end(), vect.begin(), vect.end());
}

template <typename LocalBoundary, typename GlobalBoundary>
bool ttk::DiscreteMorseSandwichMPI::mergeGlobalBoundaries(
  std::vector<bool> &onBoundary,
  LocalBoundary &s2LocalBoundary,
  GlobalBoundary &s2GlobalBoundary,
  LocalBoundary &pTauLocalBoundary,
  GlobalBoundary &pTauGlobalBoundary) const {
  for(const auto e : pTauLocalBoundary) {
    onBoundary[e] = addBoundary(e, onBoundary[e], s2LocalBoundary);
  }
  bool hasChanged{false};
  for(const auto &m : pTauGlobalBoundary) {
    auto it = std::find(s2GlobalBoundary.begin(), s2GlobalBoundary.end(), m);
    if(it != s2GlobalBoundary.end()) {
      if(compareArray(m.max_, it->max_, 2)) {
        hasChanged = true;
        s2GlobalBoundary.erase(it);
        s2GlobalBoundary.insert(m);
      }
    } else {
      s2GlobalBoundary.insert(m);
      hasChanged = true;
    }
  }
  return hasChanged;
}

template <typename GlobalBoundary>
void ttk::DiscreteMorseSandwichMPI::updateMergedBoundary(
  std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
  const saddle<3> &s2,
  const SimplexId pTau,
  const ttk::SimplexId *tauOrder,
  GlobalBoundary &globalBoundary) const {
  std::vector<ttk::SimplexId> vect(10);
  vect[1] = s2.gid_;
  for(int i = 0; i < 3; i++) {
    vect[2 + i] = s2.vOrder_[i];
  }
  vect[5] = -1;
  vect[6] = pTau;
  vect[7] = ttk::MPIrank_;
  vect[8] = tauOrder[0];
  vect[9] = tauOrder[1];
  for(const auto max : globalBoundary) {
    vect.emplace_back(max.proc_);
    vect.emplace_back(max.max_[0]);
    vect.emplace_back(max.max_[1]);
  }
  vect[0] = -(vect.size() - 1);
  for(const auto max : globalBoundary) {
    sendBoundaryBuffer.at(max.proc_).insert(
      sendBoundaryBuffer.at(max.proc_).end(), vect.begin(), vect.end());
  }
}

template <typename triangulationType,
          typename GlobalBoundary,
          typename LocalBoundary>
void ttk::DiscreteMorseSandwichMPI::addEdgeToBoundary(
  const ttk::SimplexId s2Gid,
  const ttk::SimplexId pTau,
  const ttk::SimplexId edgeId,
  std::vector<bool> &onBoundary,
  GlobalBoundary &globalBoundaryIds,
  LocalBoundary &localBoundaryIds,
  const triangulationType &triangulation,
  const SimplexId *const offsets,
  std::vector<std::pair<ttk::SimplexId, ttk::SimplexId>> &ghostEdges,
  std::vector<bool> &hasChangedMax) const {
  SimplexId e{};
  triangulation.getTriangleEdge(pTau, edgeId, e);
  ttk::SimplexId rank = triangulation.getEdgeRank(e);
  if(rank == ttk::MPIrank_) {
    onBoundary[e] = addBoundary(e, onBoundary[e], localBoundaryIds);
    if(s2Gid == 8813) {
      ttk::SimplexId order[] = {-1, -1};
      fillEdgeOrder(e, offsets, triangulation, order);
      printMsg("add edge :" + std::to_string(triangulation.getEdgeGlobalId(e))
               + ", with order: " + std::to_string(order[0]) + ", "
               + std::to_string(order[1]));
    }

  } else {
    if(s2Gid == 8813) {
      ttk::SimplexId order[] = {-1, -1};
      fillEdgeOrder(e, offsets, triangulation, order);
      printMsg("add ghost edge :"
               + std::to_string(triangulation.getEdgeGlobalId(e))
               + ", with order: " + std::to_string(order[0]) + ", "
               + std::to_string(order[1]));
      // kill(getpid(), SIGINT);
    }
    ghostEdges.emplace_back(std::make_pair(e, rank));
    hasChangedMax.emplace_back(false);
    ttk::SimplexId eOrder[2];
    fillEdgeOrder(e, offsets, triangulation, eOrder);
    ttk::SimplexId currentMax[] = {-1, -1};
    getMaxOfProc(rank, currentMax, globalBoundaryIds);
    if(currentMax[0] != -1) {
      if(compareArray(currentMax, eOrder, 2)) { // TODO: correct comp?
        updateMax(maxPerProcess(rank, eOrder), globalBoundaryIds);
        hasChangedMax[hasChangedMax.size() - 1] = true;
      }
    } else {
      updateMax(maxPerProcess(rank, eOrder), globalBoundaryIds);
      hasChangedMax[hasChangedMax.size() - 1] = true;
    }
  }
};

template <typename triangulationType, typename GlobalBoundary>
void ttk::DiscreteMorseSandwichMPI::packageLocalBoundaryUpdate(
  const saddle<3> &s2,
  ttk::SimplexId *tauOrder,
  GlobalBoundary &globalBoundaryIds,
  const triangulationType &triangulation,
  std::vector<std::pair<ttk::SimplexId, ttk::SimplexId>> &ghostEdges,
  std::vector<bool> &hasChangedMax,
  std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer) const {
  switch(ghostEdges.size()) {
    case 0:
      break;
    case 1:
      updateLocalBoundary(sendBoundaryBuffer, s2,
                          triangulation.getEdgeGlobalId(ghostEdges[0].first),
                          -1, tauOrder, globalBoundaryIds,
                          ghostEdges[0].second);
      if(hasChangedMax[0]) { // Add test if other rank of propagation (outside
                             // current and updatedLocalBoundary)
        ttk::SimplexId currentMax[] = {-1, -1};
        getMaxOfProc(ghostEdges[0].second, currentMax, globalBoundaryIds);
        updateMaxBoundary(sendBoundaryBuffer, s2.gid_, currentMax,
                          globalBoundaryIds, ghostEdges[0].second);
      }
      break;
    case 2:
      if(ghostEdges[0].second == ghostEdges[1].second) {
        updateLocalBoundary(sendBoundaryBuffer, s2,
                            triangulation.getEdgeGlobalId(ghostEdges[0].first),
                            triangulation.getEdgeGlobalId(ghostEdges[1].first),
                            tauOrder, globalBoundaryIds, ghostEdges[0].second);
        if(hasChangedMax[0] || hasChangedMax[1]) {
          ttk::SimplexId currentMax[] = {-1, -1};
          getMaxOfProc(ghostEdges[0].second, currentMax, globalBoundaryIds);
          updateMaxBoundary(sendBoundaryBuffer, s2.gid_, currentMax,
                            globalBoundaryIds, ghostEdges[0].second);
        }
      } else {
        updateLocalBoundary(sendBoundaryBuffer, s2,
                            triangulation.getEdgeGlobalId(ghostEdges[0].first),
                            -1, tauOrder, globalBoundaryIds,
                            ghostEdges[0].second);
        if(hasChangedMax[0]) {
          ttk::SimplexId currentMax[] = {-1, -1};
          getMaxOfProc(ghostEdges[0].second, currentMax, globalBoundaryIds);
          updateMaxBoundary(sendBoundaryBuffer, s2.gid_, currentMax,
                            globalBoundaryIds, ghostEdges[0].second);
        }
        updateLocalBoundary(sendBoundaryBuffer, s2,
                            triangulation.getEdgeGlobalId(ghostEdges[1].first),
                            -1, tauOrder, globalBoundaryIds,
                            ghostEdges[1].second);
        if(hasChangedMax[1]) {
          ttk::SimplexId currentMax[] = {-1, -1};
          getMaxOfProc(ghostEdges[1].second, currentMax, globalBoundaryIds);
          updateMaxBoundary(sendBoundaryBuffer, s2.gid_, currentMax,
                            globalBoundaryIds, ghostEdges[1].second);
        }
      }
      break;
    case 3:
      printErr("NOT SUPPOSED TO BE HERE");
      // printErr("Rank of pTau:
      // "+std::to_string(triangulation.getTriangleRank(pTau)));
      kill(getpid(), SIGINT);
  }
};

template <typename triangulationType,
          typename GlobalBoundary,
          typename LocalBoundary>
SimplexId ttk::DiscreteMorseSandwichMPI::eliminateBoundariesSandwich(
  const saddle<3> &s2,
  std::vector<bool> &onBoundary,
  std::vector<GlobalBoundary> &s2GlobalBoundaries,
  std::vector<LocalBoundary> &s2LocalBoundaries,
  std::vector<SimplexId> &partners,
  std::vector<int> &s1Locks,
  std::vector<int> &s2Locks,
  const std::vector<saddle<2>> &saddles1,
  const std::vector<saddle<3>> &saddles2,
  const triangulationType &triangulation,
  const SimplexId *const offsets,
  std::vector<std::vector<ttk::SimplexId>> &sendBoundaryBuffer,
  std::vector<std::vector<ttk::SimplexId>> &sendComputeBuffer) const {
  int lock;
  auto &localBoundaryIds{s2LocalBoundaries[s2.lid_]};
  auto &globalBoundaryIds{s2GlobalBoundaries[s2.lid_]};
  if(s2.gid_ == 8813) {
    printMsg("Process " + std::to_string(s2.gid_));
    // kill(getpid(), SIGINT);
  }
  const auto clearOnBoundary = [&localBoundaryIds, &onBoundary]() {
    // clear the onBoundary vector (set everything to false)
    for(const auto e : localBoundaryIds) {
      onBoundary[e] = false;
    }
  };

  if(!isEmpty(localBoundaryIds, globalBoundaryIds)) {
    // restore previously computed s2 boundary
    for(const auto e : localBoundaryIds) {
      onBoundary[e] = true;
    }
  } else {
    // init cascade with s2 triangle boundary (3 edges)
    std::vector<std::pair<ttk::SimplexId, ttk::SimplexId>> ghostEdges;
    std::vector<bool> hasChangedMax;
    for(ttk::SimplexId i = 0; i < 3; ++i) {
      addEdgeToBoundary(s2.gid_, triangulation.getTriangleLocalId(s2.gid_), i,
                        onBoundary, globalBoundaryIds, localBoundaryIds,
                        triangulation, offsets, ghostEdges, hasChangedMax);
    }
    if(ghostEdges.size() > 0) {
      ttk::SimplexId tauOrder[2] = {-1, -1};
      // tau: youngest edge on boundary
      const auto tau{*localBoundaryIds.begin()};
      fillEdgeOrder(tau, offsets, triangulation, tauOrder);
      this->packageLocalBoundaryUpdate(s2, tauOrder, globalBoundaryIds,
                                       triangulation, ghostEdges, hasChangedMax,
                                       sendBoundaryBuffer);
    }
  }

  // lock the 2-saddle to ensure that only one thread can perform the
  // boundary expansion
  do {
#pragma omp atomic capture
    {
      lock = s2Locks[s2.lid_];
      s2Locks[s2.lid_] = 1;
    };
  } while(lock == 1);
  // kill(getpid(), SIGINT);
  while(!isEmpty(localBoundaryIds, globalBoundaryIds)) {
    ttk::SimplexId tauOrder[2] = {-1, -1};
    // tau: youngest edge on boundary
    const auto tau{*localBoundaryIds.begin()};
    fillEdgeOrder(tau, offsets, triangulation, tauOrder);
    if(s2.gid_ == 8813) {
      printMsg("Selected tau: "
               + std::to_string(triangulation.getEdgeGlobalId(tau)));
    }
    if(triangulation.getEdgeRank(tau) != ttk::MPIrank_) {
      printMsg("ERROR HERE for tau "
               + std::to_string(triangulation.getEdgeGlobalId(tau)));
      kill(getpid(), SIGINT);
    }
    if(!globalBoundaryIds.empty()) {
      const auto globMax{*globalBoundaryIds.begin()};
      if(!compareArray(globMax.max_, tauOrder, 2)) {
        sendComputeBuffer.at(globMax.proc_).emplace_back(s2.gid_);
        updateMaxBoundary(sendBoundaryBuffer, s2.gid_, tauOrder,
                          globalBoundaryIds, ttk::MPIrank_);
        clearOnBoundary();
#pragma omp atomic write
        s2Locks[s2.lid_] = 0;
        return 0;
      }
    }

    // use the Discrete Gradient to find a triangle paired to tau
    auto pTau{this->dg_.getPairedCell(Cell{1, tau}, triangulation)};
    // pTau is a regular triangle
    // add pTau triangle boundary (3 edges)
    if(pTau != -1) {
      std::vector<std::pair<ttk::SimplexId, ttk::SimplexId>> ghostEdges;
      std::vector<bool> hasChangedMax;
      std::vector<ttk::SimplexId> newMax;
      for(SimplexId i = 0; i < 3; ++i) {
        this->addEdgeToBoundary(s2.gid_, pTau, i, onBoundary, globalBoundaryIds,
                                localBoundaryIds, triangulation, offsets,
                                ghostEdges, hasChangedMax);
      }
      this->packageLocalBoundaryUpdate(s2, tauOrder, globalBoundaryIds,
                                       triangulation, ghostEdges, hasChangedMax,
                                       sendBoundaryBuffer);
    }
    bool critical{false};
    ttk::SimplexId saddleTau{-1};
    if(pTau == -1) {
      // TODO: make it usafe and fast
      ttk::SimplexId gid = triangulation.getEdgeGlobalId(tau);
      auto it = globalToLocalSaddle1_.find(gid);
      if(globalToLocalSaddle1_.end() != it) {
        saddleTau = it->second;
      } else {
        printErr("PROBLEM HERE");
      }
      // maybe tau is critical and paired to a critical triangle
      // TODO: works in distributed?
      do {
#ifdef TTK_ENABLE_OPENMP
#pragma omp atomic read
#endif // TTK_ENABLE_OPENMP
        pTau = partners[saddleTau];
        if(pTau == -1 || s2LocalBoundaries[pTau].empty()) {
          break;
        }
      } while(*s2LocalBoundaries[pTau].begin() != tau);

      critical = true;
    }
    if(pTau == -1) {
      // tau is critical and not paired

      // compare-and-swap from "Towards Lockfree Persistent Homology"
      // using locks over 1-saddles instead of atomics (OpenMP compatibility)
      do {
#pragma omp atomic capture
        {
          lock = s1Locks[saddleTau];
          s1Locks[saddleTau] = 1;
        };
      } while(lock == 1);

      const auto cap = partners[saddleTau];
      if(partners[saddleTau] == -1) {
        partners[saddleTau] = s2.lid_;
      }
#pragma omp atomic write
      s1Locks[saddleTau] = 0;
      // cleanup before exiting
      clearOnBoundary();
#pragma omp atomic write
      s2Locks[s2.lid_] = 0;
      if(cap == -1) {
        // Update global boundary
        updateMaxBoundary(sendBoundaryBuffer, s2.gid_, tauOrder,
                          globalBoundaryIds, ttk::MPIrank_);
        clearOnBoundary();
        return tau;
      } else {
        clearOnBoundary();
        return this->eliminateBoundariesSandwich(
          s2, onBoundary, s2GlobalBoundaries, s2LocalBoundaries, partners,
          s1Locks, s2Locks, saddles1, saddles2, triangulation, offsets,
          sendBoundaryBuffer, sendComputeBuffer);
      }

    } else {
      // expand boundary
      if(critical) {
        if(saddles2[pTau] < s2) {
          // pTau is an already-paired 2-saddle
          // merge pTau boundary into s2 boundary

          // make sure that pTau boundary is not modified by another
          // thread while we merge the two boundaries...
          do {
#pragma omp atomic capture
            {
              lock = s2Locks[pTau];
              s2Locks[pTau] = 1;
            };
          } while(lock == 1);
          mergeGlobalBoundaries(onBoundary, localBoundaryIds, globalBoundaryIds,
                                s2LocalBoundaries[pTau],
                                s2GlobalBoundaries[pTau]);
          updateMergedBoundary(
            sendBoundaryBuffer, s2, pTau, tauOrder, globalBoundaryIds);
#pragma omp atomic write
          s2Locks[pTau] = 0;
          /*if(this->Compute2SaddlesChildren) {
            this->s2Children_[s2Mapping[s2]].emplace_back(s2Mapping[pTau]);
          }*/

        } else if(saddles2[pTau] > s2) {

          // compare-and-swap from "Towards Lockfree Persistent
          // Homology" using locks over 1-saddles
          do {
#pragma omp atomic capture
            {
              lock = s1Locks[saddleTau];
              s1Locks[saddleTau] = 1;
            };
          } while(lock == 1);

          const auto cap = partners[saddleTau];
          if(partners[saddleTau] == pTau) {
            partners[saddleTau] = s2.lid_;
          }
#pragma omp atomic write
          s1Locks[saddleTau] = 0;
          updateMaxBoundary(sendBoundaryBuffer, s2.gid_, tauOrder,
                            globalBoundaryIds, ttk::MPIrank_);
          if(cap == pTau) {
            // cleanup before exiting
            clearOnBoundary();
#pragma omp atomic write
            s2Locks[s2.lid_] = 0;
            return this->eliminateBoundariesSandwich(
              saddles2[pTau], onBoundary, s2GlobalBoundaries, s2LocalBoundaries,
              partners, s1Locks, s2Locks, saddles1, saddles2, triangulation,
              offsets, sendBoundaryBuffer, sendComputeBuffer);
          }
        }
      }
    }
  }

  // cleanup before exiting
  clearOnBoundary();
#pragma omp atomic write
  s2Locks[s2.lid_] = 0;
  return -1;
}

template <typename triangulationType,
          typename GlobalBoundary,
          typename LocalBoundary,
          typename compareEdges>
void ttk::DiscreteMorseSandwichMPI::receiveBoundaryUpdate(
  std::vector<ttk::SimplexId> &recvBoundaryBuffer,
  std::vector<int> &s2Locks,
  std::vector<GlobalBoundary> &globalBoundaries,
  std::vector<LocalBoundary> &localBoundaries,
  std::vector<saddle<3>> &saddles2,
  triangulationType &triangulation,
  compareEdges &cmpEdges) const {
  std::vector<ttk::SimplexId> newGids;
#pragma omp declare reduction (merge : std::vector<ttk::SimplexId>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for num_threads(threadNumber_) schedule(static) \
  reduction(merge                                                    \
            : newGids)
  for(ttk::SimplexId i = 0; i < recvBoundaryBuffer.size(); i++) {
    if(recvBoundaryBuffer[i] < -1) {
      ttk::SimplexId size = -recvBoundaryBuffer[i];
      if(size > 4) {
        auto it = globalToLocalSaddle2_.find(recvBoundaryBuffer[i + 1]);
        if(it == globalToLocalSaddle2_.end()) {
          newGids.emplace_back(recvBoundaryBuffer[i + 1]);
        }
      }
    }
  }
  ttk::SimplexId currentLastElement = saddles2.size();
  TTK_PSORT(this->threadNumber_, newGids.begin(), newGids.end());
  auto last = std::unique(newGids.begin(), newGids.end());
  newGids.erase(last, newGids.end());
  ttk::SimplexId newGidSize = newGids.size();
  printMsg("new gid: " + std::to_string(newGidSize));
#pragma omp parallel master
  {
#pragma omp task
    for(ttk::SimplexId i = 0; i < newGidSize; i++) {
      globalToLocalSaddle2_[newGids[i]] = currentLastElement + i;
    }
#pragma omp task
    saddles2.resize(currentLastElement + newGidSize);
#pragma omp task
    s2Locks.resize(currentLastElement + newGidSize, 0);
  }
  globalBoundaries.resize(currentLastElement
                          + newGidSize); // TODO: put in task?
  localBoundaries.resize(
    currentLastElement + newGidSize, LocalBoundary(cmpEdges));
  /*#pragma omp parallel for num_threads(threadNumber_) schedule(static) \
    shared(s2Locks, saddles2, globalBoundaries, localBoundaries)*/
  for(ttk::SimplexId i = 0; i < recvBoundaryBuffer.size(); i++) {
    if(recvBoundaryBuffer[i] < -1) {
      int lock;
      if(recvBoundaryBuffer[i + 1] == 9244) {
        printMsg("ReceiveBoundary");
        // kill(getpid(), SIGINT);
      }
      ttk::SimplexId size = -recvBoundaryBuffer[i];
      ttk::SimplexId lid
        = globalToLocalSaddle2_.find(recvBoundaryBuffer[i + 1])->second;
      // This is a simple max update
      if(size == 4) {
        ttk::SimplexId rank = recvBoundaryBuffer[i + 2];
        ttk::SimplexId newMax[]
          = {recvBoundaryBuffer[i + 3], recvBoundaryBuffer[i + 4]};
        // atomic lock
        do {
#pragma omp atomic capture
          {
            lock = s2Locks[lid];
            s2Locks[lid] = 1;
          };
        } while(lock == 1);
        // globalBoundaries[lid].erase(maxPerProcess(rank));
        // insert should erase the existing one
        globalBoundaries[lid].insert(maxPerProcess(rank, newMax));
#pragma omp atomic write
        s2Locks[lid] = 0;
      } else {
        // This is either a merge order or a addition of local edges
        if(recvBoundaryBuffer[i + 5] == -1) {
          // This is a merge order
          ttk::SimplexId pTau = recvBoundaryBuffer[i + 6];
          auto it = globalToLocalSaddle2_.find(pTau);
          if(it != globalToLocalSaddle2_.end()) {
            // pTau is present on this process, therefore both the local and
            // global boundary need to be updated
            ttk::SimplexId pTauLid = it->second;
            // s is present
            do {
#pragma omp atomic capture
              {
                lock = s2Locks[lid];
                s2Locks[lid] = 1;
              };
            } while(lock == 1);
            saddle<3> &s{saddles2[lid]};
            GlobalBoundary &globalBoundary = globalBoundaries[lid];
            for(int j = 7; j < size; j += 3) {
              if(recvBoundaryBuffer[i + j] != ttk::MPIrank_) {
                ttk::SimplexId newMax[] = {
                  recvBoundaryBuffer[i + j + 1], recvBoundaryBuffer[i + j + 2]};
                globalBoundary.emplace(
                  maxPerProcess(recvBoundaryBuffer[i + j], newMax));
              }
            }
            do {
#pragma omp atomic capture
              {
                lock = s2Locks[pTauLid];
                s2Locks[pTauLid] = 1;
              };
            } while(lock == 1);

            auto &pTauLocalBoundary = localBoundaries[pTauLid];
            if(s.lid_ != -1) {
              auto &s2LocalBoundary = localBoundaries[s.lid_];
              for(const auto e : pTauLocalBoundary) {
                auto ite = s2LocalBoundary.find(e);
                if(ite == s2LocalBoundary.end()) {
                  s2LocalBoundary.emplace(e);
                } else {
                  s2LocalBoundary.erase(ite);
                }
              }
            } else {
              localBoundaries[s.lid_].merge(pTauLocalBoundary);
            }
#pragma omp atomic write
            s2Locks[pTauLid] = 0;
            if(s.gid_ == -1) {
              s.gid_ = recvBoundaryBuffer[i + 1];
              for(int j = 0; j < 3; j++) {
                s.vOrder_[j] = recvBoundaryBuffer[i + 2 + j];
              }
              s.lid_ = lid;
            }
#pragma omp atomic write
            s2Locks[lid] = 0;
          } else {
            // pTau is not present on this process, therefore only the global
            // boundary needs to be update s is necessarily present
            do {
#pragma omp atomic capture
              {
                lock = s2Locks[lid];
                s2Locks[lid] = 1;
              };
            } while(lock == 1);

            saddle<3> &s{saddles2[lid]};
            if(s.gid_ == -1) {
              s.gid_ = recvBoundaryBuffer[i + 1];
              for(int j = 0; j < 3; j++) {
                s.vOrder_[j] = recvBoundaryBuffer[i + 2 + j];
              }
              s.lid_ = lid;
            }
            GlobalBoundary &globalBoundary = globalBoundaries[lid];
            for(int j = 7; j < size; j += 3) {
              if(recvBoundaryBuffer[i + j] != ttk::MPIrank_) {
                ttk::SimplexId newMax[] = {
                  recvBoundaryBuffer[i + j + 1], recvBoundaryBuffer[i + j + 2]};
                globalBoundary.emplace(
                  maxPerProcess(recvBoundaryBuffer[i + j], newMax));
              }
            }
#pragma omp atomic write
            s2Locks[lid] = 0;
          }
        } else {
          // This is an addition of local edges
          ttk::SimplexId leid1
            = triangulation.getEdgeLocalId(recvBoundaryBuffer[i + 5]);
          ttk::SimplexId leid2
            = triangulation.getEdgeLocalId(recvBoundaryBuffer[i + 6]);
          do {
#pragma omp atomic capture
            {
              lock = s2Locks[lid];
              s2Locks[lid] = 1;
            };
          } while(lock == 1);
          saddle<3> &s{saddles2[lid]};
          LocalBoundary &localBoundary = localBoundaries[lid];
          localBoundary.emplace(leid1);
          if(leid2 != -1) {
            localBoundary.emplace(leid2);
          }
          if(s.lid_ == -1) {
            GlobalBoundary &globalBoundary = globalBoundaries[lid];
            for(int j = 7; j < size; j += 3) {
              if(recvBoundaryBuffer[i + j] != ttk::MPIrank_) {
                ttk::SimplexId newMax[] = {
                  recvBoundaryBuffer[i + j + 1], recvBoundaryBuffer[i + j + 2]};
                globalBoundary.emplace(
                  maxPerProcess(recvBoundaryBuffer[i + j], newMax));
              }
            }
            s.gid_ = recvBoundaryBuffer[i + 1];
            for(int j = 0; j < 3; j++) {
              s.vOrder_[j] = recvBoundaryBuffer[i + 2 + j];
            }
            s.lid_ = lid;
          }
#pragma omp atomic write
          s2Locks[lid] = 0;
        }
      }
    }
  }
}

template <typename triangulationType>
void ttk::DiscreteMorseSandwichMPI::getSaddleSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const bool exportBoundaries,
  std::vector<GeneratorType> &boundaries,
  const std::vector<SimplexId> &critical1Saddles,
  const std::vector<SimplexId> &critical2Saddles,
  const std::vector<SimplexId> &crit1SaddlesOrder,
  const std::vector<SimplexId> &crit2SaddlesOrder,
  const triangulationType &triangulation,
  const SimplexId *const offsets) const {

  Timer tm2{};
  const auto nSadExtrPairs = pairs.size();

  // 1- and 2-saddles yet to be paired
  std::vector<SimplexId> saddles1Gid{}, saddles2Gid{};
  // filter out already paired 1-saddles (edge id)

#pragma omp declare reduction (merge : std::vector<ttk::SimplexId>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : saddles1Gid) schedule(static)
  for(size_t i = 0; i < critical1Saddles.size(); i++) {
    const auto s1 = critical1Saddles[i];
    ttk::SimplexId gid = triangulation.getEdgeGlobalId(s1);
    auto it = globalToLocalSaddle1_.find(triangulation.getEdgeGlobalId(s1));
    if(it == globalToLocalSaddle1_.end()) {
      saddles1Gid.emplace_back(gid);
    } else {
      if(saddleToPairedMin_[it->second] < 0) {
        saddles1Gid.emplace_back(gid);
      }
    }
  }

#pragma omp parallel for reduction(merge : saddles2Gid) schedule(static)
  for(size_t i = 0; i < critical2Saddles.size(); i++) {
    const auto s2 = critical2Saddles[i];
    ttk::SimplexId gid = triangulation.getTriangleGlobalId(s2);
    auto it = globalToLocalSaddle2_.find(triangulation.getTriangleGlobalId(s2));
    if(it == globalToLocalSaddle2_.end()) {
      saddles2Gid.emplace_back(gid);
    } else {
      if(saddleToPairedMax_[it->second] < 0) {
        saddles2Gid.emplace_back(gid);
      }
    }
  }
  printMsg("Unpaired saddles 1 and 2 extracted");
  globalToLocalSaddle1_.clear();
  globalToLocalSaddle2_.clear();
  std::vector<saddle<2>> saddles1(saddles1Gid.size());
  std::vector<saddle<3>> saddles2(saddles2Gid.size());
  for(size_t i = 0; i < saddles1.size(); i++) {
    globalToLocalSaddle1_.emplace(saddles1Gid[i], i);
  }
  for(size_t i = 0; i < saddles2.size(); i++) {
    globalToLocalSaddle2_.emplace(saddles2Gid[i], i);
  }
  // kill(getpid(), SIGINT);
#pragma omp parallel for num_threads(threadNumber_) schedule(static)
  for(size_t i = 0; i < saddles1.size(); i++) {
    auto &s1{saddles1[i]};
    s1.gid_ = saddles1Gid[i];
    ttk::SimplexId lid = triangulation.getEdgeLocalId(s1.gid_);
    s1.lid_ = i;
    s1.order_ = crit1SaddlesOrder[lid];
    fillEdgeOrder(lid, offsets, triangulation, s1.vOrder_);
  }

#pragma omp parallel for num_threads(threadNumber_) schedule(static)
  for(size_t i = 0; i < saddles2.size(); i++) {
    auto &s2{saddles2[i]};
    s2.gid_ = saddles2Gid[i];
    ttk::SimplexId lid = triangulation.getTriangleLocalId(s2.gid_);
    s2.lid_ = i;
    s2.order_ = crit2SaddlesOrder[lid];
    fillTriangleOrder(lid, offsets, triangulation, s2.vOrder_);
  }
  // sort every triangulation edges by filtration order
  const auto &edgesFiltrOrder{crit1SaddlesOrder};

  auto &onBoundary{this->onBoundary_};
  auto &edgeTrianglePartner{this->edgeTrianglePartner_};
  edgeTrianglePartner.resize(saddles1.size(), -1);
  std::function<bool(const ttk::SimplexId, const ttk::SimplexId)> cmpEdges
    = [&edgesFiltrOrder](const ttk::SimplexId a, const ttk::SimplexId b) {
        return edgesFiltrOrder[a] > edgesFiltrOrder[b];
      };
  const auto cmpMaxPerProcess
    = [](const maxPerProcess a, const maxPerProcess b) {
        if(a.proc_ == b.proc_) {
          return false;
        }
        for(int i = 0; i < 2; i++) {
          if(a.max_[i] != b.max_[i]) {
            return a.max_[i] < b.max_[i];
          }
        }
        return false;
      };

  using GlobalBoundary = std::set<maxPerProcess, std::less<>>;
  using LocalBoundary = std::set<ttk::SimplexId, decltype(cmpEdges)>;

  std::vector<GlobalBoundary> s2GlobalBoundaries(saddles2.size());
  std::vector<LocalBoundary> s2LocalBoundaries(1, LocalBoundary(cmpEdges));
  s2LocalBoundaries.resize(saddles2.size(), LocalBoundary(cmpEdges));

  // one lock per 1-saddle
  std::vector<int> s1Locks(saddles1.size(), 0);
  // one lock per 2-saddle
  std::vector<int> s2Locks(saddles2.size(), 0);
  std::vector<std::vector<ttk::SimplexId>> sendBoundaryBuffer(ttk::MPIsize_);
  std::vector<std::vector<ttk::SimplexId>> sendComputeBuffer(ttk::MPIsize_);
  // compute 2-saddles boundaries in parallel
  for(ttk::SimplexId i = 0; i < critical1Saddles.size(); i++) {
    if(triangulation.getEdgeGlobalId(critical1Saddles[i]) == 4324) {
      printMsg("HERE");
    }
  }
  printMsg("Preproc done");
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic) \
  firstprivate(onBoundary) shared(s1Locks, s2Locks)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < saddles2.size(); ++i) {
    // 2-saddles sorted in increasing order
    const auto &s2 = saddles2[i];
    this->eliminateBoundariesSandwich(
      s2, onBoundary, s2GlobalBoundaries, s2LocalBoundaries,
      edgeTrianglePartner, s1Locks, s2Locks, saddles1, saddles2, triangulation,
      offsets, sendBoundaryBuffer, sendComputeBuffer);
  }

  // Start communication phase
  ttk::SimplexId hasSentMessages{10};
  MPI_Datatype MPI_SimplexId = getMPIType(hasSentMessages);
  std::vector<std::vector<ttk::SimplexId>> recvBoundaryBuffer(ttk::MPIsize_);
  std::vector<std::vector<ttk::SimplexId>> recvComputeBuffer(ttk::MPIsize_);
  while(hasSentMessages > 0) {
    printMsg("Communication phase");
    ttk::SimplexId localSentMessageNumber{0};
    std::vector<MPI_Request> sendRequests(ttk::MPIsize_ - 1);
    std::vector<MPI_Request> recvRequests(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> sendStatus(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> recvStatus(ttk::MPIsize_ - 1);
    std::vector<ttk::SimplexId> sendMessageSize(ttk::MPIsize_, 0);
    std::vector<ttk::SimplexId> recvMessageSize(ttk::MPIsize_, 0);
    std::vector<int> recvCompleted(ttk::MPIsize_ - 1, 0);
    std::vector<int> sendCompleted(ttk::MPIsize_ - 1, 0);
    int recvPerformedCount = 0;
    int recvPerformedCountTotal = 0;
    for(int i = 0; i < ttk::MPIsize_; i++) {
      // Send size of Sendbuffer
      if(i != ttk::MPIrank_) {
        sendMessageSize[i] = sendBoundaryBuffer[i].size();
        localSentMessageNumber += sendMessageSize[i];
      }
    }
    MPI_Alltoall(sendMessageSize.data(), 1, MPI_SimplexId,
                 recvMessageSize.data(), 1, MPI_SimplexId, ttk::MPIcomm_);
    std::vector<MPI_Request> sendRequestsData(ttk::MPIsize_ - 1);
    std::vector<MPI_Request> recvRequestsData(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> recvStatusData(ttk::MPIsize_ - 1);
    int recvCount = 0;
    int sendCount = 0;
    int r = 0;
    for(int i = 0; i < ttk::MPIsize_; i++) {
      if((sendMessageSize[i] > 0)) {
        MPI_Isend(sendBoundaryBuffer[i].data(), sendMessageSize[i],
                  MPI_SimplexId, i, 1, ttk::MPIcomm_,
                  &sendRequestsData[sendCount]);
        sendCount++;
      }
      if((recvMessageSize[i] > 0)) {
        recvBoundaryBuffer[i].resize(recvMessageSize[i]);
        MPI_Irecv(recvBoundaryBuffer[i].data(), recvMessageSize[i],
                  MPI_SimplexId, i, 1, ttk::MPIcomm_,
                  &recvRequestsData[recvCount]);
        recvCount++;
      }
    }
    recvPerformedCountTotal = 0;
    while(recvPerformedCountTotal < recvCount) {
      MPI_Waitsome(recvCount, recvRequestsData.data(), &recvPerformedCount,
                   recvCompleted.data(), recvStatusData.data());

      if(recvPerformedCount > 0) {
        for(int i = 0; i < recvPerformedCount; i++) {
          r = recvStatusData[i].MPI_SOURCE;
          receiveBoundaryUpdate(recvBoundaryBuffer[r], s2Locks,
                                s2GlobalBoundaries, s2LocalBoundaries, saddles2,
                                triangulation, cmpEdges);
        }
        recvPerformedCountTotal += recvPerformedCount;
      }
    }
    MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);
    printMsg("Update boundaries");
    // Exchange computation signals
    recvPerformedCount = 0;
    recvPerformedCountTotal = 0;
    for(int i = 0; i < ttk::MPIsize_; i++) {
      // Send size of Sendbuffer
      if(i != ttk::MPIrank_) {
        sendMessageSize[i] = sendComputeBuffer[i].size();
        localSentMessageNumber += sendMessageSize[i];
      }
    }
    MPI_Alltoall(sendMessageSize.data(), 1, MPI_SimplexId,
                 recvMessageSize.data(), 1, MPI_SimplexId, ttk::MPIcomm_);
    recvCount = 0;
    sendCount = 0;
    for(int i = 0; i < ttk::MPIsize_; i++) {
      if((sendMessageSize[i] > 0)) {
        MPI_Isend(sendComputeBuffer[i].data(), sendMessageSize[i],
                  MPI_SimplexId, i, 1, ttk::MPIcomm_,
                  &sendRequestsData[sendCount]);
        sendCount++;
      }
      if((recvMessageSize[i] > 0)) {
        recvComputeBuffer[i].resize(recvMessageSize[i]);
        MPI_Irecv(recvComputeBuffer[i].data(), recvMessageSize[i],
                  MPI_SimplexId, i, 1, ttk::MPIcomm_,
                  &recvRequestsData[recvCount]);
        recvCount++;
      }
    }
    recvPerformedCountTotal = 0;
    while(recvPerformedCountTotal < recvCount) {
      MPI_Waitsome(recvCount, recvRequestsData.data(), &recvPerformedCount,
                   recvCompleted.data(), recvStatusData.data());

      if(recvPerformedCount > 0) {
        for(int i = 0; i < recvPerformedCount; i++) {
          r = recvStatusData[i].MPI_SOURCE;
#pragma omp parallel for num_threads(threadNumber_) firstprivate(onBoundary) \
  schedule(dynamic)
          for(ttk::SimplexId j = 0; j < recvMessageSize[r]; j++) {
            ttk::SimplexId lid
              = globalToLocalSaddle2_.find(recvComputeBuffer[r][j])->second;
            const auto &s2 = saddles2[lid];
            this->eliminateBoundariesSandwich(
              s2, onBoundary, s2GlobalBoundaries, s2LocalBoundaries,
              edgeTrianglePartner, s1Locks, s2Locks, saddles1, saddles2,
              triangulation, offsets, sendBoundaryBuffer, sendComputeBuffer);
          }
        }
        recvPerformedCountTotal += recvPerformedCount;
      }
    }
    MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);
    printMsg("Reruns done");
    // Stop condition computation
    MPI_Allreduce(&localSentMessageNumber, &hasSentMessages, 1, MPI_SimplexId,
                  MPI_SUM, ttk::MPIcomm_);
  }
  Timer tmseq{};

  // extract saddle-saddle pairs from computed boundaries
  for(size_t i = 0; i < edgeTrianglePartner.size(); ++i) {
    if(edgeTrianglePartner[i] != -1) {
      const auto s1 = saddles1[i].gid_;
      const auto s2 = saddles2[edgeTrianglePartner[i]].gid_;
      // we found a pair
      pairs.emplace_back(s1, s2, 1);
      // paired1Saddles[s1] = true;
      // paired2Saddles[s2] = true;
    }
  }

  /*if(exportBoundaries) {
    boundaries.resize(s2Boundaries.size());
    for(size_t i = 0; i < boundaries.size(); ++i) {
      const auto &boundSet{s2Boundaries[i]};
      if(boundSet.empty()) {
        continue;
      }
      boundaries[i] = {
        {boundSet.begin(), boundSet.end()},
        saddles2[i],
        std::array<SimplexId, 2>{
          this->dg_.getCellGreaterVertex(Cell{2, saddles2[i]}, triangulation),
          this->dg_.getCellGreaterVertex(
            Cell{1, *boundSet.begin()}, triangulation),
        }};
    }
  }*/

  const auto nSadSadPairs = pairs.size() - nSadExtrPairs;

  this->printMsg(
    "Computed " + std::to_string(nSadSadPairs) + " saddle-saddle pairs", 1.0,
    tm2.getElapsedTime(), this->threadNumber_);

  this->printMsg("saddle-saddle pairs sequential part", 1.0,
                 tmseq.getElapsedTime(), 1, debug::LineMode::NEW);
}

template <typename triangulationType>
void ttk::DiscreteMorseSandwichMPI::extractCriticalCells(
  std::array<std::vector<SimplexId>, 4> &criticalCellsByDim,
  std::array<std::vector<SimplexId>, 4> &critCellsOrder,
  const SimplexId *const offsets,
  const triangulationType &triangulation,
  const bool sortEdges) const {

  Timer tm{};

  this->dg_.getCriticalPoints(criticalCellsByDim, triangulation);

  /*this->printMsg("Extracted critical cells", 1.0, tm.getElapsedTime(),
                 localThreadNumber, debug::LineMode::NEW);*/

  // memory allocations
  auto &critEdges{this->critEdges_};
  if(!sortEdges) {
    critEdges.resize(criticalCellsByDim[1].size());
  }
  std::vector<TriangleSimplex> critTriangles(criticalCellsByDim[2].size());
  std::vector<TetraSimplex> critTetras(criticalCellsByDim[3].size());

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  {
    if(sortEdges) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp for nowait
#endif // TTK_ENABLE_OPENMP
      for(size_t i = 0; i < critEdges.size(); ++i) {
        critEdges[i].fillEdge(i, offsets, triangulation);
      }
    } else {
#ifdef TTK_ENABLE_OPENMP
#pragma omp for nowait
#endif // TTK_ENABLE_OPENMP
      for(size_t i = 0; i < critEdges.size(); ++i) {
        critEdges[i].fillEdge(criticalCellsByDim[1][i], offsets, triangulation);
      }
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp for nowait
#endif // TTK_ENABLE_OPENMP
    for(size_t i = 0; i < critTriangles.size(); ++i) {
      critTriangles[i].fillTriangle(
        criticalCellsByDim[2][i], offsets, triangulation);
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp for
#endif // TTK_ENABLE_OPENMP
    for(size_t i = 0; i < critTetras.size(); ++i) {
      critTetras[i].fillTetra(criticalCellsByDim[3][i], offsets, triangulation);
    }
  }

  TTK_PSORT(this->threadNumber_, critEdges.begin(), critEdges.end());
  TTK_PSORT(this->threadNumber_, critTriangles.begin(), critTriangles.end());
  TTK_PSORT(this->threadNumber_, critTetras.begin(), critTetras.end());

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  {
#ifdef TTK_ENABLE_OPENMP
#pragma omp for nowait
#endif // TTK_ENABLE_OPENMP
    for(size_t i = 0; i < critEdges.size(); ++i) {
      critCellsOrder[1][critEdges[i].id_] = i;
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp for nowait
#endif // TTK_ENABLE_OPENMP
    for(size_t i = 0; i < critTriangles.size(); ++i) {
      criticalCellsByDim[2][i] = critTriangles[i].id_;
      critCellsOrder[2][critTriangles[i].id_] = i;
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp for
#endif // TTK_ENABLE_OPENMP
    for(size_t i = 0; i < critTetras.size(); ++i) {
      criticalCellsByDim[3][i] = critTetras[i].id_;
      critCellsOrder[3][critTetras[i].id_] = i;
    }
  }

  if(sortEdges) {
    TTK_PSORT(this->threadNumber_, criticalCellsByDim[1].begin(),
              criticalCellsByDim[1].end(),
              [&critCellsOrder](const SimplexId a, const SimplexId b) {
                return critCellsOrder[1][a] < critCellsOrder[1][b];
              });
  } else {
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
    for(size_t i = 0; i < critEdges.size(); ++i) {
      criticalCellsByDim[1][i] = critEdges[i].id_;
    }
  }

  /*this->printMsg("Extracted & sorted critical cells", 1.0,
     tm.getElapsedTime(), this->threadNumber_, debug::LineMode::NEW);*/
}

template <typename triangulationType>
int ttk::DiscreteMorseSandwichMPI::computePersistencePairs(
  std::vector<PersistencePair> &pairs,
  const SimplexId *const offsets,
  const triangulationType &triangulation,
  const bool ignoreBoundary,
  const bool compute2SaddlesChildren) {

  // allocate memory
  this->alloc(triangulation);
#ifdef TTK_ENABLE_MPI_TIME
  ttk::Timer t_mpi;
  ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
#endif
  Timer tm{};
  pairs.clear();
  const auto dim = this->dg_.getDimensionality();
  this->Compute2SaddlesChildren = compute2SaddlesChildren;

  // get every critical cell sorted them by dimension
  std::array<std::vector<SimplexId>, 4> criticalCellsByDim{};
  // holds the critical cells order
  auto &critCellsOrder{this->critCellsOrder_};

  this->extractCriticalCells(
    criticalCellsByDim, critCellsOrder, offsets, triangulation, dim == 3);

#ifdef TTK_ENABLE_MPI_TIME
  double elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
  if(ttk::MPIrank_ == 0) {
    printMsg("Extract critical cells performed using "
             + std::to_string(ttk::MPIsize_)
             + " MPI processes lasted :" + std::to_string(elapsedTime));
  }
#endif
  /* // if minima are paired
   auto &pairedMinima{this->pairedCritCells_[0]};
   // if 1-saddles are paired
   auto &paired1Saddles{this->pairedCritCells_[1]};
   // if 2-saddles are paired
   auto &paired2Saddles{this->pairedCritCells_[dim - 1]};
   // if maxima are paired
   auto &pairedMaxima{this->pairedCritCells_[dim]};*/

  // connected components (global min/max pair)
  size_t nConnComp{};
  if(dim > 2 && UseTasks) {
    int minSadThreadNumber = std::max(1, static_cast<int>(threadNumber_ / 2));
    int maxSadThreadNumber = std::max(1, threadNumber_ - minSadThreadNumber);
    int taskNumber = std::min(2, threadNumber_);
    omp_set_nested(1);
    std::vector<PersistencePair> sadMaxPairs;
    MPI_Comm minSadComm;
    MPI_Comm_dup(ttk::MPIcomm_, &minSadComm);
    MPI_Comm sadMaxComm;
    MPI_Comm_dup(ttk::MPIcomm_, &sadMaxComm);
#pragma omp parallel master num_threads(taskNumber) \
  firstprivate(dim, sadMaxComm, minSadComm)
    {
#pragma omp task
      {
        this->getMinSaddlePairs(pairs, criticalCellsByDim[1], critCellsOrder[1],
                                criticalCellsByDim[0], offsets, nConnComp,
                                triangulation, minSadComm, minSadThreadNumber);
      }
      // saddle - maxima pairs
#pragma omp task
      {
        this->getMaxSaddlePairs(
          sadMaxPairs, criticalCellsByDim[dim - 1], critCellsOrder[dim - 1],
          criticalCellsByDim[dim], critCellsOrder[dim], triangulation,
          ignoreBoundary, offsets, sadMaxComm, maxSadThreadNumber);
      }
    }
    omp_set_nested(0);
    MPI_Comm_free(&minSadComm);
    MPI_Comm_free(&sadMaxComm);
    pairs.insert(pairs.end(), sadMaxPairs.begin(), sadMaxPairs.end());
  } else {
    // minima - saddle pairs
    this->getMinSaddlePairs(pairs, criticalCellsByDim[1], critCellsOrder[1],
                            criticalCellsByDim[0], offsets, nConnComp,
                            triangulation, ttk::MPIcomm_, threadNumber_);
    // saddle - maxima pairs
    this->getMaxSaddlePairs(pairs, criticalCellsByDim[dim - 1],
                            critCellsOrder[dim - 1], criticalCellsByDim[dim],
                            critCellsOrder[dim], triangulation, ignoreBoundary,
                            offsets, ttk::MPIcomm_, threadNumber_);
  }

  // saddle - saddle pairs
  if(dim == 3 && !criticalCellsByDim[1].empty()
     && !criticalCellsByDim[2].empty() && this->ComputeSadSad) {
    std::vector<GeneratorType> tmp{};
    this->getSaddleSaddlePairs(pairs, false, tmp, criticalCellsByDim[1],
                               criticalCellsByDim[2], critCellsOrder[1],
                               critCellsOrder[2], triangulation, offsets);
  }
  // TODO: implement following
  /*if(std::is_same<triangulationType, ttk::ExplicitTriangulation>::value) {
    // create infinite pairs from non-paired 1-saddles, 2-saddles and maxima
    size_t nHandles{}, nCavities{}, nNonPairedMax{};
    if((dim == 2 && !ignoreBoundary && this->ComputeMinSad
        && this->ComputeSadMax)
       || (dim == 3 && this->ComputeMinSad && this->ComputeSadSad)) {
      // non-paired 1-saddles
      for(const auto s1 : criticalCellsByDim[1]) {
        if(!paired1Saddles[s1]) {
          paired1Saddles[s1] = true;
          // topological handles
          pairs.emplace_back(s1, -1, 1);
          nHandles++;
        }
      }
    }
    if(dim == 3 && !ignoreBoundary && this->ComputeSadMax
       && this->ComputeSadSad) {
      // non-paired 2-saddles
      for(const auto s2 : criticalCellsByDim[2]) {
        if(!paired2Saddles[s2]) {
          paired2Saddles[s2] = true;
          // cavities
          pairs.emplace_back(s2, -1, 2);
          nCavities++;
        }
      }
    }
    if(dim == 2 && !ignoreBoundary && this->ComputeSadMax) {
      // non-paired maxima
      for(const auto max : criticalCellsByDim[dim]) {
        if(!pairedMaxima[max]) {
          pairs.emplace_back(max, -1, 2);
          nNonPairedMax++;
        }
      }
    }

    int nBoundComp
      = (dim == 3 ? nCavities : nHandles) + nConnComp - nNonPairedMax;
    nBoundComp = std::max(nBoundComp, 0);

    // print Betti numbers
    const std::vector<std::vector<std::string>> rows{
      {" #Connected components", std::to_string(nConnComp)},
      {" #Topological handles", std::to_string(nHandles)},
      {" #Cavities", std::to_string(nCavities)},
      {" #Boundary components", std::to_string(nBoundComp)},
    };

    this->printMsg(rows, debug::Priority::DETAIL);
  }*/
  /*this->printMsg(
    "Computed " + std::to_string(pairs.size()) + " persistence pairs", 1.0,
    tm.getElapsedTime(), this->threadNumber_);*/

  // this->displayStats(pairs, criticalCellsByDim, pairedMinima, paired1Saddles,
  //                   paired2Saddles, pairedMaxima);

  // free memory
  this->clear();

#ifdef TTK_ENABLE_MPI_TIME
  elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
  if(ttk::MPIrank_ == 0) {
    printMsg("Computation of persistence pairs performed using "
             + std::to_string(ttk::MPIsize_)
             + " MPI processes lasted :" + std::to_string(elapsedTime));
  }
#endif

  return 0;
}