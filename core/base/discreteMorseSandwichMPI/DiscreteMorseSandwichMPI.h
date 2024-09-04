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
      char ghostPresenceSize_;
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
      MPI_Datatype types[]
        = {MPI_SimplexId, MPI_SimplexId, MPI_SimplexId, MPI_CHAR, MPI_CHAR};
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
      // printMsg("create_struct done");
      MPI_Type_commit(&MPI_MessageType);
      // printMsg("commit done");
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

    inline void preconditionTriangulation(AbstractTriangulation *const data) {
      this->dg_.preconditionTriangulation(data);
    }

    inline void setInputOffsets(const SimplexId *const offsets) {
      printMsg("setInputOffsets");
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
    int getSaddle1ToMinima(const std::vector<SimplexId> &criticalEdges,
                           const triangulationType &triangulation,
                           const SimplexId *const offsets,
                           std::vector<std::vector<extremaNode<1>>> &res,
                           std::vector<std::vector<char>> &ghostPresence,
                           std::unordered_map<ttk::SimplexId, std::vector<char>>
                             &localGhostPresenceMap) const;

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
    template <typename triangulationType,
              typename GFS,
              typename GFSN,
              typename OB>
    std::vector<std::vector<SimplexId>>
      getSaddle2ToMaxima(const std::vector<SimplexId> &criticalCells,
                         const GFS &getFaceStar,
                         const GFSN &getFaceStarNumber,
                         const OB &isOnBoundary,
                         const triangulationType &triangulation) const;

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
                           const triangulationType &triangulation) const;

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
                           const std::vector<SimplexId> &critMaxsOrder,
                           const triangulationType &triangulation,
                           const bool ignoreBoundary,
                           const SimplexId *const offsets);

    template <typename triangulationType, int sizeExtr, int sizeSad>
    void computeMaxSaddlePairs(
      std::vector<PersistencePair> &pairs,
      const std::vector<SimplexId> &criticalSaddles,
      const std::vector<SimplexId> &critSaddlesOrder,
      const std::vector<SimplexId> &critMaxsOrder,
      const triangulationType &triangulation,
      const bool ignoreBoundary,
      const SimplexId *const offsets,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema);
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
                              const triangulationType &triangulation) const;

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
    int processTriplet(saddleEdge<sizeSad> sv,
                       std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                       std::vector<ttk::SimplexId> &extremaToPairedSaddle,
                       std::vector<saddleEdge<sizeSad>> &saddles,
                       std::vector<extremaNode<sizeExtr>> &extremas,
                       bool increasing,
                       std::vector<std::vector<char>> &ghostPresence,
                       std::vector<std::vector<messageType<sizeExtr, sizeSad>>>
                         &sendBuffer) const;

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
    void storeRerunToSend(
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      saddleEdge<sizeSad> &sv) const;

    template <int sizeExtr, int sizeSad>
    void addPair(const saddleEdge<sizeSad> &sad,
                 const extremaNode<sizeExtr> &extr,
                 std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                 std::vector<ttk::SimplexId> &extremaToPairedSaddle) const;

    template <int sizeExtr, int sizeSad>
    void removePair(const saddleEdge<sizeSad> &sad,
                    const extremaNode<sizeExtr> &extr,
                    std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                    std::vector<ttk::SimplexId> &extremaToPairedSaddle) const;

    template <int sizeExtr, int sizeSad>
    struct extremaNode<sizeExtr> &
      getRep(extremaNode<sizeExtr> *extr,
             saddleEdge<sizeSad> *sv,
             bool increasing,
             std::vector<extremaNode<sizeExtr>> &extremas,
             std::vector<saddleEdge<sizeSad>> &saddles) const;

    template <int sizeExtr, int sizeSad>
    void receiveElement(
      messageType<sizeExtr, sizeSad> &element,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      std::vector<saddleEdge<sizeSad>> &saddles,
      std::vector<extremaNode<sizeExtr>> &extremas,
      std::vector<ttk::SimplexId> &extremaToPairedSaddle,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
      std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
      std::vector<std::vector<char>> &ghostPresence,
      char sender,
      bool increasing) const;

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
      std::vector<PersistencePair> &pairs,
      const SimplexId pairDim,
      std::vector<extremaNode<sizeExtr>> &extremas,
      std::vector<saddleEdge<sizeSad>> &saddles,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
      std::vector<ttk::SimplexId> &extremaToPairedSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
      std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
      std::vector<std::vector<char>> ghostPresence,
      MPI_Datatype &MPI_MessageType,
      bool isFirstTime) const;

    template <int sizeExtr, int sizeSad>
    void extractPairs(std::vector<PersistencePair> &pairs,
                      std::vector<extremaNode<sizeExtr>> &extremas,
                      std::vector<saddleEdge<sizeSad>> &saddles,
                      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
                      bool increasing,
                      const int pairDim) const;

    template <int sizeExtr, int sizeSad>
    int computePairNumbers(
      std::vector<saddleEdge<sizeSad>> &saddles,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema) const;
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
    template <typename triangulationType, typename Container>
    SimplexId
      eliminateBoundariesSandwich(const SimplexId s2,
                                  std::vector<bool> &onBoundary,
                                  std::vector<Container> &s2Boundaries,
                                  const std::vector<SimplexId> &s2Mapping,
                                  const std::vector<SimplexId> &s1Mapping,
                                  std::vector<SimplexId> &partners,
                                  std::vector<Lock> &s1Locks,
                                  std::vector<Lock> &s2Locks,
                                  const triangulationType &triangulation) const;

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
      Timer tm{};
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
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->edgeTrianglePartner_.resize(
            triangulation.getNumberOfEdges(), -1);
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->onBoundary_.resize(triangulation.getNumberOfEdges(), false);
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->s2Mapping_.resize(triangulation.getNumberOfTriangles(), -1);
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->s1Mapping_.resize(triangulation.getNumberOfEdges(), -1);
        }
        for(int i = 0; i < dim + 1; ++i) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->pairedCritCells_[i].resize(
            this->dg_.getNumberOfCells(i, triangulation), false);
        }
        for(int i = 1; i < dim + 1; ++i) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp task
#endif
          this->critCellsOrder_[i].resize(
            this->dg_.getNumberOfCells(i, triangulation), -1);
        }
      }
      this->printMsg("Memory allocations", 1.0, tm.getElapsedTime(), 1,
                     debug::LineMode::NEW);
    }

    void clear() {
      Timer tm{};
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
      this->printMsg(
        "Memory cleanup", 1.0, tm.getElapsedTime(), 1, debug::LineMode::NEW);
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
  };
} // namespace ttk

template <typename triangulationType>
int ttk::DiscreteMorseSandwichMPI::getSaddle1ToMinima(
  const std::vector<SimplexId> &criticalEdges,
  const triangulationType &triangulation,
  const SimplexId *const offsets,
  std::vector<std::vector<extremaNode<1>>> &res,
  std::vector<std::vector<char>> &ghostPresence,
  std::unordered_map<ttk::SimplexId, std::vector<char>> &localGhostPresenceMap)
  const {

  Timer tm{};
  const std::vector<int> neighbors = triangulation.getNeighborRanks();
  const std::map<int, int> neighborsToId = triangulation.getNeighborsToId();
  int neighborNumber = neighbors.size();
  res.resize(criticalEdges.size(), std::vector<extremaNode<1>>());
  std::vector<std::vector<std::vector<vpathToSend>>> sendBufferThread(
    threadNumber_);
  std::vector<std::vector<std::vector<vpathFinished<1>>>>
    sendFinishedVPathBufferThread(threadNumber_);
  // TODO: PUT IN ALLOC?
  ghostPresence.resize(
    triangulation.getNumberOfVertices(), std::vector<char>());
  std::vector<Lock> extremaLocks(triangulation.getNumberOfVertices());
  std::vector<Lock> saddleLocks(criticalEdges.size());
  for(int i = 0; i < this->threadNumber_; i++) {
    sendBufferThread.at(i).resize(neighborNumber);
    sendFinishedVPathBufferThread.at(i).resize(ttk::MPIsize_);
  }
  ttk::SimplexId localElementNumber = 2 * criticalEdges.size();
  ttk::SimplexId totalElement{0};
  const auto followVPath = [this, &triangulation, &neighborsToId,
                            &localElementNumber, &extremaLocks, &res,
                            &saddleLocks, &ghostPresence, &sendBufferThread,
                            &sendFinishedVPathBufferThread,
                            offsets](const SimplexId v, ttk::SimplexId saddleId,
                                     char saddleRank, int threadNumber) {
    std::vector<Cell> vpath{};
    this->dg_.getDescendingPath(Cell{0, v}, vpath, triangulation);
    const Cell &lastCell = vpath.back();
    if(lastCell.dim_ == 0) {
      ttk::SimplexId extremaId = triangulation.getVertexGlobalId(lastCell.id_);
      int rank = triangulation.getVertexRank(lastCell.id_);
      if(rank != ttk::MPIrank_) {
        sendBufferThread.at(threadNumber)
          .at(neighborsToId.find(rank)->second)
          .emplace_back(vpathToSend{.saddleId_ = saddleId,
                                    .extremaId_ = extremaId,
                                    .saddleRank_ = saddleRank});
      } else {
#pragma omp atomic update
        localElementNumber--;
        if(this->dg_.isCellCritical(lastCell)) {
          extremaLocks.at(lastCell.id_).lock();
          auto &ghost{ghostPresence.at(lastCell.id_)};
          auto it = find(ghost.begin(), ghost.end(), saddleRank);
          if(it == ghost.end()) {
            ghost.push_back(saddleRank);
          }
          extremaLocks.at(lastCell.id_).unlock();
          if(saddleRank == ttk::MPIrank_) {
            ttk::SimplexId vOrd[] = {offsets[lastCell.id_]};
            extremaNode<1> n(extremaId, -1, offsets[lastCell.id_], Rep{-1, -1},
                             static_cast<char>(ttk::MPIrank_), vOrd);
            // We store it in the current rank
            // TODO: only locks for second one?
            saddleLocks[saddleId].lock();
            res[saddleId].emplace_back(n);
            saddleLocks[saddleId].unlock();
          } else {
            // We store it to send it back to whoever will own the extrema
            sendFinishedVPathBufferThread.at(threadNumber)
              .at(saddleRank)
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
#pragma omp atomic update
      localElementNumber--;
    }
  };
  ttk::Memory m{};
  printMsg("Memory use: " + std::to_string(m.getTotalUsage()));
  // follow vpaths from 1-saddles to minima
#pragma omp parallel shared(extremaLocks, localElementNumber) \
  num_threads(threadNumber_)
  {
    int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static)
    for(size_t i = 0; i < criticalEdges.size(); ++i) {
      // critical edge vertices
      SimplexId v0{}, v1{};
      triangulation.getEdgeVertex(criticalEdges[i], 0, v0);
      triangulation.getEdgeVertex(criticalEdges[i], 1, v1);

      // follow vpath from each vertex of the critical edge
      followVPath(v0, i, ttk::MPIrank_, threadNumber);
      followVPath(v1, i, ttk::MPIrank_, threadNumber);
    }
  }
  // Send receive elements
  MPI_Datatype MPI_SimplexId = getMPIType(static_cast<ttk::SimplexId>(0));
  MPI_Datatype MPI_MessageType;
  this->createVpathMPIType(MPI_MessageType);
  MPI_Allreduce(&localElementNumber, &totalElement, 1, MPI_SimplexId, MPI_SUM,
                ttk::MPIcomm_);
  std::vector<std::vector<vpathToSend>> sendBuffer(neighborNumber);
  std::vector<std::vector<vpathToSend>> recvBuffer(neighborNumber);
  bool keepWorking = (totalElement != 0);
  while(keepWorking) {
#pragma omp parallel for schedule(static, 1)
    for(int j = 0; j < neighborNumber; j++) {
      sendBuffer.at(j).clear();
      for(int i = 0; i < this->threadNumber_; i++) {
        sendBuffer.at(j).insert(sendBuffer.at(j).end(),
                                sendBufferThread.at(i).at(j).begin(),
                                sendBufferThread.at(i).at(j).end());
        // TODO: check if ok with memory
        sendBufferThread.at(i).at(j).clear();
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
      sendMessageSize[i] = sendBuffer.at(i).size();
      MPI_Isend(&sendMessageSize[i], 1, MPI_SimplexId, neighbors[i], 0,
                ttk::MPIcomm_, &sendRequests[i]);
      MPI_Irecv(&recvMessageSize[i], 1, MPI_SimplexId, neighbors[i], 0,
                ttk::MPIcomm_, &recvRequests[i]);
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
            r = neighbors[i];
            if((sendMessageSize[rankId] > 0)) {
              MPI_Isend(sendBuffer.at(rankId).data(), sendMessageSize[rankId],
                        MPI_MessageType, r, 1, ttk::MPIcomm_,
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
              recvBuffer.at(rankId).resize(recvMessageSize[rankId]);
              MPI_Irecv(recvBuffer.at(rankId).data(), recvMessageSize[rankId],
                        MPI_MessageType, r, 1, ttk::MPIcomm_,
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
#pragma omp parallel
          {
            int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static)
            for(int j = 0; j < recvMessageSize[rankId]; j++) {
              struct vpathToSend element = recvBuffer.at(rankId).at(j);
              ttk::SimplexId v
                = triangulation.getVertexLocalId(element.extremaId_);
              if(element.saddleId_ == 61) {
                printMsg("Received saddle 207");
              }
              followVPath(
                v, element.saddleId_, element.saddleRank_, threadNumber);
            }
          }
        }
        recvPerformedCountTotal += recvPerformedCount;
      }
    }
    MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);
    // Stop condition computation
    MPI_Allreduce(&localElementNumber, &totalElement, 1, MPI_SimplexId, MPI_SUM,
                  ttk::MPIcomm_);
    keepWorking = (totalElement != 0);
  }
  // Create ghostPresence and send finished VPath back
  std::vector<std::vector<char>> ghostPresenceToSend(ttk::MPIsize_);
  std::vector<std::vector<vpathFinished<1>>> finishedVPathToSend(ttk::MPIsize_);
  std::vector<std::vector<std::vector<char>>> ghostPresenceToSendThread(
    threadNumber_);
  std::vector<std::vector<std::vector<vpathFinished<1>>>>
    finishedVPathToSendThread(threadNumber_);
  std::vector<std::vector<std::vector<ttk::SimplexId>>> ghostPerThread(
    threadNumber_);
  std::vector<std::vector<ttk::SimplexId>> ghostCounterThread(threadNumber_);
  for(int i = 0; i < threadNumber_; i++) {
    ghostPresenceToSendThread.at(i).resize(ttk::MPIsize_);
    finishedVPathToSendThread.at(i).resize(ttk::MPIsize_);
    ghostPerThread.at(i).resize(ttk::MPIsize_);
    ghostCounterThread.at(i).resize(ttk::MPIsize_, 0);
  }

#pragma omp parallel num_threads(threadNumber_) firstprivate(ghostPerThread)
  {
    int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static, 1)
    for(int j = 0; j < threadNumber_; j++) {
      for(int i = 0; i < ttk::MPIsize_; i++) {
        for(int k = 0; k < sendFinishedVPathBufferThread.at(j).at(i).size();
            k++) {
          // Find owner by applying the following rule:
          // if the current rank is in ghostPresence, then the current rank is
          // the owner if not, it is the rank with the lowest rank id that is
          // the owner
          auto vp = sendFinishedVPathBufferThread.at(j).at(i).at(k);
          ttk::SimplexId lid = triangulation.getVertexLocalId(vp.extremaId_);
          auto &ghost{ghostPresence[lid]};
          auto it = std::find(
            ghost.begin(), ghost.end(), static_cast<char>(ttk::MPIrank_));
          if(it != ghost.end()) {
            // The rank of the extrema is the current rank
            // We store to send the finished vpath
            vp.extremaRank_ = ttk::MPIrank_;
            vp.ghostPresenceSize_ = 0;
          } else {
            // The rank of the extrema is NOT the current rank
            // We find the smallest rank
            auto minRank = std::min_element(ghost.begin(), ghost.end());
            vp.extremaRank_ = (*minRank);
            ghostCounterThread.at(threadNumber).at(i) += ghost.size();
            vp.ghostPresenceSize_ = ghostCounterThread.at(threadNumber).at(i);
            // Send the ghostPresence to that rank
            ghostPresenceToSendThread.at(threadNumber)
              .at(i)
              .insert(ghostPresenceToSendThread.at(threadNumber).at(i).end(),
                      ghost.begin(), ghost.end());
          }
          finishedVPathToSendThread.at(threadNumber).at(i).emplace_back(vp);
        }
      }
    }
  }
  // Merge the vectors
#pragma omp parallel for schedule(static, 1)
  for(int j = 0; j < ttk::MPIsize_; j++) {
    ttk::SimplexId ghostCounter{0};
    for(int i = 0; i < this->threadNumber_; i++) {
      std::transform(finishedVPathToSendThread.at(i).at(j).begin(),
                     finishedVPathToSendThread.at(i).at(j).end(),
                     finishedVPathToSendThread.at(i).at(j).begin(),
                     [this, &ghostCounter](vpathFinished<1> &vp) {
                       if(vp.ghostPresenceSize_ != 0) {
                         vp.ghostPresenceSize_ += ghostCounter;
                       }
                       return vp;
                     });
      ghostPresenceToSend.at(j).insert(
        ghostPresenceToSend.at(j).end(),
        ghostPresenceToSendThread.at(i).at(j).begin(),
        ghostPresenceToSendThread.at(i).at(j).end());
      finishedVPathToSend.at(j).insert(
        finishedVPathToSend.at(j).end(),
        finishedVPathToSendThread.at(i).at(j).begin(),
        finishedVPathToSendThread.at(i).at(j).end());
      ghostCounter += ghostPerThread.at(i).at(j).size();
    }
  }
  /*for(int i = 0; i < ttk::MPIsize_; i++) {
    for(int j = 0; j < finishedVPathToSend.at(i).size(); j++) {
      printMsg("Element: "
               + std::to_string(finishedVPathToSend.at(i).at(j).extremaId_));
    }
  }*/
  // printMsg("Start ghostPresence comm");
  // Send/Recv them
  std::vector<ttk::SimplexId> recvMessageSize(2 * ttk::MPIsize_, 0);
  std::vector<ttk::SimplexId> sendMessageSize(2 * ttk::MPIsize_, 0);
  std::vector<MPI_Request> requests(4 * ttk::MPIsize_, MPI_REQUEST_NULL);
  std::vector<std::vector<vpathFinished<1>>> recvVPathFinished(ttk::MPIsize_);
  std::vector<std::vector<char>> recvGhostPresence(ttk::MPIsize_);
  MPI_Datatype MPI_FinishedVPathMPIType;
  createFinishedVpathMPIType<1>(MPI_FinishedVPathMPIType);
  int count{0};
  for(int i = 0; i < ttk::MPIsize_; i++) {
    if(i != ttk::MPIrank_) {
      sendMessageSize[2 * i] = finishedVPathToSend[i].size();
      sendMessageSize[2 * i + 1] = ghostPresenceToSend[i].size();
      MPI_Isend(sendMessageSize.data() + 2 * i, 2, MPI_SimplexId, i, 0,
                ttk::MPIcomm_, &requests[count]);
      MPI_Irecv(recvMessageSize.data() + 2 * i, 2, MPI_SimplexId, i, 0,
                ttk::MPIcomm_, &requests[count + 1]);
      count += 2;
    }
  }
  MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
  /*for(int i = 0; i < ttk::MPIsize_; i++) {
    if(i != ttk::MPIrank_) {
      printMsg("Sendsize: " + std::to_string(sendMessageSize[2 * i]) + ", "
               + std::to_string(sendMessageSize[2 * i + 1]));
      printMsg("recvsize: " + std::to_string(recvMessageSize[2 * i]) + ", "
               + std::to_string(recvMessageSize[2 * i + 1]));
    }
  }*/
  count = 0;
  // Exchange of the data
  for(int i = 0; i < ttk::MPIsize_; i++) {
    recvVPathFinished[i].resize(recvMessageSize[2 * i]);
    recvGhostPresence[i].resize(recvMessageSize[2 * i + 1]);
    if(recvMessageSize[2 * i] > 0) {
      MPI_Irecv(recvVPathFinished[i].data(), recvMessageSize[2 * i],
                MPI_FinishedVPathMPIType, i, 1, ttk::MPIcomm_,
                &requests[count]);
      count++;
    }
    if(sendMessageSize[2 * i] > 0) {
      MPI_Isend(finishedVPathToSend[i].data(), sendMessageSize[2 * i],
                MPI_FinishedVPathMPIType, i, 1, ttk::MPIcomm_,
                &requests[count]);
      count++;
    }
    if(recvMessageSize[2 * i + 1] > 0) {
      MPI_Irecv(recvGhostPresence[i].data(), recvMessageSize[2 * i + 1],
                MPI_CHAR, i, 2, ttk::MPIcomm_, &requests[count]);
      count++;
    }
    if(sendMessageSize[2 * i + 1] > 0) {
      MPI_Isend(ghostPresenceToSend[i].data(), sendMessageSize[2 * i + 1],
                MPI_CHAR, i, 2, ttk::MPIcomm_, &requests[count]);
      count++;
    }
  }
  MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
  /*for(int i = 0; i < ttk::MPIsize_; i++) {
    for(int j = 0; j < recvVPathFinished.at(i).size(); j++) {
      printMsg("Received Element: "
               + std::to_string(recvVPathFinished.at(i).at(j).extremaId_));
    }
  }
  printMsg("Receive ghostPresence");*/
  for(int i = 0; i < ttk::MPIsize_; i++) {
    //#pragma omp parallel for schedule(static)
    for(int j = 0; j < recvMessageSize[2 * i]; j++) {
      // Receive element: create VPath and add it to the list
      auto &vp{recvVPathFinished.at(i).at(j)};
      ttk::SimplexId beginGhost
        = (j == 0) ? 0 : recvVPathFinished.at(i).at(j - 1).ghostPresenceSize_;
      extremaNode<1> n(vp.extremaId_, -1, vp.vOrder_[0], Rep{-1, -1},
                       vp.extremaRank_, vp.vOrder_);
      if(vp.saddleId_ == 61) {
        printMsg("Received saddleId 207 after all the computation");
      }
      /*printMsg("Received element: " + std::to_string(vp.saddleId_) + ", "
               + std::to_string(vp.extremaId_));*/
      // We store it in the current rank
      saddleLocks[vp.saddleId_].lock();
      res[vp.saddleId_].emplace_back(n);
      saddleLocks[vp.saddleId_].unlock();
      if(vp.ghostPresenceSize_ != 0) {
        // Add the received ghostPresence to the local ghostPresence
        // If there is only one process, then the extrema won't be on the
        // boundary of the new graph, there is no need to record it
        if(vp.ghostPresenceSize_ - beginGhost > 1) {
          std::vector<char> ghost{};
          ghost.insert(ghost.end(),
                       recvGhostPresence.at(i).begin() + beginGhost,
                       recvGhostPresence.at(i).begin()
                         + static_cast<ttk::SimplexId>(vp.ghostPresenceSize_));
          ttk::SimplexId lid = triangulation.getVertexLocalId(vp.extremaId_);
          // If the extrema is not locally present in the triangulation,
          // Add the entry to the map
          if(lid == -1) {
#pragma omp critical
            { localGhostPresenceMap[vp.extremaId_] = ghost; }
          } else {
            extremaLocks[lid].lock();
            ghostPresence.at(lid) = ghost;
            extremaLocks[lid].unlock();
          }
        }
      }
    }
  }
  this->printMsg("Computed the descending 1-separatrices", 1.0,
                 tm.getElapsedTime(), this->threadNumber_,
                 debug::LineMode::NEW);

  return 0;
}

template <typename triangulationType, typename GFS, typename GFSN, typename OB>
std::vector<std::vector<SimplexId>>
  ttk::DiscreteMorseSandwichMPI::getSaddle2ToMaxima(
    const std::vector<SimplexId> &criticalCells,
    const GFS &getFaceStar,
    const GFSN &getFaceStarNumber,
    const OB &isOnBoundary,
    const triangulationType &triangulation) const {

  Timer tm{};

  const auto dim = this->dg_.getDimensionality();
  std::vector<std::vector<SimplexId>> res(criticalCells.size());

  // follow vpaths from 2-saddles to maxima
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(size_t i = 0; i < criticalCells.size(); ++i) {
    const auto sid = criticalCells[i];
    auto &maxs = res[i];

    const auto followVPath
      = [this, dim, &maxs, &triangulation](const SimplexId v) {
          std::vector<Cell> vpath{};
          this->dg_.getAscendingPath(Cell{dim, v}, vpath, triangulation);
          const Cell &lastCell = vpath.back();
          if(lastCell.dim_ == dim && this->dg_.isCellCritical(lastCell)) {
            maxs.emplace_back(lastCell.id_);
          } else if(lastCell.dim_ == dim - 1) {
            maxs.emplace_back(-1);
          }
        };

    const auto starNumber = getFaceStarNumber(sid);

    for(SimplexId j = 0; j < starNumber; ++j) {
      SimplexId cellId{};
      getFaceStar(sid, j, cellId);
      followVPath(cellId);
    }

    if(isOnBoundary(sid)) {
      // critical saddle is on boundary
      maxs.emplace_back(-1);
    }
  }

  this->printMsg("Computed the ascending 1-separatrices", 1.0,
                 tm.getElapsedTime(), this->threadNumber_,
                 debug::LineMode::NEW);

  return res;
}

template <typename triangulationType>
void ttk::DiscreteMorseSandwichMPI::getMinSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const std::vector<ttk::SimplexId> &criticalEdges,
  const std::vector<ttk::SimplexId> &critEdgesOrder,
  const std::vector<ttk::SimplexId> &criticalExtremas,
  const SimplexId *const offsets,
  size_t &nConnComp,
  const triangulationType &triangulation) const {

  ttk::SimplexId totalNumberOfVertices{-1};
  MPI_Datatype MPI_SimplexId = getMPIType(totalNumberOfVertices);
  ttk::SimplexId localNumberOfVertices = triangulation.getNumberOfVertices();

  MPI_Allreduce(&localNumberOfVertices, &totalNumberOfVertices, 1,
                MPI_SimplexId, MPI_SUM, ttk::MPIcomm_);

  ttk::SimplexId localMinOffset{totalNumberOfVertices};
  ttk::SimplexId globalMinOffset{-1};
  ttk::SimplexId localMin;

  if(criticalExtremas.size() > 0) {
    // extracts the global pair
    localMin
      = *std::min_element(criticalExtremas.begin(), criticalExtremas.end(),
                          [offsets](const SimplexId a, const SimplexId b) {
                            return offsets[a] < offsets[b];
                          });
    if(triangulation.getVertexRank(localMin) == ttk::MPIrank_) {
      localMinOffset = offsets[localMin];
    }
  }

  MPI_Allreduce(&localMinOffset, &globalMinOffset, 1, MPI_SimplexId, MPI_MIN,
                ttk::MPIcomm_);

  if(this->ComputeMinSad) {
    // minima - saddle pairs
    Timer tm{};
    std::vector<std::vector<extremaNode<1>>> saddle1ToMinima;
    std::unordered_map<ttk::SimplexId, std::vector<char>> localGhostPresenceMap;
    std::vector<std::vector<char>> localGhostPresenceVector;
    this->getSaddle1ToMinima(criticalEdges, triangulation, offsets,
                             saddle1ToMinima, localGhostPresenceVector,
                             localGhostPresenceMap);
    Timer tmseq{};
    auto &saddleToPairedExtrema{this->saddleToPairedMin_};
    auto &extremaToPairedSaddle{this->minToPairedSaddle_};
    auto &globalToLocalSaddle{this->globalToLocalSaddle1_};
    std::vector<saddleEdge<2>> saddles{};
    std::vector<extremaNode<1>> extremas{};
    std::vector<std::vector<char>> ghostPresence{};
    saddles.reserve(saddle1ToMinima.size());
    extremas.reserve(2 * saddle1ToMinima.size());
    globalToLocalSaddle.reserve(saddle1ToMinima.size());
    std::unordered_map<ttk::SimplexId, ttk::SimplexId> globalToLocalExtrema{};
    globalToLocalExtrema.reserve(2 * saddle1ToMinima.size());
    ghostPresence.reserve(2 * saddle1ToMinima.size());
    std::vector<ttk::SimplexId> globalMinLid;
    ttk::SimplexId saddle1ToMinimaNumber = saddle1ToMinima.size();
    ttk::SimplexId totalNumberOfPairs = criticalExtremas.size();
    MPI_Allreduce(MPI_IN_PLACE, &totalNumberOfPairs, 1, MPI_SimplexId, MPI_SUM,
                  ttk::MPIcomm_);
    std::vector<char> ghosts{};
    // Add ghostPresence
    for(size_t i = 0; i < saddle1ToMinimaNumber; ++i) {
      auto &mins = saddle1ToMinima[i];
      const auto s1 = criticalEdges[i];
      // remove duplicates
      std::sort(mins.begin(), mins.end());
      const auto last = std::unique(mins.begin(), mins.end());
      mins.erase(last, mins.end());
      if(mins.size() != 2) {
        continue;
      }
      ttk::SimplexId vOrd[2];
      fillEdgeOrder(s1, offsets, triangulation, vOrd);
      saddleEdge e
        = saddleEdge<2>(triangulation.getEdgeGlobalId(s1), critEdgesOrder[s1],
                        vOrd, static_cast<char>(ttk::MPIrank_));
      for(int j = 0; j < 2; j++) {
        // ttk::SimplexId gid = triangulation.getVertexGlobalId(mins[j]);
        ttk::SimplexId lid{static_cast<ttk::SimplexId>(extremas.size())};
        auto pair = globalToLocalExtrema.try_emplace(mins[j].gid_, lid);
        if(pair.second) {
          mins[j].lid_ = lid;
          mins[j].rep_.extremaId_ = lid;
          extremas.emplace_back(mins[j]);
          if(globalMinOffset == mins[j].vOrder_[0]) {
            // Mark this as the global min not to be paired
            globalMinLid.emplace_back(lid);
          }
          ttk::SimplexId triangLid
            = triangulation.getVertexLocalId(mins[j].gid_);
          if(triangLid == -1) {
            auto it = localGhostPresenceMap.find(mins[j].gid_);
            if(it != localGhostPresenceMap.end()) {
              ghosts = localGhostPresenceMap[mins[j].gid_];
            } else {
              ghosts.resize(0);
            }
          } else {
            ghosts = localGhostPresenceVector[triangLid];
          }
          ghostPresence.emplace_back(ghosts);
        } else {
          lid = (*pair.first).second;
        }
        e.t_[j] = lid;
      }
      saddles.emplace_back(e);
    }
    const auto cmpSadMin = [=, &extremas](const saddleEdge<2> &s0,
                                          const saddleEdge<2> &s1) -> bool {
      if(&s0 != &s1) {
        if(s0.order_ != -1 && s1.order_ != -1) {
          return s0.order_ < s1.order_;
        }
        for(size_t i = 0; i < 2; i++) {
          if(s0.vOrder_[i] != s1.vOrder_[i]) {
            return s0.vOrder_[i] < s1.vOrder_[i];
          }
        }
      }
      return extremas[s0.t_[0]].vOrder_ > extremas[s1.t_[0]].vOrder_;
    };

    // TRI des arcs
    if(ttk::MPIsize_ == 1) {
      TTK_PSORT(this->threadNumber_, saddles.begin(), saddles.end(), cmpSadMin);
    } else {
      auto rng = std::default_random_engine{0};
      std::shuffle(std::begin(saddles), std::end(saddles), rng);
    }
    // Mise en place des lid des arcs

#pragma omp declare reduction (merge :std::unordered_map<ttk::SimplexId,ttk::SimplexId>:omp_out.merge(omp_in))
#pragma omp parallel for reduction(merge : globalToLocalSaddle) schedule(static)
    for(int i = 0; i < saddles.size(); i++) {
      auto &s{saddles[i]};
      s.lid_ = i;
      globalToLocalSaddle[s.gid_] = i;
    }
    extremaToPairedSaddle.resize(globalToLocalExtrema.size(), -1);
    saddleToPairedExtrema.resize(globalToLocalSaddle.size(), -1);

    MPI_Datatype MPI_MessageType;
    createMPIMessageType<1, 2>(MPI_MessageType);
    tripletsToPersistencePairs<1, 2>(
      pairs, 0, extremas, saddles, saddleToPairedExtrema, extremaToPairedSaddle,
      globalToLocalSaddle, globalToLocalExtrema, ghostPresence, MPI_MessageType,
      true);
    ttk::SimplexId nMinSadPairs
      = computePairNumbers<1, 2>(saddles, saddleToPairedExtrema);
    char rerunNeeded{0};
    // printMsg("STOP HERE");
    // kill(getpid(), SIGINT);
    for(const auto lid : globalMinLid) {
      if(extremaToPairedSaddle[lid] != -1
         && saddles[extremaToPairedSaddle[lid]].rank_ == ttk::MPIrank_) {
        saddleToPairedExtrema[extremaToPairedSaddle[lid]] = -2;
        printMsg("Marked "
                 + std::to_string(saddles[extremaToPairedSaddle[lid]].gid_)
                 + " as needing rerun");
        rerunNeeded = 1;
      }
      printMsg("Not marked for extrema " + std::to_string(extremas[lid].gid_)
               + " extremaToPairedSaddle: "
               + std::to_string(extremaToPairedSaddle[lid]));
    }
    MPI_Allreduce(
      MPI_IN_PLACE, &rerunNeeded, 1, MPI_CHAR, MPI_LOR, ttk::MPIcomm_);
    MPI_Allreduce(
      MPI_IN_PLACE, &nMinSadPairs, 1, MPI_SimplexId, MPI_SUM, ttk::MPIcomm_);
    printErr("Total number of pairs computed: " + std::to_string(nMinSadPairs));

    while((nMinSadPairs != totalNumberOfPairs - 1) || rerunNeeded) {
      printMsg("Re-computation, rerun needed: " + std::to_string(rerunNeeded));
      tripletsToPersistencePairs<1, 2>(
        pairs, 0, extremas, saddles, saddleToPairedExtrema,
        extremaToPairedSaddle, globalToLocalSaddle, globalToLocalExtrema,
        ghostPresence, MPI_MessageType, false);
      nMinSadPairs = computePairNumbers<1, 2>(saddles, saddleToPairedExtrema);
      rerunNeeded = 0;
      for(const auto lid : globalMinLid) {
        if(extremaToPairedSaddle[lid] != -1
           && saddles[extremaToPairedSaddle[lid]].rank_ == ttk::MPIrank_) {
          saddleToPairedExtrema[extremaToPairedSaddle[lid]] = -2;
          rerunNeeded = 1;
        }
      }
      MPI_Allreduce(
        MPI_IN_PLACE, &rerunNeeded, 1, MPI_CHAR, MPI_LOR, ttk::MPIcomm_);
      MPI_Allreduce(
        MPI_IN_PLACE, &nMinSadPairs, 1, MPI_SimplexId, MPI_SUM, ttk::MPIcomm_);
    }
    extractPairs<1, 2>(
      pairs, extremas, saddles, saddleToPairedExtrema, false, 0);
    /*std::ofstream myfile;
    printMsg("Start writing file");
    myfile.open("/home/eveleguillou/experiment/DiscreteMorseSandwich/"
                + std::to_string(ttk::MPIsize_) + "_pairs_"
                + std::to_string(ttk::MPIrank_) + ".csv");
    myfile << "min,sad\n";
    for(ttk::SimplexId i = 0; i < pairs.size(); i++) {
      myfile << std::to_string(pairs[i].birth) + ","
                  + std::to_string(pairs[i].death) + "\n";
    }
    myfile.close();
    printMsg("Finish writing file");*/

    // non-paired minima
    if(totalNumberOfPairs > 1) {
#pragma omp parallel for shared(pairs, nConnComp) \
  num_threads(this->threadNumber_)
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
      if(globalMinOffset == localMinOffset) {
        pairs.emplace_back(triangulation.getVertexGlobalId(localMin), -1, 0);
        nConnComp++;
      }
    }

    this->printMsg("min-saddle pairs sequential part", 1.0,
                   tmseq.getElapsedTime(), 1, debug::LineMode::NEW);

    this->printMsg(
      "Computed " + std::to_string(nMinSadPairs) + " min-saddle pairs", 1.0,
      tm.getElapsedTime(), this->threadNumber_);
  } else {
    if(globalMinOffset == localMinOffset) {
      pairs.emplace_back(triangulation.getVertexGlobalId(localMin), -1, 0);
      nConnComp++;
    }
  }
}

template <typename triangulationType, int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::computeMaxSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const std::vector<SimplexId> &criticalSaddles,
  const std::vector<SimplexId> &critSaddlesOrder,
  const std::vector<SimplexId> &critMaxsOrder,
  const triangulationType &triangulation,
  const bool ignoreBoundary,
  const SimplexId *const offsets,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema) {
  Timer tm{};
  const auto dim = this->dg_.getDimensionality();
  auto saddle2ToMaxima
    = dim == 3
        ? getSaddle2ToMaxima(
          criticalSaddles,
          [&triangulation](const SimplexId a, const SimplexId i, SimplexId &r) {
            return triangulation.getTriangleStar(a, i, r);
          },
          [&triangulation](const SimplexId a) {
            return triangulation.getTriangleStarNumber(a);
          },
          [&triangulation](const SimplexId a) {
            return triangulation.isTriangleOnBoundary(a);
          },
          triangulation)
        : getSaddle2ToMaxima(
          criticalSaddles,
          [&triangulation](const SimplexId a, const SimplexId i, SimplexId &r) {
            return triangulation.getEdgeStar(a, i, r);
          },
          [&triangulation](const SimplexId a) {
            return triangulation.getEdgeStarNumber(a);
          },
          [&triangulation](const SimplexId a) {
            return triangulation.isEdgeOnBoundary(a);
          },
          triangulation);

  Timer tmseq{};
  std::function<ttk::SimplexId(ttk::SimplexId)> getSaddleGlobalId;
  if(dim == 3) {
    getSaddleGlobalId = [&triangulation](ttk::SimplexId lid) {
      return triangulation.getTriangleGlobalId(lid);
    };
  } else {
    getSaddleGlobalId = [&triangulation](ttk::SimplexId lid) {
      return triangulation.getEdgeGlobalId(lid);
    };
  }
  std::function<ttk::SimplexId(ttk::SimplexId)> getMaxGlobalId;
  if(dim == 3) {
    getMaxGlobalId = [&triangulation](ttk::SimplexId lid) {
      return triangulation.getCellGlobalId(lid);
    };
  } else {
    getMaxGlobalId = [&triangulation](ttk::SimplexId lid) {
      return triangulation.getTriangleGlobalId(lid);
    };
  }
  std::function<void(const ttk::SimplexId, ttk::SimplexId *)>
    fillExtremaNodeOrder;
  std::function<void(const ttk::SimplexId, ttk::SimplexId *)>
    fillSaddleEdgeOrder;
  if(dim == 3) {
    fillExtremaNodeOrder = [this, &triangulation, offsets](
                             const ttk::SimplexId id, ttk::SimplexId *vOrd) {
      return fillTetraOrder(id, offsets, triangulation, vOrd);
    };
    fillSaddleEdgeOrder = [this, &triangulation, offsets](
                            const ttk::SimplexId id, ttk::SimplexId *vOrd) {
      return fillTriangleOrder(id, offsets, triangulation, vOrd);
    };
  } else {
    fillExtremaNodeOrder = [this, &triangulation, offsets](
                             const ttk::SimplexId id, ttk::SimplexId *vOrd) {
      return fillTriangleOrder(id, offsets, triangulation, vOrd);
    };
    fillSaddleEdgeOrder = [this, &triangulation, offsets](
                            const ttk::SimplexId id, ttk::SimplexId *vOrd) {
      return fillEdgeOrder(id, offsets, triangulation, vOrd);
    };
  }

  auto &saddleToPairedExtrema{this->saddleToPairedMax_};
  auto &extremaToPairedSaddle{this->maxToPairedSaddle_};

  std::vector<saddleEdge<sizeSad>> saddles{};
  std::vector<extremaNode<sizeExtr>> extremas{};
  std::vector<std::vector<char>> ghostPresence{};
  saddles.reserve(saddle2ToMaxima.size());
  extremas.reserve(2 * saddle2ToMaxima.size());
  ghostPresence.reserve(2 * saddle2ToMaxima.size());
  globalToLocalExtrema.reserve(2 * saddle2ToMaxima.size());
  if(dim == 3) {
    globalToLocalSaddle.reserve(saddle2ToMaxima.size());
  } else {
    globalToLocalSaddle.reserve(globalToLocalSaddle.size()
                                + saddle2ToMaxima.size());
  }
  ttk::SimplexId saddle2ToMaximaNumber
    = static_cast<ttk::SimplexId>(saddle2ToMaxima.size());
  for(ttk::SimplexId i = 0; i < saddle2ToMaximaNumber; i++) {
    auto &maxs = saddle2ToMaxima[i];
    // remove duplicates
    std::sort(
      maxs.begin(), maxs.end(), [](const SimplexId a, const SimplexId b) {
        // positive values (actual maxima) before negative ones
        // (boundary component id)
        if(a * b >= 0) {
          return a < b;
        } else {
          return a > b;
        }
      });
    const auto last = std::unique(maxs.begin(), maxs.end());
    maxs.erase(last, maxs.end());

    // remove "doughnut" configurations: two ascending separatrices
    // leading to the same maximum/boundary component
    if(maxs.size() != 2) {
      continue;
    }

    const auto s2 = criticalSaddles[i];
    bool pairedSaddle = false;
    ttk::SimplexId gid{-1};
    if(dim != 3) {
      gid = getSaddleGlobalId(s2);
      auto it = globalToLocalSaddle.find(gid);
      if((it != globalToLocalSaddle.end())
         && (saddleToPairedMin_[it->second] > -1)) {
        pairedSaddle = true;
      }
    }
    if(!pairedSaddle) {
      if(gid == -1) {
        gid = getSaddleGlobalId(s2);
      }
      ttk::SimplexId vOrd[sizeSad];
      fillSaddleEdgeOrder(s2, vOrd);
      saddleEdge e = saddleEdge<sizeSad>(
        gid, critSaddlesOrder[s2], vOrd, static_cast<char>(ttk::MPIrank_));
      for(int j = 0; j < 2; j++) {
        if(maxs[j] > -1) {
          gid = getMaxGlobalId(maxs[j]);
          ttk::SimplexId lid{static_cast<ttk::SimplexId>(extremas.size())};
          auto pair = globalToLocalExtrema.try_emplace(gid, lid);
          if(pair.second) {
            ttk::SimplexId vOrd2[sizeExtr];
            fillExtremaNodeOrder(maxs[j], vOrd2);
            extremaNode n = extremaNode<sizeExtr>(
              gid, lid, critMaxsOrder[maxs[j]], Rep{lid, -1},
              static_cast<char>(ttk::MPIrank_), vOrd2);
            extremas.emplace_back(n);
          } else {
            lid = (*pair.first).second;
          }
          e.t_[j] = lid;
        }
      }
      saddles.emplace_back(e);
    }
  }
  const auto cmpSadMax
    = [this, &extremas](
        const saddleEdge<sizeSad> &s0, const saddleEdge<sizeSad> &s1) -> bool {
    if(&s0 != &s1) {
      if(s0.order_ != -1 && s1.order_ != -1) {
        return s0.order_ > s1.order_;
      }
      for(size_t i = 0; i < sizeSad; i++) {
        if(s0.vOrder_[i] != s1.vOrder_[i]) {
          return s0.vOrder_[i] < s1.vOrder_[i];
        }
      }
      return s0.gid_ > s1.gid_;
    }

    auto t0 = extremas[s0.t_[1]];
    auto t1 = extremas[s1.t_[1]];
    if(t0.order_ != -1 && t1.order_ != -1) {
      return t0.order_ < t1.order_;
    }
    for(size_t i = 0; i < sizeExtr; i++) {
      if(t0.vOrder_[i] != t1.vOrder_[i]) {
        return t0.vOrder_[i] < t1.vOrder_[i];
      }
    }
    return true;
  };
  // TRI des arcs
  TTK_PSORT(this->threadNumber_, saddles.begin(), saddles.end(), cmpSadMax);
  // auto rng = std::default_random_engine{0};
  // std::shuffle(std::begin(saddles), std::end(saddles), rng);
#pragma omp declare reduction (merge :std::unordered_map<ttk::SimplexId,ttk::SimplexId>:omp_out.merge(omp_in))
#pragma omp parallel for reduction(merge : globalToLocalSaddle) schedule(static)
  for(int i = 0; i < saddle2ToMaximaNumber; i++) {
    auto &s{saddles[i]};
    s.lid_ = i;
    globalToLocalSaddle[s.gid_] = i;
  }

  extremaToPairedSaddle.resize(globalToLocalExtrema.size(), -1);
  saddleToPairedExtrema.resize(saddle2ToMaxima.size(), -1);
  const auto nMinSadPairs = pairs.size();
  MPI_Datatype MPI_MessageType;
  createMPIMessageType<sizeExtr, sizeSad>(MPI_MessageType);
  tripletsToPersistencePairs<sizeExtr, sizeSad>(
    pairs, dim - 1, extremas, saddles, saddleToPairedExtrema,
    extremaToPairedSaddle, globalToLocalSaddle, globalToLocalExtrema,
    ghostPresence, MPI_MessageType, true);

  const auto nSadMaxPairs = pairs.size() - nMinSadPairs;

  this->printMsg(
    "Computed " + std::to_string(nSadMaxPairs) + " saddle-max pairs", 1.0,
    tm.getElapsedTime(), this->threadNumber_);
  this->printMsg("saddle-max pairs sequential part", 1.0,
                 tmseq.getElapsedTime(), 1, debug::LineMode::NEW);
}
template <typename triangulationType>
void ttk::DiscreteMorseSandwichMPI::getMaxSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const std::vector<SimplexId> &criticalSaddles,
  const std::vector<SimplexId> &critSaddlesOrder,
  const std::vector<SimplexId> &critMaxsOrder,
  const triangulationType &triangulation,
  const bool ignoreBoundary,
  const SimplexId *const offsets) {
  Timer t{};
  const auto dim = this->dg_.getDimensionality();
  auto &globalToLocalSaddle{dim == 3 ? this->globalToLocalSaddle2_
                                     : this->globalToLocalSaddle1_};
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> globalToLocalExtrema{};
  if(dim > 1 && this->ComputeSadMax) {
    if(dim == 3) {
      computeMaxSaddlePairs<triangulationType, 4, 3>(
        pairs, criticalSaddles, critSaddlesOrder, critMaxsOrder, triangulation,
        ignoreBoundary, offsets, globalToLocalSaddle, globalToLocalExtrema);
    } else {
      computeMaxSaddlePairs<triangulationType, 3, 2>(
        pairs, criticalSaddles, critSaddlesOrder, critMaxsOrder, triangulation,
        ignoreBoundary, offsets, globalToLocalSaddle, globalToLocalExtrema);
    }
  }
  if(ignoreBoundary) {
    // post-process saddle-max pairs: remove the one with the global
    // maximum (if it exists) to be (more) compatible with FTM
    const auto it
      = std::find_if(pairs.begin(), pairs.end(), [&](const PersistencePair &p) {
          if(p.type < dim - 1) {
            return false;
          }
          // TODO: won't work in distributed
          const Cell cmax{dim, triangulation.getCellLocalId(p.death)};
          const auto vmax{this->getCellGreaterVertex(cmax, triangulation)};
          return offsets[vmax] == triangulation.getNumberOfVertices() - 1;
        });

    if(it != pairs.end()) {
      // remove saddle-max pair with global maximum
      this->saddleToPairedMax_[globalToLocalSaddle[(*it).death]] = -1;
      this->maxToPairedSaddle_[globalToLocalExtrema[(*it).birth]] = -1;
      pairs.erase(it);
    }
  }
}

// get representative of current extremum
template <int sizeExtr, int sizeSad>
struct ttk::DiscreteMorseSandwichMPI::extremaNode<sizeExtr> &
  ttk::DiscreteMorseSandwichMPI::getRep(
    extremaNode<sizeExtr> *extr,
    saddleEdge<sizeSad> *sv,
    bool increasing,
    std::vector<extremaNode<sizeExtr>> &extremas,
    std::vector<saddleEdge<sizeSad>> &saddles) const {
  auto currentNode = extr;
  if(currentNode->rep_.extremaId_ == -1) {
    return extremas[currentNode->lid_];
  }
  auto rep = &extremas[extr->rep_.extremaId_];
  saddleEdge<sizeSad> *s;
  while((*rep) != (*currentNode)) {
    // printMsg("In getRep: "+std::to_string(sv->gid_)+",
    // "+std::to_string(currentNode->gid_)+", "+std::to_string(rep->gid_));
    /*if(sv->gid_ == 364822) {
      printMsg("saddle " + std::to_string(sv->gid_) + "("
               + std::to_string(sv->vOrder_[0]) + ")" + ", "
               + std::to_string(extr->gid_) + ", "
               + std::to_string(currentNode->gid_) + ", by "
               + std::to_string(saddles[currentNode->rep_.saddleId_].gid_) + "("
               + std::to_string(saddles[currentNode->rep_.saddleId_].vOrder_[0])
               + ")" + ", next: " + std::to_string(rep->gid_));
    }*/
    if(currentNode->rep_.extremaId_ == -1) {
      break;
    }
    /*if(sv->gid_ == 1112) {
      printMsg("saddle " + std::to_string(sv->gid_) + "("
               + std::to_string(sv->vOrder_[0]) + ","
               + std::to_string(sv->order_) + ")" + ", "
               + std::to_string(extr->gid_) + ", "
               + std::to_string(currentNode->gid_) + "("
               + std::to_string(saddles[currentNode->rep_.saddleId_].gid_) + ","
               + std::to_string(saddles[currentNode->rep_.saddleId_].vOrder_[0])
               + ")" + ", next: " + std::to_string(rep->gid_) + ", "
               + std::to_string(rep->rep_.saddleId_) + "("
               + std::to_string(saddles[rep->rep_.saddleId_].gid_) + "), "
               + std::to_string(currentNode->rep_.saddleId_));
    }*/
    // Test if ghost
    // if(currentNode->rank_ != ttk::MPIrank_) {
    //  if (sv->gid_ == 1112){
    //   printMsg("saddle "+std::to_string(sv->gid_)+",
    //   "+std::to_string(extr->gid_)+", "+std::to_string(currentNode->gid_)+",
    //   next: "+std::to_string(rep->gid_)+",
    //   rank:"+std::to_string(currentNode->rank_));
    //  }
    //  return extremas[currentNode->lid_];
    //}
    if(currentNode->rep_.saddleId_ > -1) {
      s = &saddles[currentNode->rep_.saddleId_];
      if((s->gid_ == sv->gid_) || (((*s) < (*sv)) == increasing)) {
        break;
      }
    }
    currentNode = rep;
    if(currentNode->rep_.extremaId_ == -1) {
      break;
    }
    rep = &extremas[currentNode->rep_.extremaId_];
  }
  /*if(sv->gid_ == 364822) {
    printMsg("final saddle " + std::to_string(sv->gid_) + ", "
             + std::to_string(extr->gid_) + ", "
             + std::to_string(currentNode->gid_));
  }*/
  return extremas[currentNode->lid_];
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::addPair(
  const saddleEdge<sizeSad> &sad,
  const extremaNode<sizeExtr> &extr,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle) const {
  /*if(extr.gid_ == 8 || extr.gid_ == 232 || extr.gid_ == 125 || extr.gid_ ==
     190
     || extr.gid_ == 317 || extr.gid_ == 190 || sad.gid_ == 652
     || sad.gid_ == 1112 || sad.gid_ == 1112 || sad.gid_ == 98
     || sad.gid_ == 98) {*/
  /*if(sad.gid_ == 264) {
    printMsg("AddPair: " + std::to_string(sad.gid_) + "("
             + std::to_string(sad.vOrder_[0]) + "," + std::to_string(sad.order_)
             + "," + std::to_string(sad.rank_) + "), "
             + std::to_string(extr.gid_) + "(" + std::to_string(extr.vOrder_[0])
             + "," + std::to_string(extr.rank_) + "), "
             + std::to_string(extr.lid_));
    kill(getpid(), SIGINT);
  }*/
  saddleToPairedExtrema[sad.lid_] = extr.lid_;
  extremaToPairedSaddle[extr.lid_] = sad.lid_;
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::removePair(
  const saddleEdge<sizeSad> &sad,
  const extremaNode<sizeExtr> &extr,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle) const {
  /*if(sad.gid_ == 1112 || sad.gid_ == 278 || extr.gid_ == 326
     || sad.gid_ == 170) {
    printMsg("removePair: " + std::to_string(sad.gid_) + "("
             + std::to_string(sad.vOrder_[0]) + "," + std::to_string(sad.order_)
             + "," + std::to_string(sad.rank_) + "), "
             + std::to_string(extr.gid_) + "(" + std::to_string(extr.vOrder_[0])
             + "," + std::to_string(extr.rank_) + ")");
  }*/

  if(extremaToPairedSaddle[extr.lid_] == sad.lid_
     || extremaToPairedSaddle[extr.lid_] == -1) {
    extremaToPairedSaddle[extr.lid_] = -1;
  } else {
    printErr("HAPPENING HERE FOR " + std::to_string(sad.gid_) + " and "
             + std::to_string(extr.gid_) + " (true: "
             + std::to_string(extremaToPairedSaddle[extr.lid_]) + ")");
    extremaToPairedSaddle[saddleToPairedExtrema[sad.lid_]] = -1;
  }
  saddleToPairedExtrema.at(sad.lid_) = -1;
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::tripletsToPersistencePairs(
  std::vector<PersistencePair> &pairs,
  const SimplexId pairDim,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  std::vector<std::vector<char>> ghostPresence,
  MPI_Datatype &MPI_MessageType,
  bool isFirstTime) const {
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> sendBuffer(
    ttk::MPIsize_, std::vector<messageType<sizeExtr, sizeSad>>());
  const bool increasing = (pairDim > 0);
  // Timer tm{};
  // saddleToPairedExtremaTime = tm.getElapsedTime();
  if(isFirstTime) {
    for(const auto &s : saddles) {
      processTriplet<sizeExtr, sizeSad>(
        s, saddleToPairedExtrema, extremaToPairedSaddle, saddles, extremas,
        increasing, ghostPresence, sendBuffer);
    }
  } else {
    for(const auto &s : saddles) {
      if(saddleToPairedExtrema[s.lid_] == -2) {
        processTriplet<sizeExtr, sizeSad>(
          s, saddleToPairedExtrema, extremaToPairedSaddle, saddles, extremas,
          increasing, ghostPresence, sendBuffer);
      }
    }
  }
  const auto cmpSadMin
    = [=](const messageType<sizeExtr, sizeSad> &elt0,
          const messageType<sizeExtr, sizeSad> &elt1) -> bool {
    if(elt0.s_ != elt0.s_) {
      for(size_t i = 0; i < sizeSad; i++) {
        if(elt0.sOrder_[i] != elt1.sOrder_[i]) {
          return elt0.sOrder_[i] < elt1.sOrder_[i]; // TODO: add increasing
        }
      }
    }
    return elt0.t1Order_[0] > elt1.t1Order_[0];
  };
  // Receive elements
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> recvBuffer(
    ttk::MPIsize_, std::vector<messageType<sizeExtr, sizeSad>>());
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> tempSendBuffer(
    ttk::MPIsize_, std::vector<messageType<sizeExtr, sizeSad>>());
  ttk::SimplexId hasSentMessages{1};
  MPI_Datatype MPI_SimplexId = getMPIType(hasSentMessages);
  while(hasSentMessages > 0) {
    if(ttk::MPIrank_ == 0)
      printMsg("While loop iteration: " + std::to_string(hasSentMessages));
    ttk::SimplexId localSentMessageNumber{0};
    std::vector<MPI_Request> sendRequests(ttk::MPIsize_ - 1);
    std::vector<MPI_Request> recvRequests(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> sendStatus(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> recvStatus(ttk::MPIsize_ - 1);
    std::vector<ttk::SimplexId> sendMessageSize(ttk::MPIsize_, 0);
    std::vector<ttk::SimplexId> recvMessageSize(ttk::MPIsize_, 0);
    std::vector<int> recvCompleted(ttk::MPIsize_ - 1, 0);
    std::vector<int> sendCompleted(ttk::MPIsize_ - 1, 0);
    int sendPerformedCount = 0;
    int recvPerformedCount = 0;
    int sendPerformedCountTotal = 0;
    int recvPerformedCountTotal = 0;
    int count = 0;
    for(int i = 0; i < ttk::MPIsize_; i++) {
      // Send size of Sendbuffer
      if(i != ttk::MPIrank_) {
        sendMessageSize[i] = sendBuffer.at(i).size();
        localSentMessageNumber += sendMessageSize[i];
        MPI_Isend(&sendMessageSize[i], 1, MPI_SimplexId, i, 0, ttk::MPIcomm_,
                  &sendRequests[count]);
        MPI_Irecv(&recvMessageSize[i], 1, MPI_SimplexId, i, 0, ttk::MPIcomm_,
                  &recvRequests[count]);
        count++;
      }
    }
    std::vector<MPI_Request> sendRequestsData(ttk::MPIsize_ - 1);
    std::vector<MPI_Request> recvRequestsData(ttk::MPIsize_ - 1);
    std::vector<MPI_Status> recvStatusData(ttk::MPIsize_ - 1);
    int recvCount = 0;
    int sendCount = 0;
    int r;
    while((sendPerformedCountTotal < ttk::MPIsize_ - 1
           || recvPerformedCountTotal < ttk::MPIsize_ - 1)) {
      if(sendPerformedCountTotal < ttk::MPIsize_ - 1) {
        MPI_Waitsome(ttk::MPIsize_ - 1, sendRequests.data(),
                     &sendPerformedCount, sendCompleted.data(),
                     sendStatus.data());
        if(sendPerformedCount > 0) {
          for(int i = 0; i < sendPerformedCount; i++) {
            r = sendCompleted[i];
            if(ttk::MPIrank_ <= sendCompleted[i]) {
              r++;
            }
            if((sendMessageSize[r] > 0)) {
              MPI_Isend(sendBuffer.at(r).data(), sendMessageSize[r],
                        MPI_MessageType, r, 1, ttk::MPIcomm_,
                        &sendRequestsData[sendCount]);
              sendCount++;
            }
          }
          sendPerformedCountTotal += sendPerformedCount;
        }
      }
      if(recvPerformedCountTotal < ttk::MPIsize_ - 1) {
        MPI_Waitsome(ttk::MPIsize_ - 1, recvRequests.data(),
                     &recvPerformedCount, recvCompleted.data(),
                     recvStatus.data());
        if(recvPerformedCount > 0) {
          for(int i = 0; i < recvPerformedCount; i++) {
            r = recvStatus[i].MPI_SOURCE;
            if((recvMessageSize[r] > 0)) {
              recvBuffer.at(r).resize(recvMessageSize[r]);
              MPI_Irecv(recvBuffer.at(r).data(), recvMessageSize[r],
                        MPI_MessageType, r, 1, ttk::MPIcomm_,
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
          // TODO: Don't forget to sort them first
          auto rng = std::default_random_engine{0};
          std::shuffle(
            std::begin(recvBuffer.at(r)), std::end(recvBuffer.at(r)), rng);
          /*std::sort(
            recvBuffer.at(r).begin(), recvBuffer.at(r).end(), cmpSadMin);*/
          //#pragma omp parallel for schedule(static)
          for(ttk::SimplexId j = 0; j < recvMessageSize[r]; j++) {
            receiveElement<sizeExtr, sizeSad>(
              recvBuffer.at(r).at(j), globalToLocalSaddle, globalToLocalExtrema,
              saddles, extremas, extremaToPairedSaddle, saddleToPairedExtrema,
              tempSendBuffer, ghostPresence, static_cast<char>(r), increasing);
          }
        }
        recvPerformedCountTotal += recvPerformedCount;
      }
    }
    MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);
    /*for(int i = 0; i < ttk::MPIsize_; i++) {
      for(ttk::SimplexId j = 0; j < recvMessageSize[i]; j++) {
        receiveElement<sizeExtr, sizeSad>(
          recvBuffer.at(i).at(j), globalToLocalSaddle, globalToLocalExtrema,
          saddles, extremas, extremaToPairedSaddle, saddleToPairedExtrema,
          tempSendBuffer, ghostPresence, static_cast<char>(i), increasing);
      }
    }*/
    // Stop condition computation
    MPI_Allreduce(&localSentMessageNumber, &hasSentMessages, 1, MPI_SimplexId,
                  MPI_SUM, ttk::MPIcomm_);
    if(hasSentMessages) {
      for(int i = 0; i < ttk::MPIsize_; i++) {
        sendBuffer.at(i).clear();
        sendBuffer.at(i).insert(sendBuffer.at(i).end(),
                                tempSendBuffer.at(i).begin(),
                                tempSendBuffer.at(i).end());
        tempSendBuffer.at(i).clear();
        recvBuffer.at(i).clear();
      }
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
  const int pairDim) const {
  ttk::SimplexId saddleNumber = saddleToPairedExtrema.size();

#ifdef TTK_ENABLE_OPENMP
#pragma omp declare reduction (merge : std::vector<PersistencePair> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : pairs) schedule(static)
#endif
  for(int i = 0; i < saddleNumber; i++) {
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
int ttk::DiscreteMorseSandwichMPI::computePairNumbers(
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema) const {
  ttk::SimplexId saddleNumber = saddleToPairedExtrema.size();
  ttk::SimplexId computedSaddleNumber{0};
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for reduction(+ : computedSaddleNumber) schedule(static)
#endif
  for(ttk::SimplexId i = 0; i < saddleNumber; i++) {
    if(saddleToPairedExtrema[i] > -1 && saddles[i].rank_ == ttk::MPIrank_) {
      computedSaddleNumber++;
    }
  }
  return computedSaddleNumber;
}

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::receiveElement(
  messageType<sizeExtr, sizeSad> &element,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalSaddle,
  std::unordered_map<ttk::SimplexId, ttk::SimplexId> &globalToLocalExtrema,
  std::vector<saddleEdge<sizeSad>> &saddles,
  std::vector<extremaNode<sizeExtr>> &extremas,
  std::vector<ttk::SimplexId> &extremaToPairedSaddle,
  std::vector<ttk::SimplexId> &saddleToPairedExtrema,
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  std::vector<std::vector<char>> &ghostPresence,
  char sender,
  bool increasing) const {
  /*if(element.s_ == 6349 && ttk::MPIrank_ == 2) {
    printMsg("Receive element " + std::to_string(element.s_) + " from "
             + std::to_string(sender)  + ", "
             + std::to_string(element.t1_) + ", "
             + std::to_string(element.t2_));
    printMsg("globalToLocalExtrema:
  "+std::to_string(globalToLocalExtrema[element.t1_]));
    //kill(getpid(), SIGINT);
  }  */
  const auto getSaddle = [this, &globalToLocalSaddle, &saddles](
                           const ttk::SimplexId saddleGid,
                           ttk::SimplexId *vOrder, const char rank) {
    auto it = globalToLocalSaddle.find(saddleGid);
    if(it != globalToLocalSaddle.end()) {
      return saddles[it->second];
    }
    return saddleEdge<sizeSad>(saddleGid, -1, vOrder, rank);
  };

  const auto getLocalExtrema = [this, &globalToLocalExtrema,
                                &extremas](const ttk::SimplexId extremaGid) {
    auto it = globalToLocalExtrema.find(extremaGid);
    if(it != globalToLocalExtrema.end()) {
      return it->second;
    }
    return static_cast<ttk::SimplexId>(-1);
  };

  const auto addSaddle = [this, &globalToLocalSaddle, &saddles,
                          &saddleToPairedExtrema](saddleEdge<sizeSad> s) {
    if(s.lid_ == -1 && s.gid_ > -1) {
      globalToLocalSaddle[s.gid_] = saddles.size();
      s.lid_ = saddles.size();
      saddles.emplace_back(s);
      s = saddles.back();
      // saddleToPairedExt.resize(saddleToPairedExt.size()+1, -1);
      saddleToPairedExtrema.emplace_back(static_cast<ttk::SimplexId>(-1));
    }
    return s;
  };

  const auto getUpdatedT1 =
    [this, &globalToLocalExtrema, &extremas, &increasing, &saddles](
      const ttk::SimplexId extremaGid, messageType<sizeExtr, sizeSad> &elt,
      saddleEdge<sizeSad> s) {
      extremaNode<sizeExtr> rep;
      ttk::SimplexId lid{-1};
      auto it = globalToLocalExtrema.find(extremaGid);
      if(it != globalToLocalExtrema.end()) {
        lid = it->second;
        auto e{extremas[lid]};
        rep = getRep(&e, &s, increasing, extremas, saddles);
        if((rep.gid_ != e.gid_)
           && !(
             compareArray(s.vOrder_, saddles[e.rep_.saddleId_].vOrder_, sizeSad)
             == increasing)) {
          printErr("YES IT HAPPENS");
        }
        // TODO: check niveau de rep/extr
        if(rep.gid_ != e.gid_) {
          lid = globalToLocalExtrema.find(rep.gid_)->second;
          elt.t1_ = rep.gid_;
          for(int i = 0; i < sizeExtr; i++) {
            elt.t1Order_[i] = rep.vOrder_[i];
          }
          elt.t1Rank_ = rep.rank_;
          elt.hasBeenModified_ = 1;
          saddleEdge<sizeSad> s1;
          if(rep.rep_.saddleId_ > -1) {
            s1 = saddles[rep.rep_.saddleId_];
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
            }
            if(compareArray(s1Loc.vOrder_, elt.s1Order_, sizeSad)
               == increasing) {
              elt.s1_ = s1Loc.gid_;
              elt.s1Rank_ = s1Loc.rank_;
              for(int i = 0; i < sizeSad; i++) {
                elt.s1Order_[i] = s1Loc.vOrder_[i];
              }
            }
          }
        }
      }
      return lid;
    };

  const auto getUpdatedT2 =
    [this, &globalToLocalExtrema, &extremas, &increasing, &saddles](
      const ttk::SimplexId extremaGid, messageType<sizeExtr, sizeSad> &elt,
      saddleEdge<sizeSad> s) {
      extremaNode<sizeExtr> rep;
      ttk::SimplexId lid{-1};
      auto it = globalToLocalExtrema.find(extremaGid);
      if(it != globalToLocalExtrema.end()) {
        lid = it->second;
        auto e{extremas[lid]};
        rep = getRep(&e, &s, increasing, extremas, saddles);
        // TODO: check niveau de rep/extr
        if((rep.gid_ != e.gid_)
           && !(
             compareArray(s.vOrder_, saddles[e.rep_.saddleId_].vOrder_, sizeSad)
             == increasing)) {
          printErr("YES IT HAPPENS");
        }
        if(rep.gid_ != e.gid_) {
          lid = globalToLocalExtrema.find(rep.gid_)->second;
          elt.t2_ = rep.gid_;
          for(int i = 0; i < sizeExtr; i++) {
            elt.t2Order_[i] = rep.vOrder_[i];
          }
          elt.t2Rank_ = rep.rank_;
          elt.hasBeenModified_ = 1;
          saddleEdge<sizeSad> s2;
          if(rep.rep_.saddleId_ > -1) {
            s2 = saddles[rep.rep_.saddleId_];
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
            }
            if(compareArray(s2Loc.vOrder_, elt.s2Order_, sizeSad)
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
      return lid;
    };

  const auto swapT1T2 = [this, &increasing, &sendBuffer](
                          messageType<sizeExtr, sizeSad> &elt,
                          ttk::SimplexId &t1Lid, ttk::SimplexId &t2Lid) {
    if(elt.t1_ != elt.t2_) {
      bool pairedR2 = elt.s2_ != -1;
      bool isR2Invalid
        = ((elt.s2_ != -1)
           && (compareArray(elt.s2Order_, elt.sOrder_, sizeSad) == increasing));
      if(isR2Invalid)
        pairedR2 = false;
      bool pairedR1 = elt.s1_ != -1;
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

  const auto addLocalExtrema
    = [this, &extremas, &globalToLocalExtrema, &extremaToPairedSaddle](
        ttk::SimplexId &lid, const ttk::SimplexId gid, char rank,
        ttk::SimplexId *vOrder) {
        if(lid == -1) {
          lid = extremas.size();
          extremaNode n
            = extremaNode<sizeExtr>(gid, lid, -1, Rep{-1, -2}, rank, vOrder);
          extremas.emplace_back(n);
          extremaToPairedSaddle.emplace_back(-1);
          globalToLocalExtrema[gid] = lid;
        }
      };

  /*if(element.s_ == 1112 || element.s_ == 170) {
    printMsg("Receive element " + std::to_string(element.s_) + ", "
             + std::to_string(element.t1_) + ", " + std::to_string(element.t2_)
             + " with hasBeenModified: "
             + std::to_string(element.hasBeenModified_) + " from "
             + std::to_string(sender));
    //kill(getpid(), SIGINT);
  }*/

  /*if(element.s_ == 170 || element.s_ == 650) {
    printMsg("Receive element " + std::to_string(element.s_) + ", "
             + std::to_string(element.t1_) + ", " + std::to_string(element.t2_)
             + " with hasBeenModified: "
             + std::to_string(element.hasBeenModified_) + " from "
             + std::to_string(sender));
    // kill(getpid(), SIGINT);
  }*/

  struct saddleEdge<sizeSad> s
    = getSaddle(element.s_, element.sOrder_, element.sRank_);
  if(s.rank_ == ttk::MPIrank_ && element.t1_ == -1 && element.t2_ == -1) {
    if(saddleToPairedExtrema[s.lid_] > -1) {
      removePair(saddles[s.lid_], extremas[saddleToPairedExtrema[s.lid_]],
                 saddleToPairedExtrema, extremaToPairedSaddle);
    }
    processTriplet<sizeExtr>(saddles[s.lid_], saddleToPairedExtrema,
                             extremaToPairedSaddle, saddles, extremas,
                             increasing, ghostPresence, sendBuffer);
    return;
  }
  ttk::SimplexId t1Lid = getUpdatedT1(element.t1_, element, s);
  ttk::SimplexId t2Lid = getUpdatedT2(element.t2_, element, s);
  /*if(element.s_ == 6349 && element.t1_ == 39 && element.t2_ == 675) {
    printMsg("Updated element " + std::to_string(element.s_) + ", "
             + std::to_string(element.t1_) + ", "
             + std::to_string(element.t2_)+ " hasBeenMod:
  "+std::to_string(element.hasBeenModified_));
    //kill(getpid(), SIGINT);
  }*/

  /*if(element.s_ == 650 || element.s_ == 170) {
    printMsg("Updated element " + std::to_string(element.s_) + ", "
             + std::to_string(element.t1_) + ", "
             + std::to_string(element.t2_));
    // kill(getpid(), SIGINT);
  }*/


  swapT1T2(element, t1Lid, t2Lid);

  if(element.sRank_ != ttk::MPIrank_) {
    // Send now if the extrema rep is -1
    if(t1Lid == -1 || extremas[t1Lid].rep_.extremaId_ == -1) {
      element.hasBeenModified_ = 1;
      sendBuffer.at(element.t1Rank_).emplace_back(element);
    } else {
      // Send back to owner of s if hasBeenModified
      if(element.hasBeenModified_) {
        element.hasBeenModified_ = 0; // TODO: intérêt?
        sendBuffer.at(s.rank_).emplace_back(element);
        element.hasBeenModified_ = 1;
      }
      // If t1 present locally
      extremaNode<sizeExtr> &rep1 = extremas[t1Lid];
      // Who is t1 paired with?
      ttk::SimplexId ls1Id = extremaToPairedSaddle[t1Lid];
      if(element.t1_ == element.t2_) {
        if(s.lid_ > -1 && saddleToPairedExtrema[s.lid_] > -1) {
          auto &t1{extremas[saddleToPairedExtrema[s.lid_]]};
          extremaToPairedSaddle[saddleToPairedExtrema[s.lid_]] = -1;
          if(t1.gid_ == rep1.gid_) {
            t1.rep_.extremaId_ = t1.lid_;
            t1.rep_.saddleId_ = -1;
            if(element.t1Rank_ != ttk::MPIrank_ && element.t1Rank_ != sender) {
              sendBuffer.at(element.t1Rank_).emplace_back(element);
            } else {
              if(t1.rank_ == ttk::MPIrank_
                 && ghostPresence[t1.lid_].size() > 1) {
                for(const auto rank : ghostPresence[t1.lid_]) {
                  if((rank != ttk::MPIrank_)
                     && ((rank == sender && element.hasBeenModified_)
                         || rank != sender)) {
                    sendBuffer.at(rank).emplace_back(element);
                  }
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
          if((ls1 < s) == increasing) {
            // s1 is incorrect
            ttk::SimplexId oldSaddleId = rep1.rep_.saddleId_;
            removePair(ls1, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
            s = addSaddle(s);
            addLocalExtrema(
              t2Lid, element.t2_, element.t2Rank_, element.t2Order_);
            addPair(s, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
            rep1.rep_.extremaId_ = t2Lid;
            rep1.rep_.saddleId_ = s.lid_;
            // If t1 owned but with !ghostPresence.empty() -> send to
            // ghostPresence
            saddleEdge<sizeSad> lst1;
            saddleEdge<sizeSad> lst2;
            if(rep1.rep_.saddleId_ > -1) {
              lst1 = saddles[rep1.rep_.saddleId_];
            }
            if(extremas[t2Lid].rep_.saddleId_ > -1) {
              lst2 = saddles[extremas[t2Lid].rep_.saddleId_];
            }
            storeMessageToSend<sizeExtr, sizeSad>(
              ghostPresence, sendBuffer, s, lst1, lst2, rep1, extremas[t2Lid],
              sender, element.hasBeenModified_);
            // Re-compute for incorrect s1
            if(saddles[ls1.lid_].rank_ == ttk::MPIrank_) {
              processTriplet<sizeExtr, sizeSad>(
                saddles[ls1.lid_], saddleToPairedExtrema, extremaToPairedSaddle,
                saddles, extremas, increasing, ghostPresence, sendBuffer);
            } else {
              // Send s1 for re-computation
              storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddleId]);
            }
          }
        } else {
          // Update rep + send update to other processes
          s = addSaddle(s);
          addLocalExtrema(
            t2Lid, element.t2_, element.t2Rank_, element.t2Order_);
          // printMsg("Add saddles: "+std::to_string(s.gid_)+", s2:
          // "+std::to_string(s2.gid_)); kill(getpid(), SIGINT);
          addPair(s, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
          if(rep1.rep_.extremaId_ != -1) {
            rep1.rep_.extremaId_ = t2Lid;
            rep1.rep_.saddleId_ = s.lid_;
          }
        }
      }
    }
  } else {
    if(t1Lid != -1 && extremas[t1Lid].rep_.extremaId_ == -1
       && element.hasBeenModified_ == 1) {
      sendBuffer.at(element.t1Rank_).emplace_back(element);
    } else {
      if(element.t1_ == element.t2_) {
        if(saddleToPairedExtrema[s.lid_] > -1) {
          auto &t1{extremas[saddleToPairedExtrema[s.lid_]]};
          extremaToPairedSaddle[saddleToPairedExtrema[s.lid_]] = -1;
          ttk::SimplexId r1Lid = getLocalExtrema(element.t1_);
          if(r1Lid != -1 && t1.gid_ == extremas[r1Lid].gid_) {
            if(t1.rep_.extremaId_ != -1) {
              t1.rep_.extremaId_ = t1.lid_;
              t1.rep_.saddleId_ = -1;
            }
          }
          if(t1.rank_ == ttk::MPIrank_ && ghostPresence[t1.lid_].size() > 1) {
            for(const auto rank : ghostPresence[t1.lid_]) {
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
        addLocalExtrema(t1Lid, element.t1_, element.t1Rank_, element.t1Order_);
        // If saddle is not paired at all
        if(extremaToPairedSaddle[t1Lid] < 0
           && saddleToPairedExtrema[s.lid_] < 0) {
          // If the paired extrema is the global min, re-computation is
          // triggered
          extremaNode<sizeExtr> &rep1 = extremas[t1Lid];
          addPair(s, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
          ttk::SimplexId oldSaddle = rep1.rep_.saddleId_;
          // s2 = addSaddle(s2);
          addLocalExtrema(
            t2Lid, element.t2_, element.t2Rank_, element.t2Order_);
          extremaNode<sizeExtr> &rep2 = extremas[t2Lid];
          if(rep1.rep_.extremaId_ != -1) {
            rep1.rep_.extremaId_ = t2Lid;
            rep1.rep_.saddleId_ = s.lid_;
          }
          if(rep1.rank_ != ttk::MPIrank_
             || (ghostPresence[rep1.lid_].size() > 1)) {
            saddleEdge<sizeSad> ls1;
            saddleEdge<sizeSad> ls2;
            if(oldSaddle > -1) {
              ls1 = saddles[oldSaddle];
            }
            if(rep2.rep_.saddleId_ > -1) {
              ls2 = saddles[rep2.rep_.saddleId_];
            }
            storeMessageToSend<sizeExtr, sizeSad>(
              ghostPresence, sendBuffer, s, ls1, ls2, rep1, rep2);
          }
        }
        bool isPairedWithWrongSaddle
          = ((extremaToPairedSaddle[t1Lid] > -1)
             && (extremaToPairedSaddle[t1Lid] != s.lid_));
        struct saddleEdge<sizeSad> sCurrent;
        if(extremaToPairedSaddle[t1Lid] > -1) {
          sCurrent = saddles[extremaToPairedSaddle[t1Lid]];
        }
        bool isPairedWithWrongExtrema
          = ((saddleToPairedExtrema[s.lid_] > -1)
             && (saddleToPairedExtrema[s.lid_] != t1Lid));

        if(isPairedWithWrongExtrema) {
          extremaNode<sizeExtr> &extr = extremas[saddleToPairedExtrema[s.lid_]];
          removePair(s, extr, saddleToPairedExtrema, extremaToPairedSaddle);
          if(extr.rep_.extremaId_ != -1) {
            extr.rep_.extremaId_ = extr.lid_;
            extr.rep_.saddleId_ = -1;
          }
        }
        extremaNode<sizeExtr> &rep1 = extremas[t1Lid];
        // extrema paired to wrong saddle
        if(isPairedWithWrongSaddle && ((sCurrent < s) == increasing)) {
          // If the paired extrema is the global min, re-computation is
          // triggered The new saddle is right, the old one is false
          removePair(
            sCurrent, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
        }
        if((isPairedWithWrongSaddle && ((sCurrent < s) == increasing))
           || isPairedWithWrongExtrema) {
          addPair(s, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
          ttk::SimplexId oldSaddle = rep1.rep_.saddleId_;
          // s2 = addSaddle(s2);
          addLocalExtrema(
            t2Lid, element.t2_, element.t2Rank_, element.t2Order_);
          extremaNode<sizeExtr> &rep2 = extremas[t2Lid];
          if(rep1.rep_.extremaId_ != -1) {
            rep1.rep_.extremaId_ = t2Lid;
            rep1.rep_.saddleId_ = s.lid_;
          }
          if(rep1.rank_ != ttk::MPIrank_
             || (ghostPresence[rep1.lid_].size() > 1)) {
            saddleEdge<sizeSad> ls1;
            saddleEdge<sizeSad> ls2;
            if(oldSaddle > -1) {
              ls1 = saddles[oldSaddle];
            }
            if(rep2.rep_.saddleId_ > -1) {
              ls2 = saddles[rep2.rep_.saddleId_];
            }
            storeMessageToSend<sizeExtr, sizeSad>(
              ghostPresence, sendBuffer, s, ls1, ls2, rep1, rep2);
          }
          if(sCurrent.gid_ > -1 && sCurrent.gid_ != s.gid_) {
            if(sCurrent.rank_ == ttk::MPIrank_) {
              processTriplet<sizeExtr, sizeSad>(
                sCurrent, saddleToPairedExtrema, extremaToPairedSaddle, saddles,
                extremas, increasing, ghostPresence, sendBuffer);
            } else {
              storeRerunToSend<sizeExtr>(sendBuffer, sCurrent);
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
    sendBuffer.at(rep1.rank_).emplace_back(m);
  } else {
    for(auto r : ghostPresence[rep1.lid_]) {
      if(sender != r || hasBeenModified) {
        sendBuffer.at(r).emplace_back(m);
      }
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
    sendBuffer.at(rep1.rank_).emplace_back(m);
  } else {
    for(auto r : ghostPresence[rep1.lid_]) {
      if(sender != r || hasBeenModified) {
        sendBuffer.at(r).emplace_back(m);
      }
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
  sendBuffer.at(rep1.rank_).emplace_back(m);
};

template <int sizeExtr, int sizeSad>
void ttk::DiscreteMorseSandwichMPI::storeRerunToSend(
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer,
  saddleEdge<sizeSad> &sv) const {
  messageType<sizeExtr, sizeSad> m
    = messageType<sizeExtr, sizeSad>(sv.gid_, sv.vOrder_, sv.rank_);
  /*printErr("IN RERUN TO SEND: " + std::to_string(sv.gid_) + " to "
           + std::to_string(sv.rank_));*/
  sendBuffer.at(sv.rank_).emplace_back(m);
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
  std::vector<std::vector<messageType<sizeExtr, sizeSad>>> &sendBuffer) const {
  /*if(sv.gid_ == 112) {
    printErr("Process saddle " + std::to_string(sv.gid_));
    // kill(getpid(), SIGINT);
  }*/
  // rep1 is either last correct in local or a ghost
  auto &rep1 = getRep(&extremas[sv.t_[0]], &sv, increasing, extremas, saddles);
  bool pairedR1 = extremaToPairedSaddle[rep1.lid_] > -1;
  bool isR1Invalid = ((rep1.rep_.saddleId_ > -1)
                      && ((saddles[rep1.rep_.saddleId_] < sv) == increasing));
  ttk::SimplexId oldSaddle{-1};
  if(isR1Invalid)
    pairedR1 = false;
  if(sv.t_[1] < 0) {
    // deal with "shadow" triplets (a 2-saddle with only one
    // ascending 1-separatrix leading to an unique maximum)
    if(!pairedR1 && saddleToPairedExtrema[sv.lid_] == -1) {

      // when considering the boundary, the "-1" of the triplets
      // indicate a virtual maximum of infinite persistence on the
      // boundary component. a pair is created with the other
      // maximum
      if(isR1Invalid) {
        oldSaddle = rep1.rep_.saddleId_;
        removePair(saddles[rep1.rep_.saddleId_], rep1, saddleToPairedExtrema,
                   extremaToPairedSaddle);
      }
      addPair(sv, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
      // If extrema is has local id, then is present in local TODO: CAREFUL:
      // NOT TRUE extrema can be present in triangulation but not graph
      if(rep1.rank_ != ttk::MPIrank_ || (ghostPresence[rep1.lid_].size() > 1)) {
        saddleEdge<sizeSad> s1;
        if(oldSaddle > -1) {
          s1 = saddles[oldSaddle];
        }
        storeMessageToSend<sizeExtr, sizeSad>(
          ghostPresence, sendBuffer, sv, s1, rep1);
      }
      // If ghost or on the border: send message, do what happens next?
      if(rep1.rep_.extremaId_ != -1) {
        rep1.rep_.extremaId_ = rep1.lid_;
        rep1.rep_.saddleId_ = sv.lid_;
      }
      if(isR1Invalid) {
        if(saddles[oldSaddle].rank_ == ttk::MPIrank_) {
          return processTriplet<sizeExtr, sizeSad>(
            saddles[oldSaddle], saddleToPairedExtrema, extremaToPairedSaddle,
            saddles, extremas, increasing, ghostPresence, sendBuffer);
        } else {
          storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddle]);
        }
      }
    }
    return 0;
  }
  auto &rep2 = getRep(&extremas[sv.t_[1]], &sv, increasing, extremas, saddles);
  /*if(sv.gid_ == 650) {
    printErr("Process saddle " + std::to_string(sv.gid_)
             + ", rep1: " + std::to_string(rep1.gid_)
             + ", rep2: " + std::to_string(rep2.gid_));
  }*/
  bool pairedR2 = extremaToPairedSaddle[rep2.lid_] > -1;
  bool isR2Invalid = ((rep2.rep_.saddleId_ > -1)
                      && ((saddles[rep2.rep_.saddleId_] < sv) == increasing));
  if(isR2Invalid)
    pairedR2 = false;
  if(rep2.rep_.extremaId_ == -1) {
    // Send to owner to continue computation
    storeMessageToSendToRepOwner(sendBuffer, sv, saddles, rep2, rep1);
    return 0;
  } else {
    if(rep1.rep_.extremaId_ == -1) {
      // Send to owner to continue computation
      storeMessageToSendToRepOwner(sendBuffer, sv, saddles, rep1, rep2);
      return 0;
    }
  }
  if(rep1.gid_ != rep2.gid_) {
    if((((rep2 < rep1) == increasing) || pairedR1) && !pairedR2) {
      if(isR2Invalid) {
        oldSaddle = rep2.rep_.saddleId_;
        removePair(saddles[rep2.rep_.saddleId_], rep2, saddleToPairedExtrema,
                   extremaToPairedSaddle);
      }
      addPair(sv, rep2, saddleToPairedExtrema, extremaToPairedSaddle);
      rep2.rep_.extremaId_ = rep1.lid_;
      rep2.rep_.saddleId_ = sv.lid_;
      // send to other processes if necessary
      if(rep2.rank_ != ttk::MPIrank_ || (ghostPresence[rep2.lid_].size() > 1)) {
        saddleEdge<sizeSad> s1;
        saddleEdge<sizeSad> s2;
        if(rep1.rep_.saddleId_ > -1) {
          s1 = saddles[rep1.rep_.saddleId_];
        }
        if(oldSaddle > -1) {
          s2 = saddles[oldSaddle];
        }
        storeMessageToSend<sizeExtr, sizeSad>(
          ghostPresence, sendBuffer, sv, s2, s1, rep2, rep1);
      }
      if(isR2Invalid) {
        if(saddles[oldSaddle].rank_ == ttk::MPIrank_
           && saddles[oldSaddle].gid_ != sv.gid_) {
          return processTriplet<sizeExtr, sizeSad>(
            saddles[oldSaddle], saddleToPairedExtrema, extremaToPairedSaddle,
            saddles, extremas, increasing, ghostPresence, sendBuffer);
        } else {
          storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddle]);
        }
      }
    } else {
      if(!pairedR1) {
        if(isR1Invalid) {
          oldSaddle = rep1.rep_.saddleId_;
          removePair(saddles[rep1.rep_.saddleId_], rep1, saddleToPairedExtrema,
                     extremaToPairedSaddle);
        }
        addPair(sv, rep1, saddleToPairedExtrema, extremaToPairedSaddle);
        rep1.rep_.extremaId_ = rep2.lid_;
        rep1.rep_.saddleId_ = sv.lid_;
        if(rep1.rank_ != ttk::MPIrank_
           || (ghostPresence[rep1.lid_].size() > 1)) {
          saddleEdge<sizeSad> s1;
          saddleEdge<sizeSad> s2;
          if(oldSaddle > -1) {
            s1 = saddles[oldSaddle];
          }
          if(rep2.rep_.saddleId_ > -1) {
            s2 = saddles[rep2.rep_.saddleId_];
          }
          storeMessageToSend<sizeExtr, sizeSad>(
            ghostPresence, sendBuffer, sv, s1, s2, rep1, rep2);
        }
        if(isR1Invalid) {
          if(saddles[oldSaddle].rank_ == ttk::MPIrank_
             && saddles[oldSaddle].gid_ != sv.gid_) {
            return processTriplet<sizeExtr, sizeSad>(
              saddles[oldSaddle], saddleToPairedExtrema, extremaToPairedSaddle,
              saddles, extremas, increasing, ghostPresence, sendBuffer);
          } else {
            storeRerunToSend<sizeExtr>(sendBuffer, saddles[oldSaddle]);
          }
        }
      }
    }
  } else {
    if(saddleToPairedExtrema[sv.lid_] > -1) {
      auto &t1{extremas[saddleToPairedExtrema[sv.lid_]]};
      extremaToPairedSaddle[saddleToPairedExtrema[sv.lid_]] = -1;
      if(t1.gid_ == rep1.gid_) {
        t1.rep_.extremaId_ = t1.lid_;
        t1.rep_.saddleId_ = -1;
        if(rep1.rank_ != ttk::MPIrank_
           || (ghostPresence[rep1.lid_].size() > 1)) {
          saddleEdge<sizeSad> s1;
          saddleEdge<sizeSad> s2;
          if(rep1.rep_.saddleId_ > -1) {
            s1 = saddles[rep1.rep_.saddleId_];
          }
          if(rep2.rep_.saddleId_ > -1) {
            s2 = saddles[rep2.rep_.saddleId_];
          }
          storeMessageToSend<sizeExtr, sizeSad>(
            ghostPresence, sendBuffer, sv, s1, s2, rep1, rep2);
        }
      }
      saddleToPairedExtrema[sv.lid_] = static_cast<ttk::SimplexId>(-2);
    }
  }

  return 0;
};

template <typename triangulationType, typename Container>
SimplexId ttk::DiscreteMorseSandwichMPI::eliminateBoundariesSandwich(
  const SimplexId s2,
  std::vector<bool> &onBoundary,
  std::vector<Container> &s2Boundaries,
  const std::vector<SimplexId> &s2Mapping,
  const std::vector<SimplexId> &s1Mapping,
  std::vector<SimplexId> &partners,
  std::vector<Lock> &s1Locks,
  std::vector<Lock> &s2Locks,
  const triangulationType &triangulation) const {

  auto &boundaryIds{s2Boundaries[s2Mapping[s2]]};

  const auto addBoundary = [&boundaryIds, &onBoundary](const SimplexId e) {
    // add edge e to boundaryIds/onBoundary modulo 2
    if(!onBoundary[e]) {
      boundaryIds.emplace(e);
      onBoundary[e] = true;
    } else {
      const auto it = boundaryIds.find(e);
      boundaryIds.erase(it);
      onBoundary[e] = false;
    }
  };

  const auto clearOnBoundary = [&boundaryIds, &onBoundary]() {
    // clear the onBoundary vector (set everything to false)
    for(const auto e : boundaryIds) {
      onBoundary[e] = false;
    }
  };

  if(!boundaryIds.empty()) {
    // restore previously computed s2 boundary
    for(const auto e : boundaryIds) {
      onBoundary[e] = true;
    }
  } else {
    // init cascade with s2 triangle boundary (3 edges)
    for(SimplexId i = 0; i < 3; ++i) {
      SimplexId e{};
      triangulation.getTriangleEdge(s2, i, e);
      addBoundary(e);
    }
  }

  // lock the 2-saddle to ensure that only one thread can perform the
  // boundary expansion
  s2Locks[s2Mapping[s2]].lock();

  while(!boundaryIds.empty()) {
    // tau: youngest edge on boundary
    const auto tau{*boundaryIds.begin()};
    // use the Discrete Gradient to find a triangle paired to tau
    auto pTau{this->dg_.getPairedCell(Cell{1, tau}, triangulation)};
    bool critical{false};
    if(pTau == -1) {
      // maybe tau is critical and paired to a critical triangle
      do {
#ifdef TTK_ENABLE_OPENMP
#pragma omp atomic read
#endif // TTK_ENABLE_OPENMP
        pTau = partners[tau];
        if(pTau == -1 || s2Boundaries[s2Mapping[pTau]].empty()) {
          break;
        }
      } while(*s2Boundaries[s2Mapping[pTau]].begin() != tau);

      critical = true;
    }
    if(pTau == -1) {
      // tau is critical and not paired

      // compare-and-swap from "Towards Lockfree Persistent Homology"
      // using locks over 1-saddles instead of atomics (OpenMP compatibility)
      s1Locks[s1Mapping[tau]].lock();
      const auto cap = partners[tau];
      if(partners[tau] == -1) {
        partners[tau] = s2;
      }
      s1Locks[s1Mapping[tau]].unlock();

      // cleanup before exiting
      clearOnBoundary();
      s2Locks[s2Mapping[s2]].unlock();
      if(cap == -1) {
        return tau;
      } else {
        return this->eliminateBoundariesSandwich(
          s2, onBoundary, s2Boundaries, s2Mapping, s1Mapping, partners, s1Locks,
          s2Locks, triangulation);
      }

    } else {
      // expand boundary
      if(critical && s2Mapping[pTau] != -1) {
        if(s2Mapping[pTau] < s2Mapping[s2]) {
          // pTau is an already-paired 2-saddle
          // merge pTau boundary into s2 boundary

          // make sure that pTau boundary is not modified by another
          // thread while we merge the two boundaries...
          s2Locks[s2Mapping[pTau]].lock();
          for(const auto e : s2Boundaries[s2Mapping[pTau]]) {
            addBoundary(e);
          }
          s2Locks[s2Mapping[pTau]].unlock();
          if(this->Compute2SaddlesChildren) {
            this->s2Children_[s2Mapping[s2]].emplace_back(s2Mapping[pTau]);
          }

        } else if(s2Mapping[pTau] > s2Mapping[s2]) {

          // compare-and-swap from "Towards Lockfree Persistent
          // Homology" using locks over 1-saddles
          s1Locks[s1Mapping[tau]].lock();
          const auto cap = partners[tau];
          if(partners[tau] == pTau) {
            partners[tau] = s2;
          }
          s1Locks[s1Mapping[tau]].unlock();

          if(cap == pTau) {
            // cleanup before exiting
            clearOnBoundary();
            s2Locks[s2Mapping[s2]].unlock();
            return this->eliminateBoundariesSandwich(
              pTau, onBoundary, s2Boundaries, s2Mapping, s1Mapping, partners,
              s1Locks, s2Locks, triangulation);
          }
        }
      } else { // pTau is a regular triangle
        // add pTau triangle boundary (3 edges)
        for(SimplexId i = 0; i < 3; ++i) {
          SimplexId e{};
          triangulation.getTriangleEdge(pTau, i, e);
          addBoundary(e);
        }
      }
    }
  }

  // cleanup before exiting
  clearOnBoundary();
  s2Locks[s2Mapping[s2]].unlock();
  return -1;
}

template <typename triangulationType>
void ttk::DiscreteMorseSandwichMPI::getSaddleSaddlePairs(
  std::vector<PersistencePair> &pairs,
  const bool exportBoundaries,
  std::vector<GeneratorType> &boundaries,
  const std::vector<SimplexId> &critical1Saddles,
  const std::vector<SimplexId> &critical2Saddles,
  const std::vector<SimplexId> &crit1SaddlesOrder,
  const triangulationType &triangulation) const {

  Timer tm2{};
  const auto nSadExtrPairs = pairs.size();

  // 1- and 2-saddles yet to be paired
  std::vector<SimplexId> saddles1{}, saddles2{};
  std::vector<std::vector<SimplexId>> saddlesThread(
    this->threadNumber_, std::vector<ttk::SimplexId>());

  // filter out already paired 1-saddles (edge id)

#pragma omp parallel num_threads(threadNumber_)
  {
    int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static)
    for(const auto s1 : critical1Saddles) {
      auto it = globalToLocalSaddle1_.find(triangulation.getEdgeGlobalId(s1));
      if((it == globalToLocalSaddle1_.end())
         || (saddleToPairedMin_[it->second] == -1)) {
        saddlesThread[threadNumber].emplace_back(s1);
      }
    }
  }
  for(int i = 0; i < threadNumber_; i++) {
    saddles1.insert(
      saddles1.end(), saddlesThread[i].begin(), saddlesThread[i].end());
    saddlesThread[i].clear();
  }
  // filter out already paired 2-saddles (triangle id)
#pragma omp parallel num_threads(threadNumber_)
  {
    int threadNumber = omp_get_thread_num();
#pragma omp for schedule(static)
    for(const auto s2 : critical2Saddles) {
      auto it = globalToLocalSaddle2_.find(triangulation.getEdgeGlobalId(s2));
      if((it == globalToLocalSaddle2_.end())
         || (saddleToPairedMax_[it->second] == -1)) {
        saddlesThread[threadNumber].emplace_back(s2);
      }
    }
  }
  for(int i = 0; i < threadNumber_; i++) {
    saddles2.insert(
      saddles2.end(), saddlesThread[i].begin(), saddlesThread[i].end());
  }
  saddlesThread.clear();
  if(this->Compute2SaddlesChildren) {
    this->s2Children_.resize(saddles2.size());
  }

  // sort every triangulation edges by filtration order
  const auto &edgesFiltrOrder{crit1SaddlesOrder};

  auto &onBoundary{this->onBoundary_};
  auto &edgeTrianglePartner{this->edgeTrianglePartner_};

  const auto cmpEdges
    = [&edgesFiltrOrder](const SimplexId a, const SimplexId b) {
        return edgesFiltrOrder[a] > edgesFiltrOrder[b];
      };
  using Container = std::set<SimplexId, decltype(cmpEdges)>;
  std::vector<Container> s2Boundaries(saddles2.size(), Container(cmpEdges));

  // unpaired critical triangle id -> index in saddle2 vector
  auto &s2Mapping{this->s2Mapping_};
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < saddles2.size(); ++i) {
    s2Mapping[saddles2[i]] = i;
  }

  // unpaired critical edge id -> index in saddle1 vector
  auto &s1Mapping{this->s1Mapping_};
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < saddles1.size(); ++i) {
    s1Mapping[saddles1[i]] = i;
  }

  // one lock per 1-saddle
  std::vector<Lock> s1Locks(saddles1.size());
  // one lock per 2-saddle
  std::vector<Lock> s2Locks(saddles2.size());

  // compute 2-saddles boundaries in parallel

#ifdef TTK_ENABLE_OPENMP4
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic) \
  firstprivate(onBoundary)
#endif // TTK_ENABLE_OPENMP4
  for(size_t i = 0; i < saddles2.size(); ++i) {
    // 2-saddles sorted in increasing order
    const auto s2 = saddles2[i];
    this->eliminateBoundariesSandwich(s2, onBoundary, s2Boundaries, s2Mapping,
                                      s1Mapping, edgeTrianglePartner, s1Locks,
                                      s2Locks, triangulation);
  }

  Timer tmseq{};

  // extract saddle-saddle pairs from computed boundaries
  for(size_t i = 0; i < saddles2.size(); ++i) {
    if(!s2Boundaries[i].empty()) {
      const auto s2 = saddles2[i];
      const auto s1 = *s2Boundaries[i].begin();
      // we found a pair
      pairs.emplace_back(s1, s2, 1);
      // paired1Saddles[s1] = true;
      // paired2Saddles[s2] = true;
    }
  }

  if(exportBoundaries) {
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
  }

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

  this->printMsg("Extracted critical cells", 1.0, tm.getElapsedTime(),
                 this->threadNumber_, debug::LineMode::NEW);

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

  this->printMsg("Extracted & sorted critical cells", 1.0, tm.getElapsedTime(),
                 this->threadNumber_, debug::LineMode::NEW);
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
  this->getMinSaddlePairs(pairs, criticalCellsByDim[1], critCellsOrder[1],
                          criticalCellsByDim[0], offsets, nConnComp,
                          triangulation);
  // saddle - maxima pairs
  /*this->getMaxSaddlePairs(pairs, criticalCellsByDim[dim - 1],
                          critCellsOrder[dim - 1], critCellsOrder[dim],
                          triangulation, ignoreBoundary, offsets);

  // saddle - saddle pairs
  if(dim == 3 && !criticalCellsByDim[1].empty()
     && !criticalCellsByDim[2].empty() && this->ComputeSadSad) {
    std::vector<GeneratorType> tmp{};
    this->getSaddleSaddlePairs(pairs, false, tmp, criticalCellsByDim[1],
                               criticalCellsByDim[2], critCellsOrder[1],
                               triangulation);
  }*/
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

  this->printMsg(
    "Computed " + std::to_string(pairs.size()) + " persistence pairs", 1.0,
    tm.getElapsedTime(), this->threadNumber_);

  // this->displayStats(pairs, criticalCellsByDim, pairedMinima, paired1Saddles,
  //                   paired2Saddles, pairedMaxima);

  // free memory
  this->clear();

  return 0;
}