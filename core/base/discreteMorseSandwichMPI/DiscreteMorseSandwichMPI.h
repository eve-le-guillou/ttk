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

    // template<typename datatype>
    struct extremaNode {
      ttk::SimplexId gid_{-1};
      ttk::SimplexId lid_{-1};
      ttk::SimplexId order_{-1};
      // datatype scalar_;
      Rep rep_;
      char rank_{static_cast<char>(ttk::MPIrank_)};

      bool operator==(const extremaNode &t1) {
        return this->gid_ == t1.gid_;
      }

      bool operator!=(const extremaNode &t1) {
        return this->gid_ != t1.gid_;
      }

      bool operator<(const extremaNode &t1) {
        if(this->gid_ == t1.gid_) {
          return false;
        }
        if(this->order_ != -1 && t1.order_ != -1) {
          return this->order_ < t1.order_;
        }
        // if (t0.scalar_ != t1.scalar_){ //TODO: access to scalars?
        //  return t0.scalar_ < t1.scalar_;
        //}
        return this->gid_ < t1.gid_;
      }
    };
    // template<typename datatype>
    struct saddleEdge {
      ttk::SimplexId gid_{-1};
      ttk::SimplexId lid_{-1};
      ttk::SimplexId order_{-1};
      // datatype scalar_;
      std::array<ttk::SimplexId, 2> t_{-1, -1};
      char rank_{static_cast<char>(ttk::MPIrank_)};

      bool operator==(const saddleEdge &s1) {
        return this->gid_ == s1.gid_;
      }

      bool operator<(const saddleEdge &s1) {
        if(this->gid_ == s1.gid_) {
          return false;
        }
        if(this->order_ != -1 && s1.order_ != -1) {
          return this->order_ < s1.order_;
        }
        // if (s0.scalar_ != s1.scalar_){
        //  return s0.scalar_ < s1.scalar_;
        //}
        // return s0.gid_ < s1.gid_;
        // TODO: deal with this
        return false;
      }
    };

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
    /*template<typename datatype>
    void storeMessageToSend();*/

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
    std::vector<std::vector<SimplexId>>
      getSaddle1ToMinima(const std::vector<SimplexId> &criticalEdges,
                         const triangulationType &triangulation) const;

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
    void tripletsToPersistencePairs(
      std::vector<PersistencePair> &pairs,
      const SimplexId pairDim,
      std::vector<extremaNode> &extremas,
      std::vector<saddleEdge> &saddles,
      std::vector<ttk::SimplexId> &saddleToPairedExtrema,
      std::vector<ttk::SimplexId> &extremaToPairedSaddle,
      /*std::vector<std::vector<ttk::SimplexId>> ghostPresence*/
      float &getRepTime,
      float &getPostTreatmentTime) const;

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
        //#ifdef TTK_ENABLE_OPENMP
        //#pragma omp task
        //#endif // TTK_ENABLE_OPENMP
        // this->saddleToPairedMin_.resize(
        //  this->dg_.getNumberOfCells(1, triangulation), -1);
        /*#ifdef TTK_ENABLE_OPENMP
        #pragma omp task
        #endif // TTK_ENABLE_OPENMP
                this->minToPairedSaddle_.resize(
                  this->dg_.getNumberOfCells(0, triangulation), -1);*/
        if(dim > 1) {
          //#ifdef TTK_ENABLE_OPENMP
          //#pragma omp task
          //#endif // TTK_ENABLE_OPENMP
          // this->saddleToPairedMax_.resize(
          //  this->dg_.getNumberOfCells(dim - 1, triangulation), -1);
          /*#ifdef TTK_ENABLE_OPENMP
          #pragma omp task
          #endif // TTK_ENABLE_OPENMP
                    this->maxToPairedSaddle_.resize(
                      this->dg_.getNumberOfCells(dim, triangulation), -1);*/
        }
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
                     debug::LineMode::NEW, debug::Priority::DETAIL);
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
      this->printMsg("Memory cleanup", 1.0, tm.getElapsedTime(), 1,
                     debug::LineMode::NEW, debug::Priority::DETAIL);
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
std::vector<std::vector<SimplexId>>
  ttk::DiscreteMorseSandwichMPI::getSaddle1ToMinima(
    const std::vector<SimplexId> &criticalEdges,
    const triangulationType &triangulation) const {

  Timer tm{};

  std::vector<std::vector<SimplexId>> res(criticalEdges.size());

  // follow vpaths from 1-saddles to minima
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(size_t i = 0; i < criticalEdges.size(); ++i) {
    auto &mins = res[i];

    const auto followVPath = [this, &mins, &triangulation](const SimplexId v) {
      std::vector<Cell> vpath{};
      this->dg_.getDescendingPath(Cell{0, v}, vpath, triangulation);
      const Cell &lastCell = vpath.back();
      if(lastCell.dim_ == 0 && this->dg_.isCellCritical(lastCell)) {
        mins.emplace_back(lastCell.id_);
      }
    };

    // critical edge vertices
    SimplexId v0{}, v1{};
    triangulation.getEdgeVertex(criticalEdges[i], 0, v0);
    triangulation.getEdgeVertex(criticalEdges[i], 1, v1);

    // follow vpath from each vertex of the critical edge
    followVPath(v0);
    followVPath(v1);
  }

  this->printMsg("Computed the descending 1-separatrices", 1.0,
                 tm.getElapsedTime(), this->threadNumber_, debug::LineMode::NEW,
                 debug::Priority::DETAIL);

  return res;
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
                 tm.getElapsedTime(), this->threadNumber_, debug::LineMode::NEW,
                 debug::Priority::DETAIL);

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
  if(this->ComputeMinSad) {
    // minima - saddle pairs
    Timer t{};
    Timer tm{};
    auto saddle1ToMinima = getSaddle1ToMinima(criticalEdges, triangulation);
    float getTripletsTime = t.getElapsedTime();
    t.reStart();
    Timer tmseq{};

    auto &saddleToPairedExtrema{this->saddleToPairedMin_};
    auto &extremaToPairedSaddle{this->minToPairedSaddle_};
    auto &globalToLocalSaddle{this->globalToLocalSaddle1_};
    std::vector<saddleEdge> saddles{};
    std::vector<extremaNode> extremas{};
    saddles.reserve(saddle1ToMinima.size());
    extremas.reserve(2 * saddle1ToMinima.size());
    globalToLocalSaddle.reserve(saddle1ToMinima.size());
    std::unordered_map<ttk::SimplexId, ttk::SimplexId> globalToLocalExtrema{};
    globalToLocalExtrema.reserve(2 * saddle1ToMinima.size());
    float reserveTime = t.getElapsedTime();
    ttk::SimplexId saddle1ToMinimaNumber = saddle1ToMinima.size();
    t.reStart();
    /*#ifdef TTK_ENABLE_OPENMP
    #pragma omp declare reduction (merge : std::vector<saddleEdge>
    :omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp parallel for reduction(merge                       \
                                       : saddles) schedule(static) \
      shared(extremas, globalToLocalExtrema)
    #endif*/
    for(size_t i = 0; i < saddle1ToMinima.size(); ++i) {
      auto &mins = saddle1ToMinima[i];
      const auto s1 = criticalEdges[i];
      // remove duplicates
      std::sort(mins.begin(), mins.end());
      const auto last = std::unique(mins.begin(), mins.end());
      mins.erase(last, mins.end());
      if(mins.size() != 2) {
        continue;
      }
      // TODO: scalars!
      saddleEdge e{.gid_ = triangulation.getEdgeGlobalId(s1),
                   .order_ = critEdgesOrder[s1],
                   /*.scalar_ = saddleScalars[s1],*/ .rank_
                   = static_cast<char>(ttk::MPIrank_)};
      for(int j = 0; j < 2; j++) {
        ttk::SimplexId gid = triangulation.getVertexGlobalId(mins[j]);
        ttk::SimplexId lid{-1};
        //#pragma omp critical // TODO: better way?
        //        {
        auto it = globalToLocalExtrema.find(gid);
        if(it == globalToLocalExtrema.end()) {
          lid = extremas.size();
          extremaNode n{
            .gid_ = gid,
            .lid_ = lid,
            .order_ = offsets[mins[j]],
            /*.scalar_ = extremaScalars[mins[i]],*/ .rep_ = Rep{lid, -1},
            .rank_ = static_cast<char>(ttk::MPIrank_)};
          extremas.emplace_back(n);
          globalToLocalExtrema[gid] = lid;
          } else {
            lid = it->second;
          }
          //}
          e.t_[j] = lid;
      }
      saddles.emplace_back(e);
    }
    float preTreatmentTime = t.getElapsedTime();
    t.reStart();
    const auto cmpSadMin
      = [=, &extremas](const saddleEdge &s0, const saddleEdge &s1) -> bool {
      if(&s0 != &s1) {
        if(s0.order_ != -1 && s1.order_ != -1) {
          return s0.order_ < s1.order_;
        }
        // if (s0.scalar_ != s0.scalar_){ //TODO: access to scalars?
        //  return s1.scalar_ < s1.scalar_;
        //}
        // return s0.gid_ < s1.gid_;
      } else {
        if(extremas[s0.t_[0]].order_ != -1 && extremas[s1.t_[0]].order_ != -1) {
          return extremas[s0.t_[0]].order_ > extremas[s1.t_[0]].order_;
        }
        // if (t0.scalar_ != t1.scalar_){ //TODO: access to scalars?
        //  return t0.scalar_ > t1.scalar_;
        //}
        // return t0.gid_ > t1.gid_;
      }
      // TODO: deal with this
      return false;
    };
    // TRI des arcs
    TTK_PSORT(this->threadNumber_, saddles.begin(), saddles.end(), cmpSadMin);
    float sortingTime = t.getElapsedTime();
    t.reStart();
    // Mise en place des lid des arcs

    // auto rng = std::default_random_engine{0};
    // std::shuffle(std::begin(saddles), std::end(saddles), rng);

    //#pragma omp declare reduction (merge
    //:std::unordered_map<ttk::SimplexId,ttk::SimplexId>:omp_out.insert(omp_in.begin(),omp_in.end()))
    //#pragma omp parallel for reduction(merge \
//                                   : globalToLocalSaddle) schedule(static)
    //                                   \
// num_threads(this->threadNumber_)
    for(int i = 0; i < saddle1ToMinimaNumber; i++) {
      auto &s{saddles[i]};
      s.lid_ = i;
      globalToLocalSaddle[s.gid_] = i;
    }
    float dictTime = t.getElapsedTime();
    t.reStart();
    extremaToPairedSaddle.resize(globalToLocalExtrema.size(), -1);
    saddleToPairedExtrema.resize(saddle1ToMinima.size(), -1);
    float resizeTime = t.getElapsedTime();
    t.reStart();
    float getRepTime{0}, postTreatmentTime{0};
    tripletsToPersistencePairs(
      pairs, 0, extremas, saddles, saddleToPairedExtrema,
      extremaToPairedSaddle /*, ghostPresence*/, getRepTime, postTreatmentTime);
    const auto nMinSadPairs = pairs.size();

    this->printMsg(
      "Computed " + std::to_string(nMinSadPairs) + " min-saddle pairs", 1.0,
      tm.getElapsedTime(), this->threadNumber_);
    this->printMsg("triplets creation time for min-saddle took "
                   + std::to_string(getTripletsTime) + "s");
    this->printMsg("reserve for min-saddle took " + std::to_string(reserveTime)
                   + "s");
    this->printMsg("pre treatment for min-saddle took "
                  + std::to_string(preTreatmentTime) + "s");
    this->printMsg("sorting for min-saddle took " + std::to_string(sortingTime)
                   + "s");
    this->printMsg("dict for min-saddle took " + std::to_string(dictTime)
                   + "s");
    this->printMsg("resize for min-saddle took " + std::to_string(resizeTime)
                   + "s");
    this->printMsg("getRep time for min-saddle took "
                   + std::to_string(getRepTime) + "s");
    this->printMsg("post treatment time for min-saddle took "
                   + std::to_string(postTreatmentTime) + "s");

    this->printMsg("min-saddle pairs sequential part", 1.0,
                   tmseq.getElapsedTime(), 1, debug::LineMode::NEW,
                   debug::Priority::VERBOSE);

    // non-paired minima
#pragma omp parallel for reduction(+:nConnComp) shared(pairs) num_threads(this->threadNumber_)
    for(const auto min : criticalExtremas) {
      ttk::SimplexId gid = triangulation.getVertexGlobalId(min);
      auto it = globalToLocalExtrema.find(gid);
      if(it == globalToLocalExtrema.end()
         || extremaToPairedSaddle[it->second] == -1) {
#pragma omp critical
        {
          pairs.emplace_back(gid, -1, 0);
          nConnComp++;
        }
      }
    }
  } else {
    // still extract the global pair
    const auto globMin{
      *std::min_element(criticalExtremas.begin(), criticalExtremas.end(),
                        [offsets](const SimplexId a, const SimplexId b) {
                          return offsets[a] < offsets[b];
                        })};
    // TODO: in distributed, MPI_MIN reduce
    pairs.emplace_back(triangulation.getVertexGlobalId(globMin), -1, 0);
    nConnComp++;
  }
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
    Timer tm{};
    // Timer t{};
    auto saddle2ToMaxima
      = dim == 3 ? getSaddle2ToMaxima(
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
                   [&triangulation](
                     const SimplexId a, const SimplexId i, SimplexId &r) {
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
    // float getTripletsTime = t.getElapsedTime();
    // t.reStart();
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
    auto &saddleToPairedExtrema{this->saddleToPairedMax_};
    auto &extremaToPairedSaddle{this->maxToPairedSaddle_};
    // std::iota(firstRep.begin(), firstRep.end(), 0);
    std::vector<saddleEdge> saddles{};
    std::vector<extremaNode> extremas{};
    // t.reStart();
    saddles.reserve(saddle2ToMaxima.size());
    extremas.reserve(2 * saddle2ToMaxima.size());
    globalToLocalExtrema.reserve(2 * saddle2ToMaxima.size());
    if(dim == 3) {
      globalToLocalSaddle.reserve(saddle2ToMaxima.size());
    } else {
      globalToLocalSaddle.reserve(globalToLocalSaddle.size()
                                  + saddle2ToMaxima.size());
    }
    /*#ifdef TTK_ENABLE_OPENMP
    #pragma omp declare reduction (merge : std::vector<saddleEdge> :
    omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp parallel for reduction(merge                       \
                                       : saddles) schedule(static) \
      shared(extremas, globalToLocalExtrema)
    #endif*/
    t.reStart();
    for(size_t i = saddle2ToMaxima.size() - 1; i >= 0; --i) {
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
           && (saddleToPairedMin_[it->second] != -1)) {
          pairedSaddle = true;
          // printMsg("Paired saddle: "+std::to_string(gid)+" to
          // "+std::to_string(saddleToPairedMin_[it->second]));
        }
      }
      if(!pairedSaddle) {
        if(gid == -1) {
          gid = getSaddleGlobalId(s2);
        }
        saddleEdge e{.gid_ = gid,
                     .order_ = critSaddlesOrder[s2],
                     .rank_ = static_cast<char>(ttk::MPIrank_)};
        for(int j = 0; j < 2; j++) {
          if(maxs[j] != -1) {
            gid = getMaxGlobalId(maxs[j]);
            ttk::SimplexId lid{-1};
            //#pragma omp critical // TODO: better way?
            //            {
            auto it = globalToLocalExtrema.find(gid);
            if(it == globalToLocalExtrema.end()) {
              lid = extremas.size();
              extremaNode n{.gid_ = gid,
                            .lid_ = lid,
                            .order_ = critMaxsOrder[maxs[j]],
                            .rep_ = Rep{lid, -1},
                            .rank_ = static_cast<char>(ttk::MPIrank_)};
              extremas.emplace_back(n);
              globalToLocalExtrema[gid] = lid;
              } else {
                lid = it->second;
              }
              //}
              e.t_[j] = lid;
          }
        }
        saddles.emplace_back(e);
      }
    }
    float preTreatmentTime = t.getElapsedTime();
    const auto cmpSadMax
      = [this, &extremas](const saddleEdge &s0, const saddleEdge &s1) -> bool {
      if(&s0 != &s1) {
        if(s0.order_ != -1 && s1.order_ != -1) {
          return s0.order_ > s1.order_;
        }
        // if (s0.scalar_ != s1.scalar_){
        //  return s0.scalar_ < s1.scalar_;
        //}
        return s0.gid_ > s1.gid_;
      } else {
        if(extremas[s0.t_[0]].order_ != -1 && extremas[s1.t_[0]].order_ != -1) {
          return extremas[s0.t_[0]].order_ < extremas[s1.t_[0]].order_;
        }
        // if (s0.t_[0].scalar_ != t1.scalar_){ //TODO: access to scalars?
        //  return t0.scalar_ > t1.scalar_;
        //}
        // return t0.gid_ < t1.gid;_;
      }
      // TODO: deal with this
      return false;
    };
    // TRI des arcs
    t.reStart();
    TTK_PSORT(this->threadNumber_, saddles.begin(), saddles.end(), cmpSadMax);
    float sortingTime = t.getElapsedTime();
    ttk::SimplexId saddle2ToMaximaNumber
      = static_cast<ttk::SimplexId>(saddle2ToMaxima.size());
    //#pragma omp declare reduction (merge : std::unordered_map<ttk::SimplexId,
    // ttk::SimplexId> : omp_out.insert(omp_out.end(), omp_in.begin(),
    // omp_in.end())) #pragma omp parallel for reduction(merge :
    // globalToLocalSaddle) schedule(static)
    // auto rng = std::default_random_engine{0};
    // std::shuffle(std::begin(saddles), std::end(saddles), rng);
    for(int i = 0; i < saddle2ToMaximaNumber; i++) {
      auto &s{saddles[i]};
      s.lid_ = i;
      globalToLocalSaddle[s.gid_] = i;
    }

    extremaToPairedSaddle.resize(globalToLocalExtrema.size(), -1);
    saddleToPairedExtrema.resize(saddle2ToMaxima.size(), -1);
    float getRepTime{0}, postTreatmentTime{0}, saddleToPairedExtremaTime{0};
    const auto nMinSadPairs = pairs.size();
    // float preTreatmentTime = t.getElapsedTime();
    tripletsToPersistencePairs(pairs, dim - 1, extremas, saddles,
                               saddleToPairedExtrema, extremaToPairedSaddle,
                               getRepTime, postTreatmentTime);
    const auto nSadMaxPairs = pairs.size() - nMinSadPairs;

    this->printMsg(
      "Computed " + std::to_string(nSadMaxPairs) + " saddle-max pairs", 1.0,
      tm.getElapsedTime(), this->threadNumber_);
    this->printMsg("pre treatment for max-saddle took "
                   + std::to_string(preTreatmentTime) + "s");
    this->printMsg("sorting for max-saddle took " + std::to_string(sortingTime)
                   + "s");
    /*this->printMsg("triplets creation time for saddle-max took "
                  + std::to_string(getTripletsTime) + "s");
    this->printMsg("svToR init for saddle-max took " + std::to_string(svToRInit)
                  + "s");
    this->printMsg("pre treatment for saddle-max took "
                  + std::to_string(preTreatmentTime) + "s");
    this->printMsg("getRep time for  saddle-max took "
                  + std::to_string(getRepTime) + "s");
    this->printMsg("saddleCreatTime for  saddle-max took "
                  + std::to_string(saddleToPairedExtremaTime) + "s");
    this->printMsg("post treatment time for  saddle-max took "
                  + std::to_string(postTreatmentTime) + "s");*/
    this->printMsg("saddle-max pairs sequential part", 1.0,
                   tmseq.getElapsedTime(), 1, debug::LineMode::NEW,
                   debug::Priority::VERBOSE);
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
                 tmseq.getElapsedTime(), 1, debug::LineMode::NEW,
                 debug::Priority::VERBOSE);
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
                 this->threadNumber_, debug::LineMode::NEW,
                 debug::Priority::VERBOSE);

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
                 this->threadNumber_, debug::LineMode::NEW,
                 debug::Priority::DETAIL);
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
  this->getMaxSaddlePairs(pairs, criticalCellsByDim[dim - 1],
                          critCellsOrder[dim - 1], critCellsOrder[dim],
                          triangulation, ignoreBoundary, offsets);

  // saddle - saddle pairs
  if(dim == 3 && !criticalCellsByDim[1].empty()
     && !criticalCellsByDim[2].empty() && this->ComputeSadSad) {
    std::vector<GeneratorType> tmp{};
    this->getSaddleSaddlePairs(pairs, false, tmp, criticalCellsByDim[1],
                               criticalCellsByDim[2], critCellsOrder[1],
                               triangulation);
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

  this->printMsg(
    "Computed " + std::to_string(pairs.size()) + " persistence pairs", 1.0,
    tm.getElapsedTime(), this->threadNumber_);

  // this->displayStats(pairs, criticalCellsByDim, pairedMinima, paired1Saddles,
  //                   paired2Saddles, pairedMaxima);

  // free memory
  this->clear();

  return 0;
}