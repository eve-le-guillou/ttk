/// \ingroup base
/// \class ttk::IntegralLines
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \date March 2016
///
/// \brief TTK processing package for the computation of edge-based integral
/// lines of the gradient of an input scalar field defined on a PL manifold.
///
/// Given a list of sources, the package produces forward or backward integral
/// lines along the edges of the input triangulation.
///
/// \sa ttkIntegralLines.cpp %for a usage example.

#pragma once

// base code includes
#include <DataSetAttributes.h>
#include <Geometry.h>
#include <Triangulation.h>

// std includes
#include <iterator>
#include <limits>
#include <unordered_set>
#define IS_ELEMENT_TO_PROCESS 0
#define FINISHED_ELEMENT 1
#define STOP_WORKING 2
#if TTK_ENABLE_MPI
#include <mpi.h>
#endif

struct Message {
  int Id;
  float DistanceFromSeed;
  int SeedIdentifier;
};

namespace ttk {
  enum Direction { Forward = 0, Backward };

  class IntegralLines : virtual public Debug {

  public:
    IntegralLines();
    ~IntegralLines() override;

    template <class triangulationType>
    inline float getDistance(const triangulationType *triangulation,
                             const SimplexId &a,
                             const SimplexId &b) const {
      float p0[3];
      triangulation->getVertexPoint(a, p0[0], p0[1], p0[2]);
      float p1[3];
      triangulation->getVertexPoint(b, p1[0], p1[1], p1[2]);

      return Geometry::distance(p0, p1, 3);
    }

    template <typename dataType, class triangulationType>
    inline float getGradient(const triangulationType *triangulation,
                             const SimplexId &a,
                             const SimplexId &b,
                             dataType *scalars) const {
      return std::fabs(static_cast<float>(scalars[b] - scalars[a]))
             / getDistance<triangulationType>(triangulation, a, b);
    }

    template <typename dataType,
              class triangulationType = ttk::AbstractTriangulation>
    int execute(triangulationType *);

    template <typename dataType,
              class Compare,
              class triangulationType = ttk::AbstractTriangulation>
    int execute(Compare, const triangulationType *) const;

    template <typename dataType,
              class triangulationType = ttk::AbstractTriangulation>
    void create_task(const triangulationType *triangulation,
                     std::vector<std::vector<SimplexId>> *trajectories,
                     std::vector<std::vector<double>> *distanceFromSeed,
                     ttk::SimplexId v,
                     int index,
                     int *finishedElement,
                     const SimplexId *offsets,
                     dataType *scalars,
                     MPI_Datatype message_type,
                     std::vector<int> *seedIdentifier) const;

    inline void setVertexNumber(const SimplexId &vertexNumber) {
      vertexNumber_ = vertexNumber;
    }

    inline void setSeedNumber(const SimplexId &seedNumber) {
      seedNumber_ = seedNumber;
    }

    inline void setDirection(int direction) {
      direction_ = direction;
    }

    void setMyRank(int rank) {
      this->MyRank = rank;
    }

    void setNumberOfProcesses(int number) {
      this->NumberOfProcesses = number;
    }

#if TTK_ENABLE_MPI
    void setMPIComm(MPI_Comm comm) {
      this->MPIComm = comm;
    }

    void setGlobalElementToCompute(int number) {
      this->GlobalElementToCompute = number;
    }

    SimplexId getLocalIdFromGlobalId(int globalId) {
      bool found = (GlobalIdsArray[0] == globalId);
      ;
      int i = 0;
      while(!found) {
        i++;
        found = (GlobalIdsArray[i] == globalId);
      }
      return (SimplexId)i;
    }

    void setGlobalIdsArray(long int *array) {
      this->GlobalIdsArray = array;
    }

    void setPointGhostArray(unsigned char *array) {
      this->PointGhostArray = array;
    }

    void setProcessId(int *processId) {
      this->ProcessId = processId;
    }
#endif

    int preconditionTriangulation(
      ttk::AbstractTriangulation *triangulation) const {
      return triangulation->preconditionVertexNeighbors();
    }

    inline void setInputScalarField(void *data) {
      inputScalarField_ = data;
    }

    /**
     * @pre For this function to behave correctly in the absence of
     * the VTK wrapper, ttk::preconditionOrderArray() needs to be
     * called to fill the @p data buffer prior to any
     * computation (the VTK wrapper already includes a mecanism to
     * automatically generate such a preconditioned buffer).
     * @see examples/c++/main.cpp for an example use.
     */
    inline void setInputOffsets(const SimplexId *const data) {
      inputOffsets_ = data;
    }

    inline void setVertexIdentifierScalarField(SimplexId *const data) {
      vertexIdentifierScalarField_ = data;
    }

    inline void
      setOutputTrajectories(std::vector<std::vector<SimplexId>> *trajectories) {
      outputTrajectories_ = trajectories;
    }

    inline void setOutputDistanceFromSeed(
      std::vector<std::vector<double>> *distanceFromSeed) {
      outputDistanceFromSeed_ = distanceFromSeed;
    }

  protected:
    SimplexId vertexNumber_;
    SimplexId seedNumber_;
    int direction_;
    void *inputScalarField_;
    const SimplexId *inputOffsets_;
    SimplexId *vertexIdentifierScalarField_;
    std::vector<std::vector<SimplexId>> *outputTrajectories_;
    std::vector<std::vector<double>> *outputDistanceFromSeed_;
    std::vector<SimplexId> *outputSeedIdentifier_;
    int MyRank;
    int NumberOfProcesses;
#if TTK_ENABLE_MPI
    MPI_Comm MPIComm;
    int GlobalElementToCompute;
    long int *GlobalIdsArray;
    unsigned char *PointGhostArray;
    int *ProcessId;
#endif
  };
} // namespace ttk

template <typename dataType, class triangulationType>
void ttk::IntegralLines::create_task(
  const triangulationType *triangulation,
  std::vector<std::vector<SimplexId>> *trajectories,
  std::vector<std::vector<double>> *distanceFromSeed,
  ttk::SimplexId v,
  int index,
  int *finishedElement,
  const SimplexId *offsets,
  dataType *scalars,
  MPI_Datatype message_type,
  std::vector<int> *seedIdentifier) const {
  struct Message m;
  (*trajectories)[index].push_back(v);
  (*distanceFromSeed)[index].push_back(0);
  if((*seedIdentifier)[index] == -1)
    (*seedIdentifier)[index] = this->GlobalIdsArray[v];
  double distance = 0;
  float p0[3];
  float p1[3];
  triangulation->getVertexPoint(v, p0[0], p0[1], p0[2]);
  bool isMax{};
  while(!isMax) {
    SimplexId vnext{-1};
    float fnext = std::numeric_limits<float>::min();
    SimplexId neighborNumber = triangulation->getVertexNeighborNumber(v);
    for(SimplexId k = 0; k < neighborNumber; ++k) {
      SimplexId n;
      triangulation->getVertexNeighbor(v, k, n);

      if((direction_ == static_cast<int>(Direction::Forward))
         xor (offsets[n] < offsets[v])) {
        const float f = getGradient<dataType, triangulationType>(
          triangulation, v, n, scalars);
        if(f > fnext) {
          vnext = n;
          fnext = f;
        }
      }
    }

    if(vnext == -1) {
      isMax = true;
#pragma omp atomic update
      (*finishedElement)++;
    } else {
      if(this->PointGhostArray[v] && ttk::type::DUPLICATEPOINT) {
        isMax = true;
        // TODO: SeedIdentifier
        m.Id = this->GlobalIdsArray[v], m.DistanceFromSeed = distance,
        m.SeedIdentifier = 0;
        MPI_Send(&m, 1, message_type, this->ProcessId[vnext],
                 IS_ELEMENT_TO_PROCESS, this->MPIComm);
      } else {
        v = vnext;
        (*trajectories)[index].push_back(v);

        triangulation->getVertexPoint(v, p1[0], p1[1], p1[2]);
        distance += Geometry::distance(p0, p1, 3);
        p0[0] = p1[0];
        p0[1] = p1[1];
        p0[2] = p1[2];
        (*distanceFromSeed)[index].push_back(distance);
      }
    }
  }
}

template <typename dataType, class triangulationType>
int ttk::IntegralLines::execute(triangulationType *triangulation) {
  MPI_Datatype message_type;
  MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_INT};
  int lengths[] = {1, 1, 1};
  const long int mpi_offsets[]
    = {offsetof(Message, Id), offsetof(Message, DistanceFromSeed),
       offsetof(Message, SeedIdentifier)};

  MPI_Type_create_struct(3, lengths, mpi_offsets, types, &message_type);
  MPI_Type_commit(&message_type);
  const SimplexId *offsets = inputOffsets_;
  SimplexId *identifiers = vertexIdentifierScalarField_;
  dataType *scalars = static_cast<dataType *>(inputScalarField_);
  std::vector<std::vector<SimplexId>> *trajectories = outputTrajectories_;
  std::vector<std::vector<double>> *distanceFromSeed = outputDistanceFromSeed_;
  std::vector<int> *seedIdentifier = outputSeedIdentifier_;
  Timer t;
  int *finishedElement = 0;
  // get the seeds
  std::unordered_set<SimplexId> isSeed;
  for(SimplexId k = 0; k < seedNumber_; ++k)
    isSeed.insert(identifiers[k]);
  std::vector<SimplexId> seeds(isSeed.begin(), isSeed.end());
  isSeed.clear();

  trajectories->resize(seeds.size());
  distanceFromSeed->resize(seeds.size());
  seedIdentifier->assign(seeds.size(), -1);
  bool keepWorking = true;
#pragma omp parallel shared(finishedElement, keepWorking)
  {
    struct Message m;
#pragma omp single nowait
    {
      MPI_Status *status;
      while(keepWorking) {
        MPI_Recv(&m, 3, message_type, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 this->MPIComm, status);
        int stat = status->MPI_TAG;
        switch(stat) {
          case IS_ELEMENT_TO_PROCESS: {
            int currentSize = seeds.size();
            trajectories->resize(currentSize + 1);
            distanceFromSeed->resize(currentSize + 1);
            int elementId = m.Id;
            seedIdentifier->push_back(m.SeedIdentifier);
#pragma omp task firstprivate(elementId, currentSize)
            {
              this->create_task<dataType, triangulationType>(
                triangulation, trajectories, distanceFromSeed,
                this->getLocalIdFromGlobalId(m.Id), currentSize,
                finishedElement, offsets, scalars, message_type,
                seedIdentifier);
            }
            break;
          }
          case FINISHED_ELEMENT: {
            this->GlobalElementToCompute -= m.Id;
            if(this->GlobalElementToCompute == 0) {
#pragma omp atomic write
              keepWorking = false;
              for(int i = 1; i < this->NumberOfProcesses; i++) {
                MPI_Send(&m, 1, message_type, i, STOP_WORKING, this->MPIComm);
              }
            }
            break;
          }
          case STOP_WORKING: {
#pragma omp atomic write
            keepWorking = false;
            break;
          }
          default:
            break;
        }
      }
    }

#pragma omp single nowait
    {
      for(SimplexId i = 0; i < (SimplexId)seeds.size(); ++i) {
        SimplexId v{seeds[i]};
#pragma omp task firstprivate(v, i)
        {
          // TODO:
          this->create_task<dataType, triangulationType>(
            triangulation, trajectories, distanceFromSeed, v, i,
            finishedElement, offsets, scalars, message_type, seedIdentifier);
        }
      }
      while(keepWorking) {
#pragma omp taskwait
        if((*finishedElement) > 0) {
#pragma omp atomic write
          m.Id = (*finishedElement);
          MPI_Send(&m, 1, message_type, 0, FINISHED_ELEMENT, this->MPIComm);
#pragma omp atomic write
          (*finishedElement) = 0;
        }
      }
    }
  }
  {
    std::stringstream msg;
    msg << "Processed " << vertexNumber_ << " points";
    this->printMsg(msg.str(), 1, t.getElapsedTime(), 1);
  }

  return 0;
}

template <typename dataType, class Compare, class triangulationType>
int ttk::IntegralLines::execute(Compare cmp,
                                const triangulationType *triangulation) const {
  const auto offsets = inputOffsets_;
  SimplexId *identifiers
    = static_cast<SimplexId *>(vertexIdentifierScalarField_);
  dataType *scalars = static_cast<dataType *>(inputScalarField_);
  std::vector<std::vector<SimplexId>> *trajectories = outputTrajectories_;

  Timer t;

  // get the seeds
  std::unordered_set<SimplexId> isSeed;
  for(SimplexId k = 0; k < seedNumber_; ++k)
    isSeed.insert(identifiers[k]);
  std::vector<SimplexId> seeds(isSeed.begin(), isSeed.end());
  isSeed.clear();

  trajectories->resize(seeds.size());
  for(SimplexId i = 0; i < (SimplexId)seeds.size(); ++i) {
    SimplexId v{seeds[i]};
    (*trajectories)[i].push_back(v);

    bool isMax{};
    while(!isMax) {
      SimplexId vnext{-1};
      float fnext = std::numeric_limits<float>::min();
      SimplexId neighborNumber = triangulation->getVertexNeighborNumber(v);
      for(SimplexId k = 0; k < neighborNumber; ++k) {
        SimplexId n;
        triangulation->getVertexNeighbor(v, k, n);

        if((direction_ == static_cast<int>(Direction::Forward))
           xor (offsets[n] < offsets[v])) {
          const float f
            = getGradient<dataType, triangulationType>(v, n, scalars);
          if(f > fnext) {
            vnext = n;
            fnext = f;
          }
        }
      }

      if(vnext == -1)
        isMax = true;
      else {
        v = vnext;
        (*trajectories)[i].push_back(v);

        if(cmp(v))
          isMax = true;
      }
    }
  }

  {
    std::stringstream msg;
    msg << "Processed " << vertexNumber_ << " points";
    this->printMsg(msg.str(), 1, t.getElapsedTime(), 1);
  }

  return 0;
}
