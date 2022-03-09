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
                     std::vector<int> *seedIdentifier,
                     int *taskCounter,
                     int *globalElementToCompute,
                     bool *keepWorking) const;

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
      bool found = (this->GlobalIdsArray[0] == globalId);
      int i = 0;
      while(!found) {
        i++;
        found = (this->GlobalIdsArray[i] == globalId);
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

    inline void
      setOutputSeedIdentifier(std::vector<SimplexId> *seedIdentifier) {
      outputSeedIdentifier_ = seedIdentifier;
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
  std::vector<int> *seedIdentifier,
  int *taskCounter,
  int *globalElementToCompute,
  bool *keepWorking) const {
  printMsg("Start create task for seed " + std::to_string(v));
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
    int elt;
    if(vnext == -1) {
      isMax = true;
#pragma omp atomic update
      (*finishedElement)++;
#pragma omp atomic read
      elt = (*finishedElement);
      printMsg("finishedElement: " + std::to_string(elt));
    } else {
      v = vnext;
      triangulation->getVertexPoint(v, p1[0], p1[1], p1[2]);
      distance += Geometry::distance(p0, p1, 3);

      if(this->PointGhostArray[v] && ttk::type::DUPLICATEPOINT) {
        printMsg("Sending element " + std::to_string(this->GlobalIdsArray[v])
                 + " to process to process "
                 + std::to_string(this->ProcessId[v]));
        isMax = true;
        m.Id = this->GlobalIdsArray[v];
        m.DistanceFromSeed = distance;
        m.SeedIdentifier = seedIdentifier->at(index);
        MPI_Send(&m, 1, message_type, this->ProcessId[v], IS_ELEMENT_TO_PROCESS,
                 this->MPIComm);
        printMsg("Sent element " + std::to_string(m.Id)
                 + " to process to process "
                 + std::to_string(this->ProcessId[v]));
      } else {
        (*trajectories)[index].push_back(v);

        p0[0] = p1[0];
        p0[1] = p1[1];
        p0[2] = p1[2];
        (*distanceFromSeed)[index].push_back(distance);
      }
    }
  }
  int tempTask;
  int seed;
  int totalSeed;
#pragma omp atomic update
  (*taskCounter)--;
#pragma omp atomic read
  tempTask = (*taskCounter);
  printMsg("TempTask: " + std::to_string(tempTask));
  if(tempTask == 0) {
#pragma omp atomic read
    seed = (*finishedElement);
    if(seed > 0) {
      printMsg("Taskwait done, sending number of finished elements");
      m.Id = seed;
      if(this->MyRank != 0) {
        MPI_Send(&m, 1, message_type, 0, FINISHED_ELEMENT, this->MPIComm);
      } else {
#pragma omp atomic update
        (*globalElementToCompute) -= m.Id;
#pragma omp atomic read
        totalSeed = (*globalElementToCompute);
        if(totalSeed == 0) {
          printMsg("totalSeed: " + std::to_string(totalSeed));
#pragma omp atomic write
          (*keepWorking) = false;
          for(int i = 1; i < this->NumberOfProcesses; i++) {
            MPI_Send(&m, 1, message_type, i, STOP_WORKING, this->MPIComm);
          }
        }
      }
#pragma omp atomic update
      (*finishedElement) -= seed;
      printMsg("Taskwait done, number of finished elements sent");
    }
  }
  printMsg("End create task for seed "
           + std::to_string((*trajectories)[index][0]));
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
  int finishedElement = 0;
  int taskCounter = seedNumber_;

  // get the seeds
  std::unordered_set<SimplexId> isSeed;
  std::string str = "";
  for(SimplexId k = 0; k < seedNumber_; ++k) {
    isSeed.insert(identifiers[k]);
    str += std::to_string(this->GlobalIdsArray[identifiers[k]]) + " ";
  }
  printMsg("Seeds: " + str);
  std::vector<SimplexId> seeds(isSeed.begin(), isSeed.end());
  isSeed.clear();
  int globalElementToCompute = this->GlobalElementToCompute;
  printMsg("GlobalElementToCompute: "
           + std::to_string(this->GlobalElementToCompute));
  int totalSeed;
  trajectories->resize(seeds.size());
  distanceFromSeed->resize(seeds.size());
  seedIdentifier->assign(seeds.size(), -1);
  bool keepWorking = true;
  bool keepWorkingAux = keepWorking;
#pragma omp parallel shared(                                         \
  finishedElement, keepWorking, globalElementToCompute, taskCounter) \
  firstprivate(keepWorkingAux)
  {
    struct Message m;

#pragma omp single nowait
    {
#pragma omp task
      {
        printMsg("Creating tasks");
        for(SimplexId i = 0; i < seedNumber_; ++i) {
          SimplexId v{seeds[i]};
#pragma omp task firstprivate(v, i)
          {
            this->create_task<dataType, triangulationType>(
              triangulation, trajectories, distanceFromSeed, v, i,
              &finishedElement, offsets, scalars, message_type, seedIdentifier,
              &taskCounter, &globalElementToCompute, &keepWorking);
          }
          printMsg("Task created");
        }
      }
      MPI_Status status;
      while(keepWorkingAux) {
        printMsg("Start receiving messages");
        MPI_Recv(&m, 3, message_type, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 this->MPIComm, &status);
        printMsg("Message received");
        int stat = status.MPI_TAG;
        switch(stat) {
          case IS_ELEMENT_TO_PROCESS: {
            int currentSize = seeds.size();
            trajectories->resize(currentSize + 1);
            distanceFromSeed->resize(currentSize + 1);
            int elementId = m.Id;
            seedIdentifier->push_back(m.SeedIdentifier);
            printMsg("Received element " + std::to_string(m.SeedIdentifier)
                     + " to process");
#pragma omp task firstprivate(elementId, currentSize)
            {
#pragma omp atomic update
              taskCounter++;
              this->create_task<dataType, triangulationType>(
                triangulation, trajectories, distanceFromSeed,
                this->getLocalIdFromGlobalId(m.Id), currentSize,
                &finishedElement, offsets, scalars, message_type,
                seedIdentifier, &taskCounter, &globalElementToCompute,
                &keepWorking);
            }
            printMsg("Task created from message");
            break;
          }
          case FINISHED_ELEMENT: {
            printMsg("Received finished element");
#pragma omp atomic update
            globalElementToCompute -= m.Id;
#pragma omp atomic read
            totalSeed = globalElementToCompute;
            if(totalSeed == 0) {
#pragma omp atomic write
              keepWorking = false;
              printMsg("Proc 0 tells all processes to stop working");
              for(int i = 1; i < this->NumberOfProcesses; i++) {
                MPI_Send(&m, 1, message_type, i, STOP_WORKING, this->MPIComm);
              }
            }
            break;
          }
          case STOP_WORKING: {
            printMsg("I was told to stop working");
#pragma omp atomic write
            keepWorking = false;
            break;
          }
          default:
            break;
        }
#pragma omp atomic read
        keepWorkingAux = keepWorking;
      }
      {
        std::stringstream msg;
        msg << "Processed " << vertexNumber_ << " points";
        this->printMsg(msg.str(), 1, t.getElapsedTime(), 1);
      }
    }
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
