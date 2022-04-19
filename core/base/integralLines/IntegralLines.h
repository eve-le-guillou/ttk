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
#include <ArrayLinkedList.h>
#include <DataSetAttributes.h>
#include <Geometry.h>
#include <Triangulation.h>
// std includes
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <unordered_set>
#define IS_ELEMENT_TO_PROCESS 0
#define FINISHED_ELEMENT 1
#define STOP_WORKING 2
#define TABULAR_SIZE 50
#if TTK_ENABLE_MPI
#include <mpi.h>
#endif

struct Message {
  int Id1;
  int Id2;
  double DistanceFromSeed1;
  double DistanceFromSeed2;
  int SeedIdentifier;
};

namespace ttk {

#if TTK_ENABLE_MPI
  static int finishedElement;
  static int taskCounter;
  static bool keepWorking;
  static int globalElementCounter;
#endif

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
                     std::vector<SimplexId> *trajectory,
                     std::vector<double> *distanceFromSeed,
                     ttk::SimplexId v,
                     const SimplexId *offsets,
                     dataType *scalars,
                     int seedIdentifier) const;

    inline void setVertexNumber(const SimplexId &vertexNumber) {
      vertexNumber_ = vertexNumber;
    }

    inline void setSeedNumber(const SimplexId &seedNumber) {
      seedNumber_ = seedNumber;
    }

    inline void setDirection(int direction) {
      direction_ = direction;
    }


#if TTK_ENABLE_MPI

    void createMessageType() {
      MPI_Datatype types[]
        = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
      int lengths[] = {1, 1, 1, 1, 1};
      const long int mpi_offsets[]
        = {offsetof(Message, Id1), offsetof(Message, Id2),
           offsetof(Message, DistanceFromSeed1),
           offsetof(Message, DistanceFromSeed2),
           offsetof(Message, SeedIdentifier)};
      MPI_Type_create_struct(
        5, lengths, mpi_offsets, types, &(this->MessageType));
      MPI_Type_commit(&(this->MessageType));
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

    inline void setOutputTrajectories(
      ArrayLinkedList<std::vector<SimplexId>, TABULAR_SIZE> *trajectories) {
      outputTrajectories_ = trajectories;
    }

    inline void setOutputDistancesFromSeed(
      ArrayLinkedList<std::vector<double>, TABULAR_SIZE> *distancesFromSeed) {
      outputDistancesFromSeed_ = distancesFromSeed;
    }

    inline void setOutputSeedIdentifiers(
      ArrayLinkedList<int, TABULAR_SIZE> *seedIdentifiers) {
      outputSeedIdentifiers_ = seedIdentifiers;
    }

    inline void setGlobalToLocal(std::map<SimplexId, SimplexId> map) {
      this->globalToLocal = map;
    }

  protected:
    SimplexId vertexNumber_;
    SimplexId seedNumber_;
    int direction_;
    void *inputScalarField_;
    const SimplexId *inputOffsets_;
    SimplexId *vertexIdentifierScalarField_;
    ArrayLinkedList<std::vector<SimplexId>, TABULAR_SIZE> *outputTrajectories_;
    ArrayLinkedList<std::vector<double>, TABULAR_SIZE>
      *outputDistancesFromSeed_;
    ArrayLinkedList<int, TABULAR_SIZE> *outputSeedIdentifiers_;
    std::map<SimplexId, SimplexId> globalToLocal;
  };
} // namespace ttk

template <typename dataType, class triangulationType>
void ttk::IntegralLines::create_task(const triangulationType *triangulation,
                                     std::vector<SimplexId> *trajectory,
                                     std::vector<double> *distanceFromSeed,
                                     ttk::SimplexId v,
                                     const SimplexId *offsets,
                                     dataType *scalars,
                                     int seedIdentifier) const {
#if TTK_ENABLE_MPI
  struct Message m = {0, 0, 0, 0};
#endif
  double distance = (*distanceFromSeed)[0];
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
#if TTK_ENABLE_MPI
#pragma omp atomic update
      (finishedElement)++;
#endif
    } else {
      v = vnext;
      triangulation->getVertexPoint(v, p1[0], p1[1], p1[2]);
      distance += Geometry::distance(p0, p1, 3);
#if TTK_ENABLE_MPI
      if(this->PointGhostArray[v] && ttk::type::DUPLICATEPOINT) {
        int finished;
#pragma omp atomic read
        finished = finishedElement;
        // if (this->MyRank == this->ProcessId[v]){
        // printMsg("Sending element " + std::to_string(this->GlobalIdsArray[v])
        //          + " to process to process "
        //          + std::to_string(this->ProcessId[v])
        //          + " finishedElement: " + std::to_string(finished));
        // }

        isMax = true;
        m.Id2 = this->GlobalIdsArray[v];
        m.Id1 = this->GlobalIdsArray[trajectory->back()];
        m.DistanceFromSeed2 = distance;
        m.SeedIdentifier = seedIdentifier;
        m.DistanceFromSeed1 = distanceFromSeed->back();
        MPI_Send(&m, 1, this->MessageType, this->ProcessId[v],
                 IS_ELEMENT_TO_PROCESS, this->MPIComm);
        // printMsg("Sent element " + std::to_string(m.Id)
        //          + " to process to process "
        //          + std::to_string(this->ProcessId[v]));
      } else {
#endif
        // (*trajectory).push_back(v);

        // p0[0] = p1[0];
        // p0[1] = p1[1];
        // p0[2] = p1[2];
        // (*distanceFromSeed).push_back(distance);
#if TTK_ENABLE_MPI
      }
      (*trajectory).push_back(v);

      p0[0] = p1[0];
      p0[1] = p1[1];
      p0[2] = p1[2];
      (*distanceFromSeed).push_back(distance);
#endif
    }
  }
#if TTK_ENABLE_MPI
  int tempTask;
  int seed;
  int totalSeed;
#pragma omp atomic update
  taskCounter--;
#pragma omp atomic read
  tempTask = taskCounter;
  if(tempTask == 0) {
#pragma omp critical
    {
#pragma omp atomic read
      seed = (finishedElement);
#pragma omp atomic update
      (finishedElement) -= seed;
    }
    if(seed > 0) {
      // printMsg("Taskwait done, sending number of finished elements");
      m.Id1 = seed;
      if(this->MyRank != 0) {
        MPI_Send(&m, 1, this->MessageType, 0, FINISHED_ELEMENT, this->MPIComm);
        // printMsg("finishedElement: " + std::to_string(seed));
      } else {
#pragma omp atomic update
        globalElementCounter -= m.Id1;
#pragma omp atomic read
          totalSeed = globalElementCounter;
          if(totalSeed == 0) {
#pragma omp atomic write
            (keepWorking) = false;
            for(int i = 0; i < this->NumberOfProcesses; i++) {
              MPI_Send(
                &m, 1, this->MessageType, i, STOP_WORKING, this->MPIComm);
          }
        }
        }

      // printMsg("Taskwait done, number of finished elements sent");
      }
    }

#endif
}

template <typename dataType, class triangulationType>
int ttk::IntegralLines::execute(triangulationType *triangulation) {

#if TTK_ENABLE_MPI
  this->createMessageType();
  finishedElement = 0;
  taskCounter = seedNumber_;
  globalElementCounter = this->GlobalElementToCompute;
  // if(this->MyRank == 0)
  //   printMsg("GlobalElementToCompute: "
  //            + std::to_string(this->GlobalElementToCompute));
  int totalSeed;
  keepWorking = true;
  bool keepWorkingAux = keepWorking;
  int provided;
  MPI_Query_thread(&provided);
  printMsg("LEVEL OF THREAD SUPPORT: " + std::to_string(provided));
#endif

  const SimplexId *offsets = inputOffsets_;
  SimplexId *identifiers = vertexIdentifierScalarField_;
  dataType *scalars = static_cast<dataType *>(inputScalarField_);
  ArrayLinkedList<std::vector<SimplexId>, TABULAR_SIZE> *trajectories
    = outputTrajectories_;
  ArrayLinkedList<std::vector<double>, TABULAR_SIZE> *distancesFromSeed
    = outputDistancesFromSeed_;
  ArrayLinkedList<int, TABULAR_SIZE> *seedIdentifiers = outputSeedIdentifiers_;
  Timer t;
  // get the seeds
  std::unordered_set<SimplexId> isSeed;
  for(SimplexId k = 0; k < seedNumber_; ++k) {
    isSeed.insert(identifiers[k]);
    // if (this->GlobalIdsArray[identifiers[k]] == 81 ||
    // this->GlobalIdsArray[identifiers[k]] == 273){
    //     printMsg("Seed 91 or 273 present in identifier");
    //   }
  }
  // printMsg("isSeed: "+std::to_string(isSeed.size()));
  // printMsg("seedNumber_: "+std::to_string(seedNumber_));
  // printMsg("Seed number: " + std::to_string(seedNumber_));
  std::vector<SimplexId> seeds(isSeed.begin(), isSeed.end());
  // if (seeds.size() != seedNumber_){
  //   printMsg("ERRROR");
  // }
  // printMsg("seeds size: "+std::to_string(seeds.size()));
  // printMsg("last element of seeds: "+std::to_string(seeds[seeds.size()-1]));
  // printMsg("last element of identifiers:
  // "+std::to_string(identifiers[seedNumber_-1])); printMsg("seeds.size()-1
  // element of identifiers: "+std::to_string(identifiers[seeds.size()-1]));
  // printMsg("seedNumber_-1 element of seeds:
  // "+std::to_string(seeds[seedNumber_-1]));

  // for (int i =0; i<seedNumber_; i++){
  //   if ( this->GlobalIdsArray[seeds[i]] == 81 ||
  //   this->GlobalIdsArray[seeds[i]] == 273){
  //     printMsg("Seed 81 or 273 present in ");
  // }

  //     }
  isSeed.clear();
#if TTK_ENABLE_MPI
#pragma omp parallel shared(                                       \
  finishedElement, keepWorking, globalElementCounter, taskCounter) \
  firstprivate(keepWorkingAux)
  {
#else
#pragma omp parallel
  {
#endif
#pragma omp single nowait
    {
      for(SimplexId i = 0; i < seedNumber_; ++i) {
        SimplexId v{seeds[i]};
        std::vector<int> *trajectory
          = trajectories->addArrayElement(std::vector<int>(1, v));
        std::vector<double> *distanceFromSeed
          = distancesFromSeed->addArrayElement(std::vector<double>(1, 0));
        int seedIdentifier;
#if TTK_ENABLE_MPI
        seedIdentifier = this->GlobalIdsArray[v];
#else
        seedIdentifier = v;
#endif
        seedIdentifiers->addArrayElement(seedIdentifier);
#pragma omp task firstprivate(v, i)
          {
            this->create_task<dataType, triangulationType>(
              triangulation, trajectory, distanceFromSeed, v, offsets, scalars,
              seedIdentifier);
          }
      }
#if TTK_ENABLE_MPI
      MPI_Status status;
      struct Message m = {0, 0, 0, 0};
      while(keepWorkingAux) {
        // printMsg("Start receiving messages");
        MPI_Recv(&m, 3, this->MessageType, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 this->MPIComm, &status);
        // printMsg("Message Received");
        int stat = status.MPI_TAG;
        switch(stat) {
          case IS_ELEMENT_TO_PROCESS: {
            // printMsg("Message received: m.Id:" + std::to_string(m.Id)
            //          + ", m.SeedIdentifier:" +
            //          std::to_string(m.SeedIdentifier)
            //          + ", m.DistanceFromSeed:"
            //          + std::to_string(m.DistanceFromSeed));
            std::vector<int> *trajectory
              = trajectories->addArrayElement(std::vector<int>{
                this->globalToLocal[m.Id1], this->globalToLocal[m.Id2]});
            std::vector<double> *distanceFromSeed
              = distancesFromSeed->addArrayElement(
                std::vector<double>{m.DistanceFromSeed1, m.DistanceFromSeed2});
            int elementId = m.Id2;
            int identifier = m.SeedIdentifier;
            seedIdentifiers->addArrayElement(identifier);
#pragma omp atomic update
            (taskCounter)++;
#pragma omp task firstprivate(elementId, identifier)
            {

              this->create_task<dataType, triangulationType>(
                triangulation, trajectory, distanceFromSeed,
                this->globalToLocal[elementId], offsets, scalars, identifier);
            }
            break;
          }
          case FINISHED_ELEMENT: {
            // printMsg("Received finished element");
#pragma omp atomic update
            (globalElementCounter) -= m.Id1;
#pragma omp atomic read
              totalSeed = (globalElementCounter);
              // printMsg("totalSeed: " + std::to_string(totalSeed));
              if(totalSeed == 0) {
#pragma omp atomic write
                (keepWorking) = false;
                // printMsg("Proc 0 tells all processes to stop working");
                for(int i = 1; i < this->NumberOfProcesses; i++) {
                  MPI_Send(
                    &m, 1, this->MessageType, i, STOP_WORKING, this->MPIComm);
              }
            }
            break;
          }
          case STOP_WORKING: {
            // printMsg("I was told to stop working");
#pragma omp atomic write
            (keepWorking) = false;
            break;
          }
          default:
            break;
        }
#pragma omp atomic read
        keepWorkingAux = (keepWorking);
      }

#endif
      {
        std::stringstream msg;
        msg << "Processed " << vertexNumber_ << " points";
        this->printMsg(msg.str(), 1, t.getElapsedTime(), 1);
      }
    }
  }
  return 0;
}

// template <typename dataType, class Compare, class triangulationType>
// int ttk::IntegralLines::execute(Compare cmp,
//                                 const triangulationType *triangulation) const
//                                 {
//   const auto offsets = inputOffsets_;
//   SimplexId *identifiers
//     = static_cast<SimplexId *>(vertexIdentifierScalarField_);
//   dataType *scalars = static_cast<dataType *>(inputScalarField_);
//   std::vector<std::vector<SimplexId>> *trajectories = outputTrajectories_;

//   Timer t;

//   get the seeds
//   std::unordered_set<SimplexId> isSeed;
//   for(SimplexId k = 0; k < seedNumber_; ++k)
//     isSeed.insert(identifiers[k]);
//   std::vector<SimplexId> seeds(isSeed.begin(), isSeed.end());
//   isSeed.clear();

//   trajectories->resize(seeds.size());
//   for(SimplexId i = 0; i < (SimplexId)seeds.size(); ++i) {
//     SimplexId v{seeds[i]};
//     (*trajectories)[i].push_back(v);

//     bool isMax{};
//     while(!isMax) {
//       SimplexId vnext{-1};
//       float fnext = std::numeric_limits<float>::min();
//       SimplexId neighborNumber = triangulation->getVertexNeighborNumber(v);
//       for(SimplexId k = 0; k < neighborNumber; ++k) {
//         SimplexId n;
//         triangulation->getVertexNeighbor(v, k, n);

//         if((direction_ == static_cast<int>(Direction::Forward))
//            xor (offsets[n] < offsets[v])) {
//           const float f
//             = getGradient<dataType, triangulationType>(v, n, scalars);
//           if(f > fnext) {
//             vnext = n;
//             fnext = f;
//           }
//         }
//       }

//       if(vnext == -1)
//         isMax = true;
//       else {
//         v = vnext;
//         (*trajectories)[i].push_back(v);

//         if(cmp(v))
//           isMax = true;
//       }
//     }
//   }

//   {
//     std::stringstream msg;
//     msg << "Processed " << vertexNumber_ << " points";
//     this->printMsg(msg.str(), 1, t.getElapsedTime(), 1);
//   }

//   return 0;
// }
