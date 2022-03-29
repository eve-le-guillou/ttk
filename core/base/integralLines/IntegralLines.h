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
#include <LinkedList.h>
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
  int Id;
  double DistanceFromSeed;
  int SeedIdentifier;
};

namespace ttk {

  static int finishedElement;
  static int taskCounter;
  static bool keepWorking;
  static int globalElementCounter;
  static int sent;
  static int received;
  static int localyProcessed;

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
                     int index,

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
      MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_INT};
      int lengths[] = {1, 1, 1};
      const long int mpi_offsets[]
        = {offsetof(Message, Id), offsetof(Message, DistanceFromSeed),
           offsetof(Message, SeedIdentifier)};
      MPI_Type_create_struct(
        3, lengths, mpi_offsets, types, &(this->MessageType));
      MPI_Type_commit(&(this->MessageType));
    }

    int getLocalIdFromGlobalId(int globalId) {
      bool found = (this->GlobalIdsArray[0] == globalId);
      int i = 0;
      while((!found) && (i < vertexNumber_)) {
        i++;
        found = (this->GlobalIdsArray[i] == globalId);
      }
      if(i >= vertexNumber && !found) {
        std::cout << "Error in getLocalIdFromGlobalId" << std::endl;
      }
      return i;
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
      LinkedList<std::vector<SimplexId> *, TABULAR_SIZE> *trajectories) {
      outputTrajectories_ = trajectories;
    }

    inline void setOutputDistancesFromSeed(
      LinkedList<std::vector<double> *, TABULAR_SIZE> *distancesFromSeed) {
      outputDistancesFromSeed_ = distancesFromSeed;
    }

    inline void
      setOutputSeedIdentifiers(LinkedList<int, TABULAR_SIZE> *seedIdentifiers) {
      outputSeedIdentifiers_ = seedIdentifiers;
    }

  protected:
    SimplexId vertexNumber_;
    SimplexId seedNumber_;
    int direction_;
    void *inputScalarField_;
    const SimplexId *inputOffsets_;
    SimplexId *vertexIdentifierScalarField_;
    LinkedList<std::vector<SimplexId> *, TABULAR_SIZE> *outputTrajectories_;
    LinkedList<std::vector<double> *, TABULAR_SIZE> *outputDistancesFromSeed_;
    LinkedList<int, TABULAR_SIZE> *outputSeedIdentifiers_;
  };
} // namespace ttk

template <typename dataType, class triangulationType>
void ttk::IntegralLines::create_task(const triangulationType *triangulation,
                                     std::vector<SimplexId> *trajectory,
                                     std::vector<double> *distanceFromSeed,
                                     ttk::SimplexId v,
                                     int index,
                                     const SimplexId *offsets,
                                     dataType *scalars,
                                     int seedIdentifier) const {
  // printMsg("Start create task for seed identifier " +
  // std::to_string((*seedIdentifier)[index]));
  struct Message m = {0, 0, 0};
  //(*trajectory).push_back(v);
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
#pragma omp atomic update
      localyProcessed++;
// #pragma omp atomic update
//       taskCounter--;
// #pragma omp atomic read
//       elt = (finishedElement);
//       int counts;
// #pragma omp atomic read
//       counts = taskCounter;

//       int sends, receiveds;
// #pragma omp atomic read
//       sends = sent;
// #pragma omp atomic read
//       receiveds = received;
//       int sum = elt+counts+sends-receiveds;
//       if (sum != 1395 && sum != 468){
//         printMsg("finishedElement: " + std::to_string(elt)+ " taskCount:
//         "+std::to_string(counts)+" sent: "+std::to_string(sends)+ " received:
//         "+std::to_string(receiveds)); printMsg("The sum is:
//         "+std::to_string(elt+counts+sends-receiveds));

//       }
#endif
    } else {
      v = vnext;
      triangulation->getVertexPoint(v, p1[0], p1[1], p1[2]);
      distance += Geometry::distance(p0, p1, 3);
#if TTK_ENABLE_MPI
      if(this->PointGhostArray[v] && ttk::type::DUPLICATEPOINT) {
        // #pragma atomic update
        // taskCounter--;
        // int count;
        int finished;
// #pragma omp atomic read
// count = taskCounter;
#pragma atomic read
        finished = finishedElement;
#pragma omp atomic update
        sent++;
        printMsg("Sending element " + std::to_string(this->GlobalIdsArray[v])
                 + " to process to process "
                 + std::to_string(this->ProcessId[v])
                 + " finishedElement: " + std::to_string(finished));
        isMax = true;
        m.Id = this->GlobalIdsArray[v];
        m.DistanceFromSeed = distance;
        m.SeedIdentifier = seedIdentifier;
        MPI_Send(&m, 1, this->MessageType, this->ProcessId[v],
                 IS_ELEMENT_TO_PROCESS, this->MPIComm);
        printMsg("Sent element " + std::to_string(m.Id)
                 + " to process to process "
                 + std::to_string(this->ProcessId[v]));
      } else {
#endif
        (*trajectory).push_back(v);

        p0[0] = p1[0];
        p0[1] = p1[1];
        p0[2] = p1[2];
        (*distanceFromSeed).push_back(distance);
#if TTK_ENABLE_MPI
      }
#endif
    }
  }
#if TTK_ENABLE_MPI
  int tempTask;
  int seed;
  int totalSeed;
#pragma omp critical(communication)
  {
    taskCounter--;
    tempTask = taskCounter;
    // printMsg("TempTask: " + std::to_string(tempTask));
    if(tempTask == 0) {
#pragma omp atomic read
      seed = (finishedElement);
      if(seed > 0) {
        // printMsg("Taskwait done, sending number of finished elements");
        m.Id = seed;
        if(this->MyRank != 0) {
          MPI_Send(
            &m, 1, this->MessageType, 0, FINISHED_ELEMENT, this->MPIComm);
          printMsg("finishedElement: " + std::to_string(seed));
        } else {
          // #pragma omp critical
          // {
          globalElementCounter -= m.Id;
          totalSeed = globalElementCounter;
          // }
          int counting;
#pragma omp atomic read
          counting = taskCounter;
          printMsg("rank 0: totalSeed: " + std::to_string(totalSeed)
                   + " taskCounter: " + std::to_string(counting));
          if(totalSeed == 0) {
#pragma omp atomic write
            (keepWorking) = false;
            for(int i = 0; i < this->NumberOfProcesses; i++) {
              MPI_Send(
                &m, 1, this->MessageType, i, STOP_WORKING, this->MPIComm);
          }
        }
      }
#pragma omp atomic update
      (finishedElement) -= seed;
#pragma omp atomic read
      seed = finishedElement;
#pragma omp atomic read
      tempTask = (taskCounter);
      int elt;
#pragma omp atomic read
      elt = (finishedElement);
      int counts;
#pragma omp atomic read
      counts = taskCounter;

      int sends, receiveds;
#pragma omp atomic read
      sends = sent;
#pragma omp atomic read
      receiveds = received;
      printMsg(
        "Taskwait done, number of finished elements sent, new finishedElement: "
        + std::to_string(seed) + " and taskCount:" + std::to_string(tempTask)
        + " and localy processed: " + std::to_string(localyProcessed));
      printMsg("finishedElement: " + std::to_string(elt) + " taskCount: "
               + std::to_string(counts) + " sent: " + std::to_string(sends)
               + " received: " + std::to_string(receiveds));
      }
    }
    // printMsg("End create task for seed "
    //+ std::to_string((*trajectories)[index][0]));
  }
#endif
}

template <typename dataType, class triangulationType>
int ttk::IntegralLines::execute(triangulationType *triangulation) {

#if TTK_ENABLE_MPI
  this->createMessageType();
  finishedElement = 0;
  taskCounter = seedNumber_;
  sent = 0;
  received = 0;
  localyProcessed = 0;
  globalElementCounter = this->GlobalElementToCompute;
  if(this->MyRank == 0)
    printMsg("GlobalElementToCompute: "
             + std::to_string(this->GlobalElementToCompute));
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
  LinkedList<std::vector<SimplexId> *, TABULAR_SIZE> *trajectories
    = outputTrajectories_;
  LinkedList<std::vector<double> *, TABULAR_SIZE> *distancesFromSeed
    = outputDistancesFromSeed_;
  LinkedList<int, TABULAR_SIZE> *seedIdentifiers = outputSeedIdentifiers_;
  Timer t;
  // get the seeds
  std::unordered_set<SimplexId> isSeed;
  for(SimplexId k = 0; k < seedNumber_; ++k) {
    isSeed.insert(identifiers[k]);
  }
  printMsg("Seed number: " + std::to_string(seedNumber_));
  std::vector<SimplexId> seeds(isSeed.begin(), isSeed.end());
  isSeed.clear();
  int currentSize = seeds.size();
  MPI_Barrier(this->MPIComm);
  printMsg("Bouh0");
#if TTK_ENABLE_MPI
#pragma omp parallel shared(                                             \
  finishedElement, keepWorking, globalElementCounter, taskCounter, sent, \
  received, localyProcessed) firstprivate(keepWorkingAux, currentSize)
  {
#else
#pragma omp parallel
  {
#endif
#pragma omp single nowait
    {
      //#pragma omp task
      //      {
      for(SimplexId i = 0; i < seedNumber_; ++i) {
        SimplexId v{seeds[i]};
        std::vector<int> *trajectory = new std::vector<int>(1, v);
        trajectories->addArrayElement(trajectory);
        std::vector<double> *distanceFromSeed = new std::vector<double>(1, 0);
        distancesFromSeed->addArrayElement(distanceFromSeed);
        seedIdentifiers->addArrayElement(this->GlobalIdsArray[v]);
#pragma omp task firstprivate(v, i)
          {
#if TTK_ENABLE_MPI
            this->create_task<dataType, triangulationType>(
              triangulation, trajectory, distanceFromSeed, v, i, offsets,
              scalars, this->GlobalIdsArray[v]/*, &finishedElement, &taskCounter,
              &globalElementToCompute, &keepWorking*/);
#else
            this->create_task<dataType, triangulationType>(
              triangulation, trajectory, distanceFromSeed, v, i, offsets,
              scalars, seedIdentifier);
#endif
          }
          //        }
      }
      MPI_Barrier(this->MPIComm);
      // printMsg("size of trajectory list:
      // "+std::to_string(trajectories->sizeList)); printMsg("seed number:
      // "+std::to_string(seedNumber_));

      // printMsg("Bouh1");
      // MPI_Barrier(this->MPIComm);
#if TTK_ENABLE_MPI
      MPI_Status status;
      struct Message m = {0, 0, 0};
      while(keepWorkingAux) {
        // MPI_Barrier(this->MPIComm);
        printMsg("Start receiving messages");
        MPI_Recv(&m, 3, this->MessageType, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 this->MPIComm, &status);
        // MPI_Barrier(this->MPIComm);
        printMsg("Message Received");
        int stat = status.MPI_TAG;
        switch(stat) {
          case IS_ELEMENT_TO_PROCESS: {
            printMsg("Message received: m.Id:" + std::to_string(m.Id)
                     + ", m.SeedIdentifier:" + std::to_string(m.SeedIdentifier)
                     + ", m.DistanceFromSeed:"
                     + std::to_string(m.DistanceFromSeed));
#pragma omp atomic update
            received++;
            // trajectories->resize(currentSize + 1);
            // distancesFromSeed->resize(currentSize + 1);
            std::vector<int> *trajectory
              = new std::vector<int>(1, this->getLocalIdFromGlobalId(m.Id));
            trajectories->addArrayElement(trajectory);
            std::vector<double> *distanceFromSeed
              = new std::vector<double>(1, m.DistanceFromSeed);
            distancesFromSeed->addArrayElement(distanceFromSeed);
            int elementId = m.Id;
            int identifier = m.SeedIdentifier;
            // seedIdentifier->push_back(m.SeedIdentifier);
            seedIdentifiers->addArrayElement(identifier);
#pragma omp critical(communication)
            (taskCounter)++;
#pragma omp task firstprivate(elementId, currentSize, identifier)
            {

              // printMsg("Create task for:
              // "+std::to_string(seedIdentifiers->at(currentSize))+
              // "currentSize:
              // "+std::to_string(currentSize));
              this->create_task<dataType, triangulationType>(
                triangulation, trajectory, distanceFromSeed,
                this->getLocalIdFromGlobalId(m.Id), currentSize, offsets,
                scalars, identifier);
            }
            // printMsg("Task created from message");
            currentSize++;
            break;
          }
          case FINISHED_ELEMENT: {
#pragma omp critical(communication)
            {
              printMsg("Received finished element");
#pragma omp atomic update
              (globalElementCounter) -= m.Id;
#pragma omp atomic read
              totalSeed = (globalElementCounter);
              printMsg("totalSeed: " + std::to_string(totalSeed));
              if(totalSeed == 0) {
#pragma omp atomic write
                (keepWorking) = false;
                printMsg("Proc 0 tells all processes to stop working");
                for(int i = 1; i < this->NumberOfProcesses; i++) {
                  MPI_Send(
                    &m, 1, this->MessageType, i, STOP_WORKING, this->MPIComm);
              }
            }
            }
            break;
          }
          case STOP_WORKING: {
            printMsg("I was told to stop working");
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
  MPI_Barrier(this->MPIComm);
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
