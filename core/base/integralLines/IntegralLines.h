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
#define TABULAR_SIZE 50
#if TTK_ENABLE_MPI
#define IS_ELEMENT_TO_PROCESS 0
#define FINISHED_ELEMENT 1
#define STOP_WORKING 2
#include <mpi.h>
#endif

#if TTK_ENABLE_MPI
struct Message {
  int Id1;
  int Id2;
  int Id3;
  int Id4;
  double DistanceFromSeed1;
  double DistanceFromSeed2;
  double DistanceFromSeed3;
  double DistanceFromSeed4;
  int SeedIdentifier;
};
#endif

namespace ttk {

#if TTK_ENABLE_MPI
  static int finishedElement;
  static int taskCounter;
  static bool keepWorking;
  static int globalElementCounter;
  static std::vector<std::vector<SimplexId> *> unfinishedTraj;
  static std::vector<std::vector<double> *> unfinishedDist;
  static std::vector<int> unfinishedSeed;
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
        = {MPI_INT,    MPI_INT,    MPI_INT,    MPI_INT, MPI_DOUBLE,
           MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
      int lengths[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
      const long int mpi_offsets[] = {offsetof(Message, Id1),
                                      offsetof(Message, Id2),
                                      offsetof(Message, Id3),
                                      offsetof(Message, Id4),
                                      offsetof(Message, DistanceFromSeed1),
                                      offsetof(Message, DistanceFromSeed2),
                                      offsetof(Message, DistanceFromSeed3),
                                      offsetof(Message, DistanceFromSeed4),
                                      offsetof(Message, SeedIdentifier)};
      MPI_Type_create_struct(
        9, lengths, mpi_offsets, types, &(this->MessageType));
      MPI_Type_commit(&(this->MessageType));
    }

void checkEndOfComputation() const {
      Message m;
      int tempTask;
      int seed;
      int temp;
      int totalSeed;
#pragma omp atomic read
      tempTask = taskCounter;
      if(tempTask == 0) {
#pragma omp critical(resetFinishedElement)
{
#pragma omp atomic read
        seed = finishedElement;
#pragma omp atomic update
        finishedElement -= seed;
}
        if(seed > 0) {
          m.Id1 = seed;
          if(ttk::MPIrank_ != 0) {
            MPI_Send(
              &m, 1, this->MessageType, 0, FINISHED_ELEMENT, this->MPIComm);
          } else {
#pragma omp atomic update
            globalElementCounter -= m.Id1;
            // printMsg("Send "+std::to_string(m.Id1)+ " finished element to
            // myself");
#pragma omp atomic read
            totalSeed = globalElementCounter;
            if (totalSeed <=0)
		printMsg("totalSeed: "+std::to_string(totalSeed)+", received "+std::to_string(m.Id1)+" from myself");
            if(totalSeed == 0) {
#pragma omp atomic write
              (keepWorking) = false;
              for(int i = 0; i < ttk::MPIsize_; i++) {
                MPI_Send(
                  &m, 1, this->MessageType, i, STOP_WORKING, this->MPIComm);
              }
            }
          }
        }
        // printMsg("Old finishedElement: "+std::to_string(seed)+" New
        // finishedElement: "+std::to_string(temp));
      }
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
                                     const SimplexId *offsets,
                                     dataType *scalars,
                                     int seedIdentifier) const {
#if TTK_ENABLE_MPI
  struct Message m;
  int size;
#endif
  double distance = (*distanceFromSeed).back();
  ttk::SimplexId v = (*trajectory).back();
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
      if(!(this->PointGhostArray[v] && ttk::type::DUPLICATEPOINT)) {
        int temp;
#pragma omp atomic update
        finishedElement++;
        // #pragma omp atomic read
        // temp = finishedElement;
        // if (this->MyRank == 2)
        // printMsg("finishedElement: "+std::to_string(temp));
      }
#endif
    } else {
      v = vnext;
      triangulation->getVertexPoint(v, p1[0], p1[1], p1[2]);
      distance += Geometry::distance(p0, p1, 3);
      (*trajectory).push_back(v);

      p0[0] = p1[0];
      p0[1] = p1[1];
      p0[2] = p1[2];
      (*distanceFromSeed).push_back(distance);
    }
#if TTK_ENABLE_MPI
    size = trajectory->size();
    if(size > 1) {
      int processId;
      if(!(isMax && size == 3
           && (!(this->PointGhostArray[trajectory->at(size - 1)]
                 && ttk::type::DUPLICATEPOINT))
           && (this->PointGhostArray[trajectory->at(size - 2)]
               && ttk::type::DUPLICATEPOINT)
           && (this->PointGhostArray[trajectory->at(size - 3)]
               && ttk::type::DUPLICATEPOINT))) {
        if((isMax
            && (this->PointGhostArray[trajectory->at(size - 1)]
                && ttk::type::DUPLICATEPOINT))
           || (size >= 3
               && (this->PointGhostArray[trajectory->at(size - 2)]
                   && ttk::type::DUPLICATEPOINT))) {
          if(isMax) {
            m.Id4 = -1;
            m.DistanceFromSeed4 = 0;
            m.Id3 = this->GlobalIdsArray[trajectory->back()];
            processId = this->ProcessId[trajectory->back()];
            m.DistanceFromSeed3 = distanceFromSeed->back();
            m.Id2 = this->GlobalIdsArray[trajectory->at(size - 2)];
            m.DistanceFromSeed2 = distanceFromSeed->at(size - 2);
            if(size == 2) {
              m.Id1 = -1;
              m.DistanceFromSeed1 = 0;
            } else {
              m.Id1 = this->GlobalIdsArray[trajectory->at(size - 3)];
              m.DistanceFromSeed1 = distanceFromSeed->at(size - 3);
            }
          } else {
            if(size == 3) {
              m.Id1 = -1;
              m.DistanceFromSeed1 = 0;
            } else {
              m.Id1 = this->GlobalIdsArray[trajectory->at(size - 4)];
              m.DistanceFromSeed1 = distanceFromSeed->at(size - 4);
            }
            m.Id2 = this->GlobalIdsArray[trajectory->at(size - 3)];
            m.DistanceFromSeed2 = distanceFromSeed->at(size - 3);
            m.Id3 = this->GlobalIdsArray[trajectory->at(size - 2)];
            m.DistanceFromSeed3 = distanceFromSeed->at(size - 2);
            processId = this->ProcessId[trajectory->at(size - 2)];
            m.Id4 = this->GlobalIdsArray[trajectory->at(size - 1)];
            m.DistanceFromSeed4 = distanceFromSeed->at(size - 1);
            if(!(this->PointGhostArray[trajectory->at(size - 1)]
                 && ttk::type::DUPLICATEPOINT)) {
#pragma omp critical(unfinishedTrajectories)
              {
                unfinishedDist.push_back(distanceFromSeed);
                unfinishedSeed.push_back(seedIdentifier);
                unfinishedTraj.push_back(trajectory);
              }
            }
          }
          m.SeedIdentifier = seedIdentifier;
          MPI_Send(&m, 1, this->MessageType, processId, IS_ELEMENT_TO_PROCESS,
                   this->MPIComm);
          isMax = true;
        }
      }
    }
#endif
  }
#if TTK_ENABLE_MPI
//#pragma omp critical(finished)
//  {
#pragma omp atomic update
  taskCounter--;
  this->checkEndOfComputation();
//  }
#endif
}

template <typename dataType, class triangulationType>
int ttk::IntegralLines::execute(triangulationType *triangulation) {

#if TTK_ENABLE_MPI
  this->createMessageType();
  finishedElement = 0;
  taskCounter = seedNumber_;
  globalElementCounter = this->GlobalElementToCompute;
  int totalSeed;
  keepWorking = true;
  bool keepWorkingAux = keepWorking;
  // int provided;
  // MPI_Query_thread(&provided);
  // printMsg("LEVEL OF THREAD SUPPORT: " + std::to_string(provided));
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
  }
  std::vector<SimplexId> seeds(isSeed.begin(), isSeed.end());
  isSeed.clear();
#if TTK_ENABLE_MPI
#pragma omp parallel shared(                                       \
  finishedElement, keepWorking, globalElementCounter, taskCounter, \
  unfinishedDist, unfinishedTraj, unfinishedSeed) firstprivate(keepWorkingAux)
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
#pragma omp task firstprivate(seedIdentifier)
        {
          this->create_task<dataType, triangulationType>(
            triangulation, trajectory, distanceFromSeed, offsets, scalars,
            seedIdentifier);
        }
      }
#if TTK_ENABLE_MPI
      MPI_Status status;
      struct Message m;
      while(keepWorkingAux) {
        MPI_Recv(&m, 1, this->MessageType, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 this->MPIComm, &status);
        int stat = status.MPI_TAG;
        switch(stat) {
          case IS_ELEMENT_TO_PROCESS: {
            int localId1 = -1;
            std::vector<int> *trajectory;
            std::vector<double> *distanceFromSeed;
            bool isUnfinished = false;
            if(m.Id1 != -1) {
              localId1 = this->globalToLocal[m.Id1];
              if(!(this->PointGhostArray[localId1]
                   && ttk::type::DUPLICATEPOINT)) {
                isUnfinished = true;
#pragma omp critical(unfinishedTrajectories)
                {
                  int localId3 = this->globalToLocal[m.Id3];
                  for(int i = 0; i < (int)unfinishedSeed.size(); i++) {
                    if(unfinishedSeed[i] == m.SeedIdentifier
                       && (unfinishedTraj[i])->back() == localId3) {
                      trajectory = unfinishedTraj[i];
                      unfinishedTraj.erase(unfinishedTraj.begin() + i);
                      distanceFromSeed = unfinishedDist[i];
                      unfinishedDist.erase(unfinishedDist.begin() + i);
                      unfinishedSeed.erase(unfinishedSeed.begin() + i);
                      break;
                    }
                  }
                }
              } else {
                trajectory = trajectories->addArrayElement(
                  std::vector<int>(1, localId1));
                distanceFromSeed = distancesFromSeed->addArrayElement(
                  std::vector<double>(1, m.DistanceFromSeed1));
              }
            }
            if(localId1 == -1) {
              trajectory = trajectories->addArrayElement(std::vector<int>(0));
              distanceFromSeed
                = distancesFromSeed->addArrayElement(std::vector<double>(0));
            }
            int identifier = m.SeedIdentifier;
            if(!isUnfinished) {
              trajectory->push_back(this->globalToLocal[m.Id2]);
              distanceFromSeed->push_back(m.DistanceFromSeed2);
              trajectory->push_back(this->globalToLocal[m.Id3]);
              distanceFromSeed->push_back(m.DistanceFromSeed3);
              seedIdentifiers->addArrayElement(identifier);
            }
            if(m.Id4 != -1) {
              trajectory->push_back(this->globalToLocal[m.Id4]);
              distanceFromSeed->push_back(m.DistanceFromSeed4);
#pragma omp atomic update
              (taskCounter)++;
#pragma omp task firstprivate(identifier)
              {
                this->create_task<dataType, triangulationType>(
                  triangulation, trajectory, distanceFromSeed, offsets, scalars,
                  identifier);
              }
            } else {

//#pragma omp critical(finished)
//              {
                int temp;
#pragma omp atomic update
                finishedElement++;
                // #pragma omp atomic read
                // temp = finishedElement;
                // if (this->MyRank == 2)
                // printMsg("finishedElement: "+std::to_string(temp));
                this->checkEndOfComputation();
              //}
            }
            break;
          }
          case FINISHED_ELEMENT: {
#pragma omp atomic update
            (globalElementCounter) -= m.Id1;
#pragma omp atomic read
              totalSeed = (globalElementCounter);
		if (totalSeed <= 0)
               printMsg("totalSeed: "+std::to_string(totalSeed)+", received "+std::to_string(m.Id1)+" from "+std::to_string(status.MPI_SOURCE));
              if(totalSeed == 0) {
#pragma omp atomic write
                (keepWorking) = false;
                for(int i = 1; i < ttk::MPIsize_; i++) {
                  MPI_Send(
                    &m, 1, this->MessageType, i, STOP_WORKING, this->MPIComm);
                }
            }
            break;
          }
          case STOP_WORKING: {
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
