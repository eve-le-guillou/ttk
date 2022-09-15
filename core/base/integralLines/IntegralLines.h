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
#define IS_MESSAGE_SIZE 1
#include <mpi.h>
#endif

namespace ttk {

#if TTK_ENABLE_MPI
  static int finishedElement_;
  static std::vector<std::vector<SimplexId> *> unfinishedTraj;
  static std::vector<std::vector<double> *> unfinishedDist;
  static std::vector<int> unfinishedSeed;

  /*
   * For each integral line continuing on another process, we send two layers of
   * ghost cells and we keep two layers for this process, meaning that vertices
   * Id1 and Id2 belong to this process and Id3 belongs to a neighboring
   * process. Id4 either belong to this process or to the neighboring process,
   * depending on the case.
   */
  struct ElementToBeSent {
    ttk::SimplexId Id1;
    ttk::SimplexId Id2;
    ttk::SimplexId Id3;
    ttk::SimplexId Id4;
    double DistanceFromSeed1;
    double DistanceFromSeed2;
    double DistanceFromSeed3;
    double DistanceFromSeed4;
    ttk::SimplexId SeedIdentifier;
  };
#endif

  enum Direction { Forward = 0, Backward };

  class IntegralLines : virtual public Debug {

  public:
    IntegralLines();
    ~IntegralLines() override;

    template <typename dataType,
              class triangulationType = ttk::AbstractTriangulation>
    int execute(triangulationType *triangulation);

    /**
     * Computes the integral line starting at the vertex of global id
     * seedIdentifier.
     *
     */
    template <typename dataType,
              class triangulationType = ttk::AbstractTriangulation>
    void computeIntegralLine(const triangulationType *triangulation,
                             std::vector<ttk::SimplexId> *trajectory,
                             std::vector<double> *distanceFromSeed,
                             const ttk::SimplexId *offsets,
                             ttk::SimplexId seedIdentifier) const;

    /*
     * Create an OpenMP task that contains the computation of nbElement integral
     * lines. chunk_trajectory, chunk_distanceFromSeed and chunk_SeedIdentifier
     * contain pointers to trajectories, their distance from the seed and their
     * seed identifier.
     *
     */

    template <typename dataType, class triangulationType>
    void
      createTask(const triangulationType *triangulation,
                 std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectory,
                 std::vector<std::vector<double> *> &chunk_distanceFromSeed,
                 const ttk::SimplexId *offsets,
                 std::vector<ttk::SimplexId> &chunk_seedIdentifier,
                 int nbElement) const;
    /*
     * Initializes the three attributes of an integral line: the global id of
     * its seed, its trajectory, and the distances of its points with regards to
     * its seed. Then stores that the pointers to those objects in vectors to
     * use it for task creation.
     *
     */
    template <typename dataType, class triangulationType>
    void prepareForTask(
      const triangulationType *triangulation,
      std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectories,
      std::vector<std::vector<double> *> &chunk_distanceFromSeed,
      std::vector<ttk::SimplexId> &chunk_identifier,
      int startingIndex,
      int nbElement,
      std::vector<SimplexId> *seeds) const;

    template <typename dataType,
              class Compare,
              class triangulationType = ttk::AbstractTriangulation>
    int execute(Compare, const triangulationType *) const;

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

    /**
     * Checks if an integral line should be continued on another process or not.
     * If so, encapsulate the necessary data in a struct and stores it in
     * toSend_.
     *
     */
    template <class triangulationType>
    void storeToSendIfNecessary(const triangulationType *triangulation,
                                std::vector<SimplexId> *trajectory,
                                std::vector<double> *distanceFromSeed,
                                ttk::SimplexId seedIdentifier,
                                bool &isMax) const;

    /**
     * Extract the data of element to initialize the three attributes of an
     * integral line and stores their pointers the in chunk vectors at index.
     * When chunk vectors are full, the task is created and index is
     * reinitialized to 0.
     *
     */
    template <typename dataType, class triangulationType>
    void receiveElement(
      const triangulationType *triangulation,
      ElementToBeSent &element,
      std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectories,
      std::vector<std::vector<double> *> &chunk_distanceFromSeed,
      std::vector<ttk::SimplexId> &chunk_identifier,
      int &index,
      int taskSize,
      const ttk::SimplexId *offsets);

    inline void
      setToSend(std::vector<std::vector<std::vector<ElementToBeSent>>> *send) {
      toSend_ = send;
    }

    inline void setGlobalElementCounter(int counter) {
      globalElementCounter_ = counter;
    }

    inline void initializeNeighbors() {
      std::unordered_set<int> neighbors;
      getNeighbors<ttk::SimplexId>(
        neighbors, vertRankArray_, vertexNumber_, ttk::MPIcomm_);
      neighborNumber_ = neighbors.size();
      neighbors_.resize(neighborNumber_);
      int idx = 0;
      for(int neighbor : neighbors) {
        neighborsToId_[neighbor] = idx;
        neighbors_[idx] = neighbor;
        idx++;
      }
    }

    void createMessageType() {
      ttk::SimplexId id = 0;
      MPI_Datatype types[] = {getMPIType(id), getMPIType(id), getMPIType(id),
                              getMPIType(id), MPI_DOUBLE,     MPI_DOUBLE,
                              MPI_DOUBLE,     MPI_DOUBLE,     getMPIType(id)};
      int lengths[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
      const long int mpi_offsets[]
        = {offsetof(ElementToBeSent, Id1),
           offsetof(ElementToBeSent, Id2),
           offsetof(ElementToBeSent, Id3),
           offsetof(ElementToBeSent, Id4),
           offsetof(ElementToBeSent, DistanceFromSeed1),
           offsetof(ElementToBeSent, DistanceFromSeed2),
           offsetof(ElementToBeSent, DistanceFromSeed3),
           offsetof(ElementToBeSent, DistanceFromSeed4),
           offsetof(ElementToBeSent, SeedIdentifier)};
      MPI_Type_create_struct(
        9, lengths, mpi_offsets, types, &(this->MessageType));
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

    inline void
      setVertexIdentifierScalarField(std::vector<SimplexId> *const data) {
      vertexIdentifierScalarField_ = data;
    }

    inline void setOutputTrajectories(
      ArrayLinkedList<std::vector<ttk::SimplexId>, TABULAR_SIZE>
        *trajectories) {
      outputTrajectories_ = trajectories;
    }

    inline void setOutputDistancesFromSeed(
      ArrayLinkedList<std::vector<double>, TABULAR_SIZE> *distancesFromSeed) {
      outputDistancesFromSeed_ = distancesFromSeed;
    }

    inline void setOutputSeedIdentifiers(
      ArrayLinkedList<ttk::SimplexId, TABULAR_SIZE> *seedIdentifiers) {
      outputSeedIdentifiers_ = seedIdentifiers;
    }

    inline void setChunkSize(int size) {
      chunkSize_ = size;
    }

  protected:
    SimplexId vertexNumber_;
    SimplexId seedNumber_;
    SimplexId chunkSize_;
    SimplexId direction_;
    void *inputScalarField_;
    const SimplexId *inputOffsets_;
    std::vector<SimplexId> *vertexIdentifierScalarField_;
    ArrayLinkedList<std::vector<ttk::SimplexId>, TABULAR_SIZE>
      *outputTrajectories_;
    ArrayLinkedList<std::vector<double>, TABULAR_SIZE>
      *outputDistancesFromSeed_;
    ArrayLinkedList<ttk::SimplexId, TABULAR_SIZE> *outputSeedIdentifiers_;
#ifdef TTK_ENABLE_MPI
    const int *vertRankArray_{nullptr};
    std::vector<std::vector<std::vector<ElementToBeSent>>> *toSend_{nullptr};
    int neighborNumber_;
    std::unordered_map<int, int> neighborsToId_;
    std::vector<int> neighbors_;
    SimplexId keepWorking_;
    SimplexId globalElementCounter_;
    MPI_Datatype MessageType;
#endif
  };
} // namespace ttk

#ifdef TTK_ENABLE_MPI

template <typename dataType, class triangulationType>
void ttk::IntegralLines::receiveElement(
  const triangulationType *triangulation,
  ElementToBeSent &element,
  std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectories,
  std::vector<std::vector<double> *> &chunk_distanceFromSeed,
  std::vector<ttk::SimplexId> &chunk_identifier,
  int &index,
  int taskSize,
  const ttk::SimplexId *offsets) {
  ttk::SimplexId localId1 = -1;
  ttk::SimplexId identifier = element.SeedIdentifier;
  std::vector<ttk::SimplexId> *trajectory = nullptr;
  std::vector<double> *distanceFromSeed = nullptr;
  bool isUnfinished = false;
  if(element.Id1 != -1) {
    localId1 = triangulation->getVertexLocalId(element.Id1);
    // If the first vertex belong to the domain of this process
    // and is not a ghost, we are in the case of an unfinished trajectory
    if(vertRankArray_[localId1] == ttk::MPIrank_) {
      isUnfinished = true;
#pragma omp critical(unfinishedTrajectories)
      {
        // We find which trajectory it is and get its pointer and delete it from
        // the vector
        ttk::SimplexId localId3 = triangulation->getVertexLocalId(element.Id3);
        for(int i = 0; i < (int)unfinishedSeed.size(); i++) {
          if(unfinishedSeed[i] == element.SeedIdentifier
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
      trajectory = outputTrajectories_->addArrayElement(
        std::vector<ttk::SimplexId>(1, localId1));
      distanceFromSeed = outputDistancesFromSeed_->addArrayElement(
        std::vector<double>(1, element.DistanceFromSeed1));
    }
  }
  if(localId1 == -1) {
    trajectory
      = outputTrajectories_->addArrayElement(std::vector<ttk::SimplexId>(0));
    distanceFromSeed
      = outputDistancesFromSeed_->addArrayElement(std::vector<double>(0));
  }

  if(!isUnfinished) {
    trajectory->push_back(triangulation->getVertexLocalId(element.Id2));
    distanceFromSeed->push_back(element.DistanceFromSeed2);
    trajectory->push_back(triangulation->getVertexLocalId(element.Id3));
    distanceFromSeed->push_back(element.DistanceFromSeed3);
    outputSeedIdentifiers_->addArrayElement(identifier);
  }
  if(element.Id4 != -1) {
    trajectory->push_back(triangulation->getVertexLocalId(element.Id4));
    distanceFromSeed->push_back(element.DistanceFromSeed4);
    chunk_trajectories[index] = trajectory;
    chunk_identifier[index] = identifier;
    chunk_distanceFromSeed[index] = distanceFromSeed;
    if(index == taskSize - 1) {
      this->createTask<dataType, triangulationType>(
        triangulation, chunk_trajectories, chunk_distanceFromSeed, offsets,
        chunk_identifier, taskSize);
      index = 0;
    } else {
      index++;
    }
  } else {
#pragma omp atomic update seq_cst
    finishedElement_++;
  }
}

template <class triangulationType>
void ttk::IntegralLines::storeToSendIfNecessary(
  const triangulationType *triangulation,
  std::vector<SimplexId> *trajectory,
  std::vector<double> *distanceFromSeed,
  ttk::SimplexId seedIdentifier,
  bool &isMax) const {
#if TTK_ENABLE_MPI
  ElementToBeSent element
    = ElementToBeSent{-1, -1, -1, -1, 0, 0, 0, 0, seedIdentifier};
  if(ttk::isRunningWithMPI()) {
    int size = trajectory->size();
    if(size > 1) {
      int rankArray;
      if(!(isMax && size == 3
           && (vertRankArray_[trajectory->at(size - 1)] == ttk::MPIrank_)
           && (vertRankArray_[trajectory->at(size - 2)] != ttk::MPIrank_)
           && (vertRankArray_[trajectory->at(size - 3)] != ttk::MPIrank_))) {
        if((isMax
            && (vertRankArray_[trajectory->at(size - 1)] != ttk::MPIrank_))
           || (size >= 3
               && (vertRankArray_[trajectory->at(size - 2)]
                   != ttk::MPIrank_))) {
          if(isMax) {
            // The vertex last vertex of the integral line is an extremum
            element.Id4 = -1;
            element.DistanceFromSeed4 = 0;
            element.Id3 = triangulation->getVertexGlobalId(trajectory->back());
            rankArray = vertRankArray_[trajectory->back()];
            element.DistanceFromSeed3 = distanceFromSeed->back();
            element.Id2
              = triangulation->getVertexGlobalId(trajectory->at(size - 2));
            element.DistanceFromSeed2 = distanceFromSeed->at(size - 2);
            if(size == 2) {
              element.Id1 = -1;
              element.DistanceFromSeed1 = 0;
            } else {
              element.Id1
                = triangulation->getVertexGlobalId(trajectory->at(size - 3));
              element.DistanceFromSeed1 = distanceFromSeed->at(size - 3);
            }
          } else {
            if(size == 3) {
              element.Id1 = -1;
              element.DistanceFromSeed1 = 0;
            } else {
              element.Id1
                = triangulation->getVertexGlobalId(trajectory->at(size - 4));
              element.DistanceFromSeed1 = distanceFromSeed->at(size - 4);
            }
            element.Id2
              = triangulation->getVertexGlobalId(trajectory->at(size - 3));
            element.DistanceFromSeed2 = distanceFromSeed->at(size - 3);
            element.Id3
              = triangulation->getVertexGlobalId(trajectory->at(size - 2));
            element.DistanceFromSeed3 = distanceFromSeed->at(size - 2);
            rankArray = vertRankArray_[trajectory->at(size - 2)];
            element.Id4
              = triangulation->getVertexGlobalId(trajectory->at(size - 1));
            element.DistanceFromSeed4 = distanceFromSeed->at(size - 1);
            if(vertRankArray_[trajectory->at(size - 1)] == ttk::MPIrank_) {
              // Here, the last vertex of the integral line belongs to the
              // current process We store the trajectory to reuse it when we
              // receive it back later
#pragma omp critical(unfinishedTrajectories)
              {
                unfinishedDist.push_back(distanceFromSeed);
                unfinishedSeed.push_back(seedIdentifier);
                unfinishedTraj.push_back(trajectory);
              }
            }
          }
          toSend_
            ->at(neighborsToId_.find(rankArray)->second)[omp_get_thread_num()]
            .push_back(element);
          isMax = true;
        }
      }
    }
  }
#endif
}
#endif
template <typename dataType, class triangulationType>
void ttk::IntegralLines::computeIntegralLine(
  const triangulationType *triangulation,
  std::vector<SimplexId> *trajectory,
  std::vector<double> *distanceFromSeed,
  const SimplexId *offsets,
#ifdef TTK_ENABLE_MPI
  ttk::SimplexId seedIdentifier
#else
  ttk::SimplexId ttkNotUsed(seedIdentifier)
#endif
) const {
  double distance = (*distanceFromSeed).back();
  ttk::SimplexId v = (*trajectory).back();
  float p0[3];
  float p1[3];
  triangulation->getVertexPoint(v, p0[0], p0[1], p0[2]);
  bool isMax{};
  while(!isMax) {
    SimplexId vnext{-1};
    ttk::SimplexId fnext = offsets[v];
    SimplexId neighborNumber = triangulation->getVertexNeighborNumber(v);
    for(SimplexId k = 0; k < neighborNumber; ++k) {
      SimplexId n{-1};
      triangulation->getVertexNeighbor(v, k, n);
      if(direction_ == static_cast<int>(Direction::Forward)) {
        if(fnext < offsets[n]) {
          vnext = n;
          fnext = offsets[n];
        }
      } else {
        if(fnext > offsets[n]) {
          vnext = n;
          fnext = offsets[n];
        }
      }
    }

    if(vnext == -1) {
      isMax = true;
#if TTK_ENABLE_MPI
      if(ttk::isRunningWithMPI() && vertRankArray_[v] == ttk::MPIrank_) {
#pragma omp atomic update seq_cst
        finishedElement_++;
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
#ifdef TTK_ENABLE_MPI
    this->storeToSendIfNecessary<triangulationType>(
      triangulation, trajectory, distanceFromSeed, seedIdentifier, isMax);
#endif
  }
}

template <typename dataType, class triangulationType>
void ttk::IntegralLines::prepareForTask(
#ifdef TTK_ENABLE_MPI
  const triangulationType *triangulation,
#else
  const triangulationType *ttkNotUsed(triangulation),
#endif
  std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectories,
  std::vector<std::vector<double> *> &chunk_distanceFromSeed,
  std::vector<ttk::SimplexId> &chunk_identifier,
  int startingIndex,
  int nbElement,
  std::vector<SimplexId> *seeds) const {

  for(SimplexId j = 0; j < nbElement; j++) {
    SimplexId v{seeds->at(j + startingIndex)};
    chunk_trajectories[j]
      = outputTrajectories_->addArrayElement(std::vector<ttk::SimplexId>(1, v));
    chunk_distanceFromSeed[j]
      = outputDistancesFromSeed_->addArrayElement(std::vector<double>(1, 0));
#if TTK_ENABLE_MPI
    chunk_identifier[j] = triangulation->getVertexGlobalId(v);
#else
    chunk_identifier[j] = v;
#endif
    outputSeedIdentifiers_->addArrayElement(chunk_identifier[j]);
  }
}

template <typename dataType, class triangulationType>
void ttk::IntegralLines::createTask(
  const triangulationType *triangulation,
  std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectories,
  std::vector<std::vector<double> *> &chunk_distanceFromSeed,
  const ttk::SimplexId *offsets,
  std::vector<ttk::SimplexId> &chunk_identifier,
  int nbElement) const {

#pragma omp task firstprivate( \
  chunk_trajectories, chunk_distanceFromSeed, chunk_identifier)
  {
    for(int j = 0; j < nbElement; j++) {
      this->computeIntegralLine<dataType, triangulationType>(
        triangulation, chunk_trajectories[j], chunk_distanceFromSeed[j],
        offsets, chunk_identifier[j]);
    }
  }
}

template <typename dataType, class triangulationType>
int ttk::IntegralLines::execute(triangulationType *triangulation) {

#if TTK_ENABLE_MPI
  keepWorking_ = 1;
  finishedElement_ = 0;
#endif

  const SimplexId *offsets = inputOffsets_;
  std::vector<SimplexId> *seeds = vertexIdentifierScalarField_;
  Timer t;

  std::vector<std::vector<ttk::SimplexId> *> chunk_trajectories(chunkSize_);
  std::vector<std::vector<double> *> chunk_distanceFromSeed(chunkSize_);
  std::vector<ttk::SimplexId> chunk_identifier(chunkSize_);
  int taskNumber = (int)seedNumber_ / chunkSize_;
#ifdef TTK_ENABLE_OPENMP
#ifdef TTK_ENABLE_MPI
#pragma omp parallel shared(                                                 \
  finishedElement_, unfinishedDist, unfinishedTraj, unfinishedSeed, toSend_) \
  num_threads(threadNumber_)
  {
#else
#pragma omp parallel num_threads(threadNumber_)
  {
#endif // TTK_ENABLE_MPI
#pragma omp master
    {
#endif // TTK_ENABLE_OPENMP
      for(SimplexId i = 0; i < taskNumber; ++i) {
        this->prepareForTask<dataType, triangulationType>(
          triangulation, chunk_trajectories, chunk_distanceFromSeed,
          chunk_identifier, i * chunkSize_, chunkSize_, seeds);
        this->createTask<dataType, triangulationType>(
          triangulation, chunk_trajectories, chunk_distanceFromSeed, offsets,
          chunk_identifier, chunkSize_);
      }
      int rest = seedNumber_ % chunkSize_;
      if(rest > 0) {
        this->prepareForTask<dataType, triangulationType>(
          triangulation, chunk_trajectories, chunk_distanceFromSeed,
          chunk_identifier, taskNumber * chunkSize_, rest, seeds);
        this->createTask<dataType, triangulationType>(
          triangulation, chunk_trajectories, chunk_distanceFromSeed, offsets,
          chunk_identifier, rest);
      }
#ifdef TTK_ENABLE_OPENMP
    }
  }
#endif
#ifdef TTK_ENABLE_MPI
  if(ttk::isRunningWithMPI()) {
    int i;
    int finishedElementReceived = 0;
    std::vector<int> sendMessageSize(neighborNumber_);
    std::vector<int> recvMessageSize(neighborNumber_);
    std::vector<std::vector<ElementToBeSent>> send_buf(neighborNumber_);
    std::vector<std::vector<ElementToBeSent>> recv_buf(neighborNumber_);
    for(i = 0; i < neighborNumber_; i++) {
      send_buf.reserve((int)seedNumber_ * 0.005);
      recv_buf.reserve((int)seedNumber_ * 0.005);
    }
    std::vector<MPI_Request> requests(2 * neighborNumber_, MPI_REQUEST_NULL);
    std::vector<MPI_Status> statuses(4 * neighborNumber_);
    int taskSize;
    int index;
    int totalMessageSize;
    while(keepWorking_) {
      // Exchange of the number of integral lines finished on all processes
      MPI_Allreduce(&finishedElement_, &finishedElementReceived, 1, MPI_INTEGER,
                    MPI_SUM, ttk::MPIcomm_);
      finishedElement_ = 0;
      // Update the number of integral lines left to compute
      globalElementCounter_ -= finishedElementReceived;
      // Stop working in case there are no more computation to be done
      if(globalElementCounter_ == 0) {
        keepWorking_ = 0;
      }
      if(keepWorking_) {
        totalMessageSize = 0;
        // Preparation of the send buffers and exchange of the size of messages
        // to be sent
        for(i = 0; i < neighborNumber_; i++) {
          for(int j = 0; j < threadNumber_; j++) {
            send_buf[i].insert(send_buf[i].end(), toSend_->at(i)[j].begin(),
                               toSend_->at(i)[j].end());
            toSend_->at(i)[j].clear();
          }
          sendMessageSize[i] = (int)send_buf[i].size();
          MPI_Isend(&sendMessageSize[i], 1, MPI_INTEGER, neighbors_[i],
                    IS_MESSAGE_SIZE, ttk::MPIcomm_, &requests[2 * i]);
          MPI_Irecv(&recvMessageSize[i], 1, MPI_INTEGER, neighbors_[i],
                    IS_MESSAGE_SIZE, ttk::MPIcomm_, &requests[2 * i + 1]);
        }
        MPI_Waitall(2 * neighborNumber_, requests.data(), MPI_STATUSES_IGNORE);
        // Exchange of the data
        for(i = 0; i < neighborNumber_; i++) {
          if(recv_buf[i].size() < (size_t)recvMessageSize[i]) {
            recv_buf[i].resize(recvMessageSize[i]);
          }
          if(recvMessageSize[i] > 0) {
            MPI_Irecv(recv_buf[i].data(), recvMessageSize[i], this->MessageType,
                      neighbors_[i], IS_ELEMENT_TO_PROCESS, ttk::MPIcomm_,
                      &requests[2 * i]);
            totalMessageSize += recvMessageSize[i];
          }

          if(sendMessageSize[i] > 0) {
            MPI_Isend(send_buf[i].data(), sendMessageSize[i], this->MessageType,
                      neighbors_[i], IS_ELEMENT_TO_PROCESS, ttk::MPIcomm_,
                      &requests[2 * i + 1]);
          }
        }
        MPI_Waitall(2 * neighborNumber_, requests.data(), MPI_STATUSES_IGNORE);
        for(i = 0; i < neighborNumber_; i++) {
          send_buf[i].clear();
        }
        // Extraction of the received data and creation of the tasks
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel shared(                                                 \
  finishedElement_, unfinishedDist, unfinishedTraj, unfinishedSeed, toSend_) \
  num_threads(threadNumber_)
        {
#pragma omp master
          {
#endif // TTK_ENABLE_OPENMP
            index = 0;
            taskSize
              = std::min(std::max(totalMessageSize / (threadNumber_ * 100),
                                  std::min(totalMessageSize, 50)),
                         chunkSize_);
            chunk_trajectories.resize(taskSize);
            chunk_distanceFromSeed.resize(taskSize);
            chunk_identifier.resize(taskSize);
            for(i = 0; i < neighborNumber_; i++) {
              for(int j = 0; j < recvMessageSize[i]; j++) {
                this->receiveElement<dataType, triangulationType>(
                  triangulation, recv_buf[i][j], chunk_trajectories,
                  chunk_distanceFromSeed, chunk_identifier, index, taskSize,
                  offsets);
              }
            }
            if(index > 0) {
              this->createTask<dataType, triangulationType>(
                triangulation, chunk_trajectories, chunk_distanceFromSeed,
                offsets, chunk_identifier, index);
            }
#ifdef TTK_ENABLE_OPENMP
          }
        }
#endif // TTK_ENABLE_OPENMP
      }
    }
  }
#endif // TTK_ENABLE_MPI
  {
    std::stringstream msg;
    msg << "Processed " << vertexNumber_ << " points";
    this->printMsg(msg.str(), 1, t.getElapsedTime(), threadNumber_);
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
