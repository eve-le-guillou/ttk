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
#define RECEIVE_SIZE 10
#if TTK_ENABLE_MPI
#define IS_ELEMENT_TO_PROCESS 0
#define FINISHED_ELEMENT 1
#define STOP_WORKING 2
#define EMPTY_MESSAGE -2
#include <mpi.h>
#endif

#if TTK_ENABLE_MPI
struct Message {
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
struct MessageAndMPIInfo {
  std::vector<Message> *m;
  int receiver;
  int tag;
  MPI_Request *request;
  int size;
};

#endif

namespace ttk {

#if TTK_ENABLE_MPI
  static int finishedElement{0};
  static int taskCounter;
  static bool keepWorking{true};
  static int globalElementCounter;
  static std::vector<std::vector<SimplexId> *> unfinishedTraj;
  static std::vector<std::vector<double> *> unfinishedDist;
  static std::vector<int> unfinishedSeed;
  static int sendCount{0};
  static int recvCount{0};
  static std::queue<MessageAndMPIInfo> sendQueue{};
  static std::vector<std::vector<Message>> multipleElementToSend;
  static std::vector<int> messageCount;
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
    void executeAlgorithm(const triangulationType *triangulation,
                          std::vector<ttk::SimplexId> *trajectory,
                          std::vector<double> *distanceFromSeed,
                          const ttk::SimplexId *offsets,
                          dataType *scalars,
                          ttk::SimplexId seedIdentifier) const;

    template <class triangulationType>
    void sendTrajectoryIfNecessary(const triangulationType *triangulation,
                                   std::vector<SimplexId> *trajectory,
                                   std::vector<double> *distanceFromSeed,
                                   ttk::SimplexId seedIdentifier,
                                   bool &isMax) const;

    template <typename dataType, class triangulationType>
    void receiveMessages(const triangulationType *triangulation,
                         const ttk::SimplexId *offsets,
                         dataType *scalars) const;

    template <typename dataType, class triangulationType>
    void
      createTask(const triangulationType *triangulation,
                 std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectory,
                 std::vector<std::vector<double> *> &chunk_distanceFromSeed,
                 const ttk::SimplexId *offsets,
                 dataType *scalars,
                 std::vector<ttk::SimplexId> &chunk_seedIdentifier,
                 int i,
                 int nbElement,
                 std::vector<SimplexId> *seeds) const;

    template <typename dataType, class triangulationType>
    void receiveElement(const triangulationType *triangulation,
                        Message &m,
                        const ttk::SimplexId *offsets,
                        dataType *scalars) const;

    inline void setVertexNumber(const SimplexId &vertexNumber) {
      vertexNumber_ = vertexNumber;
    }

    inline void setSeedNumber(const SimplexId &seedNumber) {
      seedNumber_ = seedNumber;
    }

    inline void setDirection(int direction) {
      direction_ = direction;
    }

    inline void pushMessage(Message m, int receiver, int tag) const {
      std::vector<Message> *m_ptr;
      MPI_Request *request;
      switch(tag) {
        case IS_ELEMENT_TO_PROCESS:
#pragma omp critical(sendMessageQueue)
        {
          multipleElementToSend[receiver][messageCount[receiver]] = m;
          messageCount[receiver]++;
          if(messageCount[receiver] >= messageSize_) {
            messageCount[receiver] = 0;
            m_ptr = this->sentMessages_->addArrayElement(
              multipleElementToSend[receiver]);
            request = this->sentRequests_->addArrayElement(
              MPI_Request{MPI_REQUEST_NULL});
            sendQueue.push(
              MessageAndMPIInfo{m_ptr, receiver, tag, request, messageSize_});
          }
        } break;
        default:
#pragma omp critical(sendMessageQueue)
        {
          m_ptr
            = this->sentMessages_->addArrayElement(std::vector<Message>(1, m));
          request = this->sentRequests_->addArrayElement(
            MPI_Request{MPI_REQUEST_NULL});
          sendQueue.push(MessageAndMPIInfo{m_ptr, receiver, tag, request, 1});
        } break;
      }
    };

    inline MessageAndMPIInfo popMessage() const {
      MessageAndMPIInfo m;
#pragma omp critical(sendMessageQueue)
      {
        m = sendQueue.front();
        sendQueue.pop();
      }
      return m;
    }

    inline void setSentRequests(
      ArrayLinkedList<MPI_Request, TABULAR_SIZE> *sentRequests) {
      sentRequests_ = sentRequests;
    }

    inline void setSentMessages(
      ArrayLinkedList<std::vector<Message>, TABULAR_SIZE> *sentMessages) {
      sentMessages_ = sentMessages;
    }

    inline void setMessageSize(int size) {
      messageSize_ = size;
    }

#if TTK_ENABLE_MPI

    void createMessageType() {
      ttk::SimplexId id = 0;
      MPI_Datatype types[] = {getMPIType(id), getMPIType(id), getMPIType(id),
                              getMPIType(id), MPI_DOUBLE,     MPI_DOUBLE,
                              MPI_DOUBLE,     MPI_DOUBLE,     getMPIType(id)};
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
      int tempTask;
      int seed;
//#pragma omp critical(buf)
//      {
#pragma omp atomic read seq_cst
      tempTask = taskCounter;
      if(tempTask == 0) {
// #pragma omp atomic read
//           seed = finishedElement;
// #pragma omp atomic update
//           finishedElement -= seed;
#pragma omp atomic capture seq_cst
        {
          seed = finishedElement;
          finishedElement = 0;
        }
        if(seed > 0) {
          Message m = Message{seed, -1, -1, -1, 0, 0, 0, 0, -1};
          this->pushMessage(m, 0, FINISHED_ELEMENT);
        }
      }
      //     }
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
    int chunkSize_;
    int direction_;
    void *inputScalarField_;
    const SimplexId *inputOffsets_;
    std::vector<SimplexId> *vertexIdentifierScalarField_;
    ArrayLinkedList<std::vector<ttk::SimplexId>, TABULAR_SIZE>
      *outputTrajectories_;
    ArrayLinkedList<std::vector<double>, TABULAR_SIZE>
      *outputDistancesFromSeed_;
    ArrayLinkedList<ttk::SimplexId, TABULAR_SIZE> *outputSeedIdentifiers_;
    int *rankArray_{nullptr};
    ArrayLinkedList<MPI_Request, TABULAR_SIZE> *sentRequests_;
    ArrayLinkedList<std::vector<Message>, TABULAR_SIZE> *sentMessages_;
    int messageSize_;
  };
} // namespace ttk

template <class triangulationType>
void ttk::IntegralLines::sendTrajectoryIfNecessary(
  const triangulationType *triangulation,
  std::vector<SimplexId> *trajectory,
  std::vector<double> *distanceFromSeed,
  ttk::SimplexId seedIdentifier,
  bool &isMax) const {
#if TTK_ENABLE_MPI
  Message m = Message{-1, -1, -1, -1, 0, 0, 0, 0, seedIdentifier};
  if(ttk::MPIsize_ > 1) {
    int size = trajectory->size();
    if(size > 1) {
      int rankArray;
      if(!(isMax && size == 3
           && (rankArray_[trajectory->at(size - 1)] == ttk::MPIrank_)
           && (rankArray_[trajectory->at(size - 2)] != ttk::MPIrank_)
           && (rankArray_[trajectory->at(size - 3)] != ttk::MPIrank_))) {
        if((isMax && (rankArray_[trajectory->at(size - 1)] != ttk::MPIrank_))
           || (size >= 3
               && (rankArray_[trajectory->at(size - 2)] != ttk::MPIrank_))) {
          if(isMax) {
            m.Id4 = -1;
            m.DistanceFromSeed4 = 0;
            m.Id3 = triangulation->getVertexGlobalId(trajectory->back());
            rankArray = rankArray_[trajectory->back()];
            m.DistanceFromSeed3 = distanceFromSeed->back();
            m.Id2 = triangulation->getVertexGlobalId(trajectory->at(size - 2));
            m.DistanceFromSeed2 = distanceFromSeed->at(size - 2);
            if(size == 2) {
              m.Id1 = -1;
              m.DistanceFromSeed1 = 0;
            } else {
              m.Id1
                = triangulation->getVertexGlobalId(trajectory->at(size - 3));
              m.DistanceFromSeed1 = distanceFromSeed->at(size - 3);
            }
          } else {
            if(size == 3) {
              m.Id1 = -1;
              m.DistanceFromSeed1 = 0;
            } else {
              m.Id1
                = triangulation->getVertexGlobalId(trajectory->at(size - 4));
              m.DistanceFromSeed1 = distanceFromSeed->at(size - 4);
            }
            m.Id2 = triangulation->getVertexGlobalId(trajectory->at(size - 3));
            m.DistanceFromSeed2 = distanceFromSeed->at(size - 3);
            m.Id3 = triangulation->getVertexGlobalId(trajectory->at(size - 2));
            m.DistanceFromSeed3 = distanceFromSeed->at(size - 2);
            rankArray = rankArray_[trajectory->at(size - 2)];
            m.Id4 = triangulation->getVertexGlobalId(trajectory->at(size - 1));
            m.DistanceFromSeed4 = distanceFromSeed->at(size - 1);
            if(rankArray_[trajectory->at(size - 1)] == ttk::MPIrank_) {
#pragma omp critical(unfinishedTrajectories)
              {
                unfinishedDist.push_back(distanceFromSeed);
                unfinishedSeed.push_back(seedIdentifier);
                unfinishedTraj.push_back(trajectory);
              }
            }
          }
          this->pushMessage(m, rankArray, IS_ELEMENT_TO_PROCESS);

          isMax = true;
        }
      }
    }
  }
#endif
}

template <typename dataType, class triangulationType>
void ttk::IntegralLines::receiveElement(const triangulationType *triangulation,
                                        Message &m,
                                        const ttk::SimplexId *offsets,
                                        dataType *scalars) const {
  ttk::SimplexId localId1 = -1;
  ttk::SimplexId identifier = m.SeedIdentifier;
  printMsg("identifier: " + std::to_string(identifier) + " id1: "
           + std::to_string(m.Id1) + " id1: " + std::to_string(m.Id2) + " id1: "
           + std::to_string(m.Id3) + " id1: " + std::to_string(m.Id4));
  std::vector<ttk::SimplexId> *trajectory;
  std::vector<double> *distanceFromSeed;
  bool isUnfinished = false;
  if(m.Id1 != -1) {
    localId1 = triangulation->getVertexLocalId(m.Id1);
    if(rankArray_[localId1] == ttk::MPIrank_) {
      isUnfinished = true;
#pragma omp critical(unfinishedTrajectories)
      {
        ttk::SimplexId localId3 = triangulation->getVertexLocalId(m.Id3);
        printMsg("Localid3: " + std::to_string(localId3));
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
      trajectory = outputTrajectories_->addArrayElement(
        std::vector<ttk::SimplexId>(1, localId1));
      distanceFromSeed = outputDistancesFromSeed_->addArrayElement(
        std::vector<double>(1, m.DistanceFromSeed1));
    }
  }
  if(localId1 == -1) {
    trajectory
      = outputTrajectories_->addArrayElement(std::vector<ttk::SimplexId>(0));
    distanceFromSeed
      = outputDistancesFromSeed_->addArrayElement(std::vector<double>(0));
  }

  if(!isUnfinished) {
    trajectory->push_back(triangulation->getVertexLocalId(m.Id2));
    distanceFromSeed->push_back(m.DistanceFromSeed2);
    trajectory->push_back(triangulation->getVertexLocalId(m.Id3));
    distanceFromSeed->push_back(m.DistanceFromSeed3);
    outputSeedIdentifiers_->addArrayElement(identifier);
  }
  if(m.Id4 != -1) {
    trajectory->push_back(triangulation->getVertexLocalId(m.Id4));
    distanceFromSeed->push_back(m.DistanceFromSeed4);
#pragma omp atomic update seq_cst
    (taskCounter)++;
#pragma omp task firstprivate(identifier)
    {
      this->executeAlgorithm<dataType, triangulationType>(
        triangulation, trajectory, distanceFromSeed, offsets, scalars,
        identifier);
      if(ttk::MPIsize_ > 1) {
#pragma omp atomic update seq_cst
        taskCounter--;
        this->checkEndOfComputation();
      }
    }
  } else {
#pragma omp atomic update seq_cst
    finishedElement++;
    this->checkEndOfComputation();
  }
}

template <typename dataType, class triangulationType>
void ttk::IntegralLines::receiveMessages(const triangulationType *triangulation,
                                         const ttk::SimplexId *offsets,
                                         dataType *scalars) const {
#if TTK_ENABLE_MPI
  if(ttk::MPIsize_ > 1) {
    MPI_Status status;
    std::vector<Message> m_recv(messageSize_);
    int count;
    while(keepWorking) {
      int probe;
      int out = MPI_Iprobe(
        MPI_ANY_SOURCE, MPI_ANY_TAG, this->MPIComm, &probe, &status);
      while(!out && probe) {
        printMsg("About to receive Message");
        MPI_Recv(m_recv.data(), messageSize_, this->MessageType, MPI_ANY_SOURCE,
                 MPI_ANY_TAG, this->MPIComm, &status);
        printMsg("Element: " + std::to_string(m_recv[0].Id1)
                 + " tag: " + std::to_string(status.MPI_TAG));
        MPI_Get_count(&status, this->MessageType, &count);
        // recvCount++;
        int stat = status.MPI_TAG;
        switch(stat) {
          case IS_ELEMENT_TO_PROCESS: {
            for(int i = 0; i < count; i++) {
              this->receiveElement<dataType, triangulationType>(
                triangulation, m_recv[i], offsets, scalars);
            }
            break;
          }
          case FINISHED_ELEMENT: {
            globalElementCounter -= m_recv[0].Id1;
            if(globalElementCounter < 100) {
              printMsg("GlobalElementCounter: "
                       + std::to_string(globalElementCounter));
            }
            if(globalElementCounter == 0) {
              keepWorking = false;
              for(int i = 1; i < ttk::MPIsize_; i++) {
                Message m = Message{-1, -1, -1, -1, 0, 0, 0, 0, -1};
                this->pushMessage(m, i, STOP_WORKING);
                printMsg("All processes should stop working");
              }
            }
            break;
          }
          case STOP_WORKING: {
            keepWorking = false;
            printMsg("I was told to stop working");
            break;
          }
          default:
            break;
        }
        out = MPI_Iprobe(
          MPI_ANY_SOURCE, MPI_ANY_TAG, this->MPIComm, &probe, &status);
      }
      int tempTask;
#pragma omp atomic read
      tempTask = taskCounter;
      std::vector<Message> *m_ptr;
      MPI_Request *request;
      if(tempTask == 0) {
#pragma omp critical(sendMessageQueue)
        {
          for(int i = 0; i < ttk::MPIsize_; i++) {
            if(messageCount[i] > 0) {
              m_ptr = this->sentMessages_->addArrayElement(
                multipleElementToSend[i]);
              request = this->sentRequests_->addArrayElement(
                MPI_Request{MPI_REQUEST_NULL});
              sendQueue.push(MessageAndMPIInfo{
                m_ptr, i, IS_ELEMENT_TO_PROCESS, request, messageCount[i]});
              messageCount[i] = 0;
            }
          }
        }
      }
      MessageAndMPIInfo message;
      while(!sendQueue.empty()) {
        message = this->popMessage();
        MPI_Isend(message.m->data(), message.size, this->MessageType,
                  message.receiver, message.tag, this->MPIComm,
                  message.request);
        // sendCount++;
      }
      // printMsg("sendCount: "+std::to_string(sendCount)+ " recvCount:
      // "+std::to_string(recvCount));
    }
  }
#endif
}

template <typename dataType, class triangulationType>
void ttk::IntegralLines::executeAlgorithm(
  const triangulationType *triangulation,
  std::vector<SimplexId> *trajectory,
  std::vector<double> *distanceFromSeed,
  const SimplexId *offsets,
  dataType *scalars,
  ttk::SimplexId seedIdentifier) const {
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
      if(ttk::MPIsize_ > 1 && rankArray_[v] == ttk::MPIrank_) {
#pragma omp atomic update seq_cst
        finishedElement++;
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
    this->sendTrajectoryIfNecessary<triangulationType>(
      triangulation, trajectory, distanceFromSeed, seedIdentifier, isMax);
  }
}

template <typename dataType, class triangulationType>
void ttk::IntegralLines::createTask(
  const triangulationType *triangulation,
  std::vector<std::vector<ttk::SimplexId> *> &chunk_trajectories,
  std::vector<std::vector<double> *> &chunk_distanceFromSeed,
  const ttk::SimplexId *offsets,
  dataType *scalars,
  std::vector<ttk::SimplexId> &chunk_identifier,
  int i,
  int nbElement,
  std::vector<SimplexId> *seeds) const {

  for(SimplexId j = 0; j < nbElement; j++) {
    SimplexId v{seeds->at(j + i * chunkSize_)};
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

#pragma omp task firstprivate( \
  chunk_trajectories, chunk_distanceFromSeed, chunk_identifier)
  {
    for(int j = 0; j < nbElement; j++) {
      this->executeAlgorithm<dataType, triangulationType>(
        triangulation, chunk_trajectories[j], chunk_distanceFromSeed[j],
        offsets, scalars, chunk_identifier[j]);
    }

    if(ttk::MPIsize_ > 1) {
#pragma omp atomic update seq_cst
      taskCounter -= nbElement;
      this->checkEndOfComputation();
  }
  }
}

template <typename dataType, class triangulationType>
int ttk::IntegralLines::execute(triangulationType *triangulation) {

#if TTK_ENABLE_MPI
  taskCounter = seedNumber_;
  globalElementCounter = this->GlobalElementToCompute;
  printMsg("MessageSize: " + std::to_string(messageSize_));
  if(ttk::MPIsize_ > 1) {
    multipleElementToSend.resize(ttk::MPIsize_);
    messageCount.resize(ttk::MPIsize_);
    for(int i = 0; i < ttk::MPIsize_; i++) {
      multipleElementToSend[i].resize(messageSize_);
    }
  }

#endif

  const SimplexId *offsets = inputOffsets_;
  std::vector<SimplexId> *seeds = vertexIdentifierScalarField_;
  dataType *scalars = static_cast<dataType *>(inputScalarField_);
  Timer t;

  std::vector<std::vector<ttk::SimplexId> *> chunk_trajectories(chunkSize_);
  std::vector<std::vector<double> *> chunk_distanceFromSeed(chunkSize_);
  std::vector<ttk::SimplexId> chunk_identifier(chunkSize_);
  int taskNumber = (int)seedNumber_ / chunkSize_;
#if TTK_ENABLE_MPI
#pragma omp parallel shared(finishedElement, taskCounter, unfinishedDist, \
                            unfinishedTraj, unfinishedSeed)               \
  num_threads(threadNumber_)
  {
#else
#pragma omp parallel
  {
#endif
#pragma omp single nowait
    {
      for(SimplexId i = 0; i < taskNumber; ++i) {
        this->createTask<dataType, triangulationType>(
          triangulation, chunk_trajectories, chunk_distanceFromSeed, offsets,
          scalars, chunk_identifier, i, chunkSize_, seeds);
      }
      int rest = seedNumber_ % chunkSize_;
      if(rest > 0) {
        this->createTask<dataType, triangulationType>(
          triangulation, chunk_trajectories, chunk_distanceFromSeed, offsets,
          scalars, chunk_identifier, taskNumber, rest, seeds);
      }

      this->receiveMessages<dataType, triangulationType>(
        triangulation, offsets, scalars);
    }
  }

  if(ttk::MPIsize_ > 1) {
    std::vector<MPI_Status> dummyStatus(2 * TABULAR_SIZE);
    std::list<std::array<MPI_Request, TABULAR_SIZE>>::iterator requestBlock
      = sentRequests_->list.begin();

    int sizeBlock = TABULAR_SIZE;
    while(requestBlock != sentRequests_->list.end()) {
      requestBlock++;
      if(requestBlock == sentRequests_->list.end()) {
        sizeBlock = std::min((int)TABULAR_SIZE, sentRequests_->numberOfElement);
      }
      requestBlock--;
      MPI_Waitall(sizeBlock, requestBlock->data(), dummyStatus.data());
      requestBlock++;
    }
  }
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
