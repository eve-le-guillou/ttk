/// \ingroup base
/// \class ttk::ArrayPreconditioning
/// \author Michael Will <mswill@rhrk.uni-kl.de>
/// \date 2022.
///
/// This module defines the %ArrayPreconditioning class that generates order
/// arrays from a selection of scalar field arrays.
///

#pragma once

// ttk common includes
//#include "AmsSort/AmsSort.hpp"
#include <Debug.h>
#include <Triangulation.h>
#include <psort.h>
#include <random>
#include <string>
#include <vector>

namespace ttk {

  /**
   * The ArrayPreconditioning class provides methods to generate order arrays
   * from a selection of scalar field arrays.
   */
  class ArrayPreconditioning : virtual public Debug {

  public:
    ArrayPreconditioning();

    int preconditionTriangulation(AbstractTriangulation *triangulation) {
      // Pre-condition functions.
      if(triangulation) {
#ifdef TTK_ENABLE_MPI
        triangulation->preconditionExchangeGhostVertices();
#endif // TTK_ENABLE_MPI
      }
      return 0;
    }

    template <typename DT, typename triangulationType>
    int processScalarArray(const triangulationType *triangulation,
                           ttk::SimplexId *orderArray,
                           const DT *scalarArray,
                           const size_t nVerts,
                           const int burstSize,
#ifdef TTK_ENABLE_MPI
                           std::vector<int> neighbors
#else
                           const std::vector<int> &neighbors
#endif

                           = {}) const { // start global timer
      ttk::Timer globalTimer;

      // print horizontal separator
      this->printMsg(ttk::debug::Separator::L1); // L1 is the '=' separator
      // print input parameters in table format
      this->printMsg({
        {"#Threads", std::to_string(this->threadNumber_)},
        {"#Vertices", std::to_string(nVerts)},
      });
      this->printMsg(ttk::debug::Separator::L1);

// -----------------------------------------------------------------------
// Computing order Array
// -----------------------------------------------------------------------
#ifdef TTK_ENABLE_MPI
      if(ttk::isRunningWithMPI()) {
        // ttk::produceOrdering<DT>(orderArray, scalarArray, getVertexGlobalId,
        //                         getVertexRank, getVertexLocalId, nVerts,
        //                         burstSize, neighbors);
        ttk::Timer t_mpi;
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        /*struct vertexToSort {
          double value;
          ttk::SimplexId globalId;
          ttk::SimplexId order;
        };*/
        std::vector<p_sort::vertexToSort> verticesToSort;
        verticesToSort.reserve(nVerts);
#pragma omp declare reduction (merge : std::vector<p_sort::vertexToSort> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : verticesToSort)
        for(int i = 0; i < nVerts; i++) {
          if(triangulation->getVertexRank(i) == ttk::MPIrank_) {
            verticesToSort.emplace_back(p_sort::vertexToSort{
              static_cast<double>(scalarArray[i]),
              triangulation->getVertexGlobalId(i), ttk::MPIrank_});
          }
        }
        ttk::SimplexId id = 0;
        MPI_Datatype MPI_SimplexId = getMPIType(id);
        MPI_Datatype types[] = {MPI_DOUBLE, MPI_SimplexId, MPI_SimplexId};
        int lengths[] = {1, 1, 1};
        MPI_Datatype MPI_vertexToSortType;
        const long int mpi_offsets[]
          = {offsetof(p_sort::vertexToSort, value),
             offsetof(p_sort::vertexToSort, globalId),
             offsetof(p_sort::vertexToSort, order)};
        MPI_Type_create_struct(
          3, lengths, mpi_offsets, types, &MPI_vertexToSortType);
        MPI_Type_commit(&MPI_vertexToSortType);
        /*const int kway = 64;
        const int num_levels = 3;
        std::random_device rd;
        std::mt19937_64 gen(rd());
        MPI_Comm comm = ttk::MPIcomm_;*/
        std::vector<ttk::SimplexId> vertex_distribution_buf(ttk::MPIsize_);
        std::vector<long> vertex_distribution(ttk::MPIsize_);
        ttk::SimplexId localVertexNumber = verticesToSort.size();
        MPI_Allgather(&localVertexNumber, 1, MPI_SimplexId,
                      vertex_distribution_buf.data(), 1, MPI_SimplexId,
                      ttk::MPIcomm_);
        for(int i = 0; i < ttk::MPIsize_; i++) {
          vertex_distribution[i] = vertex_distribution_buf[i];
        }
        double elapsedTime
          = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Preparation for sorting on " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        p_sort::parallel_sort<p_sort::vertexToSort, ttk::SimplexId>(
          verticesToSort, vertex_distribution.data(), MPI_COMM_WORLD);
        elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Sorting on " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        std::vector<std::vector<p_sort::vertexToSort>> verticesSorted(
          ttk::MPIsize_, std::vector<p_sort::vertexToSort>());
        std::list<std::vector<std::vector<p_sort::vertexToSort>>>
          verticesSortedThread(
            this->threadNumber_,
            std::vector<std::vector<p_sort::vertexToSort>>(
              ttk::MPIsize_, std::vector<p_sort::vertexToSort>()));
        // Compute orderOffset with MPI prefix sum
        ttk::SimplexId verticesToSortSize = verticesToSort.size();
        ttk::SimplexId orderOffset
          = std::accumulate(vertex_distribution_buf.begin(),
                            vertex_distribution_buf.begin() + ttk::MPIrank_, 0);
#pragma omp parallel firstprivate(verticesToSortSize) \
  num_threads(threadNumber_) shared(verticesSortedThread)
        {
          int rank;
          int threadNumber = omp_get_thread_num();
          std::list<std::vector<std::vector<p_sort::vertexToSort>>>::iterator it
            = verticesSortedThread.begin();
          for(int i = 0; i < threadNumber; i++)
            it++;
#pragma omp for
          for(int i = 0; i < verticesToSortSize; i++) {
            rank = verticesToSort.at(i).order;
            if(rank == ttk::MPIrank_) {
              orderArray[triangulation->getVertexLocalId(
                verticesToSort.at(i).globalId)]
                = orderOffset + i;
            } else {
              verticesToSort.at(i).order = orderOffset + i;
              it->at(rank).push_back(verticesToSort.at(i));
            }
          }
        }
        std::list<std::vector<std::vector<p_sort::vertexToSort>>>::iterator it
          = verticesSortedThread.begin();
        for(int i = 0; i < this->threadNumber_; i++) {
          for(int j = 0; j < ttk::MPIsize_; j++) {
            if(j != ttk::MPIrank_) {
              verticesSorted.at(j).insert(
                verticesSorted.at(j).end(), it->at(j).begin(), it->at(j).end());
            }
          }
          it++;
        }
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
          // Send size of verticesToSort.at(i)
          if(i != ttk::MPIrank_) {
            sendMessageSize[i] = verticesSorted.at(i).size();
            MPI_Isend(&sendMessageSize[i], 1, MPI_SimplexId, i, 0,
                      ttk::MPIcomm_, &sendRequests[count]);
            MPI_Irecv(&recvMessageSize[i], 1, MPI_SimplexId, i, 0,
                      ttk::MPIcomm_, &recvRequests[count]);
            count++;
          }
        }
        std::vector<std::vector<p_sort::vertexToSort>> recvVerticesSorted(
          ttk::MPIsize_, std::vector<p_sort::vertexToSort>());
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
                  MPI_Isend(verticesSorted.at(r).data(), sendMessageSize[r],
                            MPI_vertexToSortType, r, 1, ttk::MPIcomm_,
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
                  recvVerticesSorted.at(r).resize(recvMessageSize[r]);
                  MPI_Irecv(recvVerticesSorted.at(r).data(), recvMessageSize[r],
                            MPI_vertexToSortType, r, 1, ttk::MPIcomm_,
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
#pragma omp parallel for
              for(int j = 0; j < recvMessageSize[r]; j++) {
                orderArray[triangulation->getVertexLocalId(
                  recvVerticesSorted.at(r).at(j).globalId)]
                  = recvVerticesSorted.at(r).at(j).order;
              }
            }
            recvPerformedCountTotal += recvPerformedCount;
          }
        }
        MPI_Waitall(sendCount, sendRequestsData.data(), MPI_STATUSES_IGNORE);

        elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Post-processing for sorting on "
                   + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        ttk::exchangeGhostVertices<ttk::SimplexId, triangulationType>(
          orderArray, triangulation, ttk::MPIcomm_, 1);
        elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Ghost exchange " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
      }
#else
      this->printMsg("MPI not enabled!");
      TTK_FORCE_USE(orderArray);
      TTK_FORCE_USE(scalarArray);
      TTK_FORCE_USE(getVertexGlobalId);
      TTK_FORCE_USE(getVertexRank);
      TTK_FORCE_USE(getVertexLocalId);
      TTK_FORCE_USE(burstSize);
      TTK_FORCE_USE(neighbors);
      return 0;
#endif

      // ---------------------------------------------------------------------
      // print global performance
      // ---------------------------------------------------------------------
      {
        this->printMsg(ttk::debug::Separator::L2); // horizontal '-' separator
        this->printMsg(
          "Complete", 1, globalTimer.getElapsedTime() // global progress, time
        );
        this->printMsg(ttk::debug::Separator::L1); // horizontal '=' separator
      }

      return 1; // return success
    }

  protected:
    bool GlobalOrder{false};
  }; // ArrayPreconditioning class

} // namespace ttk
