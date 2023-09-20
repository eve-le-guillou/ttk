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
#include "AmsSort/AmsSort.hpp"
#include <Debug.h>
#include <random>
#include <vector>
namespace ttk {

  /**
   * The ArrayPreconditioning class provides methods to generate order arrays
   * from a selection of scalar field arrays.
   */
  class ArrayPreconditioning : virtual public Debug {

  public:
    ArrayPreconditioning();

    template <typename DT, typename GVGID, typename GVR, typename GVLID>
    int processScalarArray(ttk::SimplexId *orderArray,
                           const DT *scalarArray,
                           const GVGID &getVertexGlobalId,
                           const GVR &getVertexRank,
                           const GVLID &getVertexLocalId,
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
        struct vertexToSort {
          double value;
          ttk::SimplexId globalId;
          ttk::SimplexId order;
        };
        std::vector<vertexToSort> verticesToSort;
        verticesToSort.reserve(nVerts);
        for(int i = 0; i < nVerts; i++) {
          if(getVertexRank(i) == ttk::MPIrank_) {
            verticesToSort.emplace_back(
              vertexToSort{static_cast<double>(scalarArray[i]),
                           getVertexGlobalId(i), ttk::MPIrank_});
          }
        }
        ttk::SimplexId id = 0;
        MPI_Datatype MPI_SimplexId = getMPIType(id);
        MPI_Datatype types[] = {MPI_DOUBLE, MPI_SimplexId, MPI_SimplexId};
        int lengths[] = {1, 1, 1};
        MPI_Datatype MPI_vertexToSortType;
        const long int mpi_offsets[]
          = {offsetof(vertexToSort, value), offsetof(vertexToSort, globalId),
             offsetof(vertexToSort, order)};
        MPI_Type_create_struct(
          3, lengths, mpi_offsets, types, &MPI_vertexToSortType);
        MPI_Type_commit(&MPI_vertexToSortType);
        const int kway = 64;
        const int num_levels = 3;
        std::random_device rd;
        std::mt19937_64 gen(rd());
        MPI_Comm comm = ttk::MPIcomm_;
        double elapsedTime
          = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Preparation for sorting on " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);

        Ams::sort<vertexToSort>(
          MPI_vertexToSortType, verticesToSort, kway, gen, comm,
          [&](const vertexToSort a, const vertexToSort b) {
            return (b.value > a.value)
                   || (a.value == b.value && a.globalId < b.globalId);
          });
        elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Sorting on " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        std::vector<std::vector<vertexToSort>> verticesSorted(
          ttk::MPIsize_, std::vector<vertexToSort>());
        // Compute orderOffset with MPI prefix sum
        ttk::SimplexId verticesToSortSize = verticesToSort.size();
        ttk::SimplexId orderOffset;
        MPI_Exscan(&verticesToSortSize, &orderOffset, 1, MPI_SimplexId, MPI_SUM,
                   ttk::MPIcomm_);
        if(ttk::MPIrank_ == 0) {
          orderOffset = 0;
        }
        int rank = 0;
        for(int i = 0; i < verticesToSort.size(); i++) {
          rank = verticesToSort.at(i).order;
          if(rank == ttk::MPIrank_) {
            orderArray[getVertexLocalId(verticesToSort.at(i).globalId)]
              = orderOffset + i;
          } else {
            verticesToSort.at(i).order = orderOffset + i;
            verticesSorted.at(rank).emplace_back(verticesToSort.at(i));
          }
        }
        std::vector<MPI_Request> sendRequests(ttk::MPIsize_ - 1);
        std::vector<MPI_Request> recvRequests(ttk::MPIsize_ - 1);
        std::vector<ttk::SimplexId> sendMessageSize(ttk::MPIsize_, 0);
        std::vector<ttk::SimplexId> recvMessageSize(ttk::MPIsize_, 0);
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
        MPI_Waitall(
          ttk::MPIsize_ - 1, sendRequests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(
          ttk::MPIsize_ - 1, recvRequests.data(), MPI_STATUSES_IGNORE);
        std::vector<std::vector<vertexToSort>> recvVerticesSorted(
          ttk::MPIsize_, std::vector<vertexToSort>());
        int recvCount = 0;
        int sendCount = 0;
        for(int i = 0; i < ttk::MPIsize_; i++) {
          if(sendMessageSize[i] > 0) {
            MPI_Isend(verticesSorted.at(i).data(), sendMessageSize[i],
                      MPI_vertexToSortType, i, 0, ttk::MPIcomm_,
                      &sendRequests[sendCount]);
            sendCount++;
          }
          if(recvMessageSize[i] > 0) {
            recvVerticesSorted.at(i).resize(recvMessageSize[i]);
            MPI_Irecv(recvVerticesSorted.at(i).data(), recvMessageSize[i],
                      MPI_vertexToSortType, i, 0, ttk::MPIcomm_,
                      &recvRequests[recvCount]);
            recvCount++;
          }
        }

        MPI_Waitall(sendCount, sendRequests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(recvCount, recvRequests.data(), MPI_STATUSES_IGNORE);

#pragma omp parallel for
        for(int i = 0; i < ttk::MPIsize_; i++) {
          for(int j = 0; j < recvMessageSize[i]; j++) {
            orderArray[getVertexLocalId(
              recvVerticesSorted.at(i).at(j).globalId)]
              = recvVerticesSorted.at(i).at(j).order;
          }
        }
        elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Post-processing for orting on "
                   + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        ttk::exchangeGhostDataWithoutTriangulation<ttk::SimplexId,
                                                   ttk::SimplexId>(
          orderArray, getVertexRank, getVertexGlobalId, getVertexLocalId,
          nVerts, ttk::MPIcomm_, neighbors);
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
