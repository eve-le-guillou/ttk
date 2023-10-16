/// \ingroup base
/// \class ttk::ArrayPreconditioning
/// \author Michael Will <mswill@rhrk.uni-kl.de>
/// \author Eve Le Guillou <eve.le-guillou@lip6.fr>
/// \date 2022.
///
/// This module defines the %ArrayPreconditioning class that generates order
/// arrays from a selection of scalar field arrays.
///

#pragma once

// ttk common includes
#include "DataTypes.h"
#include <Debug.h>
#include <Triangulation.h>
#include <psort.h>
#include <random>
#include <string>
#include <vector>

namespace ttk {

  namespace globalOrder {

    template <typename datatype>
    struct vertexToSort {
      ttk::SimplexId globalId;
      datatype value;
      char rank;
    };

    struct sortedVertex {
      ttk::SimplexId globalId;
      ttk::SimplexId order;
    };

    template <typename datatype>
    bool comp(const vertexToSort<datatype> a, const vertexToSort<datatype> b) {
      return (b.value > a.value)
             || (a.value == b.value && a.globalId < b.globalId);
    };

    template <typename datatype>
    bool oppositeComp(const vertexToSort<datatype> a,
                      const vertexToSort<datatype> b) {
      return (a.value > b.value)
             || (a.value == b.value && a.globalId > b.globalId);
    }
  } // namespace globalOrder

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
                           const size_t nVerts) const { // start global timer
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
        ttk::Timer t_mpi;
        ttk::SimplexId id = 0;
        MPI_Datatype MPI_SimplexId = getMPIType(id);
        /*ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);

        DT v0{0};
        printMsg("is same double: "+std::to_string(std::is_same<double,
        DT>::value)); printMsg("is same float:
        "+std::to_string(std::is_same<float, DT>::value)); MPI_Datatype MPI_DT0
        = getMPIType(v0); MPI_Datatype MPI_testType; MPI_Datatype types0[] =
        {MPI_SimplexId, MPI_DT0, MPI_CHAR}; int lengths0[] = {1,1,1}; const long
        int mpi_offsets0[] = {offsetof(globalOrder::vertexToSortTest<DT>,
        globalId), offsetof(globalOrder::vertexToSortTest<DT>, value),
          offsetof(globalOrder::vertexToSortTest<DT>, order)};
        MPI_Type_create_struct(
          3, lengths0, mpi_offsets0, types0, &MPI_testType);
        MPI_Type_commit(&MPI_testType);
        MPI_Aint lb0, extent0;
        MPI_Type_get_extent(MPI_testType, &lb0, &extent0);
        printMsg("Size of testType: "+std::to_string(extent0));
        int count0 = 10000000;*/
        // std::vector<globalOrder::testStruct<DT>>fake_data(count0);
        // for (int i = 0; i < count0; i++){
        //  fake_data[i] = globalOrder::testStruct<DT>{1,static_cast<DT>(0.1)};
        //}
        // testStruct t = testStruct{std::vector<DT>(count0, 0.1)};
        // std::vector<DT> fake_data(count0, 0.1);
        /*std::vector<DT>fake_data(count0);
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if (ttk::MPIrank_%2 == 0){
          MPI_Send(fake_data.data(), count0, MPI_FLOAT, ttk::MPIrank_+1, 0,
        ttk::MPIcomm_); } else { MPI_Recv(fake_data.data(), count0, MPI_FLOAT,
        ttk::MPIrank_-1, 0, ttk::MPIcomm_, MPI_STATUS_IGNORE);
        }
        double elapsedTimeTest
          = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Templated send on " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" +
        std::to_string(elapsedTimeTest));
        }*/
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        std::vector<globalOrder::vertexToSort<DT>> verticesToSort;
        verticesToSort.reserve(nVerts);
#ifdef TTK_ENABLE_OPENMP
#pragma omp declare reduction (merge : std::vector<globalOrder::vertexToSort<DT>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : verticesToSort)
#endif
        for(ttk::SimplexId i = 0; i < nVerts; i++) {
          if(triangulation->getVertexRank(i) == ttk::MPIrank_) {
            verticesToSort.emplace_back(
              globalOrder::vertexToSort<DT>{triangulation->getVertexGlobalId(i),
                                            scalarArray[i], ttk::MPIrank_});
          }
        }
        // ttk::SimplexId id = 0;
        DT v{0};
        // MPI_Datatype MPI_SimplexId = getMPIType(id);
        MPI_Datatype MPI_DT = getMPIType(v);
        MPI_Datatype types[] = {MPI_SimplexId, MPI_DT, MPI_CHAR};
        /*std::cout << "Value of MPI_DOUBLE: "<< MPI_DOUBLE<<std::endl;
        std::cout << "Value of MPI_FLOAT: "<< MPI_FLOAT<<std::endl;
        std::cout << "Value of MPI_INT: "<< MPI_INT<<std::endl;
        std::cout << "Value of MPI_DT: "<<MPI_DT << std::endl;
        printMsg("Size of MPI_DOUBLE: "+std::to_string(sizeof(MPI_DOUBLE)));
        MPI_Aint lb, extent;
        MPI_Type_get_extent(MPI_DT, &lb, &extent);
        printMsg("Size of getMPIType: "+std::to_string(extent));
        printMsg("Offset of value:
        "+std::to_string(offsetof(globalOrder::vertexToSort<DT>, value)));
        printMsg("Offset of globalId:
        "+std::to_string(offsetof(globalOrder::vertexToSort<DT>, globalId)));*/
        int lengths[] = {1, 1, 1};
        MPI_Datatype MPI_vertexToSortType;
        const long int mpi_offsets[]
          = {offsetof(globalOrder::vertexToSort<DT>, globalId),
             offsetof(globalOrder::vertexToSort<DT>, value),
             offsetof(globalOrder::vertexToSort<DT>, rank)};
        MPI_Type_create_struct(
          3, lengths, mpi_offsets, types, &MPI_vertexToSortType);
        // MPI_Type_get_extent(MPI_vertexToSortType, &lb, &extent);
        // printMsg("lb: "+std::to_string(lb)+", extent:
        // "+std::to_string(extent));
        MPI_Type_commit(&MPI_vertexToSortType);
        // printMsg("MPI_Datatype:
        // "+std::to_string(sizeof(MPI_vertexToSortType)));

        std::vector<ttk::SimplexId> vertex_distribution(ttk::MPIsize_);
        ttk::SimplexId localVertexNumber = verticesToSort.size();
        MPI_Allgather(&localVertexNumber, 1, MPI_SimplexId,
                      vertex_distribution.data(), 1, MPI_SimplexId,
                      ttk::MPIcomm_);
        double elapsedTime
          = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Preparation for sorting on " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);

        p_sort::parallel_sort<globalOrder::vertexToSort<DT>>(
          verticesToSort, globalOrder::comp<DT>, globalOrder::oppositeComp<DT>,
          vertex_distribution, MPI_vertexToSortType, MPI_SimplexId,
          threadNumber_);
        elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        if(ttk::MPIrank_ == 0) {
          printMsg("Sorting on " + std::to_string(ttk::MPIsize_)
                   + " MPI processes lasted :" + std::to_string(elapsedTime));
        }
        ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
        MPI_Datatype types_2[] = {MPI_SimplexId, MPI_SimplexId};
        int lengths_2[] = {1, 1};
        MPI_Datatype MPI_sortedVertexType;
        const long int mpi_offsets_2[]
          = {offsetof(globalOrder::sortedVertex, globalId),
             offsetof(globalOrder::sortedVertex, order)};
        MPI_Type_create_struct(
          2, lengths_2, mpi_offsets_2, types_2, &MPI_sortedVertexType);
        MPI_Type_commit(&MPI_sortedVertexType);
        std::vector<std::vector<globalOrder::sortedVertex>> verticesSorted(
          ttk::MPIsize_, std::vector<globalOrder::sortedVertex>());
        std::list<std::vector<std::vector<globalOrder::sortedVertex>>>
          verticesSortedThread(
            this->threadNumber_,
            std::vector<std::vector<globalOrder::sortedVertex>>(
              ttk::MPIsize_, std::vector<globalOrder::sortedVertex>()));
        // Compute orderOffset with MPI prefix sum
        ttk::SimplexId verticesToSortSize = verticesToSort.size();
        ttk::SimplexId orderOffset
          = std::accumulate(vertex_distribution.begin(),
                            vertex_distribution.begin() + ttk::MPIrank_, 0);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel firstprivate(verticesToSortSize) \
  num_threads(threadNumber_) shared(verticesSortedThread)
        {
#endif
          int rank;
          int threadNumber = omp_get_thread_num();
          typename std::list<
            std::vector<std::vector<globalOrder::sortedVertex>>>::iterator it
            = verticesSortedThread.begin();
          for(int i = 0; i < threadNumber; i++)
            it++;
#ifdef TTK_ENABLE_OPENMP
#pragma omp for
#endif
          for(ttk::SimplexId i = 0; i < verticesToSortSize; i++) {
            rank = verticesToSort.at(i).rank;
            if(rank == ttk::MPIrank_) {
              orderArray[triangulation->getVertexLocalId(
                verticesToSort.at(i).globalId)]
                = orderOffset + i;
            } else {
              it->at(rank).push_back(globalOrder::sortedVertex{
                verticesToSort.at(i).globalId, orderOffset + i});
            }
          }
#ifdef TTK_ENABLE_OPENMP
        }
#endif
        typename std::list<
          std::vector<std::vector<globalOrder::sortedVertex>>>::iterator it
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
        std::vector<std::vector<globalOrder::sortedVertex>> recvVerticesSorted(
          ttk::MPIsize_, std::vector<globalOrder::sortedVertex>());
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
                            MPI_sortedVertexType, r, 1, ttk::MPIcomm_,
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
                            MPI_sortedVertexType, r, 1, ttk::MPIcomm_,
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
      TTK_FORCE_USE(triangulation);
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
