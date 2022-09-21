#include <ttkIntegralLines.h>
#include <ttkMacros.h>
#include <ttkUtils.h>

#include <ArrayLinkedList.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkDoubleArray.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPointSet.h>
#include <vtkUnstructuredGrid.h>

#include <array>

vtkStandardNewMacro(ttkIntegralLines);

ttkIntegralLines::ttkIntegralLines() {
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfOutputPorts(1);
}

ttkIntegralLines::~ttkIntegralLines() = default;

int ttkIntegralLines::FillInputPortInformation(int port, vtkInformation *info) {
  if(port == 0)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkDataSet");
  if(port == 1)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPointSet");

  return 1;
}

int ttkIntegralLines::FillOutputPortInformation(int port,
                                                vtkInformation *info) {
  if(port == 0)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkUnstructuredGrid");

  return 1;
}

template <typename triangulationType>
int ttkIntegralLines::getTrajectories(
  vtkDataSet *input,
  triangulationType *triangulation,
  ttk::ArrayLinkedList<std::vector<ttk::SimplexId>, TABULAR_SIZE> &trajectories,
  ttk::ArrayLinkedList<std::vector<double>, TABULAR_SIZE> &distancesFromSeed,
  ttk::ArrayLinkedList<ttk::SimplexId, TABULAR_SIZE> &seedIdentifiers,
  ttk::ArrayLinkedList<std::vector<ttk::SimplexId>, TABULAR_SIZE>
    &edgeIdentifiers,
  vtkUnstructuredGrid *output) {
  if(input == nullptr || output == nullptr
     || input->GetPointData() == nullptr) {
    this->printErr("Null pointers in getTrajectories parameters");
    return 0;
  }

  vtkNew<vtkUnstructuredGrid> ug{};
  vtkNew<vtkPoints> pts{};
  vtkNew<vtkDoubleArray> dist{};
  vtkNew<vtkIdTypeArray> identifier{};
  vtkNew<vtkIdTypeArray> vtkEdgeIdentifiers{};
  vtkNew<vtkIntArray> vtkRankArray{};

  dist->SetNumberOfComponents(1);
  dist->SetName("DistanceFromSeed");
  identifier->SetNumberOfComponents(1);
  identifier->SetName("SeedIdentifier");

  vtkEdgeIdentifiers->SetNumberOfComponents(1);
  vtkEdgeIdentifiers->SetName("GlobalCellIds");
  vtkRankArray->SetNumberOfComponents(1);
  vtkRankArray->SetName("RankArray");

  const auto numberOfArrays = input->GetPointData()->GetNumberOfArrays();

  std::vector<vtkDataArray *> scalarArrays{};
  scalarArrays.reserve(numberOfArrays);
  for(int k = 0; k < numberOfArrays; ++k) {
    const auto a = input->GetPointData()->GetArray(k);
    if(a->GetNumberOfComponents() == 1) {
      // only keep scalar arrays
      scalarArrays.push_back(a);
    }
  }

  std::vector<vtkSmartPointer<vtkDataArray>> inputScalars(scalarArrays.size());
  for(size_t k = 0; k < scalarArrays.size(); ++k) {
    inputScalars[k]
      = vtkSmartPointer<vtkDataArray>::Take(scalarArrays[k]->NewInstance());
    inputScalars[k]->SetNumberOfComponents(1);
    inputScalars[k]->SetName(scalarArrays[k]->GetName());
  }
  const int *cellRankArray = triangulation->getCellRankArray();
  const int *vertRankArray = triangulation->getVertRankArray();
  std::array<float, 3> p;
  std::array<vtkIdType, 2> ids;
  std::list<std::array<std::vector<ttk::SimplexId>, TABULAR_SIZE>>::iterator trajectory
    = trajectories.list.begin();
  std::list<std::array<std::vector<double>, TABULAR_SIZE>>::iterator distanceFromSeed
    = distancesFromSeed.list.begin();
  std::list<std::array<ttk::SimplexId, TABULAR_SIZE>>::iterator seedIdentifier
    = seedIdentifiers.list.begin();
  std::list<std::array<std::vector<ttk::SimplexId>, TABULAR_SIZE>>::iterator
    edgeIdentifier
    = edgeIdentifiers.list.begin();
  while(trajectory != trajectories.list.end()) {
    for(int i = 0; i < TABULAR_SIZE; i++) {
      if((*trajectory)[i].size() > 0) {
        ttk::SimplexId vertex = (*trajectory)[i].at(0);
        // init
        triangulation->getVertexPoint(vertex, p[0], p[1], p[2]);
        ids[0] = pts->InsertNextPoint(p.data());
        // distanceScalars
        dist->InsertNextTuple1((*distanceFromSeed)[i].at(0));
        identifier->InsertNextTuple1((*seedIdentifier)[i]);
        // inputScalars
        for(size_t k = 0; k < scalarArrays.size(); ++k) {
          inputScalars[k]->InsertNextTuple1(scalarArrays[k]->GetTuple1(vertex));
        }
        for(size_t j = 1; j < (*trajectory)[i].size(); ++j) {
          vertex = (*trajectory)[i].at(j);
          triangulation->getVertexPoint(vertex, p[0], p[1], p[2]);
          ids[1] = pts->InsertNextPoint(p.data());
          // distanceScalars
          dist->InsertNextTuple1((*distanceFromSeed)[i].at(j));
          identifier->InsertNextTuple1((*seedIdentifier)[i]);

          // inputScalars
          for(unsigned int k = 0; k < scalarArrays.size(); ++k)
            inputScalars[k]->InsertNextTuple1(
              scalarArrays[k]->GetTuple1(vertex));

          ug->InsertNextCell(VTK_LINE, 2, ids.data());
          vtkEdgeIdentifiers->InsertNextTuple1((*edgeIdentifier)[i].at(j));
          if(vertRankArray[(*trajectory)[i].at(j - 1)]
             == vertRankArray[vertex]) {
            vtkRankArray->InsertNextTuple1(vertRankArray[vertex]);
          } else {
            ttk::SimplexId localEdgeId
              = triangulation->getEdgeLocalId((*edgeIdentifier)[i].at(j));
            int starNumber = triangulation->getEdgeStarNumber(localEdgeId);
            ttk::SimplexId starId;
            std::vector<int> occurenceOfRank;
            std::vector<int> rank;
            triangulation->getEdgeStar(localEdgeId, 0, starId);
            rank.push_back(cellRankArray[starId / 6]);
            occurenceOfRank.push_back(1);
            std::vector<int>::iterator it;
            for(int k = 1; k < starNumber; k++) {
              triangulation->getEdgeStar(localEdgeId, k, starId);
              it = find(rank.begin(), rank.end(), cellRankArray[starId / 6]);
              if(it != rank.end()) {
                occurenceOfRank[it - rank.begin()]++;
              } else {
                occurenceOfRank.push_back(1);
                rank.push_back(cellRankArray[starId / 6]);
              }
            }
            it = std::max_element(
              occurenceOfRank.begin(), occurenceOfRank.end());
            vtkRankArray->InsertNextTuple1(
              rank[std::distance(occurenceOfRank.begin(), it)]);
          }
          // iteration
          ids[0] = ids[1];
        }
      } else {
        break;
      }
    }
    trajectory++;
    distanceFromSeed++;
    seedIdentifier++;
    edgeIdentifier++;
  }

  ug->SetPoints(pts);
  ug->GetPointData()->AddArray(dist);
  ug->GetPointData()->AddArray(identifier);
  ug->GetCellData()->AddArray(vtkRankArray);
  ug->GetCellData()->AddArray(vtkEdgeIdentifiers);
  for(unsigned int k = 0; k < scalarArrays.size(); ++k) {
    ug->GetPointData()->AddArray(inputScalars[k]);
  }

  output->ShallowCopy(ug);

  return 1;
}

int ttkIntegralLines::RequestData(vtkInformation *ttkNotUsed(request),
                                  vtkInformationVector **inputVector,
                                  vtkInformationVector *outputVector) {

  vtkDataSet *domain = vtkDataSet::GetData(inputVector[0], 0);
  vtkPointSet *seeds = vtkPointSet::GetData(inputVector[1], 0);
  vtkUnstructuredGrid *output = vtkUnstructuredGrid::GetData(outputVector, 0);

  ttk::Triangulation *triangulation = ttkAlgorithm::GetTriangulation(domain);
  vtkDataArray *inputScalars = this->GetInputArrayToProcess(0, domain);

  vtkDataArray *inputOffsets
    = this->GetOrderArray(domain, 0, 1, ForceInputOffsetScalarField);

  const ttk::SimplexId numberOfPointsInDomain = domain->GetNumberOfPoints();
  this->setVertexNumber(numberOfPointsInDomain);
  int numberOfPointsInSeeds = seeds->GetNumberOfPoints();

#ifdef TTK_ENABLE_MPI_TIME
  ttk::Timer t_mpi;
  ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
#endif
#if TTK_ENABLE_MPI
  std::vector<ttk::SimplexId> inputIdentifiers{};
  vertRankArray_ = triangulation->getVertRankArray();
  int totalSeeds;
  if(ttk::MPIsize_ > 1) {
    MPI_Reduce(&numberOfPointsInSeeds, &totalSeeds, 1, MPI_INTEGER, MPI_SUM, 0,
               ttk::MPIcomm_);
    int isDistributed;

    if(ttk::MPIrank_ == 0) {
      isDistributed = numberOfPointsInSeeds != totalSeeds;
    }
    MPI_Bcast(&isDistributed, 1, MPI_INTEGER, 0, ttk::MPIcomm_);
    MPI_Bcast(&totalSeeds, 1, MPI_INTEGER, 0, ttk::MPIcomm_);

    if(!isDistributed) {
      this->setGlobalElementCounter(totalSeeds);
      vtkDataArray *globalSeedsId;
      if(ttk::MPIrank_ == 0) {
        globalSeedsId = seeds->GetPointData()->GetArray("GlobalPointIds");
      } else {
        globalSeedsId = vtkDataArray::CreateDataArray(VTK_ID_TYPE);
      }

      if(ttk::MPIrank_ != 0) {
        globalSeedsId->SetNumberOfComponents(1);
        globalSeedsId->SetNumberOfTuples(totalSeeds);
      }
      ttk::LongSimplexId id = 0;
      MPI_Bcast(ttkUtils::GetPointer<ttk::LongSimplexId>(globalSeedsId),
                totalSeeds, ttk::getMPIType(id), 0, ttk::MPIcomm_);
      ttk::SimplexId localId = -1;
      for(int i = 0; i < totalSeeds; i++) {
        localId = triangulation->getVertexLocalId(globalSeedsId->GetTuple1(i));
        if(localId != -1 && vertRankArray_[localId] == ttk::MPIrank_) {
          inputIdentifiers.push_back(localId);
        }
      }
      numberOfPointsInSeeds = inputIdentifiers.size();
    } else {
      std::vector<ttk::SimplexId> idSpareStorage{};
      ttk::SimplexId *inputIdentifierGlobalId;
      inputIdentifierGlobalId = this->GetIdentifierArrayPtr(
        ForceInputVertexScalarField, 2, ttk::VertexScalarFieldName, seeds,
        idSpareStorage);
      ttk::SimplexId localId = 0;
      for(int i = 0; i < numberOfPointsInSeeds; i++) {
        localId = triangulation->getVertexLocalId(inputIdentifierGlobalId[i]);
        if(vertRankArray_[localId] == ttk::MPIrank_) {
          inputIdentifiers.push_back(localId);
        }
      }
      numberOfPointsInSeeds = inputIdentifiers.size();
      MPI_Allreduce(&numberOfPointsInSeeds, &totalSeeds, 1, MPI_INTEGER,
                    MPI_SUM, ttk::MPIcomm_);
      this->setGlobalElementCounter(totalSeeds);
    }
  } else {
    this->setGlobalElementCounter(numberOfPointsInSeeds);
    inputIdentifiers.resize(numberOfPointsInSeeds);
    totalSeeds = numberOfPointsInSeeds;
    std::vector<ttk::SimplexId> idSpareStorage{};
    ttk::SimplexId *inputIdentifierGlobalId;
    inputIdentifierGlobalId = this->GetIdentifierArrayPtr(
      ForceInputVertexScalarField, 2, ttk::VertexScalarFieldName, seeds,
      idSpareStorage);
    for(int i = 0; i < numberOfPointsInSeeds; i++) {
      inputIdentifiers.at(i)
        = triangulation->getVertexLocalId(inputIdentifierGlobalId[i]);
    }
  }
#else
  std::vector<ttk::SimplexId> idSpareStorage{};
  ttk::SimplexId *identifiers = this->GetIdentifierArrayPtr(
    ForceInputVertexScalarField, 2, ttk::VertexScalarFieldName, seeds,
    idSpareStorage);
  std::unordered_set<ttk::SimplexId> isSeed;
  for(ttk::SimplexId k = 0; k < numberOfPointsInSeeds; ++k) {
    isSeed.insert(identifiers[k]);
  }
  std::vector<ttk::SimplexId> inputIdentifiers(isSeed.begin(), isSeed.end());
  int totalSeeds = inputIdentifiers.size();
  isSeed.clear();
#endif

  ttk::ArrayLinkedList<std::vector<ttk::SimplexId>, TABULAR_SIZE> trajectories;
  ttk::ArrayLinkedList<std::vector<double>, TABULAR_SIZE> distancesFromSeed;
  ttk::ArrayLinkedList<ttk::SimplexId, TABULAR_SIZE> seedIdentifiers;
  ttk::ArrayLinkedList<std::vector<ttk::SimplexId>, TABULAR_SIZE>
    edgeIdentifiers;

  this->setVertexNumber(numberOfPointsInDomain);
  this->setSeedNumber(numberOfPointsInSeeds);
  this->setDirection(Direction);
  this->setInputScalarField(inputScalars->GetVoidPointer(0));
  this->setInputOffsets(ttkUtils::GetPointer<ttk::SimplexId>(inputOffsets));
  this->setVertexIdentifierScalarField(&inputIdentifiers);
  this->setOutputTrajectories(&trajectories);
  this->setOutputDistancesFromSeed(&distancesFromSeed);
  this->setOutputSeedIdentifiers(&seedIdentifiers);
  this->setOutputEdgeIdentifiers(&edgeIdentifiers);
  this->preconditionTriangulation(triangulation);
  this->setChunkSize(
    std::max(std::min(1000, (int)numberOfPointsInSeeds),
             (int)numberOfPointsInSeeds / (threadNumber_ * 100)));
#ifdef TTK_ENABLE_MPI
  std::vector<std::vector<std::vector<ttk::ElementToBeSent>>> toSend(
    ttk::MPIsize_);
  this->initializeNeighbors();
  if(ttk::MPIsize_ > 1) {
    toSend.resize(this->neighborNumber_);
    for(int i = 0; i < this->neighborNumber_; i++) {
      toSend[i].resize(this->threadNumber_);
      for(int j = 0; j < this->threadNumber_; j++) {
        toSend[i][j].reserve((int)numberOfPointsInSeeds * 0.005
                             / this->threadNumber_);
      }
    }
  }
  this->setToSend(&toSend);
  this->createMessageType();
#endif
#ifdef TTK_ENABLE_MPI_TIME
  double elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
  if(ttk::MPIrank_ == 0) {
    printMsg("Preparation performed using " + std::to_string(ttk::MPIsize_)
             + " MPI processes lasted :" + std::to_string(elapsedTime));
  }
#endif
#ifndef TTK_ENABLE_KAMIKAZE
  // triangulation problem
  if(!triangulation) {
    this->printErr("wrong triangulation.");
    return -1;
  }
  // field problem
  if(!inputScalars) {
    this->printErr("wrong scalar field.");
    return -1;
  }
  // field problem
  if(inputOffsets->GetDataType() != VTK_INT
     and inputOffsets->GetDataType() != VTK_ID_TYPE) {
    this->printErr("input offset field type not supported.");
    return -1;
  }
  // field problem
  if(!inputIdentifiers.size()) {
    this->printErr("wrong identifiers.");
    return -1;
  }
  // no points.
  if(numberOfPointsInDomain <= 0) {
    this->printErr("domain has no points.");
    return -1;
  }
  // no points.
  if(totalSeeds <= 0) {
    this->printErr("seeds have no points.");
    return -1;
  }
#endif
#ifdef TTK_ENABLE_MPI
  ttk::startMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
#endif
  int status = 0;
  ttkVtkTemplateMacro(inputScalars->GetDataType(), triangulation->getType(),
                      (status = this->execute<VTK_TT, TTK_TT>(
                         static_cast<TTK_TT *>(triangulation->getData()))));

#ifdef TTK_ENABLE_MPI_TIME
  elapsedTime = ttk::endMPITimer(t_mpi, ttk::MPIrank_, ttk::MPIsize_);
  if(ttk::MPIrank_ == 0) {
    printMsg("Computation performed using " + std::to_string(ttk::MPIsize_)
             + " MPI processes lasted :" + std::to_string(elapsedTime));
  }
#endif
#ifndef TTK_ENABLE_KAMIKAZE
  // something wrong in baseCode
  if(status != 0) {
    this->printErr("IntegralLines.execute() error code : "
                   + std::to_string(status));
    return 0;
  }
#endif

  // make the vtk trajectories
  ttkTemplateMacro(
    triangulation->getType(),
    (getTrajectories<TTK_TT>(
      domain, static_cast<TTK_TT *>(triangulation->getData()), trajectories,
      distancesFromSeed, seedIdentifiers, edgeIdentifiers, output)));

  // Write data to csv

  std::ofstream myfile;
  myfile.open("/home/eveleguillou/experiment/IntegralLines/Benchmark/"
              + std::to_string(ttk::MPIsize_) + "_proc_integraLines_"
              + std::to_string(ttk::MPIrank_) + ".csv");
  myfile << "DistanceFromSeed,SeedIdentifier,GlobalPointIds,vtkGhostType\n";
  vtkDataArray *ghostArray = output->GetPointData()->GetArray("vtkGhostType");
  vtkDataArray *seedIdentifier
    = output->GetPointData()->GetArray("SeedIdentifier");
  vtkDataArray *globalIdsForCsv
    = output->GetPointData()->GetArray("GlobalPointIds");
  vtkDataArray *distance = output->GetPointData()->GetArray("DistanceFromSeed");
  for(int i = 0; i < ghostArray->GetNumberOfTuples(); i++) {
    myfile << std::to_string(distance->GetTuple1(i)) + ","
                + std::to_string(seedIdentifier->GetTuple1(i)) + ","
                + std::to_string(globalIdsForCsv->GetTuple1(i)) + ","
                + std::to_string(ghostArray->GetTuple1(i)) + "\n";
  }
  myfile.close();

  myfile.open("/home/eveleguillou/experiment/IntegralLines/Benchmark/"
              + std::to_string(ttk::MPIsize_) + "_proc_integraLinesCellData_"
              + std::to_string(ttk::MPIrank_) + ".csv");
  myfile << "GlobalCellIds,RankArray\n";
  vtkDataArray *edgeId = output->GetCellData()->GetArray("GlobalCellIds");
  vtkDataArray *rankArray = output->GetCellData()->GetArray("RankArray");
  for(int i = 0; i < edgeId->GetNumberOfTuples(); i++) {
    myfile << std::to_string(edgeId->GetTuple1(i)) + ","
                + std::to_string(rankArray->GetTuple1(i)) + "\n";
  }
  myfile.close();

  return (int)(status == 0);
}
