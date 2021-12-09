#include <ttkScalarFieldCriticalPoints.h>

#include <vtkInformation.h>

#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSignedCharArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkInformationVector.h>
#include <vtkMultiProcessController.h>
#include <vtkMPIController.h>


#include <ttkMacros.h>
#include <ttkUtils.h>
#include <DataSetAttributes.h>

using namespace std;
using namespace ttk;

vtkStandardNewMacro(ttkScalarFieldCriticalPoints);

ttkScalarFieldCriticalPoints::ttkScalarFieldCriticalPoints() {

  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

ttkScalarFieldCriticalPoints::~ttkScalarFieldCriticalPoints() {
}

int ttkScalarFieldCriticalPoints::FillInputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0)
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
  else
    return 0;

  return 1;
}

int ttkScalarFieldCriticalPoints::FillOutputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
  else
    return 0;

  return 1;
}

int ttkScalarFieldCriticalPoints::RequestData(
  vtkInformation *ttkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector) {
  this->setThreadNumber(1);
  vtkDataSet *input = vtkDataSet::GetData(inputVector[0]);

  vtkPolyData *output = vtkPolyData::GetData(outputVector, 0);

  ttk::Triangulation *triangulation = ttkAlgorithm::GetTriangulation(input);
  if(!triangulation)
    return 0;

  if(VertexBoundary)
    triangulation->preconditionBoundaryVertices();

  // in the following, the target scalar field of the input is replaced in the
  // variable 'output' with the result of the computation.
  // if your wrapper produces an output of the same type of the input, you
  // should proceed in the same way.

  vtkDataArray *inputScalarField = this->GetInputArrayToProcess(0, inputVector);
  if(!inputScalarField)
    return 0;

  vtkDataArray *offsetField
    = this->GetOrderArray(input, 0, 1, ForceInputOffsetScalarField);

  // setting up the base layer
  this->preconditionTriangulation(triangulation);
  this->setOutput(&criticalPoints_);

  #if TTK_ENABLE_MPI
  Timer t_mpi;
  // Get processes information
  vtkMPIController *controller = vtkMPIController::SafeDownCast(
    vtkMultiProcessController::GetGlobalController());
  int myRank = controller->GetLocalProcessId();
  int numberOfProcesses = controller->GetNumberOfProcesses();
  this->setNumberOfProcesses(numberOfProcesses);
  this->setMyRank(myRank);

  controller->Barrier();
  if(myRank == 0) {
    t_mpi.reStart();
  }
  // Set GlobalIdsArray
  long int *globalPointsId = static_cast<long int *>(ttkUtils::GetVoidPointer(
    input->GetPointData()->GetArray("GlobalPointIds")));
  this->setGlobalIdsArray(globalPointsId);

  if(numberOfProcesses > 1) {

    // Set pointGhostArray
    unsigned char *pointGhostArray = static_cast<unsigned char *>(
      ttkUtils::GetVoidPointer(input->GetPointGhostArray()));
    unsigned char *cellGhostArray = static_cast<unsigned char *>(
      ttkUtils::GetVoidPointer(input->GetCellGhostArray()));
    this->setPointGhostArray(pointGhostArray);

    int vertexNumber = triangulation->getNumberOfVertices();

    // Construct bounding box
    float pt[3];
    double bounds[6] = {0, 0, 0, 0, 0, 0};
    for(vtkIdType id = 0; id < vertexNumber; id++) {
      if(!(pointGhostArray[id] & ttk::type::DUPLICATEPOINT)) {
        triangulation->getVertexPoint(id, pt[0], pt[1], pt[2]);
        bounds[0] = std::min(bounds[0], (double)pt[0]);
        bounds[1] = std::max(bounds[1], (double)pt[0]);
        bounds[2] = std::min(bounds[2], (double)pt[1]);
        bounds[3] = std::max(bounds[3], (double)pt[1]);
        bounds[4] = std::min(bounds[4], (double)pt[2]);
        bounds[5] = std::max(bounds[5], (double)pt[2]);
      }
    }

    // Send bounding boxes across all processes
    vtkSmartPointer<vtkDoubleArray> allBoundsArray
      = vtkSmartPointer<vtkDoubleArray>::New();
    allBoundsArray->SetNumberOfComponents(6);
    allBoundsArray->SetNumberOfTuples(numberOfProcesses);
    controller->AllGather(
      bounds, reinterpret_cast<double *>(allBoundsArray->GetVoidPointer(0)), 6);

    // Stores whether a vertex is on the boundary (doesn't take ghost points
    // into account)
    vtkSmartPointer<vtkUnsignedCharArray> isOnMPIBoundary
      = vtkSmartPointer<vtkUnsignedCharArray>::New();
    isOnMPIBoundary->SetNumberOfComponents(1);
    isOnMPIBoundary->SetNumberOfTuples(vertexNumber);
    isOnMPIBoundary->Fill(0);

    // Stores which process contains this vertex
    std::vector<std::vector<int>> vertex2Process(
      vertexNumber, std::vector<int>(0));

    double bb[6];
    double dPt[3];
    int cellVertexNumber = 0;
    int v_id;

    // Construct isOnMPIBoundary
    for(vtkIdType id = 0; id < triangulation->getNumberOfCells(); id++) {
      if(cellGhostArray[id] & ttk::type::DUPLICATEPOINT) {
        cellVertexNumber = triangulation->getCellVertexNumber(id);
        for(int vertex = 0; vertex < cellVertexNumber; vertex++) {
          triangulation->getCellVertex(id, vertex, v_id);
          if(!(pointGhostArray[v_id] & ttk::type::DUPLICATEPOINT)) {
            isOnMPIBoundary->SetTuple1(v_id, 1);
          }
        }
      }
    }
    // Construct vertex2Process
    for(vtkIdType id = 0; id < vertexNumber; id++) {
      for(int p = 0; p < numberOfProcesses; p++) {
        if(p == myRank) {
          vertex2Process[id].push_back(p);
        } else {
          if((*isOnMPIBoundary->GetTuple(id)) == 1) {
            triangulation->getVertexPoint(id, pt[0], pt[1], pt[2]);
            allBoundsArray->GetTuple(p, bb);
            dPt[0] = (double)pt[0];
            dPt[1] = (double)pt[1];
            dPt[2] = (double)pt[2];
            vtkBoundingBox neighborBB(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
            if(neighborBB.ContainsPoint(dPt))
              vertex2Process[id].push_back(p);
          }
        }
      }
    }
    this->setVertex2Process(vertex2Process);
    this->setIsOnMPIBoundary(
      static_cast<unsigned char *>(ttkUtils::GetVoidPointer(isOnMPIBoundary)));
  }
#endif

  printMsg("Starting computation...");
  printMsg({{"  Scalar Array", inputScalarField->GetName()},
            {"  Offset Array", offsetField ? offsetField->GetName() : "None"}});

  int status = 0;
  ttkTemplateMacro(
    triangulation->getType(),
    (status = this->execute(
       static_cast<SimplexId *>(ttkUtils::GetVoidPointer(offsetField)),
       (TTK_TT *)triangulation->getData())));
  
  if(status == -1) {
    vtkErrorMacro("Please use Ghost Arrays for parallel computation of critical points");
  }
  if(status < 0)
    return 0;
#if TTK_ENABLE_MPI
  controller->Barrier();
  if(myRank == 0) {
    printMsg("Computation performed using " + std::to_string(numberOfProcesses)
               + " MPI processes",
             1, t_mpi.getElapsedTime(), threadNumber_);
  }
#endif
  // allocate the output
  vtkNew<vtkSignedCharArray> vertexTypes{};
  vertexTypes->SetNumberOfComponents(1);
  vertexTypes->SetNumberOfTuples(criticalPoints_.size());
  vertexTypes->SetName("CriticalType");

  vtkNew<vtkPoints> pointSet{};
  pointSet->SetNumberOfPoints(criticalPoints_.size());
  vtkNew<vtkIdTypeArray> offsets{}, connectivity{};
  offsets->SetNumberOfComponents(1);
  offsets->SetNumberOfTuples(criticalPoints_.size() + 1);
  connectivity->SetNumberOfComponents(1);
  connectivity->SetNumberOfTuples(criticalPoints_.size());

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(this->threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < criticalPoints_.size(); i++) {
    std::array<double, 3> p{};
    input->GetPoint(criticalPoints_[i].first, p.data());
    pointSet->SetPoint(i, p.data());
    vertexTypes->SetTuple1(i, (float)criticalPoints_[i].second);
    offsets->SetTuple1(i, i);
    connectivity->SetTuple1(i, i);
  }
  offsets->SetTuple1(criticalPoints_.size(), criticalPoints_.size());

  vtkNew<vtkCellArray> cells{};
  cells->SetData(offsets, connectivity);
  output->SetVerts(cells);
  output->SetPoints(pointSet);
  output->GetPointData()->AddArray(vertexTypes);

  if(VertexBoundary) {
    vtkNew<vtkSignedCharArray> vertexBoundary{};
    vertexBoundary->SetNumberOfComponents(1);
    vertexBoundary->SetNumberOfTuples(criticalPoints_.size());
    vertexBoundary->SetName("IsOnBoundary");

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(size_t i = 0; i < criticalPoints_.size(); i++) {
      vertexBoundary->SetTuple1(
        i, (signed char)triangulation->isVertexOnBoundary(
             criticalPoints_[i].first));
    }

    output->GetPointData()->AddArray(vertexBoundary);
  } else {
    output->GetPointData()->RemoveArray("IsOnBoundary");
  }

  if(VertexIds) {
    vtkNew<ttkSimplexIdTypeArray> vertexIds{};
    vertexIds->SetNumberOfComponents(1);
    vertexIds->SetNumberOfTuples(criticalPoints_.size());
    vertexIds->SetName(ttk::VertexScalarFieldName);

    for(size_t i = 0; i < criticalPoints_.size(); i++) {
#if TTK_ENABLE_MPI
      vertexIds->SetTuple1(i, this->GlobalIdsArray[criticalPoints_[i].first]);
#else
      vertexIds->SetTuple1(i, criticalPoints_[i].first);
#endif
    }

    output->GetPointData()->AddArray(vertexIds);
  } else {
    output->GetPointData()->RemoveArray(ttk::VertexScalarFieldName);
  }

  if(VertexScalars) {
    for(SimplexId i = 0; i < input->GetPointData()->GetNumberOfArrays(); i++) {

      vtkDataArray *scalarField = input->GetPointData()->GetArray(i);
      vtkSmartPointer<vtkDataArray> scalarArray{scalarField->NewInstance()};

      scalarArray->SetNumberOfComponents(scalarField->GetNumberOfComponents());
      scalarArray->SetNumberOfTuples(criticalPoints_.size());
      scalarArray->SetName(scalarField->GetName());
      std::vector<double> value(scalarField->GetNumberOfComponents());
      for(size_t j = 0; j < criticalPoints_.size(); j++) {
        scalarField->GetTuple(criticalPoints_[j].first, value.data());
        scalarArray->SetTuple(j, value.data());
      }
      output->GetPointData()->AddArray(scalarArray);
    }
  } else {
    for(SimplexId i = 0; i < input->GetPointData()->GetNumberOfArrays(); i++) {
      output->GetPointData()->RemoveArray(
        input->GetPointData()->GetArray(i)->GetName());
    }
  }

  return 1;
}
