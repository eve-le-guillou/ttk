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

    // Retrieve neighbouring processes     
    vtkMPIController* controller = vtkMPIController::SafeDownCast(vtkMultiProcessController::GetGlobalController());    
    int myRank = controller->GetLocalProcessId();
    int numberOfProcesses = controller->GetNumberOfProcesses();
    this->setNumberOfProcesses(numberOfProcesses);
    this->setMyRank(myRank);

    // Get BoundingBox
    const double* oriBounds = input->GetBounds();
    vtkBoundingBox myBoundingBox(oriBounds[0],oriBounds[1],oriBounds[2],oriBounds[3],oriBounds[4],oriBounds[5]);
    myBoundingBox.Inflate();
    double bounds[6];
    myBoundingBox.GetBounds(bounds);

    int vertexNumber = triangulation->getNumberOfVertices();
    // Send bounding boxes across all processes
    vtkSmartPointer<vtkDoubleArray> allBoundsArray = vtkSmartPointer<vtkDoubleArray>::New(); 
    allBoundsArray->SetNumberOfComponents(6); 
    allBoundsArray->SetNumberOfTuples(numberOfProcesses);  
    controller->AllGather(bounds, reinterpret_cast<double*>(allBoundsArray->GetVoidPointer(0)), 6);    

    unsigned char *pointGhostArray = static_cast<unsigned char *>(ttkUtils::GetVoidPointer(input->GetPointGhostArray()));
    long int * globalPointsId = static_cast<long int *>(ttkUtils::GetVoidPointer(input->GetPointData()->GetArray("GlobalPointIds")));
    this->setGlobalIdsArray(globalPointsId);
    vtkSmartPointer<vtkUnsignedCharArray> isOnMPIBoundary = vtkSmartPointer<vtkUnsignedCharArray>::New();
    isOnMPIBoundary->SetNumberOfComponents(1);
    isOnMPIBoundary->SetNumberOfTuples(vertexNumber);
    vtkSmartPointer<vtkDoubleArray> vertex2Process = vtkSmartPointer<vtkDoubleArray>::New();
    vertex2Process->SetNumberOfComponents(numberOfProcesses);
    vertex2Process->SetNumberOfTuples(vertexNumber);
    vertex2Process->Fill(0);
    unsigned char isNeighborGhost;
    for (vtkIdType id = 0; id < vertexNumber; id++){
      isNeighborGhost =  static_cast<unsigned char>(0);
      if (!(pointGhostArray[id] & ttk::type::DUPLICATEPOINT)){
        int neighborNumber = triangulation->getVertexNeighborNumber(id);
        int i = 0;
        while ((!isNeighborGhost) & (i < neighborNumber)){
          SimplexId neighborId = 0;
          triangulation->getVertexNeighbor(id, i, neighborId);
          if (myRank == 1){
            if (globalPointsId[id] == 1445){
              }
          }
          if (pointGhostArray[neighborId] & ttk::type::DUPLICATEPOINT){
            isNeighborGhost = static_cast<unsigned char>(1);
          }
          i++;
        }
      }
      isOnMPIBoundary->SetTuple1(id, isNeighborGhost);
    }
    for (vtkIdType id = 0; id < vertexNumber; id++){
      std::vector<double> neigh(numberOfProcesses, 0);
      neigh[myRank] = 1;
      if (isOnMPIBoundary->GetTuple(id)){
        float pt[3];
        triangulation->getVertexPoint(id, pt[0], pt[1], pt[2]);
        double dPt[3] = {(double)pt[0], (double)pt[1], (double)pt[2]};
          for (int p = 0; p < numberOfProcesses; p++){
            double bb[6];
            allBoundsArray->GetTuple(p, bb);
            vtkBoundingBox neighborBB(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
            if (neighborBB.ContainsPoint(dPt))
              {
                neigh[p] = 1;
              }
          }
      }
       vertex2Process->SetTuple(id, neigh.data()); 
     }     
    
  this->setVertex2Process(static_cast<double *>(ttkUtils::GetVoidPointer(vertex2Process)));
  this->setIsOnMPIBoundary(static_cast<unsigned char *>(ttkUtils::GetVoidPointer(isOnMPIBoundary)));
  this->setPointGhostArray(pointGhostArray);   

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
      vertexIds->SetTuple1(i, criticalPoints_[i].first);
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
