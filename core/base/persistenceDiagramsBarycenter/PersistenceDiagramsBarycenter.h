/// \ingroup base
/// \class ttk::PersistenceDiagramsBarycenter
/// \author Michael Michaux <michauxmichael89@gmail.com>
/// \date August 2016.
///
/// \brief TTK processing package that takes an input ensemble data set 
/// (represented by a list of scalar fields) and which computes various 
/// vertexwise statistics (PDF estimation, bounds, moments, etc.)
///
/// \sa ttkPersistenceDiagramsBarycenter.cpp %for a usage example.

#ifndef _PERSISTENCEDIAGRAMSBARYCENTER_H
#define _PERSISTENCEDIAGRAMSBARYCENTER_H



#ifndef diagramTuple
#define diagramTuple std::tuple<ttk::ftm::idVertex, ttk::ftm::NodeType, ttk::ftm::idVertex, \
  ttk::ftm::NodeType, dataType, ttk::ftm::idVertex, \
  dataType, float, float, float, dataType, float, float, float>
#endif


#ifndef BNodeType
#define BNodeType ttk::ftm::NodeType
#define BLocalMax ttk::ftm::NodeType::Local_maximum
#define BLocalMin ttk::ftm::NodeType::Local_minimum
#define BSaddle1  ttk::ftm::NodeType::Saddle1
#define BSaddle2  ttk::ftm::NodeType::Saddle2
#define BIdVertex ttk::ftm::idVertex
#endif


// base code includes
#include                  <PersistenceDiagramsBarycenter.cpp>
#include                  <Wrapper.h>
#include                  <PersistenceDiagram.h>
#include 				  <Auction.h>
#include 				  <KDTree.h>
#include 				  <limits>
#include				  <PDBarycenter.h>

using namespace std;
using namespace ttk;

namespace ttk{
  template<typename dataType>
  class PersistenceDiagramsBarycenter : public Debug{

	public:

		PersistenceDiagramsBarycenter(){
			wasserstein_ = 2;
			geometrical_factor_ = 1;
			inputData_ = NULL;
			numberOfInputs_ = 0;
			threadNumber_ = 1;
		};

		~PersistenceDiagramsBarycenter(){
			if(inputData_)
				free(inputData_);
		};


		std::vector<std::vector<matchingTuple>> execute(std::vector<diagramTuple>* barycenter);
			
		inline int setDiagram(int idx, void* data){
			if(idx < numberOfInputs_){
			inputData_[idx] = data;
			}
			else{
			return -1;
			}
			return 0;
		}

		inline int setNumberOfInputs(int numberOfInputs){
			numberOfInputs_ = numberOfInputs;
			if(inputData_)
			free(inputData_);
			inputData_ = (void **) malloc(numberOfInputs*sizeof(void *));
			for(int i=0 ; i<numberOfInputs ; i++){
			inputData_[i] = NULL;
			}
			return 0;
		}
		
		inline void setWasserstein(const std::string &wasserstein){
			wasserstein_ = (wasserstein == "inf") ? -1 : stoi(wasserstein);
		}
		
		inline void setThreadNumber(const int &ThreadNumber){
			threadNumber_ = ThreadNumber;
		}
		
		inline void setUseProgressive(const bool use_progressive){
			use_progressive_ = use_progressive;
		}
		
		inline void setTimeLimit(const double time_limit){
			time_limit_ = time_limit;
		}
		
		template<typename type>
		static type abs(const type var) {
			return (var >= 0) ? var : -var;
		}



    protected:
	  int 					wasserstein_;
	  double                geometrical_factor_; // TODO include it in barycenter
	  
      int                   numberOfInputs_;
      void**                inputData_; //TODO : std::vector<void*>
      int 					threadNumber_;
	  bool                  use_progressive_;
	  double                time_limit_;
      
      
      int points_added_;
	  int points_deleted_;
      
      std::vector<std::vector<dataType>>      all_matchings_;
 	  std::vector<std::vector<dataType>>      all_old_matchings_;
      std::vector<BidderDiagram<dataType>>    bidder_diagrams_;
      std::vector<GoodDiagram<dataType>>	  barycenter_goods_;
  };
  
  
template <typename dataType> 
std::vector<std::vector<matchingTuple>> PersistenceDiagramsBarycenter<dataType>::execute(std::vector<diagramTuple>* barycenter){
	Timer t;
	{
	
	std::vector<std::vector<diagramTuple>*> data_min(numberOfInputs_);
	std::vector<std::vector<diagramTuple>*> data_sad(numberOfInputs_);
	std::vector<std::vector<diagramTuple>*> data_max(numberOfInputs_);
	
	std::vector<std::vector<int>> data_min_idx(numberOfInputs_);
	std::vector<std::vector<int>> data_sad_idx(numberOfInputs_);
	std::vector<std::vector<int>> data_max_idx(numberOfInputs_);
	
	std::vector<std::vector<matchingTuple>> all_matchings(numberOfInputs_);
	
	bool do_min = false;
	bool do_sad = false;
	bool do_max = false;
	
	// Create diagrams for min, saddle and max persistence pairs
	for(int i=0; i<numberOfInputs_; i++){
		data_min[i] = new std::vector<diagramTuple>;
		data_sad[i] = new std::vector<diagramTuple>;
		data_max[i] = new std::vector<diagramTuple>;
		std::vector<diagramTuple>* CTDiagram = static_cast<std::vector<diagramTuple>*>(inputData_[i]);
		
		for(int j=0; j<(int) CTDiagram->size(); ++j){
			diagramTuple t = CTDiagram->at(j);
			
			BNodeType nt1 = std::get<1>(t);
			BNodeType nt2 = std::get<3>(t);
			
			dataType dt = std::get<4>(t);
			//if (abs<dataType>(dt) < zeroThresh) continue;
			if(dt>0){
				if (nt1 == BLocalMin && nt2 == BLocalMax) {
					data_max[i]->push_back(t);
					data_max_idx[i].push_back(j);
					do_max = true;
				}
				else {
					if (nt1 == BLocalMax || nt2 == BLocalMax) {
						data_max[i]->push_back(t);
						data_max_idx[i].push_back(j);
						do_max = true;
					}
					if (nt1 == BLocalMin || nt2 == BLocalMin) {
						data_min[i]->push_back(t);
						data_min_idx[i].push_back(j);
						do_min = true;
					}
					if ((nt1 == BSaddle1 && nt2 == BSaddle2)
						|| (nt1 == BSaddle2 && nt2 == BSaddle1)) {
						data_sad[i]->push_back(t);
						data_sad_idx[i].push_back(j);
						do_sad = true;
					}
				}
			}
		}
	}
	
	std::vector<diagramTuple> barycenter_min;
	std::vector<diagramTuple> barycenter_sad;
	std::vector<diagramTuple> barycenter_max;
	
	std::vector<std::vector<matchingTuple>> matching_min, matching_sad, matching_max;
	
	/*omp_set_num_threads(1);
	#ifdef TTK_ENABLE_OPENMP
	#pragma omp parallel sections
	#endif
	{
		#ifdef TTK_ENABLE_OPENMP
		#pragma omp section
		#endif
		{*/
			if(do_min){
				std::cout << "Computing Minima barycenter..."<<std::endl;
				PDBarycenter<dataType> bary_min = PDBarycenter<dataType>();
				bary_min.setThreadNumber(threadNumber_);
				bary_min.setWasserstein(wasserstein_);
				bary_min.setNumberOfInputs(numberOfInputs_);
				bary_min.setDiagramType(0);
				bary_min.setUseProgressive(use_progressive_);
				bary_min.setTimeLimit(time_limit_);
				for(int i=0; i<numberOfInputs_; i++){
					bary_min.setDiagram(i, data_min[i]);
				}
				matching_min = bary_min.execute(barycenter_min);
			}
		/*}
		
		#ifdef TTK_ENABLE_OPENMP
		#pragma omp section
		#endif
		{*/
			if(do_sad){
				std::cout << "Computing Saddles barycenter..."<<std::endl;
				PDBarycenter<dataType> bary_sad = PDBarycenter<dataType>();
				bary_sad.setThreadNumber(threadNumber_);
				bary_sad.setWasserstein(wasserstein_);
				bary_sad.setNumberOfInputs(numberOfInputs_);
				bary_sad.setDiagramType(1);
				bary_sad.setUseProgressive(use_progressive_);
				bary_sad.setTimeLimit(time_limit_);
				for(int i=0; i<numberOfInputs_; i++){
					bary_sad.setDiagram(i, data_sad[i]);
				}
				matching_sad = bary_sad.execute(barycenter_sad);
			}
		/*}
		
		#ifdef TTK_ENABLE_OPENMP
		#pragma omp section
		#endif
		{*/
			if(do_max){
				std::cout << "Computing Maxima barycenter..."<<std::endl;
				PDBarycenter<dataType> bary_max = PDBarycenter<dataType>();
				bary_max.setThreadNumber(threadNumber_);
				bary_max.setWasserstein(wasserstein_);
				bary_max.setNumberOfInputs(numberOfInputs_);
				bary_max.setDiagramType(2);
				bary_max.setUseProgressive(use_progressive_);
				bary_max.setTimeLimit(time_limit_);
				for(int i=0; i<numberOfInputs_; i++){
					bary_max.setDiagram(i, data_max[i]);
				}
				matching_max = bary_max.execute(barycenter_max);
			}
		//}
	//}
	
	// Reconstruct matchings
	for(int i=0; i<numberOfInputs_; i++){
		
		if(do_min){
			for(unsigned int j=0; j<matching_min[i].size(); j++){
				matchingTuple t = matching_min[i][j];
				int bidder_id = std::get<0>(t);
				std::get<0>(t) = data_min_idx[i][bidder_id];
				all_matchings[i].push_back(t);
			}
		}
		
		if(do_sad){
			for(unsigned int j=0; j<matching_sad[i].size(); j++){
				matchingTuple t = matching_sad[i][j];
				int bidder_id = std::get<0>(t);
				std::get<0>(t) = data_sad_idx[i][bidder_id];
				std::get<1>(t) = std::get<1>(t) + barycenter_min.size();
				all_matchings[i].push_back(t);
			}
		}
		
		if(do_max){
			for(unsigned int j=0; j<matching_max[i].size(); j++){
				matchingTuple t = matching_max[i][j];
				int bidder_id = std::get<0>(t);
				std::get<0>(t) = data_max_idx[i][bidder_id];
				std::get<1>(t) = std::get<1>(t) + barycenter_min.size() + barycenter_sad.size();
				all_matchings[i].push_back(t);
			}
		}
	}
	// Reconstruct barcenter
	for(unsigned int j=0; j<barycenter_min.size(); j++){
		diagramTuple dt = barycenter_min[j];
		barycenter->push_back(dt);
	}
	for(unsigned int j=0; j<barycenter_sad.size(); j++){
		diagramTuple dt = barycenter_sad[j];
		//std::get<5>(dt) = barycenter->size();
		barycenter->push_back(dt);
	}
	for(unsigned int j=0; j<barycenter_max.size(); j++){
		diagramTuple dt = barycenter_max[j];
		//std::get<5>(dt) = barycenter->size();
		barycenter->push_back(dt);
	}
	
	
	for(int i=0; i<numberOfInputs_; i++){
		delete data_min[i];
		delete data_sad[i];
		delete data_max[i];
	}
	
	std::stringstream msg;
	msg << "[PersistenceDiagramsBarycenter] processed in "
		<< t.getElapsedTime() << " s. (" << threadNumber_
		<< " thread(s))."
		<< std::endl;
	dMsg(std::cout, msg.str(), timeMsg);
	return all_matchings;
	}
	
}

}
  


// if the package is a pure template class, uncomment the following line

#include <PDBarycenterImpl.h>
#include <PDBarycenter.h>
#endif 
