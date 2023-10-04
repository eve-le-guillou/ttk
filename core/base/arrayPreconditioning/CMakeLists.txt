ttk_add_base_library(arrayPreconditioning
  SOURCES
    ArrayPreconditioning.cpp
  HEADERS
    ArrayPreconditioning.h
    psort.h
  DEPENDS
    common
    triangulation
)
if(TTK_ENABLE_MPI_SORT)
  #target_compile_definitions(arrayPreconditioning PUBLIC TTK_ENABLE_MPI_SORT)
  #target_include_directories(arrayPreconditioning PUBLIC "${KADIS_INCLUDE_DIR}")
  #target_link_libraries(arrayPreconditioning PUBLIC "${KADIS_LIBRARIES}")
  #target_include_directories(arrayPreconditioning PUBLIC "${RBC_INCLUDE_DIR}")
  #target_link_libraries(arrayPreconditioning PUBLIC "${RBC_LIBRARIES}")
  #target_include_directories(arrayPreconditioning PUBLIC "${IPS2PA_INCLUDE_DIR}")
  #target_include_directories(arrayPreconditioning PUBLIC "${IPS4O_INCLUDE_DIR}")
  #target_include_directories(arrayPreconditioning PUBLIC "${TLX_INCLUDE_DIR}")
  endif()