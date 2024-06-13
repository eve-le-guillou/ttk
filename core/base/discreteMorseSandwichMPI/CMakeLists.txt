ttk_add_base_library(discreteMorseSandwichMPI
  SOURCES
    DiscreteMorseSandwichMPI.cpp
  HEADERS
    DiscreteMorseSandwichMPI.h
  DEPENDS
    discreteGradient
)
target_compile_options(discreteMorseSandwichMPI PUBLIC -O0 -g)