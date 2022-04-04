/// \ingroup base
/// \class ttk::ArrayLinkedList
/// \author Eve Le Guillou <eve.le-guillou@lip6.fr>
/// \date Mars 2022.

#include <array>
#include <list>

#pragma once

namespace ttk {
  template <typename datatype, int size>
  class ArrayLinkedList {
  public:
    std::list<std::array<datatype, size>> list;
    int numberOfElement;

    ArrayLinkedList()
      : list(std::list<std::array<datatype, size>>({})), numberOfElement(0) {
    }

    datatype *addArrayElement(datatype element) {
      numberOfElement = numberOfElement % size;
      if(numberOfElement == 0) {
        this->list.push_back(std::array<datatype, size>({}));
      }
      this->list.back().at(numberOfElement) = element;
      this->numberOfElement++;
      return &(this->list.back().at(numberOfElement - 1));
    }
  };
} // namespace ttk
