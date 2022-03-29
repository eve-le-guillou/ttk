/// \ingroup base
/// \class ttk::LinkedList
/// \author Eve Le Guillou <eve.le-guillou@lip6.fr>
/// \date Mars 2022.

#include <array>

#pragma once

namespace ttk {
  template <typename datatype, int size>
  class ListNode {
  public:
    std::array<datatype, size> *tab;
    ListNode *nextTab;

    ListNode(std::array<datatype, size> *array) : tab{array}, nextTab{NULL} {
    }
    ~ListNode() {
      delete(this->tab);
    }
  };

  template <typename datatype, int size>
  class LinkedList {
  public:
    ListNode<datatype, size> *firstTab;
    ListNode<datatype, size> *lastTab;
    int numberOfElement;

    LinkedList(ListNode<datatype, size> *node) : firstTab(node), lastTab(node) {
    }

    LinkedList() : firstTab(NULL), lastTab(NULL), numberOfElement(0) {
    }

    void addNode() {
      std::array<datatype, size> *array = new std::array<datatype, size>({});
      ListNode<datatype, size> *newNode = new ListNode<datatype, size>(array);
      if(this->firstTab == NULL) {
        this->firstTab = newNode;
      } else {
        this->lastTab->nextTab = newNode;
      }
      this->lastTab = newNode;
    }

    void addArrayElement(datatype element) {
      int index = numberOfElement % size;
      if(index == 0) {
        this->addNode();
      }
      this->lastTab->tab->at(index) = element;
      this->numberOfElement++;
    }

    ~LinkedList() {
      ListNode<datatype, size> *currentPtr = this->firstTab;
      ListNode<datatype, size> *nextPtr = this->lastTab;
      while(currentPtr != NULL) {
        nextPtr = currentPtr->nextTab;
        delete(currentPtr);
        currentPtr = nextPtr;
      }
    }
  };
} // namespace ttk
