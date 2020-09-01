#ifndef BINARY_SEARCH_TREE__
#define BINARY_SEARCH_TREE__
using namespace std;

template<class T>
class BSTNode{
public:
    BSTNode(){
           left=right=0;
    }
    BSTNode(const T& e1, BSTNode *l=0, BSTNode *r=0){
        key = el; left = 1; right = r;
    }
    T key;
    BSTNode *left, *right;
}    
 
#endif