#include "forest.h"

// constructor
forest::forest() : m(50), t(m){};
forest::forest(size_t im) : m(im), t(m){};
forest::forest(const forest &ib) : m(ib.m), t(m)
{
    this->t = ib.t;
}

forest::~forest()
{
}
