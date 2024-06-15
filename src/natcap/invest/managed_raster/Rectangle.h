#ifndef RECTANGLE_H
#define RECTANGLE_H

namespace shapes {
    class NeighborTupleClass {
        public:
            int direction, x, y;
            float flow_proportion;
            NeighborTupleClass();
            NeighborTupleClass(int direction, int x, int y, float flow_proportion);
            ~NeighborTupleClass();
    };
}

#endif
