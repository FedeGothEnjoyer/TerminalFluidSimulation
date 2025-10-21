#pragma once

#include <algorithm>

struct color {
    float r, g, b;
    color() : r(0), g(0), b(0) {}
    color(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}
    [[nodiscard]]color Clamp(){
        return {
        std::max(0.0f,std::min(1.0f,r)),
        std::max(0.0f,std::min(1.0f,g)),
        std::max(0.0f,std::min(1.0f,b))};
    }
    [[nodiscard]]color operator*(const float scalar) const {
        return {scalar * r, scalar * g, scalar * b};
    }
};

inline int ColorDifferenceSquared(color a, color b){
    int dr = a.r*255 - b.r*255;
    int dg = a.g*255 - b.g*255;
    int db = a.b*255 - b.b*255;
    int rmean = (a.r*255 + b.r*255) / 2;
    return ((512 + rmean) * dr * dr) + (1024 * dg * dg) + ((512 + (255 - rmean)) * db * db);
}