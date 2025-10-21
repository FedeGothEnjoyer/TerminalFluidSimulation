#pragma once
#include <string>
#include <rgb.h>

class Image{
    public:
        int width, height, channels;
        Image(std::string filename);
        Image(std::string filename, bool& result);
        color Sample(float _x, float _y);
        ~Image();
    private:
        unsigned char* data;
};