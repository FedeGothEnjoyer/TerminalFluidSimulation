#include "img.h"
#include <math.h>
#include "rgb.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <string>

Image::Image(std::string filename){
    data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
}
Image::Image(std::string filename, bool& result){
    data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    result = (bool)data;
}
color Image::Sample(float _x, float _y){
    int x = std::min(width-1,std::max(0,(int)round(_x*width)));
    int y = height-1-std::min(height-1,std::max(0,(int)round(_y*height)));
    int index = (y*width+x)*3;
    int r=data[index++];
    int g=data[index++];
    int b=data[index];
    return {r/255.0f,g/255.0f,b/255.0f};
}
Image::~Image(){
    if (data) stbi_image_free(data);
}