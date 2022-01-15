#include "net.h"
#include "ncnn/layer/convolution.h"

#include<stdio.h>
#include <iostream>

using namespace std;

int main()
{
    ncnn::Net squeezenet;
    squeezenet.load_param("assert/squeezenet_v1.1.param");
    squeezenet.load_model("assert/squeezenet_v1.1.bin");

    ncnn::Convolution* a = static_cast<ncnn::Convolution*>(squeezenet.layers()[1]);

    ncnn::Mat b = a->weight_data;
    const float* data = b.channel(0);
    int data_size = b.w;

    FILE* fp;
    fp = fopen("conv1_weight.bin", "w");
    fwrite(data, sizeof(float), data_size, fp);
    fclose(fp);

    return 0;
}
