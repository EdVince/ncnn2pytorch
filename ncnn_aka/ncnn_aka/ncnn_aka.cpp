#include "net.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <io.h>
#include <direct.h>

#include "ncnn/layer/input.h"
#include "ncnn/layer/convolution.h"
#include "ncnn/layer/relu.h"
#include "ncnn/layer/pooling.h"
#include "ncnn/layer/split.h"
#include "ncnn/layer/concat.h"
#include "ncnn/layer/dropout.h"
#include "ncnn/layer/softmax.h"

using namespace std;


int main()
{
    string param = "model/squeezenet_v1.1.param";
    string bin = "model/squeezenet_v1.1.bin";
    string output = "output";

    if (_access(output.c_str(), 0) == -1) {
        _mkdir(output.c_str());
    }

    ncnn::Net net;
    net.load_param(param.c_str());
    net.load_model(bin.c_str());

    for (int count = 0; count < net.layers().size(); count++) {
        ncnn::Layer* layer = net.layers()[count];
        string layer_type = layer->type;
        string txt_name = output + "/" + to_string(count) + "_" + layer_type + ".txt";
        string bin_name = output + "/" + to_string(count) + "_" + layer_type + "_";
        if (layer_type == "Input") {
            ncnn::Input* l = static_cast<ncnn::Input*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile << l->c << " " << l->h << " " << l->w << endl; // 记录c,h,w
            outfile.close();
            // 需要保存到bin的权重参数
        }
        else if (layer_type == "Convolution") {
            ncnn::Convolution* l = static_cast<ncnn::Convolution*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile << l->num_output << " " 
                << l->kernel_w << " " << l->kernel_h << " " << l->dilation_w << " " << l->dilation_h << " "
                << l->stride_w << " " << l->stride_h << " " << l->pad_left << " " << l->pad_right << " "
                << l->pad_top << " " << l->pad_bottom << " " << l->pad_value << " " << l->bias_term << " "
                << l->weight_data_size << " " << l->activation_type << endl;
            for (int x = 0; x < l->activation_params.w; x++) outfile << l->activation_params.channel(0)[x] << " "; outfile << endl;
            outfile.close();
            // 需要保存到bin的权重参数

            if (l->weight_data.dims > 0) {
                FILE* fp;
                fp = fopen((bin_name+"_weight.bin").c_str(), "w");
                fwrite(l->weight_data, sizeof(float), l->weight_data.w, fp);
                fclose(fp);
            }
            if (l->bias_data.dims > 0) {
                FILE* fp;
                fp = fopen((bin_name + "_bias.bin").c_str(), "w");
                fwrite(l->bias_data, sizeof(float), l->bias_data.w, fp);
                fclose(fp);
            }
        }
        else if (layer_type == "ReLU") {
            ncnn::ReLU* l = static_cast<ncnn::ReLU*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile << l->slope << endl;
            outfile.close();
        }
        else if (layer_type == "Pooling") {
            ncnn::Pooling* l = static_cast<ncnn::Pooling*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile << l->pooling_type << " "
                << l->kernel_w << " " << l->kernel_h << " " << l->stride_w << " " << l->stride_h << " "
                << l->pad_left << " " << l->pad_right << " " << l->pad_top << " " << l->pad_bottom << " "
                << l->global_pooling << " " << l->pad_mode << " " << l->avgpool_count_include_pad << " " << l->adaptive_pooling << " "
                << l->out_w << " " << l->out_h << endl;
            outfile.close();
        }
        else if (layer_type == "Split") {
            ncnn::Split* l = static_cast<ncnn::Split*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile.close();
        }
        else if (layer_type == "Concat") {
            ncnn::Concat* l = static_cast<ncnn::Concat*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile << l->axis << endl;
            outfile.close();
        }
        else if (layer_type == "Dropout") {
            ncnn::Dropout* l = static_cast<ncnn::Dropout*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile << l->scale << endl;
            outfile.close();
        }
        else if (layer_type == "Softmax") {
            ncnn::Softmax* l = static_cast<ncnn::Softmax*>(layer);
            // 需要保存到txt的配置参数
            ofstream outfile(txt_name, ios::trunc);
            outfile << l->name << endl;
            for (const auto& x : l->bottoms) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录bottom blob
            for (const auto& x : l->tops) outfile << net.blobs()[x].name << " "; outfile << endl; // 记录top blob
            outfile << l->axis << endl;
            outfile.close();
        }
        else {
            cout << "[fuck] unsupport layer: " << layer_type << endl;
            return -1;
        }
    }

    return 0;
}
