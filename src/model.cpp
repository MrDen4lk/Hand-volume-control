#include "model.hpp"

ResidualBlockImpl::ResidualBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride, bool downsample_needed) {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 3).stride(stride).padding(1).bias(false));
    bn1 = torch::nn::BatchNorm2d(out_planes);
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_planes, out_planes, 3).stride(1).padding(1).bias(false));;
    bn2 = torch::nn::BatchNorm2d(out_planes);

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);

    use_downsample = downsample_needed;
    if (use_downsample) {
        downsample = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 1).stride(stride).bias(false)),
            torch::nn::BatchNorm2d(out_planes)
            );
        register_module("downsample", downsample);
    }
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    auto identity = x;

    auto out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));

    if (use_downsample) {
        identity = downsample->forward(x);
    }

    out += identity;
    return torch::relu(out);
}

ResNet18HeatmapImpl::ResNet18HeatmapImpl() {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false));
    bn1 = torch::nn::BatchNorm2d(64);
    maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
    relu = torch::nn::ReLU();

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("maxpool", maxpool);
    register_module("relu", relu);

    layer1 = _make_layer(64, 64, 2);
    layer2 = _make_layer(64, 128, 2, 2);
    layer3 = _make_layer(128, 256, 2, 2);
    layer4 = _make_layer(256, 512, 2, 2);

    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);

    up1 = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 256, 4).stride(2).padding(1));
    up2 = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1));
    up3 = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1));

    register_module("up1", up1);
    register_module("up2", up2);
    register_module("up3", up3);

    final_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 21, 1));
    register_module("final_conv", final_conv);
}

torch::Tensor ResNet18HeatmapImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = relu->forward(x);
    x = maxpool->forward(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::relu(up1->forward(x));
    x = torch::relu(up2->forward(x));
    x = torch::relu(up3->forward(x));

    return final_conv->forward(x); // [B, 21, 64, 64]
}

torch::nn::Sequential ResNet18HeatmapImpl::_make_layer(int64_t in_planes, int64_t out_planes, int blocks, int64_t stride) {
    torch::nn::Sequential layers;
    layers->push_back(ResidualBlock(in_planes, out_planes, stride, stride != 1 || in_planes != out_planes));
    for (int i = 1; i < blocks; ++i) {
        layers->push_back(ResidualBlock(out_planes, out_planes));
    }
    return layers;
}
