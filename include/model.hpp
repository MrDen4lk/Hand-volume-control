#ifndef MODEL_HPP
#define MODEL_HPP

#include <torch/torch.h>

// Residual Block
struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential downsample;

    bool use_downsample = false;

    ResidualBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride = 1, bool downsample_needed = false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResidualBlock);

// ResNet18 for heatmaps
struct ResNet18HeatmapImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::ReLU relu;
    torch::nn::MaxPool2d maxpool{nullptr};

    torch::nn::Sequential layer1, layer2, layer3, layer4;

    torch::nn::ConvTranspose2d up1{nullptr}, up2{nullptr}, up3{nullptr};
    torch::nn::Conv2d final_conv{nullptr};

    ResNet18HeatmapImpl();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential _make_layer(int64_t in_planes, int64_t out_planes, int blocks, int64_t stride = 1);
};
TORCH_MODULE(ResNet18Heatmap);

#endif // MODEL_HPP