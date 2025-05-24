#include "model.hpp"

HandKeypointModelImpl::HandKeypointModelImpl() {
     conv_layers = torch::nn::Sequential(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding({1, 1})),
          torch::nn::BatchNorm2d(32),
          torch::nn::ReLU(),

          torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding({1, 1})),
          torch::nn::BatchNorm2d(64),
          torch::nn::ReLU(),

          torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding({1, 1})),
          torch::nn::BatchNorm2d(128),
          torch::nn::ReLU(),

          torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(2).padding({1, 1})),
          torch::nn::BatchNorm2d(128),
          torch::nn::ReLU(),

          torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding({1, 1})),
          torch::nn::BatchNorm2d(256),
          torch::nn::ReLU(),

          torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}))
     );

     fc_layers = torch::nn::Sequential(
          torch::nn::Linear(256, 128),
          torch::nn::ReLU(),
          torch::nn::Dropout(0.5),
          torch::nn::Linear(128, 42)
     );

     register_module("conv_layers", conv_layers);
     register_module("fc_layers", fc_layers);
}

torch::Tensor HandKeypointModelImpl::forward(torch::Tensor input) {
     input = conv_layers->forward(input);
     input = input.view({input.size(0), -1});
     input = fc_layers->forward(input);

     return input;
}

