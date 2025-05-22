#ifndef MODEL_HPP
#define MODEL_HPP

#include <torch/torch.h>

struct HandKeypointModelImpl : torch::nn::Module {
    torch::nn::Sequential conv_layers, fc_layers;

    HandKeypointModelImpl();

    torch::Tensor forward(torch::Tensor input);
};

TORCH_MODULE(HandKeypointModel);

#endif //MODEL_HPP
