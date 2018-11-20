// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/script.h>

#include "nms.h"
#include "ROIAlign.h"

using namespace at;

// thin wrapper because we cannot get it from aten in Python due to overloads
Tensor upsample_bilinear(const Tensor& inp, int64_t w, int64_t h) {
  return at::upsample_bilinear2d(inp.unsqueeze(0).unsqueeze(0), {w, h}, false).squeeze(0).squeeze(0);
}

static auto registry =
  torch::jit::RegisterOperators()
    .op("maskrcnn_benchmark::nms", &nms)
    .op("maskrcnn_benchmark::roi_align_forward(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor", &ROIAlign_forward)
    .op("maskrcnn_benchmark::upsample_bilinear", &upsample_bilinear);

