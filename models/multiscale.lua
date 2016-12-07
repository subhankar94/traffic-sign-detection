require 'torch'
local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolutionMM
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local Subsample = nn.SpatialSubSampling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout
local ContrastNorm = nn.SpatialContrastiveNormalization
local BatchNorm = nn.SpatialBatchNormalization

local model = nn.Sequential()

-- input : 3 * 40 * 40
model:add(ContrastNorm(3, torch.ones(5,5)))
model:add(Convolution(3, 12, 5, 5)) -- 12*36*36
model:add(ReLU())
model:add(Convolution(12, 16, 5, 5)) -- 12*32*32
model:add(Subsample(16, 2, 2, 2, 2)) -- 16*16*16
model:add(ReLU())
model:add(BatchNorm(16))
model:add(Convolution(16, 32, 5, 5, 1, 1, 2, 2)) -- 32*16*16
model:add(ReLU())
model:add(Convolution(32, 64, 5, 5, 1, 1, 2, 2)) -- 64*16*16
model:add(Subsample(64, 2, 2, 2, 2)) -- 64*8*8
model:add(ReLU())
model:add(View(64*8*8))
model:add(Dropout(0.3))
model:add(Linear(64*8*8, 100))
model:add(ReLU())
model:add(Linear(100, 100))
model:add(ReLU())
model:add(nn.Linear(100, 43))

return model
