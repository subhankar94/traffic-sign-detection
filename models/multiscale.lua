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

-- input : 3*32*32
-- model:add(ContrastNorm(3, torch.ones(5,5)))
-- model:add(Convolution(3, 128, 5, 5, 1, 1, 2, 2)) -- 128*32*32
-- model:add(ReLU())
model:add(Convolution(3, 256, 5, 5, 1, 1, 2, 2)) -- 256*32*32
--model:add(Subsample(16, 2, 2, 2, 2)) -- 16*16*16
model:add(ReLU())
model:add(Max(2, 2, 2, 2)) -- 256*16*16
model:add(BatchNorm(256))
model:add(Convolution(256, 512, 5, 5, 1, 1, 2, 2)) -- 512*16*16
model:add(ReLU())
model:add(Max(2, 2, 2, 2)) -- 512*8*8
model:add(Convolution(512, 1024, 5, 5)) -- 1024*4*4
model:add(Subsample(1024, 2, 2, 2, 2)) -- 1024*2*2
model:add(ReLU())
model:add(View(1024*2*2))
model:add(Dropout(0.4))
model:add(Linear(4096, 100))
model:add(ReLU())
model:add(Linear(100, 100))
model:add(ReLU())
model:add(nn.Linear(100, 43))

return model
