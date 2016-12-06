local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolutionMM
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local Subsample = nn.SpatialSubSampling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model = nn.Sequential()
local scale_1 = nn.Sequential()
local scale_2 = nn.Sequential()

-- input: 3*32*32
model:add(Convolution(3, 16, 5, 5, 1, 1, 2, 2))
model:add(Subsample(12, 2, 2, 2, 2))
model:add(ReLU)



return model
