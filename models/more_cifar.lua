local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local LCN = nn.SpatialContrastiveNormalization

local model  = nn.Sequential()

model:add(Convolution(3, 16, 5, 5, 1, 1, 2, 2))
model:add(ReLU())
model:add(Convolution(16, 64, 5, 5))
model:add(ReLU())
--model:add(LCN(64, image.gaussian(3)))
model:add(Max(2,2,2,2))
model:add(Convolution(64, 128, 5, 5))
model:add(ReLU())
model:add(Convolution(128, 128, 3, 3))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(View(2048))
model:add(Linear(2048, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
