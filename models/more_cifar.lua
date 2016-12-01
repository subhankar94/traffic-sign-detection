local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Avg = nn.SpatialAveragePooling

local model  = nn.Sequential()

model:add(Convolution(3, 16, 5, 5, 1, 1, 2, 2)) -- 16 * 32 * 32
model:add(ReLU())
model:add(Convolution(16, 64, 3, 3, 1, 1, 1, 1)) -- 64 * 32 * 32
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 64 * 16 * 16
model:add(Convolution(16, 128, 5, 5, 1, 1, 2, 2)) -- 128 * 16 * 16
model:add(ReLU())
model:add(Convolution(128, 512, 3, 3, 1, 1, 1, 1)) -- 512 * 16 * 16
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 512 * 8 * 8
model:add(Convolution(512, 1024, 5, 5, 1, 1, 2, 2)) -- 1024 * 8 * 8
model:add(ReLU())
model:add(Avg(8, 8, 1, 1)) -- 1024 * 1 * 1
model:add(View(1024))
model:add(Linear(1024, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
