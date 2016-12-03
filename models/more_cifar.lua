local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model = nn.Sequential()

model:add(Convolution(3,  16, 5, 5)) -- 16*28*28
model:add(ReLU())
model:add(Convolution(16, 64, 5, 5)) -- 64*24*24
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 64*12*12
model:add(Convolution(64, 256, 5, 5)) -- 256*8*8
model:add(ReLU())
model:add(Convolution(256, 1024, 5, 5)) -- 1024*4*4
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 1024*2*2
model:add(View(4096))
model:add(Linear(4096, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
