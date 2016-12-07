local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model = nn.Sequential()

model:add(Convolution(3,  32, 5, 5)) -- 16*28*28
model:add(ReLU())
model:add(Convolution(32, 64, 3, 3)) -- 64*26*26
model:add(ReLU())
model:add(Max(3, 3, 1, 1)) -- 64*24*24
model:add(Convolution(64, 128, 5, 5)) -- 128*20*20
model:add(ReLU())
model:add(Convolution(128, 256, 5, 5)) -- 256*16*16
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 256*8*8
model:add(View(16384))
model:add(Dropout(0.30))
model:add(Linear(16384, 4096))
model:add(ReLU())
model:add(Linear(4096, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
