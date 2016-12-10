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

-- input 3*48*48
model:add(Convolution(3,  128, 5, 5)) -- 128*44*44
model:add(ReLU())
model:add(Max(2, 2, 2, 2)) -- 128*22*22
model:add(Convolution(128, 256, 3, 3)) -- 256*20*20
model:add(ReLU())
model:add(Max(2, 2, 2, 2)) -- 256*10*10
model:add(Convolution(256, 512, 5, 5)) -- 256*6*6
model:add(ReLU())
model:add(Max(2, 2, 2, 2)) -- 512*3*3
model:add(View(4608))
model:add(Dropout(0.40))
model:add(Linear(4608, 100))
model:add(ReLU())
model:add(Linear(100, 43))

return model
