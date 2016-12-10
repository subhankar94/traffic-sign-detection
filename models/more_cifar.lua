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
model:add(Convolution(3,  64, 7, 7)) -- 64*42*42
model:add(ReLU())
model:add(Convolution(64, 128, 5, 5, 1, 1, 1)) -- 128*40*40
model:add(ReLU())
model:add(Max(2, 2, 2, 2)) -- 128*20*20
model:add(Convolution(128, 256, 5, 5)) -- 256*16*16
model:add(ReLU())
model:add(Max(2, 2, 2, 2)) -- 256*8*8
model:add(Convolution(256, 384, 5, 5)) -- 384*4*4
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 384*2*2
model:add(View(1536))
model:add(Dropout(0.40))
model:add(Linear(1536, 100))
model:add(ReLU())
model:add(Linear(100, 43))

return model
