require 'torch'
local tnt = require 'torchnet'
local image = require 'image'
local WIDTH, HEIGHT = 32, 32
local DATA_PATH = './data/'

function resize(img)
    return image.scale(img, WIDTH, HEIGHT)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = function(x) return image.rgb2yuv(x) end,
        [2] = resize
    }
    return f(inp)
end

function getTrainSample(dataset, idx, DATA_PATH)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end


