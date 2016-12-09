require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
--require 'cunn'
--require 'cudnn' -- faster convolutions

--[[
--  Hint:  Plot as much as you can.  
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)
local dbg = require "debugger"
local utils = require 'utils'
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')
local HEIGHT, WIDTH = 32, 32

torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)

function getIterator(dataset, train, pruned, permed)
   
    local dset
    if not train then 
        --iterator = tnt.DatasetIterator{
                      dset = tnt.BatchDataset{
                          batchsize = opt.batchsize,
                          dataset   = dataset
                      }
        --           }
    else
        --iterator = tnt.DatasetIterator{
                      dset = tnt.BatchDataset{
                          batchsize = opt.batchsize,
                          dataset   = tnt.ResampleDataset{
                              dataset = dataset, 
                              size    = #pruned,
                              sampler = function(dataset, idx)
                                            return pruned[permed[idx]]
                                        end
                          }
                      }
        --           }
    end
    --[[
    return tnt.ParallelDatasetIterator {
              nthread = 16,
              init    = function() 
                            local tnt    = require 'torchnet'
                            local image  = require 'image'
                            local opt    = opt
                            local DATA_PATH = DATA_PATH
                            local train  = train
                            local pruned = pruned
                            local permed = permed
                            local utils  = require 'utils'
                        end,
              closure = function() return dset end
           }
    --]]
    return tnt.DatasetIterator{dataset = dset}
end

function prune_dataset(data, trainData, epoch, maxepochs, largest, smallest)
    -- create dummy dataset that gradually approaches final distribution
    local max = largest * ((maxepochs-epoch)/maxepochs)
    -- make list of images by class
    local indices = data.__dataset.__perm
    local by_class = {}
    local current_label
    for i = 1, data.__partitionsizes[1] do -- train partition
        current_label = getTrainLabel(trainData, indices[i])[1]
        if by_class[current_label] == nil then
            by_class[current_label] = {i}
        else
            table.insert(by_class[current_label], i)
        end
    end
    -- build dummy dataset
    pruned = {}
    for class, images in pairs(by_class) do
        
        for idx, image in ipairs(images) do
            table.insert(pruned, image)
        end
        
        local class_pop = #images

        while class_pop < max do
            table.insert(pruned, images[torch.random(#images)])
            class_pop = class_pop + 1
        end
    end
    return pruned
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

local classCounts = torch.zeros(43)
for i = 1, trainData:size(1) do
  classCounts[trainData[i][9]+1] = classCounts[trainData[i][9]+1] + 1
end
local largest, smallest = torch.max(classCounts), torch.min(classCounts)

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input  = getTrainSample(trainData, idx, DATA_PATH),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input  = getTestSample(testData, idx),
            target = torch.LongTensor{testData[idx][1]}
     
        }
    end
}


local model = require("models/".. opt.model)
--model:cuda()
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
--criterion:cuda()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

print(model)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

--[[
local input  = torch.CudaTensor()
local target = torch.CudaTensor()
engine.hooks.onSample = function(state)
  input:resize(
      state.sample.input:size()
  ):copy(state.sample.input)
  target:resize(
      state.sample.target:size()
  ):copy(state.sample.target)
  state.sample.input  = input
  state.sample.target = target
end
--]]

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, total_batches)
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

local error_logs = {}
engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
    table.insert(error_logs, meter:value())
end

local epoch = 1
local maxEpochs = opt.nEpochs
while epoch <= maxEpochs do
    
    trainDataset:select('train')
    pruned = prune_dataset(trainDataset, trainData, epoch, maxEpochs, largest, smallest)
    permed = torch.randperm(#pruned)
    total_batches = torch.floor(#pruned/opt.batchsize)
    engine:train{
        network     = model,
        criterion   = criterion,
        iterator    = getIterator(trainDataset, true, pruned, permed),
        optimMethod = optim.sgd,
        maxepoch    = 1,
        config      = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    trainDataset:select('val')
    total_batches = torch.floor(trainDataset:size()/opt.batchsize)
    engine:test{
        network   = model,
        criterion = criterion,
        iterator  = getIterator(trainDataset, false, nil, nil)
    }
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames    = state.sample.target
    local _, pred      = state.network.output:max(2)
    pred               = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, 100)
    batch = batch + 1
end


engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network  = model,
    iterator = getIterator(testDataset, false, nil, nil)
}

model:clearState()
torch.save('./cnn_model', model)
torch.save('./error_logs', error_logs)

print("The End!")
