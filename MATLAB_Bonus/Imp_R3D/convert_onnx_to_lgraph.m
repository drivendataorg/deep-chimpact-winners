%% Converting ONNX to lgraph
%% 1 Importing Network from ONNX to MATLAB.

net = importONNXLayers("./convert_torch_to_onnx/R3D.onnx");
analyzeNetwork(net)
%% 2 Modifying the Imported Network
% Replacing input layer and removing flatten layer.

input_layer=[image3dInputLayer([15 180 320 3],"Name","input","Normalization","rescale-symmetric")];
lgraph = replaceLayer(net,'input',input_layer);
lgraph=removeLayers(lgraph,'Flatten_46');
lgraph=connectLayers(lgraph,'GlobalAveragePool_45','Gemm_47');
analyzeNetwork(lgraph)
save("lgarph_R3D.mat","lgraph",'-mat')