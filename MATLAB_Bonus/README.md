### MATLAB Solution Requirements:

- MATLAB 2020a (Deep Learning Toolbox, Computer Vision Toolbox, Image Processing Toolbox)
- A system with Nvidia GPU (with 16GB memory) and CUDA toolkit installed

### Steps for running the codes:

Step 1: Importing the pretrained network

-	Convert the pretrained PyTorch model to ONNX (use 'Convert_onnxruntime.py')
-	Convert the ONNX model to MATLAB lgraph (use 'convert_onnx_to_lgraph.mlx')
	
Step 2: Training the model ("main.mlx")

-	Place the original training and testing videos on "/Data/Train_videos/" and "/Data/Test_videos/" directories, respectively.
-	Select a random generation seed (we used 7 and 13) for splitting the training and validation data in section 5
-	In section 7, you can either train the model or load the two trained models (R7 or R13 based on random seed selected in the previous step)
-	The prediction for training data and testing data will be generated in Section 9 and will be saved in "/Results/" directory.
	
Step 3: Postprocessing ("postprocess.mlx")

-	Using boosted trees algorithm, the predictions from the two different random partitioning are combined and adjusted. 
	The final submission csv file will be saved in "/Results/" directory. 

The code for this solution is shared as both `.mlx` files and `.m` files.

- `.mlx` files: MATLAB live scripts. To access these files you have to download or clone the repo, and open in MATLAB.
- `.m` files: MATLAB code viewable directly on GitHub