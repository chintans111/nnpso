#include "neural_network.h"
#include <stdio.h>

#define INF 1000000000.0f
//Basic cuda error checking macro
//TODO: Add cuRAND and cuBLAS error checking macros
//TODO: Wrap all calls in relevant error checking macros
#define cudaCheckError()\
{\
	cudaError_t e = cudaGetLastError();\
	if(e != cudaSuccess)\
	{\
		printf("CUDA failure: %s%d: %s", __FILE__, __LINE__, cudaGetErrorString(e));\
		exit(EXIT_FAILURE);\
	}\
}

using namespace std;

typedef struct PSOParameters
{
	curandState_t *States = NULL;
	int NumParticles = 0;
	float *FitnessArray = NULL;
	float *PersonalBestWeights = NULL;
	float *Velocities = NULL;
	float C1 = 0.0f;
	float C2 = 0.0f;
	float XMax = 0.0f;
	float VMax = 0.0f;
} PSOParameters;

typedef struct NNParameters
{
	int Epochs = 0;
	int InputNeurons = 0;
	int HiddenLayers = 0;
	int HiddenNeurons = 0;
	int OutputNeurons = 0;
	int NetworkSize = 0;
	int MaxIOLength = 0;
	int NumVectors = 0;
	float *WeightsAndBiases = NULL;
	float *InputFeatures = NULL;
	float *IntermediateIO = NULL;
	float *OutputFeatures = NULL;
} NNParameters;

// Normalizes a vector to [-MaxValue, MaxValue]
__global__
void Normalize(float *Array, int Number, float MaxValue)
{
    int Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index < Number)
        Array[Index] = (Array[Index] - 0.5f) * MaxValue;
}

// Small kernel for device to device memory transfers
__global__
void DeviceToDevice(float *Destination, float *Source, int Size)
{
	int Index = blockIdx.x * blockDim.x + threadIdx.x;
	if(Index < Size)
		Destination[Index] = Source[Index];
}

// Sigmoid activation function
__global__
void Sigmoid(float *Input, int Size)
{
	int Index = blockIdx.x * blockDim.x + threadIdx.x;
	if(Index < Size)
    	Input[Index] = (1 / (1 + __expf(-Input[Index])));
}

// Reduce kernel
// __global__
// void Reduce(float *FitnessArray, float *PersonalBestWeights, int *Indices, int NetworkSize, int NumParticles, bool IsStage2)
// {
// 	int Index = blockDim.x * blockIdx.x + threadIdx.x;
// 	int Id = blockIdx.x;
//
// 	//Declare shared memory for staging the reduce phase
// 	__shared__ float Stage[512];
// 	__shared__ int Best[512];
//
// 	//Copy fitness and indices to shared memory
// 	Best[Id] = Indices[Index];
// 	Stage[Id] = FitnessArray[Index];
// 	__syncthreads();
//
// 	//Perform the actual reduce
// 	for (unsigned int Step = blockDim.x / 2; Step > 0; Step >>= 1)
// 	{
// 		if (Id < Step && Index + Step < NumParticles)
// 		{
// 			if (Stage[Id] > Stage[Id + Step])
// 			{
// 				Stage[Id] = Stage[Id + Step];
// 				Best[Id] = Best[Id + Step];
// 			}
// 		}
// 		__syncthreads();
// 	}
//
// 	//Copy results back into global memory
// 	if (Id == 0)
// 	{
// 		FitnessArray[Id] = Stage[0];
// 		printf("FITNESS ARRAY: %f\n", FitnessArray[Id]);
// 		Indices[Id] = Best[0];
// 	}
//
// 	//Copy particle co-ordinates to first location for stage 2
// 	if (IsStage2)
// 	{
// 		float *ActualBestWeights = PersonalBestWeights + NetworkSize * Indices[0];
// 		printf("BEST: %d\n", Indices[0]);
// 		DeviceToDevice <<<(NetworkSize - 1) / 32 + 1, 32>>> (PersonalBestWeights, ActualBestWeights, NetworkSize);
// 	}
// }

// Kernel which actually trains the data.
__global__
void TrainKernel(NNParameters NNParams, PSOParameters PSOParams)
{
    //Particle index is block index since there is one block for each particle
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
	int Id = 0;

	if(Index < PSOParams.NumParticles)
	{
		//Initialize PBest, LBest and fitness
	    float PersonalBest = INF;
		float PersonalBestX = INF;
		int LocalBestIndex = Index;
		float LocalBestX = INF;
	    float Fitness = 0.0f;

	    //Initialize chi, declare r1, r2
	    float Psi = PSOParams.C1 + PSOParams.C2;
	    float Chi = abs(2.0f / (2.0f - Psi - sqrt(Psi * Psi - 4.0f * Psi)));
		float R1, R2;

	    //Set left and right neighbours
		int Left = (PSOParams.NumParticles + Index - 1) % PSOParams.NumParticles;
		int Right = (1 + Index) % PSOParams.NumParticles;

	    //Initialize random number generator states
	    curand_init(Index, Index, 0, &PSOParams.States[Index]);
	    curandState_t LocalState = PSOParams.States[Index];

		//Pointer to weights and biases
		float *WeightsAndBiases = &NNParams.WeightsAndBiases[Index * NNParams.NetworkSize];
		float *PersonalBestWeights = &PSOParams.PersonalBestWeights[Index * NNParams.NetworkSize];

		//Input, output, matrix and temporary pointers
		float *Input;
		float *Output;
		float *Matrix;
		float *Temp;

		//Grid and block for network sized transfers
		dim3 NetworkGrid(NNParams.NetworkSize / 256 + 1, 1, 1);
		dim3 NetworkBlock(256, 1, 1);

		//cuBLAS handle initialization
		cublasHandle_t Handle;
		cublasCreate(&Handle);

		//Alpha and beta values
		float Alpha = 1.0f;
	    float Beta = 0.0f;

	    //For each epoch
	    for(int k = 0; k < NNParams.Epochs; k++)
	    {
			// if(Index == 0)
			// 	printf("Epoch: %d\n", k + 1);
			// cudaDeviceSynchronize();

			Fitness = 0.0f;

			//Main feed forward work to be done here
			//Calculate fitness, i.e. loss (MSE?)
	        for(int j = 0; j < 100; j++)
			{
				//Input hidden multiplication + biases
				Input = &NNParams.InputFeatures[NNParams.InputNeurons * j];
				Output = &NNParams.IntermediateIO[NNParams.MaxIOLength * Index];
				Matrix = &NNParams.WeightsAndBiases[NNParams.NetworkSize * Index];

				cublasSgemv(Handle, CUBLAS_OP_N,
					NNParams.HiddenNeurons, NNParams.InputNeurons, &Alpha,
					Matrix, NNParams.HiddenNeurons, Input, 1, &Beta, Output, 1);
				cudaDeviceSynchronize();

				Matrix += NNParams.InputNeurons * NNParams.HiddenNeurons;

				//Add biases
				cublasSaxpy(Handle, NNParams.HiddenNeurons,
					&Alpha, Matrix, 1, Output, 1);
				cudaDeviceSynchronize();

				//Activation function
				Sigmoid <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);
				cudaDeviceSynchronize();

				Input = Output + NNParams.MaxIOLength / 2;
				Matrix += NNParams.HiddenNeurons;

				//Hidden hidden loop
				for(int c = 1; c < NNParams.HiddenLayers; c++)
				{
					//Swap input and output
					Temp = Input;
					Input = Output;
					Output = Temp;

					//Multiply
					cublasSgemv(Handle, CUBLAS_OP_N,
						NNParams.HiddenNeurons, NNParams.HiddenNeurons, &Alpha,
						Matrix, NNParams.HiddenNeurons, Input, 1, &Beta, Output, 1);
					cudaDeviceSynchronize();

					Matrix += NNParams.HiddenNeurons * NNParams.HiddenNeurons;

					//Add biases
					cublasSaxpy(Handle, NNParams.HiddenNeurons,
						&Alpha, Matrix, 1, Output, 1);
					cudaDeviceSynchronize();

					//Activation function
					Sigmoid <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.HiddenNeurons);
					cudaDeviceSynchronize();

					Matrix += NNParams.HiddenNeurons;
				}

				//Hidden output multiplication + biases
				//Multiply
				cublasSgemv(Handle, CUBLAS_OP_N,
					NNParams.OutputNeurons, NNParams.HiddenNeurons, &Alpha,
					Matrix, NNParams.OutputNeurons, Input, 1, &Beta, Output, 1);
				cudaDeviceSynchronize();

				Matrix += NNParams.HiddenNeurons * NNParams.OutputNeurons;

				//Add biases
				cublasSaxpy(Handle, NNParams.OutputNeurons,
					&Alpha, Matrix, 1, Output, 1);
				cudaDeviceSynchronize();

				//Activation function
				Sigmoid <<<(NNParams.HiddenNeurons - 1) / 32 + 1, 32>>> (Output, NNParams.OutputNeurons);
				cudaDeviceSynchronize();

				Fitness += (NNParams.OutputFeatures[j] - Output[0]) * (NNParams.OutputFeatures[j] - Output[0]);
			}

			Fitness /= 100;
			// if(Index == 0)
				printf("PARTICLE: %d\t\tFITNESS: %f\n", Index, PersonalBest);
			__syncthreads();

			//Compare fitness to personal best so far
	        if(Fitness < PersonalBest)
	        {
				//Copy personal best values
	            PersonalBest = Fitness;
				PSOParams.FitnessArray[Index] = Fitness;
	            //Copy personal best weights and biases
				//Device to device transfer
				DeviceToDevice <<<NetworkGrid, NetworkBlock>>> (PersonalBestWeights, WeightsAndBiases, NNParams.NetworkSize);
				cudaDeviceSynchronize();
	        }
			__syncthreads();
	        //Update local best particle index (left or right)
	        if(PersonalBest > PSOParams.FitnessArray[Left])
	            LocalBestIndex = Left;
	        if(PersonalBest > PSOParams.FitnessArray[Right])
	            LocalBestIndex = Right;
			__syncthreads();

	        //Update weights and biases of each particle
			for (int i = 0; i < NNParams.NetworkSize; i++)
			{
				//Set index at which position needs to be updated
				Id = Index * NNParams.NetworkSize + i;

				//Set local best and personal best X (weights / biases)
				LocalBestX = PSOParams.PersonalBestWeights[LocalBestIndex * NNParams.NetworkSize + i];
				PersonalBestX = PSOParams.PersonalBestWeights[Index * NNParams.NetworkSize + i];

				//Generate random numbers
				R1 = curand_uniform(&LocalState);
				R2 = curand_uniform(&LocalState);

				//Update the velocity
				// velocity[id] = chi * (velocity[id] + (c1 * r1 * (p_best_x[id] - pos[id])) + (c2 * r2 * (p_best_x[local_best] - pos[id])));
				PSOParams.Velocities[Id] = Chi * (PSOParams.Velocities[Id] +
										PSOParams.C1 * R1 * (PersonalBestX - NNParams.WeightsAndBiases[Id]) +
										PSOParams.C2 * R2 * (LocalBestX - NNParams.WeightsAndBiases[Id]));

				//Ensure velocity values are within range
				if (PSOParams.Velocities[Id] > PSOParams.VMax)
					PSOParams.Velocities[Id] = PSOParams.VMax;
				if (PSOParams.Velocities[Id] < -PSOParams.VMax)
					PSOParams.Velocities[Id] = -PSOParams.VMax;

				//Update the position
				NNParams.WeightsAndBiases[Id] = NNParams.WeightsAndBiases[Id] + PSOParams.Velocities[Id];

				//Ensure position values are within range
				// if (NNParams.WeightsAndBiases[Id] > PSOParams.XMax)
				// {
				// 	NNParams.WeightsAndBiases[Id] = PSOParams.XMax;
				// 	PSOParams.Velocities[Id] = 0.0f;
				// }
				// if (NNParams.WeightsAndBiases[Id] < -PSOParams.XMax)
				// {
				// 	NNParams.WeightsAndBiases[Id] = -PSOParams.XMax;
				// 	PSOParams.Velocities[Id] = 0.0f;
				// }
				__syncthreads();
			}
	    }
	}
}

NeuralNetwork::NeuralNetwork(int InputNeurons, int HiddenLayers, int HiddenNeurons, int OutputNeurons, int NumParticles)
{
    //NN hyperparameters
    this->InputNeurons = InputNeurons;
    this->HiddenLayers = HiddenLayers;
    this->HiddenNeurons = HiddenNeurons;
    this->OutputNeurons = OutputNeurons;
    this->NumParticles = NumParticles;
    cout << "HYPERPARAMETERS SET" << endl;

    //Initialize random weights and biases on the GPU
    //Calculate total number of weights and biases for memory allocation
    int NetworkSize = ((InputNeurons + 1) * HiddenNeurons)
                                    + (((HiddenNeurons +1) * HiddenNeurons)
                                        * (HiddenLayers - 1))
                                    + ((HiddenNeurons + 1) * OutputNeurons);
	this->NetworkSize = NetworkSize;

    //Total
    int TotalWeightsAndBiases = NumParticles * NetworkSize;

    cout << "TOTAL SPACE FOR WEIGHTS AND BIASES: " << TotalWeightsAndBiases * 4 / 1024 << "KB" << endl;

    //Allocate device memory for weights and biases
    float *WeightsAndBiases;
    cudaMalloc((void**)&WeightsAndBiases, TotalWeightsAndBiases * sizeof(float));
    cout << "GPU SPACE ALLOCATED FOR WEIGHTS AND BIASES" << endl;

	//Allocate device memory for weights and biases
    float *PersonalBestWeights;
    cudaMalloc((void**)&PersonalBestWeights, TotalWeightsAndBiases * sizeof(float));
    cout << "GPU SPACE ALLOCATED FOR PERSONAL BEST WEIGHTS AND BIASES" << endl;

	//Max space to be allocated to intermediate I/O
	int MaxIOLength = 2 * max(InputNeurons, max(HiddenNeurons, OutputNeurons));
	this->MaxIOLength = MaxIOLength;
	float *IntermediateIO;
	cudaMalloc((void**)&IntermediateIO, MaxIOLength * sizeof(float));
	this->IntermediateIO = IntermediateIO;

    //Allocate device memory for velocities
    float *Velocities;
    cudaMalloc((void**)&Velocities, TotalWeightsAndBiases * sizeof(float));
    cout << "GPU SPACE ALLOCATED FOR VELOCITIES" << endl;

    //Allocate device memory for velocities
    float *FitnessArray;
    cudaMalloc((void**)&FitnessArray, NumParticles * sizeof(float));
    this->FitnessArray = FitnessArray;
    cout << "GPU SPACE ALLOCATED FOR FITNESS VALUES" << endl;

    //Initialize generator
    curandGenerator_t Gen;
	curandCreateGenerator(&Gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(Gen, time(NULL));
    cout << "CURAND GENERATOR INITIALIZED" << endl;

    //Dim3 variables for Normalize kernel
    dim3 Grid(NetworkSize, 1, 1);
    dim3 Block(NumParticles, 1, 1);

    //Generate weights and biases
    curandGenerateUniform(Gen, WeightsAndBiases, TotalWeightsAndBiases);
    Normalize <<<Grid, Block>>> (WeightsAndBiases, TotalWeightsAndBiases, 1.0f);
    this->WeightsAndBiases = WeightsAndBiases;
    cout << "WEIGHTS AND BIASES INITIALIZED ON GPU" << endl;

	//Copy generated weights and biases to personal best array for initialization
	DeviceToDevice <<<Grid, Block>>> (PersonalBestWeights, WeightsAndBiases, TotalWeightsAndBiases);
	this->PersonalBestWeights = PersonalBestWeights;

    //Generate velocities
    curandGenerateUniform(Gen, Velocities, TotalWeightsAndBiases);
    Normalize <<<Grid, Block>>> (Velocities, TotalWeightsAndBiases, 0.1f);
    this->Velocities = Velocities;
    cout << "VELOCITIES INITIALIZED ON GPU" << endl;

    //Allocate space for curand states
    curandState_t *States;
    cudaMalloc((void**)&States, NumParticles * sizeof(curandState_t));
    this->States = States;
    cout << "SPACE ALLOCATED FOR CURAND STATES" << endl;

    // float* temp = new float[TotalWeightsAndBiases];
    // cudaMemcpy(temp, WeightsAndBiases, TotalWeightsAndBiases * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < TotalWeightsAndBiases; i++)
    //     cout << temp[i] << endl;

    //Synchronize all kernel calls upto this point
    cudaDeviceSynchronize();
}

// NeuralNetwork::Load()
// Loads the input feature vectors into an array on the CPU and transfers it to
// the GPU. Method of transferring and thus training (with or without streams)
// will vary depending upon the size of input data.
void NeuralNetwork::Load(const char *FileName)
{
    int Size;
    float *InputFeatures;
    float *OutputFeatures;
    int Width = this->InputNeurons;
    fstream FIn;
    FIn.open(FileName);
    if(!FIn.fail())
    {
        cout << "FILE OPENED" << endl;
        FIn >> Size;
        InputFeatures = new float[Size * Width];
        OutputFeatures = new float[Size];
        cout << "SPACE ALLOCATED" << endl;
        int temp;

        for(int i = 0; i < Size; i++)
        {
            for(int j = 0; j < Width; j++)
            {
                FIn >> temp;
                InputFeatures[i * Width + j] = float(temp);
            }
            FIn >> temp;
            OutputFeatures[i] = float(temp);
        }
    }
    FIn.close();

    cout << "INPUT OUTPUT SPACE REQUIRED: " << Size * 24 / 1024 << "KB" << endl;
	this->NumVectors = Size;

    cout << "INPUT AND OUTPUT LOADED AND FILE CLOSED" << endl;

    //Transfer to GPU (Single cudaMemcpy() for the time being)
    float* DeviceInputFeatures;
    cudaMalloc((void**)&DeviceInputFeatures, Size * Width * sizeof(float));
    cudaMemcpy(DeviceInputFeatures, InputFeatures, Size * Width * sizeof(float), cudaMemcpyHostToDevice);
    this->InputFeatures = DeviceInputFeatures;

    float* DeviceOutputFeatures;
    cudaMalloc((void**)&DeviceOutputFeatures, Size * sizeof(float));
    cudaMemcpy(DeviceOutputFeatures, OutputFeatures, Size * sizeof(float), cudaMemcpyHostToDevice);
    this->OutputFeatures = DeviceOutputFeatures;

    cout << "INPUT AND OUTPUT TRANSFERRED TO GPU" << endl;
}

// NeuralNetwork::Train()
// Trains the network using PSO and a set number of particles in order to eliminate
// backpropogation.
// Assumes weight matrix to be in column major format.
void NeuralNetwork::Train(int Epochs)
{
	//One block per particle and one thread per block TODO: change
    dim3 Grid((this->NumParticles - 1) / 32 + 1, 1, 1);
    dim3 Block(32, 1, 1);

    cout << "GRID AND BLOCK SIZE INITIALIZED" << endl;

	//NN parameters struct
	NNParameters NNParams;
	NNParams.Epochs = Epochs;
	NNParams.InputNeurons = this->InputNeurons;
	NNParams.HiddenLayers = this->HiddenLayers;
	NNParams.HiddenNeurons = this->HiddenNeurons;
	NNParams.OutputNeurons = this->OutputNeurons;
	NNParams.NetworkSize = this->NetworkSize;
	NNParams.MaxIOLength = this->MaxIOLength;
	NNParams.NumVectors = this->NumVectors;
	NNParams.InputFeatures = this->InputFeatures;
	NNParams.IntermediateIO = this->IntermediateIO;
	NNParams.OutputFeatures = this->OutputFeatures;
	NNParams.WeightsAndBiases = this->WeightsAndBiases;

	//PSO parameters struct
	PSOParameters PSOParams;
	PSOParams.NumParticles = this->NumParticles;
	PSOParams.C1 = 2.05f;
	PSOParams.C2 = 2.05f;
	PSOParams.XMax = 1.0f;
	PSOParams.VMax = 0.1f;
	PSOParams.FitnessArray = this->FitnessArray;
	PSOParams.States = this->States;
	PSOParams.PersonalBestWeights = this->PersonalBestWeights;
	PSOParams.Velocities = this->Velocities;

	cout << "SIZE OF NNPARAMS: " << sizeof(NNParams) << endl;
	cout << "SIZE OF PSOPARAMS: " << sizeof(PSOParams) << endl;

    //Training kernel
    TrainKernel <<<Grid, Block>>> (NNParams, PSOParams);
    cudaDeviceSynchronize();

	int *Indices;
	int *HostIndices = new int[this->NumParticles];
	for(int i = 0; i < this->NumParticles; i++)
		HostIndices[i] = i;
	cudaMalloc((void**)&Indices, this->NumParticles * sizeof(int));
	cudaMemcpy(Indices, HostIndices, this->NumParticles * sizeof(int), cudaMemcpyHostToDevice);

	//Reduce to find minimum
	// Reduce <<<Grid, Block>>> (PSOParams.FitnessArray, PSOParams.PersonalBestWeights, Indices, this->NetworkSize, this->NumParticles, false);
	// cudaDeviceSynchronize();
	// Reduce <<<1, (this->NumParticles - 1) / 32 + 1>>> (PSOParams.FitnessArray, PSOParams.PersonalBestWeights, Indices, this->NetworkSize, this->NumParticles, true);
	// cudaDeviceSynchronize();
}
