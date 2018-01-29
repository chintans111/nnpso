#include "neural_network.h"
#include <stdio.h>

#define INF 1000000000
//Basic cuda error checking macro
//TODO: Add cuRAND and cuBLAS error checking macros
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

__global__
void Normalize(float *Array, int Number, float MaxValue)
{
    int Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index < Number)
        Array[Index] = (Array[Index] - 0.5f) * MaxValue;
}

__device__
float ActivationFunction(float Input)
{
    float Activation = 1 / (1 + __expf(-Input));
    return Activation;
}

__global__
void TrainKernel(int Epochs, float *WeightsAndBiases, curandState_t *States, int NumParticles, float *FitnessArray)
{
    //Particle index is block index since there is one block for each particle
    int Index = blockIdx.x;

    //Initialize PBest and LBest
    float PersonalBest = INF;
    float LocalBest = INF;
    float Fitness = 0.0f;

    //Set left and right neighbours
	int Left = (NumParticles + Index - 1) % NumParticles;
	int Right = (1 + Index) % NumParticles;

    //Initialize random number generator states
    curand_init(Index, Index, 0, &States[Index]);
    curandState_t LocalState = States[Index];

    //Initialize c1, c2, chi
    float C1 = 2.05f, C2 = 2.05f;
    float Psi = C1 + C2;
    float Chi = abs(2.0f / (2.0f - Psi - sqrt(Psi * Psi - 4.0f * Psi)));
    // printf("PSI: %.5f\t\tCHI: %.5f\n", Psi, Chi);
	float R1, R2;

    //For each epoch
    for(int i = 0; i < Epochs; i++)
    {
        Fitness = 0.0f;

        //Calculate fitness, i.e. loss (MSE?)
        //Main feed forward work to be done here


        if(Fitness < PersonalBest)
        {
            PersonalBest = Fitness;
            //Copy personal best weights and biases
            //Another small kernel launch/iteration?
        }

        //Update local best particle index (left or right)
        // if(PersonalBest > PersonalBestArray[Left])
        //     LocalBest = Left;
        // if(PersonalBest > PersonalBestArray[Right])
        //     LocalBest = Right;

        //Update weights and biases of each particle
    }

    //Find the global best particle (here? or defer to CPU?)
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
    int TotalWeightsAndBiasesPerParticle = ((InputNeurons + 1) * HiddenNeurons)
                                    + (((HiddenNeurons +1) * HiddenNeurons)
                                        * (HiddenLayers - 1))
                                    + ((HiddenNeurons + 1) * OutputNeurons);

    //Total
    int TotalWeightsAndBiases = NumParticles * TotalWeightsAndBiasesPerParticle;

    cout << "TOTAL SPACE FOR WEIGHTS AND BIASES: " << TotalWeightsAndBiases * 4 / 1024 << "KB" << endl;

    //Allocate device memory for weights and biases
    float *WeightsAndBiases;
    cudaMalloc((void**)&WeightsAndBiases, TotalWeightsAndBiases * sizeof(float));
    cout << "GPU SPACE ALLOCATED FOR WEIGHTS AND BIASES" << endl;

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
    dim3 Grid(TotalWeightsAndBiasesPerParticle, 1, 1);
    dim3 Block(NumParticles, 1, 1);

    //Generate weights and biases
    curandGenerateUniform(Gen, WeightsAndBiases, TotalWeightsAndBiases);
    Normalize <<<Grid, Block>>> (WeightsAndBiases, TotalWeightsAndBiases, 3.0f);
    this->WeightsAndBiases = WeightsAndBiases;
    cout << "WEIGHTS AND BIASES INITIALIZED ON GPU" << endl;

    //Generate velocities
    curandGenerateUniform(Gen, Velocities, TotalWeightsAndBiases);
    Normalize <<<Grid, Block>>> (Velocities, TotalWeightsAndBiases, 5.0f);
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

    //Weights and Biases to be stored in a single contiguous array
    //In Column Major format(?) (Verify) (Not necessary since only random init taking place here)

    //Pointers to required positions
    this->InputHidden = this->WeightsAndBiases;
    this->HiddenHidden = this->InputHidden
                        + ((InputNeurons + 1) * HiddenNeurons) * NumParticles;
    this->HiddenOutput = this->HiddenHidden
                        + (((HiddenNeurons +1) * HiddenNeurons)
                        * (HiddenLayers - 1)) * NumParticles;
    cout << "LAYER POINTERS SET" << endl;

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
void NeuralNetwork::Train(int Epochs)
{
    dim3 Grid(this->NumParticles, 1, 1);
    dim3 Block(1, 1, 1);

    cout << "GRID AND BLOCK SIZE INITIALIZED" << endl;

    //Allocate space for each particle's fitness
    float *Fitness;
    cudaMalloc((void**)&Fitness, this->NumParticles);

    //Training kernel
    TrainKernel <<<Grid, Block>>> (Epochs, this->WeightsAndBiases, this->States, this->NumParticles, this->FitnessArray);
    cudaDeviceSynchronize();
}
