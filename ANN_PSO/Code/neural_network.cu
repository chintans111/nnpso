#include "neural_network.h"

using namespace std;

__device__
float ActivationFunction(float Input)
{
    float Activation = 1 / (1 + __expf(-Input));
    return Activation;
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

    //Initialize Generator
    curandGenerator_t Gen;
	curandCreateGenerator(&Gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(Gen, time(NULL));
    cout << "CURAND GENERATOR INITIALIZED" << endl;

    //Generate numbers
    curandGenerateUniform(Gen, WeightsAndBiases, TotalWeightsAndBiases);
    this->WeightsAndBiases = WeightsAndBiases;
    cout << "WEIGHTS AND BIASES INITIALIZED ON GPU" << endl;

    //Allocate space for curand states
    curandState *States;
    cudaMalloc((void**)&States, NumParticles * sizeof(curandState));
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
}

// NeuralNetwork::Load()
// Loads the input feature vectors into an array on the CPU and transfers it to
// the GPU. Method of transferring and thus training (with or without streams)
// will vary depending upon the size of input data.
void NeuralNetwork::Load(const char *FileName)
{
    int Size;
    float *Input;
    float *Output;
    int Width = this->InputNeurons;
    fstream FIn;
    FIn.open(FileName);
    if(!FIn.fail())
    {
        cout << "FILE OPENED" << endl;
        FIn >> Size;
        Input = new float[Size * Width];
        Output = new float[Size];
        cout << "SPACE ALLOCATED" << endl;
        int temp;

        for(int i = 0; i < Size; i++)
        {
            for(int j = 0; j < Width; j++)
            {
                FIn >> temp;
                Input[i * Width + j] = float(temp);
            }
            FIn >> temp;
            Output[i] = float(temp);
        }
    }
    FIn.close();

    cout << "INPUT OUTPUT SPACE REQUIRED: " << Size * 24 / 1024 << "KB" << endl;

    cout << "INPUT AND OUTPUT LOADED AND FILE CLOSED" << endl;

    //Transfer to GPU (Single cudaMemcpy for the time being)
    float* DeviceInput;
    cudaMalloc((void**)&DeviceInput, Size * Width * sizeof(float));
    cudaMemcpy(DeviceInput, Input, Size * Width * sizeof(float), cudaMemcpyHostToDevice);
    this->Input = DeviceInput;

    float* DeviceOutput;
    cudaMalloc((void**)&DeviceOutput, Size * sizeof(float));
    cudaMemcpy(DeviceOutput, Output, Size * sizeof(float), cudaMemcpyHostToDevice);
    this->Output = DeviceOutput;

    cout << "INPUT AND OUTPUT TRANSFERRED TO GPU" << endl;
}

// NeuralNetwork::Train()
// Trains the network using PSO and a set number of particles in order to eliminate
// backpropogation.
