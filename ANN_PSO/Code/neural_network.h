#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <math.h>

class NeuralNetwork
{
private:
    int InputNeurons, HiddenLayers, HiddenNeurons, OutputNeurons;
    int NumParticles;
    float *Velocities, *FitnessArray, *PersonalBestWeights;
    float *WeightsAndBiases;
    float *InputFeatures, *OutputFeatures;
    curandState_t *States;

public:
    //Randomly initialize weights and biases for all particles of the swarm
    NeuralNetwork(int, int, int, int, int);

    //Load()
    //Load data from a file into the main memory/GPU memory (as needed)
    //Reshape data if needed (especially separating input features from output labels)
    //Set up streams later on if needed
    void Load(const char *);

    //Train()
    //FeedForward combined with PSO
    //Number of particles taken from constructor
    void Train(int);


    //Test()
    //Use the best set of weights and biases amongst all particles


    //Dump()
    //Dump the best set of weights and biases to a file
};

#endif
