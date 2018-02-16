#include "neural_network.h"

using namespace std;

int main()
{
    cout << "CONSTRUCTOR INVOKED" << endl;
    NeuralNetwork NN(5, 3, 3, 1, 32);

    cout << "LOAD INVOKED" << endl;
    NN.Load("train.txt");

    cout << "TRAIN INVOKED" << endl;
    NN.Train(20000, "weights.txt");

    cout << "TEST INVOKED" << endl;
    NN.Test("test.txt", "epochs20k_weights.txt");

    // NN.CheckKernel();

    // cout << "SIZE OF NN PARAMS: " << sizeof(NNParameters) << endl;
    // cout << "SIZE OF PSO PARAMS: " << sizeof(PSOParameters) << endl;
}
