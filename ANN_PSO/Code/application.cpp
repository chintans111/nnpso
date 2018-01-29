#include "neural_network.h"

using namespace std;

int main()
{
    cout << "CONSTRUCTOR INVOKED" << endl;
    NeuralNetwork NN(5, 5, 5, 1, 32);

    cout << "LOAD INVOKED" << endl;
    NN.Load("train.txt");

    // cout << "TRAIN INVOKED" << endl;
    NN.Train(100);
}
