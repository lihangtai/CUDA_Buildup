#include <jetson-inference/imageNet.h>
#include <jetson-utils/loadImages.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv){

    if(argc < 2){
        cout << "my-recognition expected image filename as argument" << endl;
        cout << "example usage: ./my-recognition image.jpg" << endl;
        return 0;
    }

    const char* imgFilename = argv[1];
    uchar3* imgPtr = nullptr;
    int imgWidth = 0;
    int imgHeight = 0;

    //load the image from disk in shared memory in uchar3 format
    if(!loadImage(imgFilename, &imgPtr, &imgWidth, &imgHeight)){
        cout << "failed to load image: " << imgFilename << endl;
        return 0; 
    }
    // key step 1: using the create function
    imageNet *net = imageNet::Create(imageNet::GOOGLENET);

    if(!net){
        cout << "failed to laod image recognition network" << endl;
    }

    float confidence = 0.0;
    // key step 2:using the Classify function in TensorRt form 
    const int classIndex = net->Claasify(imgPtr, imgWidth, imgHeight, &condifence);


    if(classIndex >= 0){
        const char* classDescription = net->GetClassDesc(classIndex);
        
        // key step3: get the recogintion result from the ILSVR12 dataset
        cout << "image is recognized as " << classDescription << " with " << confidence * 100 << " confidence" << endl;

    }
    else{
        cout << "failed to classify the image" << endl;
    }

    delete net;
    return 0;

}


