#include <iostream>

#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnOnnxParser/IOnnxParser.hpp"


int main(int argv, char** argc){
	armnnOnnxParser::IOnnxParserPtr parser = armnnOnnxParser::IOnnxParser::Create();
    std::cout << "\nmodel load start";
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile("model.onnx");
    std::cout << "\nmodel load end";

    std::cout << "\nmain end";
	return 0;
}

