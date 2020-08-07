//
// Created by sergio on 12/05/19.
//

#ifndef CPPFLOW_MODEL_H
#define CPPFLOW_MODEL_H

#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#pragma warning(push, 0)
#include <tensorflow/c/c_api.h>
#include "Tensor.h"
#pragma warning(pop)
class Tensor;

class Model {
public:
    // Pass a path to the model file and optional Tensorflow config options. See examples/load_model/main.cpp.
    explicit Model(const std::string& model_filename, const std::vector<uint8_t>& config_options = {});

    // Rule of five, moving is easy as the pointers can be copied, copying not as i have no idea how to copy
    // the contents of the pointer (i guess dereferencing won't do a deep copy)
    Model(const Model &model) = delete;
    Model(Model &&model) = default;
    Model& operator=(const Model &model) = delete;
    Model& operator=(Model &&model) = default;

    ~Model();

    void init();
    void restore(const std::string& ckpt);
    void save(const std::string& ckpt);
	void restore_savedmodel(const std::string& savedmdl);
    std::vector<std::string> get_operations() const;

    // Original Run
    void run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    // Run with references
    void run(Tensor& input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor& output);
    void run(Tensor& input, Tensor& output);

    // Run with pointers
    void run(Tensor* input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor* output);
    void run(Tensor* input, Tensor* output);

private:
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;

    // Read a file from a string
    static TF_Buffer* read(const std::string&);

    bool status_check(bool throw_exc) const;
    void error_check(bool condition, const std::string &error) const;

public:
    friend class Tensor;
};


#endif //CPPFLOW_MODEL_H
