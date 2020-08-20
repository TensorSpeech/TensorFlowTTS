//
// Created by sergio on 12/05/19.
//

#include "../include/Model.h"

Model::Model(const std::string& model_filename, const std::vector<uint8_t>& config_options) {
    this->status = TF_NewStatus();
    this->graph = TF_NewGraph();

    // Create the session.
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();

    if (!config_options.empty())
    {
        TF_SetConfig(sess_opts, static_cast<const void*>(config_options.data()), config_options.size(), this->status);
        this->status_check(true);
    }

	TF_Buffer* RunOpts = NULL;

	const char* tags = "serve";
	int ntags = 1;

	this->session = TF_LoadSessionFromSavedModel(sess_opts, RunOpts, model_filename.c_str(), &tags, ntags, this->graph, NULL, this->status);
	if (TF_GetCode(this->status) == TF_OK)
	{
		printf("TF_LoadSessionFromSavedModel OK\n");
	}
	else
	{
		printf("%s", TF_Message(this->status));
	}
    TF_DeleteSessionOptions(sess_opts);

    // Check the status
    this->status_check(true);

    // Create the graph
	TF_Graph* g = this->graph;


    this->status_check(true);
}

Model::~Model() {
    TF_DeleteSession(this->session, this->status);
    TF_DeleteGraph(this->graph);
    this->status_check(true);
    TF_DeleteStatus(this->status);
}


void Model::init() {
    TF_Operation* init_op[1] = {TF_GraphOperationByName(this->graph, "init")};

    this->error_check(init_op[0]!= nullptr, "Error: No operation named \"init\" exists");

    TF_SessionRun(this->session, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0, init_op, 1, nullptr, this->status);
    this->status_check(true);
}

void Model::save(const std::string &ckpt) {
    // Encode file_name to tensor
    size_t size = 8 + TF_StringEncodedSize(ckpt.length());
    TF_Tensor* t = TF_AllocateTensor(TF_STRING, nullptr, 0, size);
    char* data = static_cast<char *>(TF_TensorData(t));
    for (int i=0; i<8; i++) {data[i]=0;}
    TF_StringEncode(ckpt.c_str(), ckpt.size(), data + 8, size - 8, status);

    memset(data, 0, 8);  // 8-byte offset of first string.
    TF_StringEncode(ckpt.c_str(), ckpt.length(), (char*)(data + 8), size - 8, status);

    // Check errors
    if (!this->status_check(false)) {
        TF_DeleteTensor(t);
        std::cerr << "Error during filename " << ckpt << " encoding" << std::endl;
        this->status_check(true);
    }

    TF_Output output_file;
    output_file.oper = TF_GraphOperationByName(this->graph, "save/Const");
    output_file.index = 0;
    TF_Output inputs[1] = {output_file};

    TF_Tensor* input_values[1] = {t};
    const TF_Operation* restore_op[1] = {TF_GraphOperationByName(this->graph, "save/control_dependency")};
    if (!restore_op[0]) {
        TF_DeleteTensor(t);
        this->error_check(false, "Error: No operation named \"save/control_dependencyl\" exists");
    }


    TF_SessionRun(this->session, nullptr, inputs, input_values, 1, nullptr, nullptr, 0, restore_op, 1, nullptr, this->status);
    TF_DeleteTensor(t);

    this->status_check(true);
}

void Model::restore_savedmodel(const std::string & savedmdl)
{
	


}

void Model::restore(const std::string& ckpt) {

    // Encode file_name to tensor
    size_t size = 8 + TF_StringEncodedSize(ckpt.size());
    TF_Tensor* t = TF_AllocateTensor(TF_STRING, nullptr, 0, size);
    char* data = static_cast<char *>(TF_TensorData(t));
    for (int i=0; i<8; i++) {data[i]=0;}
    TF_StringEncode(ckpt.c_str(), ckpt.size(), data + 8, size - 8, status);

    // Check errors
    if (!this->status_check(false)) {
        TF_DeleteTensor(t);
        std::cerr << "Error during filename " << ckpt << " encoding" << std::endl;
        this->status_check(true);
    }

    TF_Output output_file;
    output_file.oper = TF_GraphOperationByName(this->graph, "save/Const");
    output_file.index = 0;
    TF_Output inputs[1] = {output_file};

    TF_Tensor* input_values[1] = {t};
    const TF_Operation* restore_op[1] = {TF_GraphOperationByName(this->graph, "save/restore_all")};
    if (!restore_op[0]) {
        TF_DeleteTensor(t);
        this->error_check(false, "Error: No operation named \"save/restore_all\" exists");
    }



    TF_SessionRun(this->session, nullptr, inputs, input_values, 1, nullptr, nullptr, 0, restore_op, 1, nullptr, this->status);
    TF_DeleteTensor(t);

    this->status_check(true);
}

TF_Buffer *Model::read(const std::string& filename) {
    std::ifstream file (filename, std::ios::binary | std::ios::ate);

    // Error opening the file
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return nullptr;
    }


    // Cursor is at the end to get size
    auto size = file.tellg();
    // Move cursor to the beginning
    file.seekg (0, std::ios::beg);

    // Read
    auto data = new char [size];
    file.seekg (0, std::ios::beg);
    file.read (data, size);

    // Error reading the file
    if (!file) {
        std::cerr << "Unable to read the full file: " << filename << std::endl;
        return nullptr;
    }


    // Create tensorflow buffer from read data
    TF_Buffer* buffer = TF_NewBufferFromString(data, size);

    // Close file and remove data
    file.close();
    delete[] data;

    return buffer;
}

std::vector<std::string> Model::get_operations() const {
    std::vector<std::string> result;
    size_t pos = 0;
    TF_Operation* oper;

    // Iterate through the operations of a graph
    while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr) {
        result.emplace_back(TF_OperationName(oper));
    }

    return result;
}

void Model::run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    this->error_check(std::all_of(inputs.begin(), inputs.end(), [](const Tensor* i){return i->flag == 1;}),
            "Error: Not all elements from the inputs are full");

    this->error_check(std::all_of(outputs.begin(), outputs.end(), [](const Tensor* o){return o->flag != -1;}),
            "Error: Not all outputs Tensors are valid");


    // Clean previous stored outputs
    std::for_each(outputs.begin(), outputs.end(), [](Tensor* o){o->clean();});

    // Get input operations
    std::vector<TF_Output> io(inputs.size());
    std::transform(inputs.begin(), inputs.end(), io.begin(), [](const Tensor* i) {return i->op;});

    // Get input values
    std::vector<TF_Tensor*> iv(inputs.size());
    std::transform(inputs.begin(), inputs.end(), iv.begin(), [](const Tensor* i) {return i->val;});

    // Get output operations
    std::vector<TF_Output> oo(outputs.size());
    std::transform(outputs.begin(), outputs.end(), oo.begin(), [](const Tensor* o) {return o->op;});

    // Prepare output recipients
    auto ov = new TF_Tensor*[outputs.size()];

    TF_SessionRun(this->session, nullptr, io.data(), iv.data(), inputs.size(), oo.data(), ov, outputs.size(), nullptr, 0, nullptr, this->status);
    this->status_check(true);

    // Save results on outputs and mark as full
    for (std::size_t i=0; i<outputs.size(); i++) {
        outputs[i]->val = ov[i];
        outputs[i]->flag = 1;
        outputs[i]->deduce_shape();
    }

    // Mark input as empty
    std::for_each(inputs.begin(), inputs.end(), [] (Tensor* i) {i->clean();});

    delete[] ov;
}

void Model::run(Tensor &input, Tensor &output) {
    this->run(&input, &output);
}

void Model::run(const std::vector<Tensor*> &inputs, Tensor &output) {
    this->run(inputs, &output);
}

void Model::run(Tensor &input, const std::vector<Tensor*> &outputs) {
    this->run(&input, outputs);
}

void Model::run(Tensor *input, Tensor *output) {
    this->run(std::vector<Tensor*>({input}), std::vector<Tensor*>({output}));
}

void Model::run(const std::vector<Tensor*> &inputs, Tensor *output) {
    this->run(inputs, std::vector<Tensor*>({output}));
}

void Model::run(Tensor *input, const std::vector<Tensor*> &outputs) {
    this->run(std::vector<Tensor*>({input}), outputs);
}

bool Model::status_check(bool throw_exc) const {

    if (TF_GetCode(this->status) != TF_OK) {
        if (throw_exc) {
			const char* errmsg = TF_Message(status);
			printf(errmsg);
            throw std::runtime_error(errmsg);
        } else {
            return false;
        }
    }
    return true;
}

void Model::error_check(bool condition, const std::string &error) const {
    if (!condition) {
        throw std::runtime_error(error);
    }
}
