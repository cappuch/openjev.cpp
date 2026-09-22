#pragma once

#include "json.h"
#include "llama.h"

#include <memory>
#include <string>

class laya_head {
public:
    laya_head(const std::string & path, int n_embd, int threads);
    ~laya_head();
    int max_length() const;
    common_json predict(llama_context * ctx, const llama_model * model, const common_json & request);

private:
    struct impl;
    std::unique_ptr<impl> data;
};
