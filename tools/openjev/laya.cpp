#include "laya.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

using json = common_json;

static std::vector<float> probabilities(const std::vector<float> & logits, float temperature = 1.f) {
    for (float x : logits) {
        if (!std::isfinite(x)) { throw std::runtime_error("non-finite Laya output"); }
    }
    const float top = *std::max_element(logits.begin(), logits.end());
    std::vector<float> out(logits.size());
    double sum = 0;
    for (size_t i = 0; i < out.size(); ++i) { sum += out[i] = std::exp((logits[i] - top) / temperature); }
    for (float & p : out) { p /= sum; }
    return out;
}

static std::string sanitize(std::string text) {
    size_t pos = 0;
    while ((pos = text.find("[MASK]", pos)) != std::string::npos) { text.replace(pos, 6, " "); ++pos; }
    return text;
}

struct laya_head::impl {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights{nullptr, ggml_free};
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{nullptr, ggml_backend_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    json cfg;
    int d;

    ggml_tensor * tensor(const std::string & name, int64_t n0, int64_t n1 = 1) const {
        auto * t = ggml_get_tensor(weights.get(), name.c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != n0 || t->ne[1] != n1 || t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::runtime_error("invalid Laya tensor: " + name);
        }
        return t;
    }

    ggml_tensor * linear(ggml_context * c, ggml_tensor * x, const std::string & name, int n) const {
        return ggml_add(c, ggml_mul_mat(c, tensor(name + ".weight", x->ne[0], n), x), tensor(name + ".bias", n));
    }

    ggml_tensor * norm(ggml_context * c, ggml_tensor * x, const std::string & name) const {
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), tensor(name + ".weight", d)), tensor(name + ".bias", d));
    }

    std::vector<float> evaluate(const float * embeddings, int length, int qtype,
                                const std::vector<int32_t> & markers, float & act_probability) {
        const size_t graph_size = 512;
        ggml_init_params params{ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        if (!context) { throw std::runtime_error("cannot allocate Laya graph"); }
        auto * c = context.get();
        auto * input = ggml_new_tensor_2d(c, GGML_TYPE_F32, d, length);
        auto * type = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        auto * positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, markers.size());
        ggml_set_input(input);
        ggml_set_input(type);
        ggml_set_input(positions);
        auto * h = ggml_add(c, input, ggml_get_rows(c, tensor("type_emb.weight", d, 3), type));
        const int heads = d / 64;
        for (int i = 0; i < cfg.at("head_layers").get<int>(); ++i) {
            const std::string prefix = "head.layers." + std::to_string(i);
            auto * x = norm(c, h, prefix + ".norm1");
            auto * qkv = ggml_add(c, ggml_mul_mat(c, tensor(prefix + ".self_attn.in_proj_weight", d, 3 * d), x),
                                 tensor(prefix + ".self_attn.in_proj_bias", 3 * d));
            auto part = [&](int index) {
                auto * v = ggml_cont(c, ggml_view_2d(c, qkv, d, length, qkv->nb[1], index * d * sizeof(float)));
                return ggml_permute(c, ggml_reshape_3d(c, v, 64, heads, length), 0, 2, 1, 3);
            };
            auto * q = part(0);
            auto * k = part(1);
            auto * v = ggml_cont(c, ggml_permute(c, part(2), 1, 0, 2, 3));
            auto * scores = ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, k, q), 1.f / 8.f));
            x = ggml_mul_mat(c, v, scores);
            x = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)), d, length);
            h = ggml_add(c, h, linear(c, x, prefix + ".self_attn.out_proj", d));
            x = norm(c, h, prefix + ".norm2");
            x = ggml_relu(c, linear(c, x, prefix + ".linear1", 4 * d));
            h = ggml_add(c, h, linear(c, x, prefix + ".linear2", d));
        }
        auto * selected = ggml_get_rows(c, h, positions);
        auto * logits = linear(c, ggml_gelu_erf(c, linear(c, norm(c, selected, "scorer.0"), "scorer.1", d)), "scorer.3", 1);
        auto * pooled = ggml_cont(c, ggml_view_1d(c, h, d, 0));
        ggml_set_output(logits);
        ggml_set_output(pooled);
        auto * graph = ggml_new_graph_custom(c, graph_size, false);
        ggml_build_forward_expand(graph, logits);
        ggml_build_forward_expand(graph, pooled);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        if (!ggml_gallocr_alloc_graph(alloc.get(), graph)) { throw std::runtime_error("cannot allocate Laya buffers"); }
        ggml_backend_tensor_set(input, embeddings, 0, size_t(d) * length * sizeof(float));
        const int32_t qt = qtype;
        ggml_backend_tensor_set(type, &qt, 0, sizeof(qt));
        ggml_backend_tensor_set(positions, markers.data(), 0, markers.size() * sizeof(int32_t));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) { throw std::runtime_error("Laya head failed"); }
        std::vector<float> result(markers.size()), features(d + 4);
        ggml_backend_tensor_get(logits, result.data(), 0, result.size() * sizeof(float));
        ggml_backend_tensor_get(pooled, features.data(), 0, d * sizeof(float));
        auto p = probabilities(result);
        double entropy = 0;
        for (float value : p) { entropy -= value * std::log(std::max(value, 1e-9f)); }
        std::sort(p.begin(), p.end(), std::greater<float>());
        features[d] = p[0];
        features[d + 1] = p[0] - p[1];
        features[d + 2] = entropy / std::log(double(p.size()));
        features[d + 3] = p.size() / 255.f;

        auto * act_input = ggml_new_tensor_1d(c, GGML_TYPE_F32, d + 4);
        ggml_set_input(act_input);
        auto * act = linear(c, ggml_gelu_erf(c, linear(c, act_input, "act_head.0", 256)), "act_head.2", 2);
        auto * act_graph = ggml_new_graph_custom(c, 32, false);
        ggml_build_forward_expand(act_graph, act);
        if (!ggml_gallocr_alloc_graph(alloc.get(), act_graph)) { throw std::runtime_error("cannot allocate Laya act buffers"); }
        ggml_backend_tensor_set(act_input, features.data(), 0, features.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), act_graph) != GGML_STATUS_SUCCESS) { throw std::runtime_error("Laya act head failed"); }
        std::vector<float> act_logits(2);
        ggml_backend_tensor_get(act, act_logits.data(), 0, 2 * sizeof(float));
        act_probability = probabilities(act_logits)[0];
        return result;
    }
};

laya_head::laya_head(const std::string & path, int n_embd, int threads) : data(new impl) {
    data->d = n_embd;
    ggml_context * weights = nullptr;
    gguf_init_params params{true, &weights};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(path.c_str(), params), gguf_free);
    data->weights.reset(weights);
    if (!file || !weights) { throw std::runtime_error("cannot load Laya head: " + path); }
    const auto config_key = gguf_find_key(file.get(), "laya.config");
    if (config_key < 0 || gguf_get_kv_type(file.get(), config_key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("missing Laya head configuration");
    }
    data->cfg = json::parse(gguf_get_val_str(file.get(), config_key));
    const auto & cfg = data->cfg;
    if (n_embd != 1024 || cfg.at("head_layers") != 2 || cfg.at("max_len") != 512 || cfg.at("head_max_len") != 192) {
        throw std::runtime_error("unsupported Laya head configuration (expected English checkpoint)");
    }
    auto temps = cfg.value("temperature", std::vector<float>{1, 1, 1});
    if (temps.size() != 3) { throw std::runtime_error("invalid Laya temperatures"); }
    const auto buckets = cfg.value("temperature_by_options", json::object());
    for (const auto & t : buckets.items()) { temps.push_back(t.value().get<float>()); }
    for (float t : temps) {
        if (!std::isfinite(t) || t <= 0) { throw std::runtime_error("invalid Laya temperature"); }
    }
    data->backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!data->backend) { throw std::runtime_error("Laya head requires a CPU backend"); }
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(data->backend.get()));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (set_threads) { set_threads(data->backend.get(), threads); }
    data->buffer.reset(ggml_backend_alloc_ctx_tensors(data->weights.get(), data->backend.get()));
    if (!data->buffer) { throw std::runtime_error("cannot allocate Laya weights"); }
    ggml_backend_buffer_set_usage(data->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream stream(path, std::ios::binary);
    for (int64_t i = 0; i < gguf_get_n_tensors(file.get()); ++i) {
        auto * t = ggml_get_tensor(weights, gguf_get_tensor_name(file.get(), i));
        std::vector<char> bytes(ggml_nbytes(t));
        stream.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), i));
        if (!stream.read(bytes.data(), bytes.size())) { throw std::runtime_error("truncated Laya head"); }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
}

laya_head::~laya_head() = default;
int laya_head::max_length() const { return data->cfg.at("max_len").get<int>(); }

json laya_head::predict(llama_context * ctx, const llama_model * model, const json & request) {
    const auto * vocab = llama_model_get_vocab(model);
    auto encode = [&](const std::string & text) { return common_tokenize(vocab, sanitize(text), false, true); };
    auto special = [&](const char * text) {
        const auto ids = common_tokenize(vocab, text, false, true);
        if (ids.size() != 1) { throw std::runtime_error("missing Laya special token"); }
        return ids[0];
    };
    const auto cls = special("[CLS]"), sep = special("[SEP]"), mask = special("[MASK]");
    if (request.contains("image")) { throw std::runtime_error("Laya does not score images"); }
    const auto state = encode(request.at("state").get<std::string>());
    const auto & questions = request.at("questions");
    if (!questions.is_array() || questions.empty()) { throw std::runtime_error("Laya requires a non-empty questions array"); }
    json rows = json::array();
    size_t token_count = 0;
    for (const auto & question : questions) {
        const std::string kind = question.at("type").get<std::string>();
        const int qt = kind == "choice" ? 0 : kind == "score" ? 1 : kind == "noul" ? 2 : -1;
        if (qt < 0) { throw std::runtime_error("unsupported Laya question type"); }
        const auto options = question.at("options").get<std::vector<std::string>>();
        if (options.size() < 2 || options.size() > 255 || (qt == 2 && options.size() != 2)) {
            throw std::runtime_error("Laya requires 2 to 255 options (exactly 2 for noul)");
        }
        std::vector<std::vector<llama_token>> opts;
        int total = 0;
        for (const auto & text : options) {
            auto ids = encode(" " + text);
            ids.resize(std::min(ids.size(), size_t(48)));
            ids.insert(ids.begin(), mask);
            total += ids.size();
            opts.push_back(std::move(ids));
        }
        const int budget = data->cfg.at("head_max_len").get<int>();
        if (budget - total < 16) {
            const size_t per = std::max(4, (budget - 16) / int(opts.size()));
            total = 0;
            for (auto & ids : opts) { ids.resize(std::min(ids.size(), per)); total += ids.size(); }
        }
        auto head = encode(kind + " question: " + question.at("instr").get<std::string>());
        head.resize(std::min(head.size(), size_t(std::max(8, budget - total))));
        std::vector<llama_token> tokens{cls};
        tokens.insert(tokens.end(), head.begin(), head.end());
        tokens.push_back(sep);
        std::vector<int32_t> markers;
        for (const auto & ids : opts) {
            markers.push_back(tokens.size());
            tokens.insert(tokens.end(), ids.begin(), ids.end());
        }
        tokens.push_back(sep);
        const int room = std::max(0, max_length() - int(tokens.size()) - 1);
        tokens.insert(tokens.end(), state.begin(), state.begin() + std::min(state.size(), size_t(room)));
        tokens.push_back(sep);
        tokens.resize(std::min(tokens.size(), size_t(max_length())));
        if (markers.back() >= int(tokens.size())) { throw std::runtime_error("Laya options do not fit in token budget"); }
        auto batch = llama_batch_init(tokens.size(), 0, 1);
        common_batch_clear(batch);
        for (size_t i = 0; i < tokens.size(); ++i) { common_batch_add(batch, tokens[i], i, {0}, true); }
        const int status = llama_encode(ctx, batch);
        llama_batch_free(batch);
        if (status != 0) { throw std::runtime_error("Laya encoder failed"); }
        std::vector<float> embeddings(tokens.size() * data->d);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto * values = llama_get_embeddings_ith(ctx, i);
            if (!values) { throw std::runtime_error("missing Laya encoder embedding"); }
            std::copy(values, values + data->d, embeddings.begin() + i * data->d);
        }
        float act = 0;
        const auto logits = data->evaluate(embeddings.data(), tokens.size(), qt, markers, act);
        const std::string bucket = kind + ":" + (options.size() <= 2 ? "2" : options.size() <= 5 ? "3-5" : options.size() <= 10 ? "6-10" : "11+");
        const auto temperatures = data->cfg.value("temperature", std::vector<float>{1, 1, 1});
        const float temperature = data->cfg.value("temperature_by_options", json::object()).value(bucket, temperatures[qt]);
        rows.push_back(json::object({{"logits", logits}, {"probabilities", probabilities(logits, temperature)}, {"act_probability", act}}));
        token_count += tokens.size();
    }
    return json::object({{"results", rows}, {"evaluated_tokens", token_count}, {"prefix_tokens", 0}});
}
