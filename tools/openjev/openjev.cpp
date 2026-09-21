#include "common.h"
#include "json.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;
static const char * labels[] = { "contradiction", "entailment", "neutral" };
static const char * nli_template = "Premise: {premise}\nHypothesis: {hypothesis}";

struct options {
    std::string model, mmproj, input;
    int context = 4096, batch = 512, threads = 4, gpu_layers = 99;
    bool prefix = true, latents = false;
};

static void usage() {
    std::cout << "openjev.cpp - Qwen3.5 NLI cross-encoder\n"
                 "Usage: openjev -m MODEL.gguf [options] < requests.jsonl\n"
                 "  --mmproj FILE       vision projector for image requests\n"
                 "  --input FILE        read JSONL from a file instead of stdin\n"
                 "  -c, --ctx-size N    maximum tokens per pair (default 4096)\n"
                 "  -b, --batch-size N  prefill batch size (default 512)\n"
                 "  -t, --threads N     CPU threads (default 4)\n"
                 "  -ngl N              GPU layers (default 99; 0 for CPU)\n"
                 "  --no-prefix-cache  evaluate each pair independently\n"
                 "  --latents          return final-token hidden states\n"
                 "Requests: {\"premise\":\"...\",\"hypotheses\":[\"...\"]}\n"
                 "       or {\"pairs\":[[\"premise\",\"hypothesis\"]]}\n"
                 "       or {\"question\":\"...\",\"options\":[\"...\"]}\n"
                 "       or {\"question\":\"...\",\"reference\":\"...\",\"candidate\":\"...\"}\n"
                 "Optional image: {\"image\":\"scene.png\",\"premise\":\"...\",\"hypotheses\":[\"...\"]}\n";
}

static std::string trim(const std::string & s) {
    const auto first = s.find_first_not_of(" \t\r\n\f\v");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n\f\v") - first + 1);
}

static std::string format_pair(const std::string & premise, const std::string & hypothesis) {
    return "Premise: " + trim(premise) + "\nHypothesis: " + trim(hypothesis);
}

struct batch_guard {
    llama_batch value;
    explicit batch_guard(int n) : value(llama_batch_init(n, 0, 1)) {}
    ~batch_guard() { llama_batch_free(value); }
};

class cross_encoder {
    options opt;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision{nullptr, mtmd_free};
    int n_batch = 512, n_seq_max = 5;
    uint32_t n_ctx = 0;
    size_t evaluated_tokens = 0;

    std::vector<llama_token> tokenize(const std::string & text) const {
        auto result = common_tokenize(llama_model_get_vocab(model.get()), text, false, true);
        if (result.empty() || result.size() > size_t(opt.context)) {
            throw std::runtime_error("pair exceeds --ctx-size (no silent truncation)");
        }
        return result;
    }

    void decode_packed(const std::vector<std::vector<llama_token>> & rows, size_t lo, size_t hi,
                       size_t begin, size_t end, const std::vector<llama_seq_id> & seqs) {
        auto row_end = [&](size_t r) {
            return end == 0 ? rows[r].size() : std::min(end, rows[r].size());
        };
        size_t end_max = begin;
        for (size_t r = lo; r < hi; ++r) { end_max = std::max(end_max, row_end(r)); }
        batch_guard batch(n_batch);
        size_t cursor = begin;
        while (cursor < end_max) {
            common_batch_clear(batch.value);
            while (cursor < end_max) {
                int add = 0;
                for (size_t r = lo; r < hi; ++r) {
                    if (cursor < row_end(r)) { ++add; }
                }
                if (add == 0) { ++cursor; continue; }
                if (batch.value.n_tokens > 0 && batch.value.n_tokens + add > n_batch) { break; }
                for (size_t r = lo; r < hi; ++r) {
                    if (cursor < row_end(r)) {
                        common_batch_add(batch.value, rows[r][cursor], llama_pos(cursor), {seqs[r - lo]}, true);
                    }
                }
                ++cursor;
            }
            if (batch.value.n_tokens == 0) { break; }
            if (llama_decode(ctx.get(), batch.value) != 0) {
                throw std::runtime_error("llama_decode failed");
            }
            evaluated_tokens += size_t(batch.value.n_tokens);
        }
    }

    json result(llama_seq_id seq) const {
        const float * values = llama_get_embeddings_seq(ctx.get(), seq);
        if (!values) {
            throw std::runtime_error("missing final-token output");
        }
        const int n = opt.latents ? llama_model_n_embd_out(model.get()) : 3;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(values[i])) {
                throw std::runtime_error("non-finite model output");
            }
        }
        if (opt.latents) {
            return json::object({{"latent", std::vector<float>(values, values + n)}});
        }
        const float maximum = *std::max_element(values, values + 3);
        std::vector<float> probabilities(3);
        float sum = 0;
        for (int i = 0; i < 3; ++i) {
            probabilities[i] = std::exp(values[i] - maximum);
            sum += probabilities[i];
        }
        for (float & p : probabilities) { p /= sum; }
        const int winner = int(std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
        return json::object({{"logits", std::vector<float>(values, values + 3)},
                             {"probabilities", probabilities}, {"label", labels[winner]}});
    }

    json image_pair(const std::string & premise, const std::string & hypothesis, const std::string & path) {
        if (!vision) { throw std::runtime_error("image requests require --mmproj"); }
        auto loaded = mtmd_helper_bitmap_init_from_file(vision.get(), path.c_str(), false, mtmd_helper_init_opt_default());
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(loaded.bitmap, mtmd_bitmap_free);
        if (!bitmap) { throw std::runtime_error("cannot load image: " + path); }
        if (mtmd_bitmap_is_audio(bitmap.get())) { throw std::runtime_error("expected an image, received audio"); }
        std::string p = premise;
        if (p.find(mtmd_default_marker()) == std::string::npos) {
            p = std::string(mtmd_default_marker()) + "\n" + p;
        }
        const std::string text = format_pair(p, hypothesis);
        const mtmd_input_text input{text.c_str(), text.size(), false, true};
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        const mtmd_bitmap * ptr = bitmap.get();
        if (mtmd_tokenize(vision.get(), chunks.get(), &input, &ptr, 1) != 0) {
            throw std::runtime_error("image tokenization failed");
        }
        const auto n_tokens = mtmd_helper_get_n_tokens(chunks.get());
        if (n_tokens > size_t(opt.context)) { throw std::runtime_error("image pair exceeds --ctx-size"); }
        llama_memory_clear(llama_get_memory(ctx.get()), false);
        llama_pos past = 0;
        if (mtmd_helper_eval_chunks(vision.get(), ctx.get(), chunks.get(), 0, 0, opt.batch, true, &past) != 0) {
            throw std::runtime_error("image evaluation failed");
        }
        evaluated_tokens += n_tokens;
        return result(0);
    }

public:
    explicit cross_encoder(const options & args) : opt(args) {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = opt.gpu_layers;
        model.reset(llama_model_load_from_file(opt.model.c_str(), mp));
        if (!model) { throw std::runtime_error("failed to load model"); }
        char value[256];
        if (llama_model_meta_val_str(model.get(), "openjev.nli_template", value, sizeof(value)) < 0 ||
            std::string(value) != nli_template || llama_model_n_cls_out(model.get()) != 3) {
            throw std::runtime_error("expected an openjev GGUF converted with this fork");
        }
        for (int i = 0; i < 3; ++i) {
            const char * label = llama_model_cls_label(model.get(), i);
            if (!label || std::string(label) != labels[i]) { throw std::runtime_error("invalid NLI label order"); }
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = uint32_t(opt.context) * 2;
        cp.n_seq_max = 5;
        cp.kv_unified = true;
        cp.n_batch = cp.n_ctx;
        cp.n_ubatch = uint32_t(opt.batch);
        cp.n_threads = cp.n_threads_batch = opt.threads;
        cp.embeddings = true;
        cp.pooling_type = opt.latents ? LLAMA_POOLING_TYPE_LAST : LLAMA_POOLING_TYPE_RANK;
        cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        ctx.reset(llama_init_from_model(model.get(), cp));
        if (!ctx) { throw std::runtime_error("failed to create context"); }
        n_ctx = llama_n_ctx(ctx.get());
        n_batch = int(llama_n_batch(ctx.get()));
        n_seq_max = int(llama_n_seq_max(ctx.get()));
        if (!opt.mmproj.empty()) {
            auto vp = mtmd_context_params_default();
            vp.use_gpu = opt.gpu_layers != 0;
            vp.n_threads = opt.threads;
            vision.reset(mtmd_init_from_file(opt.mmproj.c_str(), model.get(), vp));
            if (!vision) { throw std::runtime_error("failed to load vision projector"); }
        }
    }

    json predict(const json & request) {
        const auto start = std::chrono::steady_clock::now();
        evaluated_tokens = 0;
        std::vector<std::pair<std::string, std::string>> pairs;
        const std::string image = request.value("image", std::string());
        const bool rerank = request.contains("options");
        const bool grade = request.contains("candidate");
        bool shared = false;
        if (!request.is_object()) { throw std::runtime_error("request must be an object"); }
        const int forms = int(request.contains("pairs")) + int(request.contains("hypotheses")) + int(rerank) + int(grade);
        if (forms != 1) { throw std::runtime_error("provide exactly one of pairs, hypotheses, options, candidate"); }
        if ((rerank || grade) && opt.latents) { throw std::runtime_error("rerank and grade require classification mode"); }
        if (request.contains("pairs")) {
            const auto items = request.at("pairs");
            if (!items.is_array()) { throw std::runtime_error("pairs must be an array"); }
            for (const auto & pair : items) {
                if (!pair.is_array() || pair.size() != 2) { throw std::runtime_error("each pair must contain two strings"); }
                pairs.emplace_back(pair.at(0).get<std::string>(), pair.at(1).get<std::string>());
            }
        } else if (grade) {
            pairs.emplace_back(request.at("question").get<std::string>() + "\nReference answer: " + request.at("reference").get<std::string>(),
                               "Answer: " + request.at("candidate").get<std::string>());
        } else {
            const auto premise = request.at(rerank ? "question" : "premise").get<std::string>();
            const auto hypotheses = request.at(rerank ? "options" : "hypotheses").get<std::vector<std::string>>();
            for (const auto & hypothesis : hypotheses) {
                pairs.emplace_back(premise, rerank ? "The correct answer is: " + hypothesis : hypothesis);
            }
            shared = opt.prefix && pairs.size() >= 3 && image.empty();
        }
        if (pairs.empty()) { throw std::runtime_error("at least one pair is required"); }
        json rows = json::array();
        size_t prefix = 0;
        auto memory = llama_get_memory(ctx.get());
        llama_memory_clear(memory, false);
        if (!image.empty()) {
            for (const auto & pair : pairs) { rows.push_back(image_pair(pair.first, pair.second, image)); }
        } else {
            std::vector<std::vector<llama_token>> tokens;
            for (const auto & pair : pairs) {
                if (pair.first.find(mtmd_default_marker()) != std::string::npos ||
                    pair.first.find("<|image_pad|>") != std::string::npos) {
                    throw std::runtime_error("image markers require an image request");
                }
                tokens.push_back(tokenize(format_pair(pair.first, pair.second)));
            }
            if (shared) {
                // Compare complete tokenizations: token boundaries can merge across the text split.
                prefix = tokens.front().size() - 1;
                for (const auto & row : tokens) {
                    prefix = std::min(prefix, row.size() - 1);
                    size_t i = 0;
                    while (i < prefix && tokens.front()[i] == row[i]) { ++i; }
                    prefix = i;
                }
            }
            size_t i = 0;
            while (i < tokens.size()) {
                std::vector<llama_seq_id> seqs;
                size_t used = prefix, j = i;
                const int seq_room = prefix ? n_seq_max - 1 : n_seq_max;
                while (j < tokens.size() && int(j - i) < seq_room && used + tokens[j].size() <= n_ctx) {
                    used += tokens[j].size();
                    ++j;
                }
                if (j == i) { throw std::runtime_error("pair does not fit in context"); }
                if (prefix) {
                    if (i == 0) { decode_packed(tokens, 0, 1, 0, prefix, {0}); }
                    for (size_t k = i; k < j; ++k) {
                        const llama_seq_id seq = llama_seq_id(k - i + 1);
                        if (!llama_memory_seq_rm(memory, seq, -1, -1)) { throw std::runtime_error("cannot reset branch"); }
                        llama_memory_seq_cp(memory, 0, seq, -1, -1);
                        seqs.push_back(seq);
                    }
                    decode_packed(tokens, i, j, prefix, 0, seqs);
                } else {
                    llama_memory_clear(memory, false);
                    for (size_t k = i; k < j; ++k) { seqs.push_back(llama_seq_id(k - i)); }
                    decode_packed(tokens, i, j, 0, 0, seqs);
                }
                for (llama_seq_id seq : seqs) { rows.push_back(result(seq)); }
                i = j;
            }
        }
        json out = json::object({{"results", rows}, {"prefix_tokens", prefix}, {"evaluated_tokens", evaluated_tokens}});
        if (!opt.latents) {
            out["labels"] = json::array({labels[0], labels[1], labels[2]});
            if (rerank) {
                size_t best = 0;
                for (size_t i = 1; i < rows.size(); ++i) {
                    if (rows.at(i).at("probabilities").at(1).get<float>() > rows.at(best).at("probabilities").at(1).get<float>()) { best = i; }
                }
                out["index"] = best;
            }
            if (grade) { out["label"] = rows.at(0).at("label"); }
        }
        out["elapsed_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return out;
    }
};

int main(int argc, char ** argv) {
    try {
        options opt;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--no-prefix-cache") { opt.prefix = false; continue; }
            if (arg == "--latents") { opt.latents = true; continue; }
            if (i + 1 >= argc) { throw std::runtime_error("missing value for " + arg); }
            const std::string value = argv[++i];
            if (arg == "-m" || arg == "--model") { opt.model = value; }
            else if (arg == "--mmproj") { opt.mmproj = value; }
            else if (arg == "--input") { opt.input = value; }
            else {
                size_t used = 0;
                const int n = std::stoi(value, &used);
                if (used != value.size()) { throw std::runtime_error("invalid integer for " + arg); }
                if (arg == "-c" || arg == "--ctx-size") { opt.context = n; }
                else if (arg == "-b" || arg == "--batch-size") { opt.batch = n; }
                else if (arg == "-t" || arg == "--threads") { opt.threads = n; }
                else if (arg == "-ngl") { opt.gpu_layers = n; }
                else { throw std::runtime_error("unknown option: " + arg); }
            }
        }
        if (opt.model.empty() || opt.context < 1 || opt.context > 1048576 || opt.batch < 1 ||
            opt.batch > opt.context || opt.threads < 1 || opt.gpu_layers < 0) {
            throw std::runtime_error("provide -m and valid context, batch, thread and GPU settings; see --help");
        }
        std::ifstream file;
        if (!opt.input.empty()) {
            file.open(opt.input);
            if (!file) { throw std::runtime_error("cannot open input: " + opt.input); }
        }
        ggml_backend_load_all();
        llama_backend_init();
        {
            cross_encoder encoder(opt);
            std::string line;
            auto & input = opt.input.empty() ? std::cin : file;
            while (std::getline(input, line)) {
                try {
                    std::cout << encoder.predict(json::parse(line)).dump() << std::endl;
                } catch (const std::exception & e) {
                    std::cout << json::object({{"error", e.what()}}).dump() << std::endl;
                }
            }
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "openjev: " << e.what() << '\n';
        return 1;
    }
}
