#include "common.h"
#include "json.h"
#include "llama.h"
#include "laya.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = common_json;
static const char * labels[] = { "contradiction", "entailment", "neutral" };
static const char * nli_template = "Premise: {premise}\nHypothesis: {hypothesis}";
static const char * kev_specials[] = {
    "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"
};

static std::string kev_head_path(const std::string & model) {
    auto replace_end = [](std::string path, const char * from, const char * to) {
        const size_t n = std::strlen(from);
        if (path.size() >= n && path.compare(path.size() - n, n, from) == 0) {
            path.replace(path.size() - n, n, to);
            return path;
        }
        return std::string();
    };
    std::string path = replace_end(model, "-Q4_K_M.gguf", ".head.bin");
    if (path.empty()) { path = replace_end(model, "-f16.gguf", ".head.bin"); }
    if (path.empty()) { path = replace_end(model, ".gguf", ".head.bin"); }
    return path.empty() ? model + ".head.bin" : path;
}

static std::string kev_sanitize(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (text[i] == '<' && i + 1 < text.size() && text[i + 1] == '|') {
            const size_t end = text.find("|>", i + 2);
            if (end != std::string::npos) {
                out += "<\xC2\xA6";
                out.append(text, i + 2, end - (i + 2));
                out += "\xC2\xA6>";
                i = end + 2;
                continue;
            }
        }
        out += text[i++];
    }
    return out;
}

struct kev_head {
    int d = 0, dp = 0;
    float temperature = 1.f;
    std::vector<float> qw, qb, kw, kb;

    bool load(const std::string & path) {
        FILE * file = std::fopen(path.c_str(), "rb");
        if (!file) { return false; }
        char magic[8];
        uint32_t nd = 0, ndp = 0;
        if (std::fread(magic, 1, 8, file) != 8 || std::memcmp(magic, "KEVHEAD1", 8) != 0 ||
            std::fread(&nd, 4, 1, file) != 1 || std::fread(&ndp, 4, 1, file) != 1 ||
            std::fread(&temperature, 4, 1, file) != 1) {
            std::fclose(file);
            return false;
        }
        d = int(nd);
        dp = int(ndp);
        qw.resize(size_t(dp) * size_t(d));
        qb.resize(size_t(dp));
        kw.resize(size_t(dp) * size_t(d));
        kb.resize(size_t(dp));
        const bool ok = std::fread(qw.data(), 4, qw.size(), file) == qw.size() &&
                        std::fread(qb.data(), 4, qb.size(), file) == qb.size() &&
                        std::fread(kw.data(), 4, kw.size(), file) == kw.size() &&
                        std::fread(kb.data(), 4, kb.size(), file) == kb.size();
        std::fclose(file);
        return ok && d > 0 && dp > 0;
    }

    static void project(const std::vector<float> & w, const std::vector<float> & b,
                        const float * x, std::vector<float> & y, int d, int dp) {
        y.resize(size_t(dp));
        for (int i = 0; i < dp; ++i) {
            float sum = b[size_t(i)];
            const float * row = w.data() + size_t(i) * size_t(d);
            for (int j = 0; j < d; ++j) { sum += row[j] * x[j]; }
            y[size_t(i)] = sum;
        }
    }

    std::vector<float> logits(const float * decide, const std::vector<const float *> & opts) const {
        std::vector<float> q, k, out(opts.size());
        project(qw, qb, decide, q, d, dp);
        const float scale = (1.f / std::sqrt(float(dp))) / (temperature > 0.f ? temperature : 1.f);
        for (size_t i = 0; i < opts.size(); ++i) {
            project(kw, kb, opts[i], k, d, dp);
            float dot = 0;
            for (int j = 0; j < dp; ++j) { dot += k[size_t(j)] * q[size_t(j)]; }
            out[i] = dot * scale;
        }
        return out;
    }
};

struct options {
    std::string model, mmproj, input;
    int context = 4096, batch = 512, threads = 4, gpu_layers = 99;
    bool prefix = true, latents = false;
};

static void usage() {
    std::cout << "openjev.cpp - NLI, Kev pointer models and Laya decisions\n"
                 "Usage: openjev -m MODEL.gguf [options] < requests.jsonl\n"
                 "  --mmproj FILE       vision projector for image requests\n"
                 "  --input FILE        read JSONL from a file instead of stdin\n"
                 "  -c, --ctx-size N    maximum tokens per pair (default 4096)\n"
                 "  -b, --batch-size N  prefill batch size (default 512)\n"
                 "  -t, --threads N     CPU threads (default 4)\n"
                 "  -ngl N              GPU layers (default 99; 0 for CPU)\n"
                 "  --no-prefix-cache  evaluate each pair independently\n"
                 "  --latents          return final-token hidden states\n"
                 "OpenJev: {\"premise\":\"...\",\"hypotheses\":[\"...\"]}\n"
                 "Kev: {\"state\":\"...\",\"questions\":[{\"instr\":\"...\",\"options\":[\"...\"]}]}\n"
                 "Laya: Kev request shape with type (choice, score, noul) on each question\n";
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
    kev_head pointer;
    std::unique_ptr<laya_head> laya;
    bool kev = false;
    int n_batch = 512, n_seq_max = 5, n_embd = 0;
    uint32_t n_ctx = 0;
    size_t evaluated_tokens = 0;
    llama_token kev_tok[5] = {};

    std::vector<llama_token> tokenize(const std::string & text) const {
        auto result = common_tokenize(llama_model_get_vocab(model.get()), text, false, true);
        if (result.empty() || result.size() > size_t(opt.context)) {
            throw std::runtime_error("pair exceeds --ctx-size (no silent truncation)");
        }
        return result;
    }

    void decode_packed(const std::vector<std::vector<llama_token>> & rows, size_t lo, size_t hi,
                       size_t begin, size_t end, const std::vector<llama_seq_id> & seqs,
                       bool logits_all = true,
                       const std::vector<std::vector<unsigned char>> * logits_mask = nullptr,
                       std::map<std::pair<llama_seq_id, llama_pos>, std::vector<float>> * emb = nullptr) {
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
                        bool logits = logits_all;
                        if (logits_mask) { logits = (*logits_mask)[r][cursor] != 0; }
                        common_batch_add(batch.value, rows[r][cursor], llama_pos(cursor), {seqs[r - lo]}, logits);
                    }
                }
                ++cursor;
            }
            if (batch.value.n_tokens == 0) { break; }
            if (llama_decode(ctx.get(), batch.value) != 0) {
                throw std::runtime_error("llama_decode failed");
            }
            evaluated_tokens += size_t(batch.value.n_tokens);
            if (emb) {
                for (int t = 0; t < batch.value.n_tokens; ++t) {
                    if (!batch.value.logits[t]) { continue; }
                    const float * values = llama_get_embeddings_ith(ctx.get(), t);
                    if (!values) { throw std::runtime_error("missing token embedding"); }
                    const llama_seq_id seq = batch.value.n_seq_id[t] > 0 ? batch.value.seq_id[t][0] : 0;
                    (*emb)[{seq, batch.value.pos[t]}] = std::vector<float>(values, values + n_embd);
                }
            }
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

    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>
    tokenize_image_pair(const std::string & premise, const std::string & hypothesis, const mtmd_bitmap * bitmap) {
        std::string p = premise;
        if (p.find(mtmd_default_marker()) == std::string::npos) {
            p = std::string(mtmd_default_marker()) + "\n" + p;
        }
        const std::string text = format_pair(p, hypothesis);
        const mtmd_input_text input{text.c_str(), text.size(), false, true};
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        if (mtmd_tokenize(vision.get(), chunks.get(), &input, &bitmap, 1) != 0) {
            throw std::runtime_error("image tokenization failed");
        }
        if (mtmd_helper_get_n_tokens(chunks.get()) > size_t(opt.context)) {
            throw std::runtime_error("image pair exceeds --ctx-size");
        }
        return chunks;
    }

    // CLIP + image KV once, then branch each hypothesis. Re-encoding the JPEG per pair
    // left Qwen3.5 hybrid memory at pos 5 while the text suffix started at ~260.
    json image_pairs(const std::vector<std::pair<std::string, std::string>> & pairs, const std::string & path) {
        if (!vision) { throw std::runtime_error("image requests require --mmproj"); }
        auto loaded = mtmd_helper_bitmap_init_from_file(vision.get(), path.c_str(), false, mtmd_helper_init_opt_default());
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(loaded.bitmap, mtmd_bitmap_free);
        if (!bitmap) { throw std::runtime_error("cannot load image: " + path); }
        if (mtmd_bitmap_is_audio(bitmap.get())) { throw std::runtime_error("expected an image, received audio"); }
        const mtmd_bitmap * ptr = bitmap.get();
        auto first = tokenize_image_pair(pairs.front().first, pairs.front().second, ptr);
        const size_t n_chunks = mtmd_input_chunks_size(first.get());
        if (n_chunks == 0) { throw std::runtime_error("image tokenization produced no chunks"); }
        const bool split = n_chunks > 1 &&
            mtmd_input_chunk_get_type(mtmd_input_chunks_get(first.get(), n_chunks - 1)) == MTMD_INPUT_CHUNK_TYPE_TEXT;
        if (!split) {
            throw std::runtime_error("image request must tokenize as prefix chunks plus a text hypothesis");
        }
        auto memory = llama_get_memory(ctx.get());
        llama_memory_clear(memory, false);
        llama_pos prefix_past = 0;
        const size_t prefix_n = n_chunks - 1;
        for (size_t i = 0; i < prefix_n; ++i) {
            if (mtmd_helper_eval_chunk_single(vision.get(), ctx.get(), mtmd_input_chunks_get(first.get(), i),
                                              prefix_past, 0, opt.batch, false, &prefix_past) != 0) {
                throw std::runtime_error("image evaluation failed");
            }
        }
        json rows = json::array();
        const llama_seq_id branch = 1;
        for (size_t p = 0; p < pairs.size(); ++p) {
            std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> extra{nullptr, mtmd_input_chunks_free};
            const mtmd_input_chunks * chunks = first.get();
            if (p > 0) {
                extra = tokenize_image_pair(pairs[p].first, pairs[p].second, ptr);
                chunks = extra.get();
            }
            if (mtmd_input_chunks_size(chunks) != n_chunks) {
                throw std::runtime_error("image pair chunk layout changed across hypotheses");
            }
            if (!llama_memory_seq_rm(memory, branch, -1, -1)) { throw std::runtime_error("cannot reset branch"); }
            llama_memory_seq_cp(memory, 0, branch, -1, -1);
            llama_pos past = prefix_past;
            if (mtmd_helper_eval_chunk_single(vision.get(), ctx.get(), mtmd_input_chunks_get(chunks, n_chunks - 1),
                                              past, branch, opt.batch, true, &past) != 0) {
                throw std::runtime_error("image evaluation failed");
            }
            evaluated_tokens += mtmd_helper_get_n_tokens(chunks);
            rows.push_back(result(branch));
        }
        return rows;
    }

    json softmax_row(const std::vector<float> & logits) const {
        for (float v : logits) {
            if (!std::isfinite(v)) { throw std::runtime_error("non-finite model output"); }
        }
        const float maximum = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probabilities(logits.size());
        float sum = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            probabilities[i] = std::exp(logits[i] - maximum);
            sum += probabilities[i];
        }
        for (float & p : probabilities) { p /= sum; }
        return json::object({{"logits", logits}, {"probabilities", probabilities}});
    }

    json predict_kev(const json & request) {
        if (opt.latents) { throw std::runtime_error("kev requests do not support --latents"); }
        if (!request.contains("state") || !request.at("questions").is_array()) {
            throw std::runtime_error("kev requests need state and questions");
        }
        const std::string state = request.at("state").get<std::string>();
        const auto questions = request.at("questions");
        if (questions.empty()) { throw std::runtime_error("at least one question is required"); }
        auto * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> prefix = {kev_tok[0]};
        auto user = common_tokenize(vocab, kev_sanitize(state), false, false);
        const size_t max_state = std::min(size_t(opt.context), size_t(8192));
        if (user.size() + 1 > max_state) { user.resize(max_state - 1); }
        prefix.insert(prefix.end(), user.begin(), user.end());
        struct branch {
            std::vector<llama_token> tokens;
            std::vector<llama_pos> opt_pos;
            llama_pos decide = 0;
        };
        std::vector<branch> branches;
        for (const auto & q : questions) {
            if (!q.is_object() || !q.contains("instr") || !q.contains("options")) {
                throw std::runtime_error("each question needs instr and options");
            }
            const auto options = q.at("options").get<std::vector<std::string>>();
            if (options.empty() || options.size() > 255) {
                throw std::runtime_error("each question needs 1 to 255 options");
            }
            branch br;
            br.tokens.push_back(kev_tok[1]);
            auto instr = common_tokenize(vocab, kev_sanitize(q.at("instr").get<std::string>()), false, false);
            br.tokens.insert(br.tokens.end(), instr.begin(), instr.end());
            for (const auto & option : options) {
                br.tokens.push_back(kev_tok[2]);
                auto body = common_tokenize(vocab, kev_sanitize(option), false, false);
                br.tokens.insert(br.tokens.end(), body.begin(), body.end());
                br.tokens.push_back(kev_tok[3]);
                br.opt_pos.push_back(llama_pos(prefix.size() + br.tokens.size() - 1));
            }
            br.tokens.push_back(kev_tok[4]);
            br.decide = llama_pos(prefix.size() + br.tokens.size() - 1);
            if (prefix.size() + br.tokens.size() > size_t(opt.context)) {
                throw std::runtime_error("kev branch exceeds --ctx-size (no silent truncation)");
            }
            branches.push_back(std::move(br));
        }
        auto memory = llama_get_memory(ctx.get());
        llama_memory_clear(memory, false);
        std::vector<std::vector<llama_token>> prefix_row = {prefix};
        decode_packed(prefix_row, 0, 1, 0, 0, {0}, false);
        json rows = json::array();
        size_t i = 0;
        while (i < branches.size()) {
            std::vector<llama_seq_id> seqs;
            std::vector<std::vector<llama_token>> tokens;
            std::vector<std::vector<unsigned char>> mask;
            size_t used = prefix.size(), j = i;
            const int seq_room = n_seq_max - 1;
            while (j < branches.size() && int(j - i) < seq_room) {
                const size_t row = prefix.size() + branches[j].tokens.size();
                if (used + row > n_ctx) { break; }
                used += row;
                ++j;
            }
            if (j == i) { throw std::runtime_error("kev question does not fit in context"); }
            for (size_t k = i; k < j; ++k) {
                const llama_seq_id seq = llama_seq_id(k - i + 1);
                if (!llama_memory_seq_rm(memory, seq, -1, -1)) { throw std::runtime_error("cannot reset branch"); }
                llama_memory_seq_cp(memory, 0, seq, -1, -1);
                seqs.push_back(seq);
                std::vector<llama_token> row = prefix;
                row.insert(row.end(), branches[k].tokens.begin(), branches[k].tokens.end());
                std::vector<unsigned char> take(row.size(), 0);
                for (llama_pos pos : branches[k].opt_pos) { take[size_t(pos)] = 1; }
                take[size_t(branches[k].decide)] = 1;
                tokens.push_back(std::move(row));
                mask.push_back(std::move(take));
            }
            std::map<std::pair<llama_seq_id, llama_pos>, std::vector<float>> emb;
            decode_packed(tokens, 0, tokens.size(), prefix.size(), 0, seqs, false, &mask, &emb);
            for (size_t k = i; k < j; ++k) {
                const llama_seq_id seq = seqs[k - i];
                auto decide = emb.find({seq, branches[k].decide});
                if (decide == emb.end()) { throw std::runtime_error("missing decide embedding"); }
                std::vector<const float *> opts;
                for (llama_pos pos : branches[k].opt_pos) {
                    auto hit = emb.find({seq, pos});
                    if (hit == emb.end()) { throw std::runtime_error("missing option embedding"); }
                    opts.push_back(hit->second.data());
                }
                rows.push_back(softmax_row(pointer.logits(decide->second.data(), opts)));
            }
            i = j;
        }
        json out = json::object({{"results", rows}, {"prefix_tokens", prefix.size()}, {"evaluated_tokens", evaluated_tokens}});
        return out;
    }

public:
    explicit cross_encoder(const options & args) : opt(args) {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = opt.gpu_layers;
        model.reset(llama_model_load_from_file(opt.model.c_str(), mp));
        if (!model) { throw std::runtime_error("failed to load model"); }
        n_embd = llama_model_n_embd(model.get());
        char laya_file[1024];
        const int laya_file_len = llama_model_meta_val_str(model.get(), "laya.head_file", laya_file, sizeof(laya_file));
        if (laya_file_len >= 0) {
            if (laya_file_len >= int(sizeof(laya_file)) || std::filesystem::path(laya_file).filename() != laya_file) {
                throw std::runtime_error("invalid Laya companion filename");
            }
            if (opt.latents || !opt.mmproj.empty()) { throw std::runtime_error("Laya does not support --latents or --mmproj"); }
            const auto path = std::filesystem::path(opt.model).parent_path() / laya_file;
            laya.reset(new laya_head(path.string(), n_embd, opt.threads));
            if (opt.context < laya->max_length()) { throw std::runtime_error("Laya requires --ctx-size >= 512"); }
        }
        const std::string head = kev_head_path(opt.model);
        kev = !laya && pointer.load(head);
        if (kev) {
            if (pointer.d != n_embd) {
                throw std::runtime_error("kev pointer width does not match the model");
            }
            const auto * vocab = llama_model_get_vocab(model.get());
            for (int i = 0; i < 5; ++i) {
                auto ids = common_tokenize(vocab, kev_specials[i], false, true);
                if (ids.size() != 1) { throw std::runtime_error(std::string("missing kev delimiter ") + kev_specials[i]); }
                kev_tok[i] = ids[0];
            }
        } else if (!laya) {
            char value[256];
            if (llama_model_meta_val_str(model.get(), "openjev.nli_template", value, sizeof(value)) < 0 ||
                std::string(value) != nli_template || llama_model_n_cls_out(model.get()) != 3) {
                throw std::runtime_error("expected an openjev GGUF converted with this fork");
            }
            for (int i = 0; i < 3; ++i) {
                const char * label = llama_model_cls_label(model.get(), i);
                if (!label || std::string(label) != labels[i]) { throw std::runtime_error("invalid NLI label order"); }
            }
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = uint32_t(opt.context) * 2;
        cp.n_seq_max = kev ? 8 : 5;
        cp.kv_unified = true;
        cp.n_batch = cp.n_ctx;
        cp.n_ubatch = uint32_t(opt.batch);
        cp.n_threads = cp.n_threads_batch = opt.threads;
        cp.embeddings = true;
        cp.pooling_type = (kev || laya) ? LLAMA_POOLING_TYPE_NONE
                              : (opt.latents ? LLAMA_POOLING_TYPE_LAST : LLAMA_POOLING_TYPE_RANK);
        cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        if (laya) {
            cp.n_ctx = cp.n_batch = cp.n_ubatch = laya->max_length();
            cp.n_seq_max = 1;
            cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        }
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
        if (!request.is_object()) { throw std::runtime_error("request must be an object"); }
        if (laya) {
            json out = laya->predict(ctx.get(), model.get(), request);
            out["elapsed_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            return out;
        }
        const bool kev_req = request.contains("questions") && request.at("questions").is_array();
        if (kev) {
            if (!kev_req) { throw std::runtime_error("kev GGUF expects {state, questions:[{instr, options}]}"); }
            json out = predict_kev(request);
            out["elapsed_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            return out;
        }
        if (kev_req) { throw std::runtime_error("openjev NLI GGUF does not accept kev questions"); }
        std::vector<std::pair<std::string, std::string>> pairs;
        const std::string image = request.value("image", std::string());
        const bool rerank = request.contains("options");
        const bool grade = request.contains("candidate");
        bool shared = false;
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
            rows = image_pairs(pairs, image);
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
