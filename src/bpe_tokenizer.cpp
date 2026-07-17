// ============================================================
// src/bpe_tokenizer.cpp  (LENGKAP)
// ============================================================
#include "bpe_tokenizer.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

void BPETokenizer::init_base_vocab() {
    id_to_token_.clear();
    token_to_id_.clear();
    merge_info_.clear();

    id_to_token_.resize(BASE_VOCAB_SIZE);
    id_to_token_[PAD_ID] = "<pad>";
    id_to_token_[BOS_ID] = "<bos>";
    id_to_token_[EOS_ID] = "<eos>";
    id_to_token_[UNK_ID] = "<unk>";
    for (int b = 0; b < 256; ++b) {
        std::string byte_str(1, static_cast<char>(static_cast<unsigned char>(b)));
        id_to_token_[4 + b] = byte_str;
        token_to_id_[byte_str] = 4 + b;
    }
}

BPETokenizer::BPETokenizer() {
    init_base_vocab();
}

std::vector<std::string> BPETokenizer::split_segments(const std::string& text) {
    std::vector<std::string> segments;
    if (text.empty()) return segments;

    size_t start = 0;
    bool current_is_space = std::isspace(static_cast<unsigned char>(text[0])) != 0;

    for (size_t i = 1; i <= text.size(); ++i) {
        bool is_space = (i < text.size()) && (std::isspace(static_cast<unsigned char>(text[i])) != 0);
        if (i == text.size() || is_space != current_is_space) {
            segments.push_back(text.substr(start, i - start));
            start = i;
            current_is_space = is_space;
        }
    }
    return segments;
}

void BPETokenizer::train(const std::string& corpus, size_t target_vocab_size, size_t min_frequency) {
    init_base_vocab();
    if (target_vocab_size <= static_cast<size_t>(BASE_VOCAB_SIZE)) return;

    std::unordered_map<std::string, size_t> segment_freq;
    for (const auto& seg : split_segments(corpus)) segment_freq[seg]++;

    struct WordEntry { std::vector<int> symbols; size_t freq; };
    std::vector<WordEntry> words;
    words.reserve(segment_freq.size());
    for (const auto& [seg, freq] : segment_freq) {
        std::vector<int> symbols;
        symbols.reserve(seg.size());
        for (unsigned char c : seg) symbols.push_back(4 + static_cast<int>(c));
        words.push_back({std::move(symbols), freq});
    }

    while (id_to_token_.size() < target_vocab_size) {
        std::unordered_map<uint64_t, size_t> pair_counts;
        for (const auto& w : words) {
            for (size_t i = 0; i + 1 < w.symbols.size(); ++i) {
                pair_counts[pack_pair(w.symbols[i], w.symbols[i + 1])] += w.freq;
            }
        }
        if (pair_counts.empty()) break;

        uint64_t best_pair = 0;
        size_t best_count = 0;
        for (const auto& [pair, count] : pair_counts) {
            if (count > best_count) { best_count = count; best_pair = pair; }
        }
        if (best_count < min_frequency) break;

        int a = static_cast<int>(best_pair >> 32);
        int b = static_cast<int>(best_pair & 0xFFFFFFFFu);

        int new_id = static_cast<int>(id_to_token_.size());
        std::string new_token = id_to_token_[a] + id_to_token_[b];
        id_to_token_.push_back(new_token);
        token_to_id_[new_token] = new_id;
        merge_info_[best_pair] = MergeInfo{merge_info_.size(), new_id};

        for (auto& w : words) {
            std::vector<int> merged;
            merged.reserve(w.symbols.size());
            for (size_t i = 0; i < w.symbols.size(); ++i) {
                if (i + 1 < w.symbols.size() && w.symbols[i] == a && w.symbols[i + 1] == b) {
                    merged.push_back(new_id);
                    ++i;
                } else {
                    merged.push_back(w.symbols[i]);
                }
            }
            w.symbols = std::move(merged);
        }
    }
}

std::vector<int> BPETokenizer::encode_segment(const std::vector<int>& symbols) const {
    std::vector<int> current = symbols;

    while (current.size() > 1) {
        size_t best_rank = static_cast<size_t>(-1);
        size_t best_pos = static_cast<size_t>(-1);
        int best_merged_id = -1;

        for (size_t i = 0; i + 1 < current.size(); ++i) {
            auto it = merge_info_.find(pack_pair(current[i], current[i + 1]));
            if (it != merge_info_.end() && it->second.rank < best_rank) {
                best_rank = it->second.rank;
                best_pos = i;
                best_merged_id = it->second.merged_id;
            }
        }
        if (best_pos == static_cast<size_t>(-1)) break;

        std::vector<int> next;
        next.reserve(current.size() - 1);
        for (size_t i = 0; i < current.size(); ++i) {
            if (i == best_pos) {
                next.push_back(best_merged_id);
                ++i;
            } else {
                next.push_back(current[i]);
            }
        }
        current = std::move(next);
    }
    return current;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    for (const auto& seg : split_segments(text)) {
        std::vector<int> symbols;
        symbols.reserve(seg.size());
        for (unsigned char c : seg) symbols.push_back(4 + static_cast<int>(c));

        std::vector<int> encoded = encode_segment(symbols);
        result.insert(result.end(), encoded.begin(), encoded.end());
    }
    return result;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == PAD_ID || id == BOS_ID || id == EOS_ID) continue;
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
            result += id_to_token_[UNK_ID];
            continue;
        }
        result += id_to_token_[id];
    }
    return result;
}

void BPETokenizer::save(std::ostream& os) const {
    size_t num_merges = id_to_token_.size() - BASE_VOCAB_SIZE;
    os.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));

    std::vector<std::pair<int, int>> ordered_pairs(num_merges);
    for (const auto& [packed, info] : merge_info_) {
        int a = static_cast<int>(packed >> 32);
        int b = static_cast<int>(packed & 0xFFFFFFFFu);
        ordered_pairs[info.rank] = {a, b};
    }
    for (const auto& [a, b] : ordered_pairs) {
        os.write(reinterpret_cast<const char*>(&a), sizeof(a));
        os.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }
}

BPETokenizer BPETokenizer::load(std::istream& is) {
    BPETokenizer tok;
    size_t num_merges;
    is.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));

    for (size_t rank = 0; rank < num_merges; ++rank) {
        int a, b;
        is.read(reinterpret_cast<char*>(&a), sizeof(a));
        is.read(reinterpret_cast<char*>(&b), sizeof(b));

        int new_id = static_cast<int>(tok.id_to_token_.size());
        std::string new_token = tok.id_to_token_[a] + tok.id_to_token_[b];
        tok.id_to_token_.push_back(new_token);
        tok.token_to_id_[new_token] = new_id;
        tok.merge_info_[pack_pair(a, b)] = MergeInfo{rank, new_id};
    }
    return tok;
}

void BPETokenizer::save_checkpoint(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("BPETokenizer::save_checkpoint: gagal membuka file: " + path);
    save(ofs);
}

BPETokenizer BPETokenizer::load_checkpoint(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("BPETokenizer::load_checkpoint: gagal membuka file: " + path);
    return load(ifs);
}