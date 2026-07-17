// ============================================================
// include/bpe_tokenizer.h  (LENGKAP)
// ============================================================
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <ostream>
#include <istream>

class BPETokenizer {
public:
    BPETokenizer();

    void train(const std::string& corpus, size_t target_vocab_size, size_t min_frequency = 2);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    size_t vocab_size() const { return id_to_token_.size(); }
    int pad_id() const { return PAD_ID; }
    int bos_id() const { return BOS_ID; }
    int eos_id() const { return EOS_ID; }
    int unk_id() const { return UNK_ID; }

    void save(std::ostream& os) const;
    static BPETokenizer load(std::istream& is);

    // Wrapper berbasis path (konsisten dengan NeuralNetwork/SequenceModel)
    void save_checkpoint(const std::string& path) const;
    static BPETokenizer load_checkpoint(const std::string& path);

    static constexpr int PAD_ID = 0;
    static constexpr int BOS_ID = 1;
    static constexpr int EOS_ID = 2;
    static constexpr int UNK_ID = 3;
    static constexpr int BASE_VOCAB_SIZE = 260;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    struct MergeInfo { size_t rank; int merged_id; };
    std::unordered_map<uint64_t, MergeInfo> merge_info_;

    void init_base_vocab();
    static uint64_t pack_pair(int a, int b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    }

    static std::vector<std::string> split_segments(const std::string& text);
    std::vector<int> encode_segment(const std::vector<int>& symbols) const;
};