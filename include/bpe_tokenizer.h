// ============================================================
// include/bpe_tokenizer.h  (FILE BARU)
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
    BPETokenizer(); // vocab awal: 4 token khusus + 256 byte dasar

    // Latih merge dari korpus mentah. target_vocab_size termasuk 260 token dasar
    // (mis. target_vocab_size=1000 -> belajar ~740 merge). min_frequency menghentikan
    // training lebih awal kalau pasangan terbaik sudah terlalu jarang muncul.
    void train(const std::string& corpus, size_t target_vocab_size, size_t min_frequency = 2);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    size_t vocab_size() const { return id_to_token_.size(); }
    int pad_id() const { return PAD_ID; }
    int bos_id() const { return BOS_ID; }
    int eos_id() const { return EOS_ID; }
    int unk_id() const { return UNK_ID; }

    // Checkpoint: hanya daftar pasangan merge (urut) yang disimpan — token dasar
    // & string tiap token hasil merge direkonstruksi otomatis saat load.
    void save(std::ostream& os) const;
    static BPETokenizer load(std::istream& is);

    static constexpr int PAD_ID = 0;
    static constexpr int BOS_ID = 1;
    static constexpr int EOS_ID = 2;
    static constexpr int UNK_ID = 3;
    static constexpr int BASE_VOCAB_SIZE = 260; // 4 khusus + 256 byte

private:
    std::vector<std::string> id_to_token_;              // id -> byte string mentah token itu
    std::unordered_map<std::string, int> token_to_id_;   // kebalikannya (dipakai saat training)

    struct MergeInfo { size_t rank; int merged_id; };
    std::unordered_map<uint64_t, MergeInfo> merge_info_; // pack(first,second) -> rank & id hasil

    void init_base_vocab();
    static uint64_t pack_pair(int a, int b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    }

    // Pecah teks jadi segmen run-karakter (spasi vs bukan-spasi), supaya BPE tidak
    // menggabung lintas kata & spasi tetap bisa direkonstruksi persis saat decode.
    static std::vector<std::string> split_segments(const std::string& text);

    // Terapkan merge (berdasar rank, prioritas terkecil dulu) ke satu segmen sampai stabil
    std::vector<int> encode_segment(const std::vector<int>& symbols) const;
};