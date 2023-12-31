#ifndef WAVELET_TREE_HPP
#define WAVELET_TREE_HPP

#include <vector>
#include <queue>
#include <stdint.h>
#include <iostream>
#include <cassert>
#include "bit_array.hpp"

namespace wavelet {

/**
 Wavelet Tree Array Library for array processing.

 Input: A[0...n], 0 <= A[i] < k,
 Space: n log_2 k bits

 Support many queries in O(log k) time and constant for n.
 */

struct ListResult {
    ListResult(uint64_t c, uint64_t freq) : c(c), freq(freq) {}
    uint64_t c;
    uint64_t freq;
    bool operator < (const ListResult& lr) const {
        if (c != lr.c) return c < lr.c;
        return freq < lr.freq;
    }
};


class WaveletTree {
public:
    /**
     * Constructor
     */
    WaveletTree();

    /**
     * Destructor
     */
    ~WaveletTree();

    /**
     * Initialize an index from an array
     * @param An array to be initialized
     */
    void Init(const std::vector<uint64_t>& array);

    void Init(const BitArray& ba, uint64_t width, uint64_t length);

    /**
     * Clear and release the resouces
     */
    void Clear();

    /**
     * Lookup A[pos]
     * @param pos the position
     * @return return A[pos] if found, or return NOT_FOUND if pos >= length
     */
    uint64_t Lookup(uint64_t pos) const;

    /**
     * Compute the rank = the frequency of a character 'c' in the prefix of the array A[0...pos)
     * @param c Character to be examined
     * @param pos The position of the prefix (not inclusive)
     * @return The frequency of a character 'c' in the prefix of the array A[0...pos)
     *         or NOT_FOUND if c >= alphabet_num or pos > length
     */
    uint64_t Rank(uint64_t c, uint64_t pos) const;

    /**
     * Compute the select = the position of the (rank+1)-th occurence of 'c' in the array.
     * @param c Character to be examined
     * @param rank The rank of the character
     * @return The position of the (rank+1)-th occurence of 'c' in the array.
     *         or NOT_FOUND if c >= alphabet_num or rank > Freq(c)
     */
    uint64_t Select(uint64_t c, uint64_t rank) const;

    /**
     * Compute the frequency of characters c' < c in the subarray A[0...pos)
     * @param c The upper bound of the characters
     * @param pos The position of the end of the prefix (not inclusive)
     * @return The frequency of characters c' < c in the prefix of the array A[0...pos)
               or NOTFOUND if c > alphabet_num or pos > length
     */
    uint64_t RankLessThan(uint64_t c, uint64_t pos) const;

    /**
     * Compute the frequency of characters c' > c in the subarray A[0...pos)
     * @param c The lower bound of the characters
     * @param pos The position of the end of the prefix (not inclusive)
     * @return The frequency of characters c' < c in the prefix of the array A[0...pos)
               or NOTFOUND if c > alphabet_num or pos > length
     */
    uint64_t RankMoreThan(uint64_t c, uint64_t pos) const;

    /**
     * Compute the frequency of characters c' < c, c'=c, and c' > c, in the subarray A[0...pos)
     * @param c The character
     * @param pos The position of the end of the prefix (not inclusive)
     * @param rank The frefquency of c in A[0...pos)
     * @param rank_less_than The frequency of c' < c in A[0...pos)
     * @param rank_more_than The frequency of c' > c in A[0...pos)
      */
    void RankAll(uint64_t c, uint64_t pos, uint64_t& rank,
                 uint64_t& rank_less_than, uint64_t& rank_more_than) const;

    /**
     * Compute the frequency of characters min_c <= c' < max_c in the subarray A[beg_pos ... end_pos)
     * @param min_c The smallerest character to be examined
     * @param max_c The uppker bound of the character to be examined
     * @param beg_pos The beginning position of the array (inclusive)
     * @param end_pos The ending position of the array (not inclusive)
     * @return The frequency of characters min_c <= c < max_c in the subarray A[beg_pos .. end_pos)
               or NOTFOUND if max_c > alphabet_num or end_pos > length
     */
    uint64_t FreqRange(uint64_t min_c, uint64_t max_c, uint64_t beg_pos, uint64_t end_pos) const;


    /**
     * Range Quantile Query, Return the K-th smallest value in the subarray
     * @param beg_pos The beginning position
     * @param end_pos The ending position
     * @param k The order (should be smaller than end_pos - beg_pos).
     * @param pos The position where the k-th largest value appeared in the subarray A[beg_pos .. end_pos)
                  If there are many items having the k-th largest values, the smallest pos will be reporeted
     * @param val The k-th largest value appeared in the subarray A[beg_pos ... end_pos)
     */
    void QuantileRange(uint64_t beg_pos, uint64_t end_pos, uint64_t k, uint64_t& pos, uint64_t& val) const;

    /**
     * Compute the frequency of the character c
     * @param c The character to be examined
     * param Return the frequency of c in the array.
     */
    uint64_t Freq(uint64_t c) const;

    /**
     * Compute the frequency of the characters
     * @param min_c The minimum character
     * @param max_c The maximum character
     * param Return the frequency of min_c <= c < max_c in the array.
     */
    uint64_t FreqSum(uint64_t min_c, uint64_t max_c) const;

    /**
     * Return the number of alphabets in the array
     * @return The number of alphabet in the array
     */
    uint64_t alphabet_num() const;

    /**
     * Return the length of the array
     * @return The length of the array
     */
    uint64_t length() const;

    /**
     * Save the current status to a stream
     * @param os The output stream where the data is saved
     */
    void Save(std::ostream& os) const;

    /**
     * Load the current status from a stream
     * @param is The input stream where the status is saved
     */
    void Load(std::istream& is);

private:
    uint64_t GetAlphabetNum(const std::vector<uint64_t>& array) const;
    uint64_t Log2(uint64_t x) const;
    uint64_t PrefixCode(uint64_t x, uint64_t len, uint64_t total_len) const;
    static uint64_t GetMSB(uint64_t x, uint64_t pos, uint64_t len);
    static uint64_t GetLSB(uint64_t x, uint64_t pos);
    void SetArray(const std::vector<uint64_t>& array);
    void SetOccs(const std::vector<uint64_t>& array);
    void GetBegPoses(const std::vector<uint64_t>& array, uint64_t alpha_bit_num,
                     std::vector<std::vector<uint64_t> >& beg_poses) const;


    std::vector<BitArray> bit_arrays_;
    BitArray occs_;

    uint64_t alphabet_num_;
    uint64_t alphabet_bit_num_;
    uint64_t length_;
};



}


#endif // WAVELET_TREE_HPP
