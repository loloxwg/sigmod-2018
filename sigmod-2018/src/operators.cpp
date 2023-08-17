#include "operators.h"
#include <iostream>
#include <thread>
#include <cassert>
#include "threadpool.h"
#include <omp.h>

// Get materialized results
std::vector<uint64_t *> Operator::getResults() {
    std::vector<uint64_t *> result_vector;
    for (auto &c: tmp_results_) {
        result_vector.emplace_back(c.data());
    }
    return result_vector;
}

// Require a column and add it to results
bool Scan::require(SelectInfo info) {
    if (info.binding != relation_binding_)
        return false;
    assert(info.col_id < relation_.columns().size());
    result_columns_.emplace_back(relation_.columns()[info.col_id]);
    select_to_result_col_id_[info] = result_columns_.size() - 1;
    auto sum = relation_.getSum(info.col_id);
    select_to_result_col_id_sum_[info] = sum;
    return true;
}

// Run
void Scan::run() {
    // Nothing to do
    result_size_ = relation_.size();
}

// Get materialized results
std::vector<uint64_t *> Scan::getResults() {
    return result_columns_;
}

// Require a column and add it to results
bool FilterScan::require(SelectInfo info) {
    if (info.binding != relation_binding_)
        return false;
    assert(info.col_id < relation_.columns().size());
    // Use the insert method to both check and insert values
    auto [it, inserted] = select_to_result_col_id_.insert({info, tmp_results_.size()});
    if (inserted) {
        input_data_.emplace_back(relation_.columns()[info.col_id]);
        tmp_results_.emplace_back();
    }
    return true;
}

// Copy to result
void FilterScan::copy2Result(uint64_t id) {
    for (unsigned cId = 0; cId < input_data_.size(); ++cId)
        tmp_results_[cId].emplace_back(input_data_[cId][id]);
    ++result_size_;
}

// Apply filter
bool FilterScan::applyFilter(uint64_t i, FilterInfo &f) {
    auto compare_col = relation_.columns()[f.filter_column.col_id];
    auto constant = f.constant;
    switch (f.comparison) {
        case FilterInfo::Comparison::Equal:
            return compare_col[i] == constant;
        case FilterInfo::Comparison::Greater:
            return compare_col[i] > constant;
        case FilterInfo::Comparison::Less:
            return compare_col[i] < constant;
    };
    return false;
}

// Run
void FilterScan::run() {
    for (uint64_t i = 0; i < relation_.size(); ++i) {
        bool pass = true;
        for (auto &f: filters_) {
            pass &= applyFilter(i, f);
        }
        if (pass)
            copy2Result(i);
    }
}

// Require a column and add it to results
bool Join::require(SelectInfo info) {
    if (requested_columns_.count(info) == 0) {
        bool success = false;
        if (left_->require(info)) {
            requested_columns_left_.emplace_back(info);
            success = true;
        } else if (right_->require(info)) {
            success = true;
            requested_columns_right_.emplace_back(info);
        }
        if (!success)
            return false;

        tmp_results_.emplace_back();
        requested_columns_.emplace(info);
    }
    return true;
}

// Copy to result
void Join::copy2Result(uint64_t left_id, uint64_t right_id) {
    unsigned rel_col_id = 0;
    for (unsigned cId = 0; cId < copy_left_data_.size(); ++cId)
        tmp_results_[rel_col_id++].emplace_back(copy_left_data_[cId][left_id]);

    for (unsigned cId = 0; cId < copy_right_data_.size(); ++cId)
        tmp_results_[rel_col_id++].emplace_back(copy_right_data_[cId][right_id]);
    ++result_size_;
}


void Join::probePhaseParallel(uint64_t start, uint64_t end, std::vector<std::vector<uint64_t>> &temp_results_thread,
                              const uint64_t *right_key_column) {
    for (uint64_t i = start; i != end; ++i) {
        auto rightKey = right_key_column[i];
        auto range = hash_table_.equal_range(rightKey);
        temp_results_thread.resize(copy_left_data_.size() + copy_right_data_.size());
        for (auto iter = range.first; iter != range.second; ++iter) {
            copy2ResultParallel(iter->second, i, temp_results_thread);
        }
    }
}

void
Join::copy2ResultParallel(uint64_t left_id, uint64_t right_id, std::vector<std::vector<uint64_t>> &tmp_results_thread) {
    unsigned rel_col_id = 0;
    for (unsigned cId = 0; cId < copy_left_data_.size(); ++cId)
        tmp_results_thread[rel_col_id++].emplace_back(copy_left_data_[cId][left_id]);

    for (unsigned cId = 0; cId < copy_right_data_.size(); ++cId)
        tmp_results_thread[rel_col_id++].emplace_back(copy_right_data_[cId][right_id]);

    // Remove result_size_ update from here as result size will be updated after merging all thread results
}

void Join::mergeResults(std::vector<std::vector<uint64_t>> &temp_results_thread) {

    unsigned num_cols = temp_results_thread.size();
    if (num_cols == 0) {
        return;
    }

    // std::lock_guard<std::mutex> lock(merge_mutex_);
    // Insert data for each column
    for (unsigned int i = 0; i < num_cols; ++i) {
        tmp_results_[i].insert(tmp_results_[i].end(), temp_results_thread[i].begin(), temp_results_thread[i].end());
    }
    // Update result_size_ after merging  thread results
    result_size_ = tmp_results_[0].size();
}

void Join::runLeft() {
    left_->require(p_info_.left);
    left_->run();
}

// Run
void Join::run() {
    std::thread tLeft([this]() -> void { runLeft(); });
    right_->require(p_info_.right);
    right_->run();
    tLeft.join();


    // Use smaller input_ for build
    if (left_->result_size() > right_->result_size()) {
        std::swap(left_, right_);
        std::swap(p_info_.left, p_info_.right);
        std::swap(requested_columns_left_, requested_columns_right_);
    }

    auto left_input_data = left_->getResults();
    auto right_input_data = right_->getResults();

    // Resolve the input_ columns_
    unsigned res_col_id = 0;
    for (auto &info: requested_columns_left_) {
        copy_left_data_.emplace_back(left_input_data[left_->resolve(info)]);
        select_to_result_col_id_[info] = res_col_id++;
    }
    for (auto &info: requested_columns_right_) {
        copy_right_data_.emplace_back(right_input_data[right_->resolve(info)]);
        select_to_result_col_id_[info] = res_col_id++;
    }

    auto left_col_id = left_->resolve(p_info_.left);
    auto right_col_id = right_->resolve(p_info_.right);

    // Build phase
    auto left_key_column = left_input_data[left_col_id];
    hash_table_.reserve(left_->result_size() * 2);
    for (uint64_t i = 0, limit = i + left_->result_size(); i != limit; ++i) {
        hash_table_.emplace(left_key_column[i], i);
    }
    // Probe phase
    auto right_key_column = right_input_data[right_col_id];

    /////////////////////////////////////////////////////////////////////////////
    if (right_->result_size() < std::thread::hardware_concurrency() * 2.0) {

        for (uint64_t i = 0, limit = i + right_->result_size(); i != limit; ++i) {
            auto rightKey = right_key_column[i];
            auto range = hash_table_.equal_range(rightKey);
            for (auto iter = range.first; iter != range.second; ++iter) {
                /**
                 *
                 * */

                copy2Result(iter->second, i);
            }
        }
    } else {
        /////////////////////////////////////////////////////////////////////////////////////
        // New multi-threaded probe phase
        const auto hardware_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(hardware_threads);
        std::vector<std::vector<std::vector<uint64_t>>> all_tmp_results(hardware_threads);
        const auto chunk_size = right_->result_size() / hardware_threads;


        for (uint64_t i = 0; i < hardware_threads; ++i) {
            all_tmp_results[i].resize(copy_left_data_.size() + copy_right_data_.size());
            uint64_t start = i * chunk_size;
            uint64_t end = (i + 1) == hardware_threads ? right_->result_size() : (i + 1) * chunk_size;
            threads[i] = std::thread([this, start, end, &all_tmp_results, &right_key_column, i]() {
                probePhaseParallel(start, end, all_tmp_results[i], right_key_column);
            });
        }

        // Join threads and merge results
        for (uint64_t thread_index = 0; thread_index < threads.size(); ++thread_index) {
            threads[thread_index].join();
            mergeResults(all_tmp_results[thread_index]);
        }
    }
}

// Copy to result
void SelfJoin::copy2Result(uint64_t id) {
    for (unsigned cId = 0; cId < copy_data_.size(); ++cId)
        tmp_results_[cId].emplace_back(copy_data_[cId][id]);
    ++result_size_;
}

void SelfJoin::copy2ResultParallel(uint64_t id) {
    std::lock_guard<std::mutex> lock(merge_mutex_);
    for (unsigned cId = 0; cId < copy_data_.size(); ++cId)
        tmp_results_[cId].emplace_back(copy_data_[cId][id]);
    result_size_ ++;
}

// Require a column and add it to results
bool SelfJoin::require(SelectInfo info) {
    if (required_IUs_.count(info))
        return true;
    if (input_->require(info)) {
        tmp_results_.emplace_back();
        required_IUs_.emplace(info);
        return true;
    }
    return false;
}

std::unordered_map<uint64_t, std::vector<uint64_t>> buildIndex(uint64_t *col, uint64_t size) {
    std::unordered_map<uint64_t, std::vector<uint64_t>> index;
    for (uint64_t i = 0; i < size; ++i) {
        index[col[i]].emplace_back(i);
    }
    return index;
}

void SelfJoin::runThingInput() {
    input_->require(p_info_.left);
    input_->require(p_info_.right);
    input_->run();
    input_data_ = input_->getResults();
}

void SelfJoin::compareColumnsAndCopyRange(uint64_t *left_col, uint64_t *right_col, uint64_t size,uint64_t start) {
    #pragma omp simd
    for (uint64_t i = 0; i < size; ++i) {
        if (left_col[i] == right_col[i])
            copy2ResultParallel(i+start);
    }
}


// Run
void SelfJoin::run() {
    input_->require(p_info_.left);
    input_->require(p_info_.right);
    input_->run();
    input_data_ = input_->getResults();

    for (auto &iu: required_IUs_) {
        auto id = input_->resolve(iu);
        copy_data_.emplace_back(input_data_[id]);
        select_to_result_col_id_.emplace(iu, copy_data_.size() - 1);
    }

    auto num_threads = std::thread::hardware_concurrency();

    if (input_->result_size() < std::thread::hardware_concurrency() * 2) {

        auto left_col_id = input_->resolve(p_info_.left);
        auto right_col_id = input_->resolve(p_info_.right);

        auto left_col = input_data_[left_col_id];
        auto right_col = input_data_[right_col_id];
        for (uint64_t i = 0; i < input_->result_size(); ++i) {
            if (left_col[i] == right_col[i])
                copy2Result(i);
        }
    } else {
        std::vector<std::thread> threads(num_threads);

        uint64_t step = input_->result_size() / num_threads;
        uint64_t excess = input_->result_size() % num_threads;

        auto left_col_id = input_->resolve(p_info_.left);
        auto right_col_id = input_->resolve(p_info_.right);
        auto left_col = input_data_[left_col_id];
        auto right_col = input_data_[right_col_id];

        for (uint64_t i = 0; i < num_threads; ++i) {
            uint64_t start = i * step + std::min(i, excess);
            uint64_t end = start + step + (i < excess);
            // partition the range [start, end)
            auto part_left_col = left_col + start;
            auto part_right_col = right_col + start;
            threads[i] = std::thread(&SelfJoin::compareColumnsAndCopyRange, this, part_left_col, part_right_col, end- start, start);
        }

        for (auto &thread: threads) {
            thread.join();
        }
    }
}

// Run
void Checksum::run() {
    for (auto &sInfo: col_info_) {
        input_->require(sInfo);
    }

    input_->run();
    auto results = input_->getResults();
    result_size_ = input_->result_size();

    check_sums_.reserve(col_info_.size());

    for (auto &sInfo: col_info_) {
        if (result_size_ == 0) {
            check_sums_.emplace_back();
            continue;
        } else {
            auto it = select_to_result_col_id_sum_.find(sInfo);
            if (it != select_to_result_col_id_sum_.end()) {
                uint64_t sum = it->second;
                check_sums_.emplace_back(sum);
            } else {
                // Handle the case where the sum was not found in the map.
                uint64_t sum = 0;
                auto col_id = input_->resolve(sInfo);
                auto result_col = results[col_id];
#pragma omp simd
                for (auto iter = result_col; iter < (result_col + result_size_); ++iter) {
                    sum += *iter;
                }
                check_sums_.emplace_back(sum);
            }
        }
    }
}

