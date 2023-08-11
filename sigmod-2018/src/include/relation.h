#pragma once

#include <cstdint>
#include <string>
#include <vector>

using RelationId = unsigned;

class Relation {
private:
    /// Owns memory (false if it was mmaped)
    bool owns_memory_;
    /// The number of tuples
    uint64_t size_;
    /// The join column containing the keys
    std::vector<uint64_t *> columns_;
    /// The zone map sum
    std::vector<uint64_t> zone_map_sum_;
    /// The zone map max
    std::vector<uint64_t> zone_map_max_;
    /// The zone map min
    std::vector<uint64_t> zone_map_min_;

public:
    /// Constructor without mmap
    Relation(uint64_t size, std::vector<uint64_t *> &&columns, std::vector<uint64_t> &&zone_map_sum,
             std::vector<uint64_t> &&zone_map_max, std::vector<uint64_t> &&zone_map_min)
            : owns_memory_(true), size_(size), columns_(columns), zone_map_sum_(zone_map_sum),
              zone_map_max_(zone_map_max), zone_map_min_(zone_map_min) {}

    /// Constructor using mmap
    explicit Relation(const char *file_name);
    /// Delete copy constructor
    Relation(const Relation &other) = delete;
    /// Move constructor
    Relation(Relation &&other) = default;

    /// The destructor
    ~Relation();

    /// Stores a relation into a file (binary)
    void storeRelation(const std::string &file_name);
    /// Stores a relation into a file (csv)
    void storeRelationCSV(const std::string &file_name);
    /// Dump SQL: Create and load table (PostgreSQL)
    void dumpSQL(const std::string &file_name, unsigned relation_id);

    /// The number of tuples
    uint64_t size() const {
        return size_;
    }
    /// The join column containing the keys
    const std::vector<uint64_t *> &columns() const {
        return columns_;
    }

    uint64_t column_size() const;

    uint64_t getSum(uint64_t index) const;

    uint64_t getMax(uint64_t index) const;

    uint64_t getMin(uint64_t index) const;

private:
    /// Loads data from a file
    void loadRelation(const char *file_name);
};

