#include "utils.h"

#include <iostream>

// Create a dummy column
static void createColumn(std::vector<uint64_t *> &columns,
                         uint64_t num_tuples) {
    auto col = new uint64_t[num_tuples];
    columns.push_back(col);
    for (unsigned i = 0; i < num_tuples; ++i) {
        col[i] = i;
    }
}

// Create a dummy relation
Relation Utils::createRelation(uint64_t size, uint64_t num_columns) {
    std::vector<uint64_t *> columns;
    for (unsigned i = 0; i < num_columns; ++i) {
        createColumn(columns, size);
    }
    std::vector<uint64_t> zone_map_sum;
    std::vector<uint64_t> zone_map_max;
    std::vector<uint64_t> zone_map_min;

    for (unsigned j = 0; j < num_columns; ++j) {
        auto max = 0;
        auto min = INT64_MAX;
        auto sum = 0;
        for (unsigned i = 0; i < size; ++i) {
            sum += columns[j][i];
            if (columns[j][i] > max) {
                max = columns[j][i];
            }
            if (columns[j][i] < min) {
                min = columns[j][i];
            }
        }
        zone_map_sum.push_back(sum);
        zone_map_max.push_back(max);
        zone_map_min.push_back(min);
    }

    return Relation(size, move(columns), move(zone_map_sum),move(zone_map_max),move(zone_map_min));
}

// Store a relation in all formats
void Utils::storeRelation(std::ofstream &out, Relation &r, unsigned i) {
    auto base_name = "r" + std::to_string(i);
    r.storeRelation(base_name);
    r.storeRelationCSV(base_name);
    r.dumpSQL(base_name, i);
    std::cout << base_name << "\n";
    out << base_name << "\n";
}

