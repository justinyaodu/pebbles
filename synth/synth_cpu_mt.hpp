// Multi-threaded CPU-based synthesizer.

#ifndef SYNTH_CPU_MT_H
#define SYNTH_CPU_MT_H

#include <cstdint>
#include <cstring>
#include <omp.h>

#include "bitset.hpp"
#include "expr.hpp"
#include "spec.hpp"
#include "synth.hpp"

// Set experimentally.
#define TILE_SIZE 64

#define UNARY_TILE_SIZE 4096

class Synthesizer : public AbstractSynthesizer {
private:
    // The i'th bit is on iff the bank contains a term whose bitvector
    // of evaluation results is equal to i.
    ThreadSafeBitset seen;

public:
    Synthesizer(Spec spec) : AbstractSynthesizer(spec),
            seen(ThreadSafeBitset(max_distinct_terms)) {}

private:
    // Allocate the specified number of contiguous indices in the bank for new
    // terms, and return the first index in that contiguous region.
    int64_t alloc_terms(int64_t count) {
        // Increment the number of terms atomically (for thread safety), and
        // return the previous value, which is also the first free index.
        return __atomic_fetch_add(&num_terms, count, __ATOMIC_SEQ_CST);
    }

    // Add the specified number of NOT terms or variable terms to the bank.
    int64_t add_unary_terms(int64_t count, uint32_t *results, uint32_t *lefts) {
        int64_t start = alloc_terms(count);
        memcpy(&term_results[start], results, count * sizeof(uint32_t));
        memcpy(&term_lefts[start], lefts, count * sizeof(uint32_t));
        return start;
    }

    // Add the specified number of binary operator terms to the bank.
    int64_t add_binary_terms(int64_t count, uint32_t *results, uint32_t *lefts,
            uint32_t *rights) {
        int64_t start = alloc_terms(count);
        memcpy(&term_results[start], results, count * sizeof(uint32_t));
        memcpy(&term_lefts[start], lefts, count * sizeof(uint32_t));
        memcpy(&term_rights[start], rights, count * sizeof(uint32_t));
        return start;
    }

    // Add variables of the specified height to the bank.
    int64_t pass_Variable(int32_t height) {
        for (size_t i = 0; i < spec.num_vars; i++) {
            if (spec.var_heights[i] != height) {
                continue;
            }

            uint32_t result = result_mask & spec.var_values[i];
            if (seen.test_and_set(result)) {
                continue;
            }

            uint32_t left_cast = i;
            add_unary_terms(1, &result, &left_cast);

            if (result == spec.sol_result) {
                return num_terms - 1;
            }
        }

        return NOT_FOUND;
    }

    // Synthesize NOT terms.
    int64_t pass_Not(int32_t height) {
        // The operand must be a term whose height is one less than the current
        // height.
        int64_t all_lefts_start = terms_with_height_start(height - 1);
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t solution = NOT_FOUND;

        // Now that we have multiple threads, inserting new terms one at a time
        // would incur significant overhead from atomically incrementing
        // num_terms. Instead, we look at UNARY_TILE_SIZE operands at a time,
        // creating a batch of new terms, then insert the whole batch at once.
        //
        // We also want to align our memory accesses to avoid crossing cache
        // line boundaries, so we round the start index down and round the end
        // index up, then do extra bounds checks in the inner loop.
        //
        // This loop is parallelized with OpenMP.
        #pragma omp parallel for
        for (int64_t lefts_tile = all_lefts_start / UNARY_TILE_SIZE;
                lefts_tile < CEIL_DIV(all_lefts_end, UNARY_TILE_SIZE);
                lefts_tile++) {
            // Do nothing if a previous loop iteration has already found a
            // solution.
            //
            // In the single-threaded version, we could simply break out of the
            // loop. however, with OpenMP running multiple loop iterations in
            // parallel on different threads, break statements aren't supported.
            //
            // The cancel construct (#pragma omp cancel) needs an extra
            // environment variable to work properly, so this is less effort. I
            // haven't tested whether cancelling has better performance though.
            if (solution != NOT_FOUND) {
                continue;
            }

            int32_t batch_size = 0;
            uint32_t batch_results[UNARY_TILE_SIZE];
            uint32_t batch_lefts[UNARY_TILE_SIZE];

            // Loop over the operands in the tile.
            for (int64_t left = std::max(lefts_tile * UNARY_TILE_SIZE, all_lefts_start);
                    left < std::min((lefts_tile + 1) * UNARY_TILE_SIZE, all_lefts_end);
                    left++) {
                uint32_t left_result = term_results[left];
                uint32_t result = result_mask & ~left_result;
                if (seen.test_and_set(result)) {
                    continue;
                }

                // This term is new - add it to the batch.
                batch_results[batch_size] = result;
                batch_lefts[batch_size] = left;
                batch_size++;
            }

            if (batch_size == 0) {
                continue;
            }

            // Check if any of the new terms in the batch are valid solutions.
            int64_t bank_index = add_unary_terms(batch_size, batch_results, batch_lefts);
            for (int32_t i = 0; i < batch_size; i++) {
                if (batch_results[i] == spec.sol_result) {
                    // No synchronization needed, because if two threads find a
                    // solution simultaneously, it doesn't matter which we use.
                    solution = bank_index + i;
                }
            }
        }

        return solution;
    }

    // Add binary operator terms (AND, OR, XOR) to the bank.
    template <typename Op>
    friend int64_t pass_binary(Synthesizer &self, int32_t height, Op op) {
        // The left operand can be any term whose height is less than the
        // current height.
        int64_t all_lefts_end = self.terms_with_height_end(height - 1);

        // The right operand must be a term whose height is one less than the
        // current height.
        int64_t all_rights_start = self.terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        int64_t solution = Synthesizer::NOT_FOUND;

        // We need to iterate over the trapezoidal region of (left, right) pairs
        // such that:
        //
        // 1. 0 <= left < all_lefts_end
        // 2. all_rights_start <= right <= all_rights_end
        // 3. left <= right
        //
        // Our binary operators are commutative, so condition 3 ensures that we
        // don't construct redundant terms.
        //
        // Here's a diagram illustrating how we map 1D indices to coordinates on
        // the trapezoid:
        //
        //                  ..210     ..210
        //                  ..543     ..543
        // 0123456789AB <-> ..876 <-> ..876
        //                  ..BA9     ...A9
        //                  .....     ....B
        //
        // We first determine the number of tiles in the region and create a
        // linear sequence of indices (left). I'm using some letters in the
        // diagram because I ran out of base 10 digits.
        //
        // Next, we map those digits to a rectangular region (center) with the
        // same width and area as the trapezoid, and whose top edge is aligned
        // with the top edge of the trapezoid.
        //
        // Lastly, we identify the tiles in the triangular region inside the
        // rectangle and outside the trapezoid (in this case, B), and transform
        // their coordinates so they end up in the triangular region outside the
        // rectangle and inside the trapezoid.
        //
        // If you're wondering why the indices go right to left instead of left
        // to right, it's because the rectangular region isn't always a
        // rectangle. Half of the last row is missing if the trapezoid width is
        // even:
        //
        // ..3210     ..3210
        // ..7654     ..7654
        // ..BA98 <-> ...A98
        // ....DC     ....DC
        // ......     .....B
        //
        // The shape of a trapezoidal region like this can be fully described
        // with two integers, k and n. k is the index of the first column, and n
        // is the index to the right of the last column:
        //
        //   k  n
        //   |  |
        //   v  v
        // ..###.
        // ..###.
        // ..###.
        // ...##.
        // ....#.
        // ......
        //
        // See trapezoid_indexing.py for a standalone implementation of the
        // indexing algorithm.
        //
        // One more thing: instead of enumerating individual pairs of terms in
        // this trapezoidal region, we want to use square tiles of size
        // TILE_SIZE x TILE_SIZE, in order to get the same batching benefits
        // as the Not pass. Thus, k and n actually represent indices of tiles,
        // not terms.
        //
        // We round k down to the nearest tile and round n up to the nearest
        // tile. That way, the resulting trapezoidal region of tiles is
        // guaranteed to cover the trapezoidal region of pairs of terms.

        int64_t k = all_rights_start / TILE_SIZE;
        int64_t n = CEIL_DIV(all_rights_end, TILE_SIZE);

        #pragma omp parallel for
        // b is a 1D index as described above, and it uniquely identifies one of
        // the tiles covering the trapezoidal region.
        for (int64_t b = 0; b < k * (n - k) + (n - k) * (n - k + 1) / 2; b++) {
            if (solution != Synthesizer::NOT_FOUND) {
                continue;
            }

            int64_t lefts_tile = b / (n - k);
            int64_t rights_tile = n - 1 - b % (n - k);
            if (lefts_tile > rights_tile) {
                // As described above, this tile is inside the rectangle but
                // outside the trapezoid, so we need to transform its
                // coordinates.
                lefts_tile = n - (lefts_tile - (k + 1)) - 1;
                rights_tile = n - (rights_tile - k) - 1;
            }

            int32_t batch_size = 0;
            uint32_t batch_results[TILE_SIZE * TILE_SIZE];
            uint32_t batch_lefts[TILE_SIZE * TILE_SIZE];
            uint32_t batch_rights[TILE_SIZE * TILE_SIZE];

            // Use min to ensure that we don't read terms that are out of bounds
            // on the right side. However, it's okay if we're out of bounds on
            // the left side or on the wrong side of the diagonal: the terms we
            // synthesize from those pairings are still valid, they're just
            // equivalent to other terms that are in bounds. This happens rarely
            // enough (only on tiles on the perimeter) that it's not worth
            // checking for.
            for (int64_t left = lefts_tile * TILE_SIZE;
                    left < std::min((lefts_tile + 1) * TILE_SIZE, all_lefts_end);
                    left++) {
                uint32_t left_result = self.term_results[left];
                for (int64_t right = rights_tile * TILE_SIZE;
                        right < std::min((rights_tile + 1) * TILE_SIZE, all_rights_end);
                        right++) {
                    uint32_t right_result = self.term_results[right];
                    uint32_t result = self.result_mask & op(left_result, right_result);
                    if (self.seen.test_and_set(result)) {
                        continue;
                    }

                    batch_results[batch_size] = result;
                    batch_lefts[batch_size] = left;
                    batch_rights[batch_size] = right;
                    batch_size++;
                }
            }

            if (batch_size == 0) {
                continue;
            }

            int64_t bank_index = self.add_binary_terms(
                    batch_size, batch_results, batch_lefts, batch_rights);
            for (int32_t i = 0; i < batch_size; i++) {
                if (batch_results[i] == self.spec.sol_result) {
                    // No synchronization needed, because if two threads find a
                    // solution simultaneously, it doesn't matter which we use.
                    solution = bank_index + i;
                }
            }
        }

        return solution;
    }

    int64_t pass_And(int32_t height) {
        auto op = [](uint32_t a, uint32_t b) { return a & b; };
        return pass_binary(*this, height, op);
    }

    int64_t pass_Or(int32_t height) {
        auto op = [](uint32_t a, uint32_t b) { return a | b; };
        return pass_binary(*this, height, op);
    }

    int64_t pass_Xor(int32_t height) {
        auto op = [](uint32_t a, uint32_t b) { return a ^ b; };
        return pass_binary(*this, height, op);
    }
};

#endif
