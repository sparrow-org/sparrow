// Copyright 2024 Man Group Operations Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "sparrow/array_api.hpp"
#include "sparrow/arrow_interface/arrow_flag_utils.hpp"
#include "sparrow/config/config.hpp"
#include "sparrow/layout/array_access.hpp"
#include "sparrow/layout/array_factory.hpp"
#include "sparrow/layout/array_helper.hpp"
#include "sparrow/layout/array_wrapper.hpp"
#include "sparrow/layout/layout_utils.hpp"
#include "sparrow/layout/nested_value_types.hpp"
#include "sparrow/utils/contracts.hpp"
#include "sparrow/utils/crtp_base.hpp"
#include "sparrow/utils/functor_index_iterator.hpp"
#include "sparrow/utils/memory.hpp"
#include "sparrow/utils/metadata.hpp"
#include "sparrow/utils/mp_utils.hpp"
#include "sparrow/utils/nullable.hpp"

namespace sparrow
{
    class dense_union_array;
    class sparse_union_array;

    namespace detail
    {
        template <>
        struct get_data_type_from_array<sparrow::dense_union_array>
        {
            [[nodiscard]] static constexpr sparrow::data_type get()
            {
                return sparrow::data_type::DENSE_UNION;
            }
        };

        template <>
        struct get_data_type_from_array<sparrow::sparse_union_array>
        {
            [[nodiscard]] static constexpr sparrow::data_type get()
            {
                return sparrow::data_type::SPARSE_UNION;
            }
        };

        inline constexpr std::size_t union_no_child = std::numeric_limits<std::size_t>::max();

        /**
         * @brief One row of a rebuild plan.
         *
         * A row is either a copy of an existing child's row (@c child / @c row), or an inserted
         * value, in which case @c child is union_no_child and @c row indexes the plan's
         * inserted-values list.
         */
        struct union_rebuild_value
        {
            std::size_t child = union_no_child;  ///< Source child, or union_no_child when inserted
            std::size_t row = 0;                 ///< Row of the source child, or inserted-value index
            bool force_null = false;             ///< Copy the row as null (sparse union filler row)
            bool inserted = false;               ///< True when the entry is an inserted value
        };
    }

    /**
     * @brief Type trait to check if a type is a dense_union_array.
     *
     * @tparam T Type to check
     */
    template <class T>
    constexpr bool is_dense_union_array_v = std::same_as<T, dense_union_array>;

    /**
     * @brief Type trait to check if a type is a sparse_union_array.
     *
     * @tparam T Type to check
     */
    template <class T>
    constexpr bool is_sparse_union_array_v = std::same_as<T, sparse_union_array>;

    /**
     * @brief CRTP base class providing shared functionality for union array implementations.
     *
     * This class implements the common interface and logic shared between dense and sparse
     * union arrays. Union arrays can store values of different types in a single array,
     * with each element having an associated type ID that indicates which child array
     * contains the actual value.
     *
     * Key features:
     * - Type-safe heterogeneous storage
     * - Efficient type dispatch using type ID mapping
     * - STL-compatible container interface
     * - Support for both dense and sparse union layouts
     * - Arrow format compatibility
     *
     * The union array consists of:
     * - Type ID buffer: Maps each element to its corresponding child array
     * - Child arrays: Store the actual values for each supported type
     * - Type ID mapping: Translates type IDs to child array indices
     *
     * @tparam DERIVED The derived union array type (dense_union_array or sparse_union_array)
     *
     * @pre DERIVED must inherit from this class (CRTP pattern)
     * @pre DERIVED must implement element_offset() method
     * @post Provides complete union array interface
     * @post Maintains Arrow union format compatibility
     * @post Thread-safe for read operations
     *
     * @code{.cpp}
     * // Union arrays can store different types
     * dense_union_array union_arr = create_union_with_int_and_string();
     *
     * // Access elements (type is determined at runtime)
     * auto elem = union_arr[0];
     * if (elem.has_value()) {
     *     // elem could be int or string depending on type ID
     * }
     *
     * // Iteration over heterogeneous elements
     * for (const auto& element : union_arr) {
     *     // Process element of unknown type
     * }
     * @endcode
     */
    template <class DERIVED>
    class union_array_crtp_base : public crtp_base<DERIVED>
    {
    public:

        using self_type = union_array_crtp_base<DERIVED>;
        using derived_type = DERIVED;
        using inner_value_type = array_traits::inner_value_type;
        using value_type = array_traits::const_reference;
        using const_reference = array_traits::const_reference;
        using functor_type = detail::layout_bracket_functor<derived_type, value_type>;
        using const_functor_type = detail::layout_bracket_functor<const derived_type, value_type>;
        using iterator = functor_index_iterator<functor_type>;
        using const_iterator = functor_index_iterator<const_functor_type>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using size_type = std::size_t;

        using type_id_buffer_type = u8_buffer<std::uint8_t>;

        /**
         * @brief Gets the optional name of the union array.
         *
         * @return Optional string view of the array name from Arrow schema
         *
         * @post Returns nullopt if no name is set
         * @post Returned string view remains valid while array exists
         */
        [[nodiscard]] constexpr std::optional<std::string_view> name() const;

        /**
         * @brief Gets the metadata associated with the union array.
         *
         * @return Optional view of key-value metadata pairs from Arrow schema
         *
         * @post Returns nullopt if no metadata is set
         * @post Returned view remains valid while array exists
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG std::optional<key_value_view> metadata() const;

        /**
         * @brief Gets element at specified position with bounds checking.
         *
         * @param i Index of the element to access
         * @return Value from the appropriate child array
         *
         * @pre i must be < size()
         * @post Returns valid value from child array indicated by type ID
         * @post Value type depends on the type ID at position i
         *
         * @throws std::out_of_range if i >= size()
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG value_type at(size_type i) const;

        /**
         * @brief Gets element at specified position without bounds checking.
         *
         * @param i Index of the element to access
         * @return Value from the appropriate child array
         *
         * @pre i must be < size()
         * @post Returns value from child array indicated by type ID
         * @post Value type depends on the type ID at position i
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG value_type operator[](size_type i) const;

        /**
         * @brief Gets mutable element at specified position.
         *
         * @param i Index of the element to access
         * @return Value from the appropriate child array
         *
         * @pre i must be < size()
         * @post Returns value from child array indicated by type ID
         * @post Value type depends on the type ID at position i
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG value_type operator[](size_type i);

        /**
         * @brief Gets reference to the first element.
         *
         * @return Value from the appropriate child array for first element
         *
         * @pre Array must not be empty (!empty())
         * @post Returns valid value from child array
         * @post Equivalent to (*this)[0]
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG value_type front() const;

        /**
         * @brief Gets reference to the last element.
         *
         * @return Value from the appropriate child array for last element
         *
         * @pre Array must not be empty (!empty())
         * @post Returns valid value from child array
         * @post Equivalent to (*this)[size() - 1]
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG value_type back() const;

        /**
         * @brief Checks if the union array is empty.
         *
         * @return true if array contains no elements, false otherwise
         *
         * @post Return value equals (size() == 0)
         */
        [[nodiscard]] constexpr bool empty() const;

        /**
         * @brief Gets the number of elements in the union array.
         *
         * @return Number of elements in the array
         *
         * @post Returns non-negative value
         * @post Equals the number of type IDs in the type buffer
         */
        [[nodiscard]] constexpr size_type size() const;

        /**
         * @brief Gets iterator to the beginning of the array.
         *
         * @return Iterator pointing to the first element
         *
         * @post Iterator is valid for array traversal
         * @post For empty array, equals end()
         */
        [[nodiscard]] constexpr iterator begin();

        /**
         * @brief Gets iterator to the end of the array.
         *
         * @return Iterator pointing past the last element
         *
         * @post Iterator marks the end of the array range
         * @post Not dereferenceable
         */
        [[nodiscard]] constexpr iterator end();

        /**
         * @brief Gets const iterator to the beginning of the array.
         *
         * @return Const iterator pointing to the first element
         *
         * @post Iterator is valid for array traversal
         * @post For empty array, equals end()
         */
        [[nodiscard]] constexpr const_iterator begin() const;

        /**
         * @brief Gets const iterator to the end of the array.
         *
         * @return Const iterator pointing past the last element
         *
         * @post Iterator marks the end of the array range
         * @post Not dereferenceable
         */
        [[nodiscard]] constexpr const_iterator end() const;

        /**
         * @brief Gets const iterator to the beginning of the array.
         *
         * @return Const iterator pointing to the first element
         *
         * @post Iterator is valid for array traversal
         * @post Guarantees const iterator even for non-const array
         */
        [[nodiscard]] constexpr const_iterator cbegin() const;

        /**
         * @brief Gets const iterator to the end of the array.
         *
         * @return Const iterator pointing past the last element
         *
         * @post Iterator marks the end of the array range
         * @post Guarantees const iterator even for non-const array
         */
        [[nodiscard]] constexpr const_iterator cend() const;

        /**
         * @brief Gets reverse iterator to the beginning of reversed array.
         *
         * @return Const reverse iterator pointing to the last element
         *
         * @post Iterator is valid for reverse traversal
         * @post For empty array, equals rend()
         */
        [[nodiscard]] constexpr const_reverse_iterator rbegin() const;

        /**
         * @brief Gets reverse iterator to the end of reversed array.
         *
         * @return Const reverse iterator pointing before the first element
         *
         * @post Iterator marks the end of reverse traversal
         * @post Not dereferenceable
         */
        [[nodiscard]] constexpr const_reverse_iterator rend() const;

        /**
         * @brief Gets const reverse iterator to the beginning of reversed array.
         *
         * @return Const reverse iterator pointing to the last element
         *
         * @post Iterator is valid for reverse traversal
         * @post Guarantees const iterator even for non-const array
         */
        [[nodiscard]] constexpr const_reverse_iterator crbegin() const;

        /**
         * @brief Gets const reverse iterator to the end of reversed array.
         *
         * @return Const reverse iterator pointing before the first element
         *
         * @post Iterator marks the end of reverse traversal
         * @post Guarantees const iterator even for non-const array
         */
        [[nodiscard]] constexpr const_reverse_iterator crend() const;

        /**
         * @brief Inserts copies of a value before `pos`.
         *
         * Rebuilds the type-ID buffer and child arrays. Complexity is
         * O(N + U*C) for dense unions and O(N*C + U*C) for sparse unions,
         * with O(N + C) and O(N*C) temporary storage respectively. N is the
         * resulting length, C the child count, and U the number of distinct
         * inserted alternatives. Requires offset zero; batch insertions are
         * more efficient than repeated single-value insertions.
         *
         * @return Iterator to the first inserted value.
         */
        iterator insert(const_iterator pos, const_reference value, size_type count = 1);

        /**
         * @brief Inserts copies of an owned value before `pos`.
         * @return Iterator to the first inserted value.
         * @note Same complexity and offset requirement as the other insert overload.
         */
        iterator
        insert(const_iterator pos, const array_traits::value_type& value, size_type count = 1);

        /**
         * @brief Inserts a range before `pos`, repeated `count` times.
         * @return Iterator to the first inserted value.
         * @note Complexity is O(K + N) dense or O(K + N*C) sparse, plus O(U*C)
         *       for type matching; K is the input length. Single-pass ranges are
         *       materialized once when necessary.
         */
        template <std::input_iterator InputIt>
        iterator insert(const_iterator pos, InputIt first, InputIt last, size_type count = 1)
        {
            if (count == 0)
            {
                return iterator_at(static_cast<size_type>(pos - cbegin()));
            }

            if constexpr (std::forward_iterator<InputIt>)
            {
                auto input_values = std::ranges::subrange(first, last);
                if constexpr (std::same_as<std::remove_cvref_t<std::iter_reference_t<InputIt>>, array_traits::value_type>)
                {
                    return insert_materialized(pos, input_values, count);
                }
                else
                {
                    auto values = input_values
                                  | std::views::transform(
                                      [](const auto& value) -> array_traits::value_type
                                      {
                                          return array_materialize_element(value);
                                      }
                                  );
                    return insert_materialized(pos, values, count);
                }
            }
            else
            {
                std::vector<array_traits::value_type> values;
                for (; first != last; ++first)
                {
                    if constexpr (std::same_as<std::remove_cvref_t<decltype(*first)>, array_traits::value_type>)
                    {
                        values.push_back(*first);
                    }
                    else
                    {
                        values.push_back(array_materialize_element(*first));
                    }
                }
                return insert_materialized(pos, values, count);
            }
        }

        /**
         * @brief Erases the value at `pos`.
         * @return Iterator following the erased value.
         * @note Complexity is O(N) for dense unions and O(N*C) for sparse unions; requires offset zero.
         */
        iterator erase(const_iterator pos);

        /**
         * @brief Erases the half-open range [`first`, `last`).
         * @return Iterator following the erased range.
         * @note Complexity is O(N) for dense unions and O(N*C) for sparse unions; requires offset zero.
         */
        iterator erase(const_iterator first, const_iterator last);

        /**
         * @brief Appends a borrowed value.
         * @note Equivalent to inserting before `end()`; complexity is O(N) dense or O(N*C) sparse.
         */
        void push_back(const_reference value);

        /**
         * @brief Appends an owned value.
         * @note Equivalent to inserting before `end()`; complexity is O(N) dense or O(N*C) sparse.
         */
        void push_back(const array_traits::value_type& value);

        /**
         * @brief Resizes the array, default-inserting any appended values.
         * @note Complexity is O(N) dense or O(N*C) sparse when the length changes;
         *       no-op is O(1) and mutations require offset zero.
         */
        void resize(size_type new_length);

        /**
         * @brief Resizes the array, inserting `value` when it grows.
         * @note Complexity is O(N) dense or O(N*C) sparse when the length changes; requires offset zero.
         */
        void resize(size_type new_length, const_reference value);

        /**
         * @brief Resizes the array, inserting `value` when it grows.
         * @note Complexity is O(N) dense or O(N*C) sparse when the length changes; requires offset zero.
         */
        void resize(size_type new_length, const array_traits::value_type& value);

        /**
         * @brief Replaces stored values at null positions.
         *
         * Complexity is O(N); the validity state and logical nullness are unchanged.
         */
        constexpr void zero_null_values(const inner_value_type& value)
        {
            sparrow::zero_null_values(*this, value);
        }

    protected:

        static constexpr size_t TYPE_ID_MAP_SIZE = 256;

        using type_id_map = std::array<std::uint8_t, TYPE_ID_MAP_SIZE>;

        /**
         * @brief Extracts child type IDs from an Arrow union format string.
         *
         * @param format_string Arrow format string containing type IDs
         * @return Type ID for each child, in child order
         *
         * @pre format_string must be valid Arrow union format ("+ud:" or "+us:" prefix)
         */
        static std::vector<std::uint8_t> child_type_ids_from_format(std::string_view format_string);

        /**
         * @brief Creates a type-ID lookup table from child type IDs.
         *
         * @param child_type_ids Type ID for each child, in child order
         * @return Array mapping type IDs to child indices
         */
        static constexpr type_id_map make_type_id_map(std::span<const std::uint8_t> child_type_ids);

        /**
         * @brief Creates type ID mapping from child index to type ID mapping.
         *
         * @tparam R Range type containing type IDs
         * @param child_index_to_type_id Optional mapping from child index to type ID
         * @return Inverse mapping from type ID to child index
         *
         * @post If no mapping provided, uses identity mapping (0->0, 1->1, etc.)
         * @post Returned mapping enables efficient child array lookup
         */
        template <std::ranges::input_range R>
        static constexpr type_id_map
        type_id_map_from_child_to_type_id(const std::optional<R>& child_index_to_type_id);

        /**
         * @brief Creates Arrow format string for union arrays.
         *
         * @tparam R Range type for type ID mapping
         * @param dense Whether this is a dense union (true) or sparse (false)
         * @param n Number of child arrays
         * @param range Optional type ID mapping range
         * @return Arrow format string for the union
         *
         * @pre If range is provided, its size must equal n or be 0
         * @pre Range elements must be convertible to std::uint8_t
         * @post Returns valid Arrow format string ("+ud:" or "+us:" prefix)
         *
         * @throws std::invalid_argument if range size doesn't match n
         */
        template <std::ranges::input_range R>
            requires(std::convertible_to<std::ranges::range_value_t<R>, std::uint8_t>)
        static constexpr std::string
        make_format_string(bool dense, std::size_t n, const std::optional<R>& child_index_to_type_id);

        using children_type = std::vector<cloning_ptr<array_wrapper>>;

        /**
         * @brief Creates child array wrappers from Arrow proxy.
         *
         * @param proxy Arrow proxy containing child array data
         * @return Vector of child array wrappers
         *
         * @pre proxy must contain valid child array data
         * @post Returns valid children collection
         * @post Each child corresponds to a union member type
         */
        constexpr children_type make_children(arrow_proxy& proxy);

        /**
         * @brief Protected constructor from Arrow proxy.
         *
         * @param proxy Arrow proxy containing union array data and schema
         *
         * @pre proxy must contain valid Arrow union array and schema
         * @pre proxy format must be valid union format
         * @post Array is initialized with data from proxy
         * @post Type ID mapping is parsed and cached
         * @post Child arrays are created and accessible
         */
        explicit union_array_crtp_base(arrow_proxy proxy);

        /**
         * @brief Copy constructor.
         *
         * @param rhs Source union array to copy from
         *
         * @pre rhs must be in a valid state
         * @post This array contains a copy of rhs data
         * @post Child arrays and type mapping are reconstructed
         */
        constexpr union_array_crtp_base(const self_type& rhs);

        /**
         * @brief Copy assignment operator.
         *
         * @param rhs Source union array to copy from
         * @return Reference to this array
         *
         * @pre rhs must be in a valid state
         * @post This array contains a copy of rhs data
         * @post Previous data is properly released
         * @post Child arrays and type mapping are reconstructed
         */
        constexpr self_type& operator=(const self_type& rhs);

        constexpr union_array_crtp_base(self_type&& rhs) = default;
        constexpr self_type& operator=(self_type&& rhs) = default;

        template <class VALUE>
        void resize_impl(size_type new_length, VALUE&& value)
        {
            const auto current_size = size();
            if (new_length < current_size)
            {
                erase_values(new_length, current_size - new_length);
            }
            else if (new_length > current_size)
            {
                insert(cend(), std::forward<VALUE>(value), new_length - current_size);
            }
        }

        iterator iterator_at(size_type index)
        {
            return iterator(functor_type{&this->derived_cast()}, index);
        }

        /**
         * @brief Appends one rebuild entry per element in [from, to), each referring to its
         *        current child slot.
         */
        void append_rebuild_entries(
            std::vector<detail::union_rebuild_value>& entries,
            size_type from,
            size_type to
        ) const
        {
            for (size_type index = from; index < to; ++index)
            {
                entries.push_back(detail::union_rebuild_value{
                    .child = m_type_id_map[p_type_ids[index]],
                    .row = this->derived_cast().element_offset(index)
                });
            }
        }

        template <std::ranges::forward_range R>
            requires std::same_as<std::ranges::range_value_t<R>, array_traits::value_type>
        iterator insert_materialized(const_iterator pos, R&& values, size_type count)
        {
            const auto pos_index = static_cast<size_type>(pos - cbegin());
            if (count == 0 || std::ranges::empty(values))
            {
                return iterator_at(pos_index);
            }

            SPARROW_ASSERT_TRUE(m_proxy.offset() == 0);
            SPARROW_ASSERT_TRUE(pos_index <= size());

            const auto old_size = size();
            std::vector<array_traits::value_type> inserted(values.begin(), values.end());
            const auto value_count = static_cast<size_type>(inserted.size());
            std::vector<detail::union_rebuild_value> entries;
            entries.reserve(old_size + value_count * count);

            append_rebuild_entries(entries, 0, pos_index);
            for (size_type repetition = 0; repetition < count; ++repetition)
            {
                for (size_type value_index = 0; value_index < value_count; ++value_index)
                {
                    entries.push_back(detail::union_rebuild_value{
                        .row = value_index,
                        .inserted = true
                    });
                }
            }
            append_rebuild_entries(entries, pos_index, old_size);
            return this->derived_cast().rebuild_in_place(
                std::move(entries),
                std::move(inserted),
                pos_index
            );
        }

        iterator erase_values(size_type first, size_type count);

        /// @brief Rebuilds this array from @p values, the inserted values they reference, and
        ///        returns an iterator to @p return_index.
        ///
        /// Only defined in the compiled library: it needs the complete `array` type, which this
        /// header cannot include without a cycle. Each concrete layout exposes it to the library
        /// boundary as its own `rebuild_in_place`, a plain exported member -- a member of this
        /// class template cannot be exported (Clang ignores the visibility attribute on explicit
        /// instantiations of template members).
        iterator rebuild_values(
            std::vector<detail::union_rebuild_value> values,
            std::vector<array_traits::value_type> inserted,
            size_type return_index
        );

        /**
         * @brief Gets mutable reference to the Arrow proxy.
         *
         * @return Mutable reference to internal Arrow proxy
         *
         * @post Returns valid reference to Arrow proxy
         */
        [[nodiscard]] constexpr arrow_proxy& get_arrow_proxy();

        /**
         * @brief Gets const reference to the Arrow proxy.
         *
         * @return Const reference to internal Arrow proxy
         *
         * @post Returns valid const reference to Arrow proxy
         */
        [[nodiscard]] constexpr const arrow_proxy& get_arrow_proxy() const;

        arrow_proxy m_proxy;                                       ///< Internal Arrow proxy
        const std::uint8_t* p_type_ids;                            ///< Pointer to type ID buffer
        children_type m_children;                                  ///< Child array wrappers
        std::vector<std::uint8_t> m_child_type_ids;                ///< Type IDs in child order
        std::array<std::uint8_t, TYPE_ID_MAP_SIZE> m_type_id_map;  ///< Type ID to child index mapping

        friend class detail::array_access;

#if defined(__cpp_lib_format)
        friend struct std::formatter<DERIVED>;
#endif
    };

    /**
     * @brief Equality comparison operator for union arrays.
     *
     * Compares two union arrays element-wise, ensuring both type IDs and values match.
     *
     * @tparam D Union array type
     * @param lhs First union array to compare
     * @param rhs Second union array to compare
     * @return true if arrays are element-wise equal, false otherwise
     *
     * @post Returns true iff arrays have same size and all elements compare equal
     * @post Comparison includes both type IDs and actual values
     */
    template <class D>
    constexpr bool operator==(const union_array_crtp_base<D>& lhs, const union_array_crtp_base<D>& rhs);

    /**
     * @brief Dense union array implementation with offset buffer.
     *
     * Dense union arrays store an additional offset buffer that maps each element
     * to its position within the corresponding child array. This allows child arrays
     * to be densely packed (only containing values that are actually used), making
     * them more memory efficient when union elements are sparse.
     *
     * Memory layout:
     * - Type ID buffer: Maps each element to child array type
     * - Offset buffer: Maps each element to position in child array
     * - Child arrays: Contain only the values actually used
     *
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/dev/format/Columnar.html#dense-union
     *
     * @post Maintains Arrow dense union format compatibility ("+ud:")
     * @post Child arrays can be shorter than the union array length
     * @post Provides memory-efficient storage for sparse union data
     *
     * @code{.cpp}
     * // Create dense union with int and string children
     * std::vector<array> children = {int_array, string_array};
     * type_id_buffer_type type_ids = {0, 1, 0, 1};        // alternating types
     * offset_buffer_type offsets = {0, 0, 1, 1};          // positions in child arrays
     *
     * dense_union_array union_arr(std::move(children),
     *                              std::move(type_ids),
     *                              std::move(offsets));
     * @endcode
     */
    class dense_union_array : public union_array_crtp_base<dense_union_array>
    {
    public:

        using base_type = union_array_crtp_base<dense_union_array>;
        using offset_buffer_type = u8_buffer<std::uint32_t>;
        using type_id_buffer_type = typename base_type::type_id_buffer_type;

        /**
         * @brief Generic constructor for creating dense union arrays.
         *
         * Creates a dense union array from various input combinations including
         * child arrays, type IDs, offsets, and optional type mapping.
         *
         * @tparam Args Parameter pack for constructor arguments
         * @param args Constructor arguments (children, type_ids, offsets, etc.)
         *
         * @pre Args must match one of the create_proxy() overload signatures
         * @pre Args must exclude copy and move constructor signatures
         * @pre Children, type IDs, and offsets must have consistent sizes
         * @post Array is created with the specified children and configuration
         */
        template <class... Args>
            requires(mpl::excludes_copy_and_move_ctor_v<dense_union_array, Args...>)
        explicit dense_union_array(Args&&... args)
            : dense_union_array(create_proxy(std::forward<Args>(args)...))
        {
        }

        /**
         * @brief Constructs dense union array from Arrow proxy.
         *
         * @param proxy Arrow proxy containing dense union array data and schema
         *
         * @pre proxy must contain valid Arrow dense union array and schema
         * @pre proxy format must be "+ud:..."
         * @pre proxy must have type ID buffer and offset buffer
         * @post Array is initialized with data from proxy
         * @post Offset buffer pointer is cached for efficient access
         */
        SPARROW_API explicit dense_union_array(arrow_proxy proxy);

        /**
         * @brief Copy constructor.
         *
         * @param rhs Source dense union array to copy from
         *
         * @pre rhs must be in a valid state
         * @post This array contains a copy of rhs data
         * @post Offset buffer pointer is properly set
         */
        SPARROW_API dense_union_array(const dense_union_array& rhs);

        /**
         * @brief Copy assignment operator.
         *
         * @param rhs Source dense union array to copy from
         * @return Reference to this array
         *
         * @pre rhs must be in a valid state
         * @post This array contains a copy of rhs data
         * @post Previous data is properly released
         * @post Offset buffer pointer is updated
         */
        SPARROW_API dense_union_array& operator=(const dense_union_array& rhs);

        dense_union_array(dense_union_array&& rhs) = default;
        dense_union_array& operator=(dense_union_array&& rhs) = default;

    private:

        using type_id_map = typename base_type::type_id_map;

        /**
         * @brief Creates proxy for dense union array from child arrays, type IDs, and offsets.
         *
         * @tparam TYPE_MAPPING Optional type mapping range (default: std::vector<std::uint8_t>)
         * @tparam METADATA_RANGE Optional metadata range (default: std::vector<metadata_pair>)
         * @param children Child arrays for the union
         * @param element_type Type ID buffer
         * @param offsets Offset buffer
         * @param type_mapping Optional mapping from child index to type ID
         * @param name Optional name for the array
         * @param metadata Optional metadata for the array
         * @return Arrow proxy representing the dense union array
         *
         * @pre Children, type IDs, and offsets must have consistent sizes
         * @post Returns valid Arrow proxy for the dense union array
         */
        template <
            std::ranges::input_range TYPE_MAPPING = std::vector<std::uint8_t>,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
            requires(std::convertible_to<std::ranges::range_value_t<TYPE_MAPPING>, std::uint8_t>)
        [[nodiscard]] static auto create_proxy(
            std::vector<array>&& children,
            type_id_buffer_type&& element_type,
            offset_buffer_type&& offsets,
            std::optional<TYPE_MAPPING>&& type_mapping = std::nullopt,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        ) -> arrow_proxy;

        /**
         * @brief Creates proxy for dense union array from child arrays, type IDs, and offsets.
         *
         * @tparam TYPE_ID_BUFFER_RANGE Range for type ID buffer
         * @tparam OFFSET_BUFFER_RANGE Range for offset buffer
         * @tparam TYPE_MAPPING Optional type mapping range (default: std::vector<std::uint8_t>)
         * @tparam METADATA_RANGE Optional metadata range (default: std::vector<metadata_pair>)
         * @param children Child arrays for the union
         * @param element_type Type ID buffer
         * @param offsets Offset buffer
         * @param type_mapping Optional mapping from child index to type ID
         * @param name Optional name for the array
         * @param metadata Optional metadata for the array
         * @return Arrow proxy representing the dense union array
         *
         * @pre Children, type IDs, and offsets must have consistent sizes
         * @post Returns valid Arrow proxy for the dense union array
         */
        template <
            std::ranges::input_range TYPE_ID_BUFFER_RANGE,
            std::ranges::input_range OFFSET_BUFFER_RANGE,
            std::ranges::input_range TYPE_MAPPING = std::vector<std::uint8_t>,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
            requires(std::convertible_to<std::ranges::range_value_t<TYPE_MAPPING>, std::uint8_t>)
        [[nodiscard]] static arrow_proxy create_proxy(
            std::vector<array>&& children,
            TYPE_ID_BUFFER_RANGE&& element_type,
            OFFSET_BUFFER_RANGE&& offsets,
            std::optional<TYPE_MAPPING>&& type_mapping = std::nullopt,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        )
        {
            SPARROW_ASSERT_TRUE(element_type.size() == offsets.size());
            type_id_buffer_type element_type_buffer{std::move(element_type)};
            offset_buffer_type offsets_buffer{std::move(offsets)};
            return dense_union_array::create_proxy(
                std::move(children),
                std::move(element_type_buffer),
                std::move(offsets_buffer),
                std::move(type_mapping),
                std::move(name),
                std::move(metadata)
            );
        }

        /**
         * @brief Implementation of create_proxy() for dense union arrays.
         *
         * @tparam METADATA_RANGE Optional metadata range (default: std::vector<metadata_pair>)
         * @param children Child arrays for the union
         * @param element_type Type ID buffer
         * @param offsets Offset buffer
         * @param format Arrow format string
         * @param tim Type ID to child index mapping
         * @param name Optional name for the array
         * @param metadata Optional metadata for the array
         * @return Arrow proxy representing the dense union array
         *
         * @pre Children, type IDs, and offsets must have consistent sizes
         * @post Returns valid Arrow proxy for the dense union array
         */
        template <input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
        [[nodiscard]] static arrow_proxy create_proxy_impl(
            std::vector<array>&& children,
            type_id_buffer_type&& element_type,
            offset_buffer_type&& offsets,
            std::string&& format,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        );

        /**
         * @brief Implementation of create_proxy() for dense union arrays.
         *
         * @tparam METADATA_RANGE Optional metadata range (default: std::vector<metadata_pair>)
         * @param children Child arrays for the union
         * @param element_type Type ID buffer
         * @param offsets Offset buffer
         * @param format Arrow format string
         * @param tim Type ID to child index mapping
         * @param name Optional name for the array
         * @param metadata Optional metadata for the array
         * @return Arrow proxy representing the dense union array
         *
         * @pre Children, type IDs, and offsets must have consistent sizes
         * @post Returns valid Arrow proxy for the dense union array
         */
        template <
            std::ranges::input_range TYPE_ID_BUFFER_RANGE,
            std::ranges::input_range OFFSET_BUFFER_RANGE,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
        [[nodiscard]] static arrow_proxy create_proxy_impl(
            std::vector<array>&& children,
            TYPE_ID_BUFFER_RANGE&& element_type,
            OFFSET_BUFFER_RANGE&& offsets,
            std::string&& format,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        )
        {
            SPARROW_ASSERT_TRUE(std::ranges::distance(element_type) == std::ranges::distance(offsets));
            SPARROW_ASSERT_TRUE(std::ranges::distance(element_type) == children.size());
            type_id_buffer_type element_type_buffer{std::move(element_type)};
            offset_buffer_type offsets_buffer{std::move(offsets)};
            return dense_union_array::create_proxy_impl(
                std::move(children),
                std::move(element_type_buffer),
                std::move(offsets_buffer),
                std::move(format),
                std::move(name),
                std::move(metadata)
            );
        }

        /**
         * @brief Gets the offset for an element in its child array.
         *
         * @param i Index of the element in the union array
         * @return Offset of the element in its corresponding child array
         *
         * @pre i must be < size()
         * @post Returns valid offset into the appropriate child array
         * @post Used internally for element access in dense layout
         */
        SPARROW_API std::size_t element_offset(std::size_t i) const;

        SPARROW_API iterator rebuild_in_place(
            std::vector<detail::union_rebuild_value> values,
            std::vector<array_traits::value_type> inserted,
            size_type return_index
        );

        const std::int32_t* p_offsets;  ///< Pointer to offset buffer
        friend class union_array_crtp_base<dense_union_array>;
    };

    /**
     * @brief Sparse union array implementation without offset buffer.
     *
     * Sparse union arrays do not store an offset buffer. Instead, all child arrays
     * have the same length as the union array, and each element directly corresponds
     * to the same position in its child array. This is simpler but less memory
     * efficient when union elements are sparse.
     *
     * Memory layout:
     * - Type ID buffer: Maps each element to child array type
     * - Child arrays: All have the same length as the union array
     *
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/dev/format/Columnar.html#sparse-union
     *
     * @post Maintains Arrow sparse union format compatibility ("+us:")
     * @post All child arrays have the same length as the union array
     * @post Provides simpler access pattern at the cost of memory efficiency
     *
     * @code{.cpp}
     * // Create sparse union with int and string children
     * std::vector<array> children = {int_array, string_array};  // same length as union
     * type_id_buffer_type type_ids = {0, 1, 0, 1};              // alternating types
     *
     * sparse_union_array union_arr(std::move(children),
     *                               std::move(type_ids));
     * @endcode
     */
    class sparse_union_array : public union_array_crtp_base<sparse_union_array>
    {
    public:

        using base_type = union_array_crtp_base<sparse_union_array>;
        using type_id_buffer_type = typename base_type::type_id_buffer_type;

        /**
         * @brief Generic constructor for creating sparse union arrays.
         *
         * Creates a sparse union array from various input combinations including
         * child arrays, type IDs, and optional type mapping.
         *
         * @tparam Args Parameter pack for constructor arguments
         * @param args Constructor arguments (children, type_ids, etc.)
         *
         * @pre Args must match one of the create_proxy() overload signatures
         * @pre Args must exclude copy and move constructor signatures
         * @pre All child arrays must have the same length
         * @post Array is created with the specified children and configuration
         */
        template <class... Args>
            requires(mpl::excludes_copy_and_move_ctor_v<sparse_union_array, Args...>)
        explicit sparse_union_array(Args&&... args)
            : sparse_union_array(create_proxy(std::forward<Args>(args)...))
        {
        }

        /**
         * @brief Constructs sparse union array from Arrow proxy.
         *
         * @param proxy Arrow proxy containing sparse union array data and schema
         *
         * @pre proxy must contain valid Arrow sparse union array and schema
         * @pre proxy format must be "+us:..."
         * @pre proxy must have type ID buffer
         * @pre All child arrays must have same length as union array
         * @post Array is initialized with data from proxy
         */
        SPARROW_API explicit sparse_union_array(arrow_proxy proxy);

        SPARROW_API sparse_union_array(const sparse_union_array&);
        SPARROW_API sparse_union_array& operator=(const sparse_union_array&);

    private:

        using type_id_map = typename base_type::type_id_map;

        /**
         * @brief Creates proxy for sparse union array from child arrays, type IDs, and optional type mapping.
         *
         * @tparam TYPE_MAPPING Optional type mapping range (default: std::vector<std::uint8_t>)
         * @tparam METADATA_RANGE Optional metadata range (default: std::vector<metadata_pair>)
         * @param children Child arrays for the union
         * @param element_type Type ID buffer
         * @param type_mapping Optional mapping from child index to type ID
         * @param name Optional name for the array
         * @param metadata Optional metadata for the array
         * @return Arrow proxy representing the sparse union array
         *
         * @pre All child arrays must have the same length
         * @post Returns valid Arrow proxy for the sparse union array
         */
        template <
            std::ranges::input_range TYPE_MAPPING = std::vector<std::uint8_t>,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
            requires(std::convertible_to<std::ranges::range_value_t<TYPE_MAPPING>, std::uint8_t>)
        static auto create_proxy(
            std::vector<array>&& children,
            type_id_buffer_type&& element_type,
            std::optional<TYPE_MAPPING>&& type_mapping = std::nullopt,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        ) -> arrow_proxy;

        /**
         * @brief Implementation of create_proxy() for sparse union arrays.
         *
         * @tparam METADATA_RANGE Optional metadata range (default: std::vector<metadata_pair>)
         * @param children Child arrays for the union
         * @param element_type Type ID buffer
         * @param format Arrow format string
         * @param tim Type ID to child index mapping
         * @param name Optional name for the array
         * @param metadata Optional metadata for the array
         * @return Arrow proxy representing the sparse union array
         *
         * @pre All child arrays must have the same length
         * @post Returns valid Arrow proxy for the sparse union array
         */
        template <input_metadata_container METADATA_RANGE>
        static auto create_proxy_impl(
            std::vector<array>&& children,
            type_id_buffer_type&& element_type,
            std::string&& format,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        ) -> arrow_proxy;

        /**
         * @brief Gets the offset for an element in its child array.
         *
         * For sparse unions, this always returns the same index as the union array
         * since all child arrays have the same length.
         *
         * @param i Index of the element in the union array
         * @return The same index i (direct correspondence)
         *
         * @pre i must be < size()
         * @post Returns i (sparse layout uses direct indexing)
         * @post Used internally for element access in sparse layout
         */
        [[nodiscard]] SPARROW_API std::size_t element_offset(std::size_t i) const;

        SPARROW_API iterator rebuild_in_place(
            std::vector<detail::union_rebuild_value> values,
            std::vector<array_traits::value_type> inserted,
            size_type return_index
        );

        friend class union_array_crtp_base<sparse_union_array>;
    };

    /****************************************
     * union_array_crtp_base implementation *
     ****************************************/

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::child_type_ids_from_format(std::string_view format_string)
        -> std::vector<std::uint8_t>
    {
        SPARROW_ASSERT_TRUE(format_string.starts_with("+ud:") || format_string.starts_with("+us:"));
        format_string.remove_prefix(4);

        std::vector<std::uint8_t> result;
        while (!format_string.empty())
        {
            const auto separator = format_string.find(',');
            const auto token = format_string.substr(0, separator);
            std::uint32_t type_id = 0;
            const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), type_id);
            SPARROW_ASSERT_TRUE(error == std::errc{} && end == token.data() + token.size());
            SPARROW_ASSERT_TRUE(type_id <= std::numeric_limits<std::uint8_t>::max());
            result.push_back(static_cast<std::uint8_t>(type_id));
            if (separator == std::string_view::npos)
            {
                break;
            }
            format_string.remove_prefix(separator + 1);
        }
        return result;
    }

    template <class DERIVED>
    constexpr auto
    union_array_crtp_base<DERIVED>::make_type_id_map(std::span<const std::uint8_t> child_type_ids)
        -> type_id_map
    {
        type_id_map result{};
        std::array<bool, TYPE_ID_MAP_SIZE> seen{};
        for (std::size_t child_index = 0; child_index < child_type_ids.size(); ++child_index)
        {
            const auto type_id = child_type_ids[child_index];
            SPARROW_ASSERT_TRUE(child_index < TYPE_ID_MAP_SIZE);
            SPARROW_ASSERT_TRUE(!seen[type_id]);
            seen[type_id] = true;
            result[type_id] = static_cast<std::uint8_t>(child_index);
        }
        return result;
    }

    template <class DERIVED>
    template <std::ranges::input_range R>
    constexpr auto
    union_array_crtp_base<DERIVED>::type_id_map_from_child_to_type_id(const std::optional<R>& child_index_to_type_id)
        -> type_id_map
    {
        type_id_map ret{};
        if (!child_index_to_type_id.has_value())
        {
            constexpr std::array<std::uint8_t, TYPE_ID_MAP_SIZE> default_mapping = []
            {
                std::array<std::uint8_t, TYPE_ID_MAP_SIZE> arr{};
                std::iota(arr.begin(), arr.end(), 0);
                return arr;
            }();
            return default_mapping;
        }
        else
        {
            std::size_t child_index = 0;
            for (const auto type_id : *child_index_to_type_id)
            {
                SPARROW_ASSERT_TRUE(child_index < TYPE_ID_MAP_SIZE);
                const auto id = static_cast<std::size_t>(type_id);
                SPARROW_ASSERT_TRUE(id < TYPE_ID_MAP_SIZE);
                ret[id] = static_cast<std::uint8_t>(child_index++);
            }
        }
        return ret;
    }

    template <class DERIVED>
    template <std::ranges::input_range R>
        requires(std::convertible_to<std::ranges::range_value_t<R>, std::uint8_t>)
    constexpr std::string
    union_array_crtp_base<DERIVED>::make_format_string(bool dense, const std::size_t n, const std::optional<R>& range)
    {
        const auto range_size = range.has_value() ? std::ranges::size(*range) : 0;
        if (range_size != n && range_size != 0)
        {
            throw std::invalid_argument("Invalid type-id map");
        }

        std::string result = dense ? "+ud:" : "+us:";
        bool first = true;
        const auto append_type_id = [&result, &first](const auto type_id)
        {
            if (!first)
            {
                result.push_back(',');
            }
            result += std::to_string(type_id);
            first = false;
        };

        if (range_size == 0)
        {
            for (const auto type_id : std::views::iota(std::size_t{0}, n))
            {
                append_type_id(type_id);
            }
        }
        else
        {
            for (const auto type_id : *range)
            {
                append_type_id(type_id);
            }
        }
        return result;
    }

    template <class DERIVED>
    constexpr std::optional<std::string_view> union_array_crtp_base<DERIVED>::name() const
    {
        return m_proxy.name();
    }

    template <class DERIVED>
    SPARROW_CONSTEXPR_CLANG std::optional<key_value_view> union_array_crtp_base<DERIVED>::metadata() const
    {
        return m_proxy.metadata();
    }

    template <class DERIVED>
    constexpr arrow_proxy& union_array_crtp_base<DERIVED>::get_arrow_proxy()
    {
        return m_proxy;
    }

    template <class DERIVED>
    constexpr const arrow_proxy& union_array_crtp_base<DERIVED>::get_arrow_proxy() const
    {
        return m_proxy;
    }

    template <class DERIVED>
    union_array_crtp_base<DERIVED>::union_array_crtp_base(arrow_proxy proxy)
        : m_proxy(std::move(proxy))
        , p_type_ids(reinterpret_cast<std::uint8_t*>(m_proxy.buffers()[0 /*index of type-ids*/].data()))
        , m_children(make_children(m_proxy))
        , m_child_type_ids(child_type_ids_from_format(m_proxy.format()))
        , m_type_id_map(make_type_id_map(m_child_type_ids))
    {
    }

    template <class DERIVED>
    constexpr union_array_crtp_base<DERIVED>::union_array_crtp_base(const self_type& rhs)
        : self_type(rhs.m_proxy)
    {
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::operator=(const self_type& rhs) -> self_type&
    {
        if (this != &rhs)
        {
            m_proxy = rhs.m_proxy;
            p_type_ids = reinterpret_cast<std::uint8_t*>(m_proxy.buffers()[0 /*index of type-ids*/].data());
            m_children = make_children(m_proxy);
            m_child_type_ids = rhs.m_child_type_ids;
            m_type_id_map = rhs.m_type_id_map;
        }
        return *this;
    }

    template <class DERIVED>
    SPARROW_CONSTEXPR_CLANG auto union_array_crtp_base<DERIVED>::operator[](std::size_t i) const -> value_type
    {
        const auto type_id = static_cast<std::size_t>(p_type_ids[i + m_proxy.offset()]);
        const auto child_index = m_type_id_map[type_id];
        const auto offset = this->derived_cast().element_offset(i);
        return array_element(*m_children[child_index], static_cast<std::size_t>(offset));
    }

    template <class DERIVED>
    SPARROW_CONSTEXPR_CLANG auto union_array_crtp_base<DERIVED>::operator[](std::size_t i) -> value_type
    {
        return static_cast<const derived_type&>(*this)[i];
    }

    template <class DERIVED>
    constexpr std::size_t union_array_crtp_base<DERIVED>::size() const
    {
        return m_proxy.length();
    }

    template <class DERIVED>
    constexpr bool union_array_crtp_base<DERIVED>::empty() const
    {
        return size() == 0;
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::begin() -> iterator
    {
        return iterator(functor_type{&(this->derived_cast())}, 0);
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::end() -> iterator
    {
        return iterator(functor_type{&(this->derived_cast())}, this->size());
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::end() const -> const_iterator
    {
        return cend();
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::cbegin() const -> const_iterator
    {
        return const_iterator(const_functor_type{&(this->derived_cast())}, 0);
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::cend() const -> const_iterator
    {
        return const_iterator(const_functor_type{&(this->derived_cast())}, this->size());
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator{cend()};
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator{cbegin()};
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::crbegin() const -> const_reverse_iterator
    {
        return rbegin();
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::crend() const -> const_reverse_iterator
    {
        return rend();
    }

    template <class DERIVED>
    SPARROW_CONSTEXPR_CLANG auto union_array_crtp_base<DERIVED>::front() const -> value_type
    {
        return (*this)[0];
    }

    template <class DERIVED>
    SPARROW_CONSTEXPR_CLANG auto union_array_crtp_base<DERIVED>::back() const -> value_type
    {
        return (*this)[this->size() - 1];
    }

    template <class DERIVED>
    constexpr auto union_array_crtp_base<DERIVED>::make_children(arrow_proxy& proxy) -> children_type
    {
        children_type children(proxy.children().size(), nullptr);
        for (std::size_t i = 0; i < children.size(); ++i)
        {
            children[i] = array_factory(proxy.children()[i].view());
        }
        return children;
    }

    template <class D>
    constexpr bool operator==(const union_array_crtp_base<D>& lhs, const union_array_crtp_base<D>& rhs)
    {
        return std::ranges::equal(lhs, rhs);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::insert(const_iterator pos, const_reference value, size_type count)
        -> iterator
    {
        // A no-op insertion must not require offset zero (see insert_materialized).
        if (count == 0)
        {
            return iterator_at(static_cast<size_type>(pos - cbegin()));
        }
        return insert_materialized(pos, std::views::single(array_materialize_element(value)), count);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::insert(
        const_iterator pos,
        const array_traits::value_type& value,
        size_type count
    ) -> iterator
    {
        return insert_materialized(pos, std::views::single(value), count);
    }

    template <class DERIVED>
    void union_array_crtp_base<DERIVED>::push_back(const_reference value)
    {
        insert(cend(), value);
    }

    template <class DERIVED>
    void union_array_crtp_base<DERIVED>::push_back(const array_traits::value_type& value)
    {
        insert(cend(), value);
    }

    template <class DERIVED>
    void union_array_crtp_base<DERIVED>::resize(size_type new_length)
    {
        resize_impl(new_length, array_traits::value_type{});
    }

    template <class DERIVED>
    void union_array_crtp_base<DERIVED>::resize(size_type new_length, const_reference value)
    {
        resize_impl(new_length, value);
    }

    template <class DERIVED>
    void union_array_crtp_base<DERIVED>::resize(size_type new_length, const array_traits::value_type& value)
    {
        resize_impl(new_length, value);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::erase(const_iterator pos) -> iterator
    {
        const auto index = static_cast<size_type>(pos - cbegin());
        SPARROW_ASSERT_TRUE(index < size());
        return erase_values(index, 1);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::erase(const_iterator first, const_iterator last) -> iterator
    {
        const auto first_index = static_cast<size_type>(first - cbegin());
        const auto last_index = static_cast<size_type>(last - cbegin());
        SPARROW_ASSERT_TRUE(first_index <= last_index);
        SPARROW_ASSERT_TRUE(last_index <= size());
        return erase_values(first_index, last_index - first_index);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::erase_values(size_type first, size_type count) -> iterator
    {
        using rebuild_value = detail::union_rebuild_value;
        SPARROW_ASSERT_TRUE(m_proxy.offset() == 0);
        const auto current_size = size();
        SPARROW_ASSERT_TRUE(first <= current_size);
        SPARROW_ASSERT_TRUE(count <= current_size - first);
        if (count == 0)
        {
            return iterator_at(first);
        }

        std::vector<rebuild_value> values;
        values.reserve(current_size - count);
        this->append_rebuild_entries(values, 0, first);
        this->append_rebuild_entries(values, first + count, current_size);
        return this->derived_cast().rebuild_in_place(std::move(values), {}, first);
    }

    /************************************
     * Union array shared implementation *
     ************************************/

    namespace detail
    {
        template <input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
        arrow_proxy create_union_proxy_impl(
            std::vector<array>&& children,
            std::vector<buffer<std::uint8_t>>&& buffers,
            std::size_t size,
            std::string&& format,
            std::optional<std::string_view> name,
            std::optional<METADATA_RANGE> metadata
        )
        {
            const auto n_children = children.size();
            ArrowSchema** child_schemas = n_children == 0 ? nullptr : new ArrowSchema*[n_children];
            ArrowArray** child_arrays = n_children == 0 ? nullptr : new ArrowArray*[n_children];

            for (std::size_t i = 0; i < n_children; ++i)
            {
                auto& child = children[i];
                auto [flat_arr, flat_schema] = extract_arrow_structures(std::move(child));
                child_arrays[i] = new ArrowArray(std::move(flat_arr));
                child_schemas[i] = new ArrowSchema(std::move(flat_schema));
            }

            const bool is_nullable = n_children == 0
                                     || std::all_of(
                                         child_schemas,
                                         child_schemas + n_children,
                                         [](const ArrowSchema* schema)
                                         {
                                             return to_set_of_ArrowFlags(schema->flags).contains(
                                                 ArrowFlag::NULLABLE
                                             );
                                         }
                                     );

            const std::optional<std::unordered_set<sparrow::ArrowFlag>>
                flags = is_nullable
                            ? std::make_optional(std::unordered_set<sparrow::ArrowFlag>{ArrowFlag::NULLABLE})
                            : std::nullopt;

            ArrowSchema schema = make_arrow_schema(
                std::move(format),
                std::move(name),                      // name
                std::move(metadata),                  // metadata
                flags,                                // flags,
                child_schemas,                        // children
                repeat_view<bool>(true, n_children),  // children_ownership
                nullptr,                              // dictionary,
                true                                  // dictionary ownership
            );

            ArrowArray arr = make_arrow_array(
                static_cast<std::int64_t>(size),  // length
                0,                                // null_count: always 0 as the nullability is in children
                0,                                // offset
                std::move(buffers),
                child_arrays,                         // children
                repeat_view<bool>(true, n_children),  // children_ownership
                nullptr,                              // dictionary
                true
            );

            return arrow_proxy{std::move(arr), std::move(schema)};
        }
    }

    /************************************
     * dense_union_array implementation *
     ************************************/

    template <std::ranges::input_range TYPE_MAPPING, input_metadata_container METADATA_RANGE>
        requires(std::convertible_to<std::ranges::range_value_t<TYPE_MAPPING>, std::uint8_t>)
    auto dense_union_array::create_proxy(
        std::vector<array>&& children,
        type_id_buffer_type&& element_type,
        offset_buffer_type&& offsets,
        std::optional<TYPE_MAPPING>&& child_index_to_type_id,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    ) -> arrow_proxy
    {
        SPARROW_ASSERT_TRUE(element_type.size() == offsets.size());
        const auto n_children = children.size();

        std::string format = make_format_string(true /*dense union*/, n_children, child_index_to_type_id);

        return create_proxy_impl(
            std::move(children),
            std::move(element_type),
            std::move(offsets),
            std::move(format),
            std::move(name),
            std::move(metadata)
        );
    }

    template <input_metadata_container METADATA_RANGE>
    auto dense_union_array::create_proxy_impl(
        std::vector<array>&& children,
        type_id_buffer_type&& element_type,
        offset_buffer_type&& offsets,
        std::string&& format,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    ) -> arrow_proxy
    {
        SPARROW_ASSERT_TRUE(element_type.size() == offsets.size());
        const auto size = element_type.size();

        std::vector<buffer<std::uint8_t>> arr_buffs;
        arr_buffs.reserve(2);
        arr_buffs.emplace_back(std::move(element_type).extract_storage());
        arr_buffs.emplace_back(std::move(offsets).extract_storage());

        return detail::create_union_proxy_impl(
            std::move(children),
            std::move(arr_buffs),
            size,
            std::move(format),
            std::move(name),
            std::move(metadata)
        );
    }

    /*************************************
     * sparse_union_array implementation *
     *************************************/

    template <std::ranges::input_range TYPE_MAPPING, input_metadata_container METADATA_RANGE>
        requires(std::convertible_to<std::ranges::range_value_t<TYPE_MAPPING>, std::uint8_t>)
    auto sparse_union_array::create_proxy(
        std::vector<array>&& children,
        type_id_buffer_type&& element_type,
        std::optional<TYPE_MAPPING>&& child_index_to_type_id,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    ) -> arrow_proxy
    {
        const auto n_children = children.size();
        if (child_index_to_type_id.has_value())
        {
            SPARROW_ASSERT_TRUE((*child_index_to_type_id).size() == n_children);
        }

        std::string format = make_format_string(false /*is dense union*/, n_children, child_index_to_type_id);

        return create_proxy_impl(
            std::move(children),
            std::move(element_type),
            std::move(format),
            std::move(name),
            std::move(metadata)
        );
    }

    template <input_metadata_container METADATA_RANGE>
    auto sparse_union_array::create_proxy_impl(
        std::vector<array>&& children,
        type_id_buffer_type&& element_type,
        std::string&& format,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    ) -> arrow_proxy
    {
        for (const auto& child : children)
        {
            SPARROW_ASSERT_TRUE(child.size() == element_type.size());
        }
        const auto size = element_type.size();

        std::vector<buffer<std::uint8_t>> arr_buffs;
        arr_buffs.reserve(1);
        arr_buffs.emplace_back(std::move(element_type).extract_storage());

        return detail::create_union_proxy_impl(
            std::move(children),
            std::move(arr_buffs),
            size,
            std::move(format),
            std::move(name),
            std::move(metadata)
        );
    }
}

#if defined(__cpp_lib_format)

/**
 * @brief std::format formatter specialization for union arrays.
 *
 * Provides formatting support for union arrays in std::format contexts.
 * Outputs the union type, metadata, and element values.
 *
 * @tparam U Union array type (dense_union_array or sparse_union_array)
 */
template <typename U>
    requires std::derived_from<U, sparrow::union_array_crtp_base<U>>
struct std::formatter<U>
{
    /**
     * @brief Parses format specification (currently unused).
     *
     * @param ctx Format parse context
     * @return Iterator to end of format specification
     */
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();  // Simple implementation
    }

    /**
     * @brief Formats the union array for output.
     *
     * @param ar Union array to format
     * @param ctx Format context
     * @return Iterator to end of formatted output
     *
     * @post Outputs format: "DenseUnion|SparseUnion [name=NAME | size=SIZE] <elem1, elem2, ..., elemN>"
     * @post Name shows "nullptr" if not set
     */
    auto format(const U& ar, std::format_context& ctx) const
    {
        if constexpr (std::is_same_v<U, sparrow::dense_union_array>)
        {
            std::format_to(ctx.out(), "DenseUnion");
        }
        else if constexpr (std::is_same_v<U, sparrow::sparse_union_array>)
        {
            std::format_to(ctx.out(), "SparseUnion");
        }
        else
        {
            static_assert(sparrow::mpl::dependent_false<U>::value, "Unknown union array type");
            sparrow::mpl::unreachable();
        }
        const auto& proxy = ar.get_arrow_proxy();
        std::format_to(ctx.out(), " [name={} | size={}] <", proxy.name().value_or("nullptr"), proxy.length());

        std::for_each(
            ar.cbegin(),
            std::prev(ar.cend()),
            [&ctx](const auto& value)
            {
                std::format_to(ctx.out(), "{}, ", value);
            }
        );

        return std::format_to(ctx.out(), "{}>", ar.back());
    }
};

namespace sparrow
{
    /**
     * @brief Stream output operator for union arrays.
     *
     * @tparam U Union array type (dense_union_array or sparse_union_array)
     * @param os Output stream
     * @param value Union array to output
     * @return Reference to the output stream
     *
     * @post Outputs the union array using std::format formatter
     */
    template <typename U>
        requires std::derived_from<U, union_array_crtp_base<U>>
    std::ostream& operator<<(std::ostream& os, const U& value)
    {
        os << std::format("{}", value);
        return os;
    }
}

#endif
