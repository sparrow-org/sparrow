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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or mplied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sparrow/array_api.hpp"
#include "sparrow/arrow_interface/arrow_array_schema_proxy.hpp"
#include "sparrow/buffer/dynamic_bitset/dynamic_bitset.hpp"
#include "sparrow/c_interface.hpp"
#include "sparrow/debug/copy_tracker.hpp"
#include "sparrow/dictionary_reference.hpp"
#include "sparrow/layout/array_access.hpp"
#include "sparrow/layout/array_factory.hpp"
#include "sparrow/layout/array_helper.hpp"
#include "sparrow/layout/array_wrapper.hpp"
#include "sparrow/layout/nested_value_types.hpp"
#include "sparrow/primitive_array.hpp"
#include "sparrow/types/data_traits.hpp"
#include "sparrow/utils/bit.hpp"
#include "sparrow/utils/contracts.hpp"
#include "sparrow/utils/functor_index_iterator.hpp"
#include "sparrow/utils/memory.hpp"
#include "sparrow/utils/metadata.hpp"
#include "sparrow/utils/mp_utils.hpp"

namespace sparrow
{
    /**
     * Functor for accessing elements in a layout.
     *
     * @tparam Layout The layout type.
     * @tparam is_const Whether the functor provides const access.
     */
    template <class Layout, bool is_const>
    class layout_element_functor
    {
    public:

        using layout_type = Layout;
        using storage_type = std::conditional_t<is_const, const layout_type*, layout_type>;
        using return_type = std::
            conditional_t<is_const, typename layout_type::const_reference, typename layout_type::reference>;

        /**
         * Default constructor.
         */
        constexpr layout_element_functor() = default;

        /**
         * Constructs a functor with the given layout.
         *
         * @param layout_ Pointer to the layout to access.
         */
        constexpr explicit layout_element_functor(storage_type layout_)
            : p_layout(layout_)
        {
        }

        /**
         * Access operator for getting element at index.
         *
         * @param i The index of the element to access.
         * @return Reference to the element at the specified index.
         */
        [[nodiscard]] constexpr return_type operator()(std::size_t i) const
        {
            return p_layout->operator[](i);
        }

    private:

        /** Pointer to the layout being accessed. */
        storage_type p_layout;
    };

    /**
     * Forward declaration of dictionary_encoded_array.
     *
     * @tparam IT The integral type used for dictionary keys.
     */
    template <std::integral IT>
    class dictionary_encoded_array;

    namespace copy_tracker
    {
        template <typename T>
            requires mpl::is_type_instance_of_v<T, dictionary_encoded_array>
        std::string key()
        {
            return "dictionary_encoded_array";
        }
    }

    namespace detail
    {
        template <std::integral IT>
        struct get_data_type_from_array<sparrow::dictionary_encoded_array<IT>>
        {
            /**
             * Gets the data type for the dictionary keys.
             *
             * @return The data type corresponding to the key type.
             */
            [[nodiscard]] static constexpr sparrow::data_type get() noexcept
            {
                return get_data_type_from_array<primitive_array<IT>>::get();
            }
        };

        /**
         * Specialization to identify dictionary_encoded_array types.
         *
         * @tparam IT The integral type used for dictionary keys.
         */
        template <std::integral IT>
        struct is_dictionary_encoded_array<sparrow::dictionary_encoded_array<IT>>
        {
            /**
             * Returns true for dictionary_encoded_array types.
             *
             * @return Always true.
             */
            [[nodiscard]] static constexpr bool get() noexcept
            {
                return true;
            }
        };
    }

    /**
     * Checks whether T is a dictionary_encoded_array type.
     *
     * @tparam T The type to check.
     */
    template <class T>
    constexpr bool is_dictionary_encoded_array_v = detail::is_dictionary_encoded_array<T>::get();

    namespace detail
    {
        /**
         * Compares a dictionary entry with an owning value without materializing the entry when
         * the stored view and the owning value are directly comparable (e.g. string_view and
         * std::string), so lookups do not copy the dictionary values.
         */
        inline bool
        dictionary_value_equal(const array_traits::const_reference& entry, const array_traits::value_type& value)
        {
            return std::visit(
                [&value]<class N>(const N& entry_value) -> bool
                {
                    using owning_nullable = impl::materialized_nullable_t<N>;
                    const auto* other = std::get_if<owning_nullable>(
                        &static_cast<const array_traits::value_type::base_type&>(value)
                    );
                    if (other == nullptr)
                    {
                        return false;
                    }
                    if constexpr (requires { entry_value == *other; })
                    {
                        return static_cast<bool>(entry_value == *other);
                    }
                    else if constexpr (std::constructible_from<owning_nullable, const N&>)
                    {
                        return owning_nullable(entry_value) == *other;
                    }
                    else
                    {
                        return owning_nullable(
                                   typename owning_nullable::value_type(
                                       entry_value.get().begin(),
                                       entry_value.get().end()
                                   ),
                                   entry_value.has_value()
                               )
                               == *other;
                    }
                },
                static_cast<const array_traits::const_reference::base_type&>(entry)
            );
        }
    }

    /**
     * Dictionary encoded array class.
     * Dictionary encoding is a data representation technique to represent values by integers referencing a
     * dictionary usually consisting of unique values. It can be effective when you have data with many
     * repeated values.
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/dev/format/Columnar.html#dictionary-encoded-layout
     *
     * @par Buffers
     * The array owns two sets of buffers: the keys, one integer per element with an optional
     * validity bitmap, and the dictionary, a child array holding the unique values. A null element
     * stores the key 0 and has its validity bit cleared; its key is never looked up in the
     * dictionary. A non-null key is always a valid index in the dictionary child.
     *
     * @par Inserting (insert, push_back, growing resize)
     * Every inserted value is encoded by find_or_append: the dictionary values are scanned, an
     * equal entry reuses its key and a new value is appended at the end of the dictionary child
     * with a fresh key. The keys buffer is then grown, which may reallocate it, and the validity
     * bitmap is extended. Because entries are only appended, the keys of the elements already
     * present do not change; insert(pos, value, count) encodes \p value once and duplicates the
     * resulting key, while the range overloads encode each value.
     *
     * @par Modifying (set_value, dictionary_reference assignment)
     * The new value is encoded like an insertion, then the key and the validity bit of the element
     * are written in place and compact_dictionary runs: if at least one dictionary entry lost its
     * last reference, it is dropped, the dictionary child is replaced by a compacted copy, and
     * every key is rewritten to its new index. Keys are therefore only stable within one
     * operation. Assigning a null nullable adds its value to the dictionary before clearing the
     * validity bit, so the just-added entry stays unreferenced until the next set_value, erase,
     * pop_back or shrinking resize; set_value(index, nullval) and proxy assignment of nullval
     * avoid it.
     *
     * @par Deleting (erase, pop_back, shrinking resize)
     * Keys are removed from the keys buffer and compact_dictionary runs, renumbering the
     * dictionary entries that are still referenced. When no entry became unreferenced the
     * dictionary child is left untouched and only the keys buffer shrinks.
     *
     * @par Invalidations and preconditions
     * Mutations refresh the proxy's cached buffer views and reallocate the keys buffer whenever its
     * size changes. Iterators, element proxies obtained before a mutation, and references or
     * pointers into the dictionary values must not be used afterwards; element proxies are index
     * based, so a proxy whose index is still in range keeps referring to that position of the
     * array, not to the element it was created from. Mutations require an unsliced array
     * (offset() == 0) and throw std::logic_error otherwise, because rewriting the keys of a slice
     * would change elements outside of it. Mutating a non-nullable array with a null value throws
     * std::invalid_argument, and a dictionary index that does not fit the key type throws
     * std::overflow_error.
     *
     * @par Complexity
     * find_or_append is linear in the dictionary size and compact_dictionary is linear in the
     * array size plus the dictionary size, so a single mutation costs O(n + d) for n elements and
     * d dictionary entries; inserting k distinct values costs O(k * d + n).
     *
     * @tparam IT The integral type used for dictionary keys.
     */
    template <std::integral IT>
    class dictionary_encoded_array
    {
    public:

        using self_type = dictionary_encoded_array<IT>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using inner_value_type = array_traits::inner_value_type;

        using value_type = array_traits::value_type;
        using const_reference = array_traits::const_reference;

        using bitmap_type = arrow_proxy::bitmap_type;
        using bitmap_reference = bitmap_type::reference;
        using dictionary_reference_type = dictionary_reference<IT>;
        using reference = nullable<dictionary_reference_type, bitmap_reference>;

        // Iteration is read-only: mutable access goes through the operator[] proxy, so the
        // iterator and const_iterator are the same type.
        using functor_type = layout_element_functor<self_type, true>;
        using iterator = functor_index_iterator<functor_type>;
        using const_iterator = iterator;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = reverse_iterator;

        using keys_buffer_type = u8_buffer<IT>;

        /**
         * Constructs a dictionary encoded array from an arrow proxy.
         *
         * @param proxy The arrow proxy containing the array data and schema.
         */
        explicit dictionary_encoded_array(arrow_proxy proxy);

        /**
         * Copy constructor.
         *
         * @param other The dictionary_encoded_array to copy from.
         */
        constexpr dictionary_encoded_array(const self_type& other);

        /**
         * Copy assignment operator.
         *
         * @param other The dictionary_encoded_array to copy from.
         * @return Reference to this dictionary_encoded_array.
         */
        constexpr self_type& operator=(const self_type& other);

        /**
         * Move constructor.
         *
         * @param other The dictionary_encoded_array to move from.
         */
        constexpr dictionary_encoded_array(self_type&& other);

        /**
         * Move assignment operator.
         *
         * @param other The dictionary_encoded_array to move from.
         * @return Reference to this dictionary_encoded_array.
         */
        constexpr self_type& operator=(self_type&& other);

        /**
         * Gets the name of the array.
         *
         * @return Optional name of the array.
         */
        [[nodiscard]] constexpr std::optional<std::string_view> name() const;

        /**
         * Gets the metadata of the array.
         *
         * @return Optional metadata of the array.
         */
        [[nodiscard]] std::optional<key_value_view> metadata() const;

        /**
         * Gets the number of elements in the array.
         *
         * @return The number of elements.
         */
        [[nodiscard]] constexpr size_type size() const;

        /**
         * Checks if the array is empty.
         *
         * @return true if the array is empty, false otherwise.
         */
        [[nodiscard]] constexpr bool empty() const;

        /**
         * Access operator for getting element at index.
         *
         * @param i The index of the element to access.
         * @return Constant reference to the element at the specified index.
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG const_reference operator[](size_type i) const;

        /**
         * Access operator for getting a mutable element proxy at index.
         *
         * @param i The index of the element to access.
         * @return Mutable reference to the element at the specified index.
         *
         * The proxy is index based and writes through set_value, so assigning to it performs the
         * dictionary update and compaction described there. Any mutation of the array invalidates
         * the proxy and the iterators obtained before it.
         *
         * @throws std::logic_error if the array is not nullable.
         *
         * @note Assigning a null nullable (\c dict[i] = make_nullable<T>(v, false)) runs the proxy
         * assignment with \c v, adding it to the dictionary if absent, and only then clears the
         * validity flag. The just-added entry can therefore stay unreferenced until the next
         * \ref erase, \ref set_value, \ref pop_back or shrinking \ref resize. \ref set_value with
         * \ref nullval and direct proxy assignment (\c dict[i].get() = nullval) avoid it.
         * Assigning a null \ref value_type (\c dict[i] = value_type{nullable<T>{}}) marks the
         * element valid: use one of the paths above instead.
         */
        [[nodiscard]] reference operator[](size_type i);

        /**
         * Gets a constant iterator to the beginning of the array.
         *
         * @return Constant iterator to the beginning.
         */
        [[nodiscard]] constexpr const_iterator begin() const;

        /**
         * Gets a constant iterator to the end of the array.
         *
         * @return Constant iterator to the end.
         */
        [[nodiscard]] constexpr const_iterator end() const;

        /**
         * Gets a constant iterator to the beginning of the array.
         *
         * @return Constant iterator to the beginning.
         */
        [[nodiscard]] constexpr const_iterator cbegin() const;

        /**
         * Gets a constant iterator to the end of the array.
         *
         * @return Constant iterator to the end.
         */
        [[nodiscard]] constexpr const_iterator cend() const;

        /**
         * Gets a constant reverse iterator to the beginning of the array.
         *
         * @return Constant reverse iterator to the beginning.
         */
        [[nodiscard]] constexpr const_reverse_iterator rbegin() const;

        /**
         * Gets a constant reverse iterator to the end of the array.
         *
         * @return Constant reverse iterator to the end.
         */
        [[nodiscard]] constexpr const_reverse_iterator rend() const;

        /**
         * Gets a constant reverse iterator to the beginning of the array.
         *
         * @return Constant reverse iterator to the beginning.
         */
        [[nodiscard]] constexpr const_reverse_iterator crbegin() const;

        /**
         * Gets a constant reverse iterator to the end of the array.
         *
         * @return Constant reverse iterator to the end.
         */
        [[nodiscard]] constexpr const_reverse_iterator crend() const;

        /**
         * Gets a reference to the first element.
         *
         * @return Constant reference to the first element.
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG const_reference front() const;

        /**
         * Gets a reference to the last element.
         *
         * @return Constant reference to the last element.
         */
        [[nodiscard]] SPARROW_CONSTEXPR_CLANG const_reference back() const;

        /**
         * Replaces the element at the given index.
         *
         * A value already present in the dictionary reuses its entry, a new value is appended to
         * the dictionary values, and entries left without any reference are removed.
         *
         * The key and the validity bit of the element are written in place, then the dictionary is
         * compacted: when an entry lost its last reference, the dictionary child is replaced and
         * every key is renumbered. Element proxies and iterators obtained before the call are
         * invalidated.
         *
         * @param index The index of the element to replace.
         * @param value The new value.
         *
         * @pre \p index must be less than size().
         * @pre The array must not be sliced (offset() == 0).
         *
         * @throws std::invalid_argument if \p value is null and the array is not nullable.
         */
        void set_value(size_type index, const value_type& value);

        /**
         * Replaces the element at the given index.
         *
         * The nullable is converted with to_value_type and handled like
         * set_value(size_type, const value_type&): the value is encoded, the key and the validity
         * bit are written, and the dictionary is compacted.
         *
         * @tparam T The nullable value type.
         * @tparam B The nullable flag type.
         * @param index The index of the element to replace.
         * @param value The new value.
         *
         * @pre \p index must be less than size().
         * @pre The array must not be sliced (offset() == 0).
         *
         * @throws std::invalid_argument if \p value is null and the array is not nullable.
         */
        template <class T, mpl::boolean_like B>
            requires dictionary_assignable_nullable<T, B>
        void set_value(size_type index, const nullable<T, B>& value);

        /**
         * Replaces the element at the given index with null.
         *
         * The validity bit is cleared and the key set to 0 without looking anything up in the
         * dictionary, then the dictionary is compacted like any other replacement: entries that
         * are no longer referenced are dropped and the remaining keys are renumbered.
         *
         * @param index The index of the element to replace.
         * @param null_value The null sentinel.
         *
         * @pre \p index must be less than size().
         * @pre The array must not be sliced (offset() == 0).
         *
         * @throws std::invalid_argument if the array is not nullable.
         */
        void set_value(size_type index, nullval_t null_value);

        /**
         * Inserts \p count copies of \p value before \p pos.
         *
         * \p value is encoded once and the resulting key duplicated \p count times. An absent
         * value is appended to the dictionary, the keys buffer is grown and may reallocate, and
         * the keys of the elements already present keep their index. Iterators and element proxies
         * are invalidated.
         *
         * @param pos The position before which the values are inserted.
         * @param value The value to insert.
         * @param count The number of copies to insert.
         * @return An iterator to the first inserted element.
         *
         * @pre The array must not be sliced (offset() == 0) when \p count is not 0.
         *
         * @throws std::invalid_argument if \p value is null and the array is not nullable.
         */
        iterator insert(const_iterator pos, const value_type& value, size_type count = 1);

        /**
         * Inserts \p count copies of \p value before \p pos.
         *
         * The nullable is converted with to_value_type and inserted like
         * insert(const_iterator, const value_type&, size_type): a single key is encoded and
         * duplicated, the dictionary grows when the value is absent, and the keys buffer may
         * reallocate.
         *
         * @tparam T The nullable value type.
         * @tparam B The nullable flag type.
         * @param pos The position before which the values are inserted.
         * @param value The value to insert.
         * @param count The number of copies to insert.
         * @return An iterator to the first inserted element.
         *
         * @pre The array must not be sliced (offset() == 0) when \p count is not 0.
         */
        template <class T, mpl::boolean_like B>
            requires dictionary_assignable_nullable<T, B>
        iterator insert(const_iterator pos, const nullable<T, B>& value, size_type count = 1);

        /**
         * Inserts the values in <tt>[first, last)</tt> before \p pos.
         *
         * The values are materialized before the array is modified, so inserting a range of the
         * array into itself is supported. Each value is then encoded separately: values already in
         * the dictionary reuse their entry, absent ones are appended, and the keys buffer is grown
         * once for the whole range. Iterators and element proxies are invalidated.
         *
         * @param pos The position before which the values are inserted.
         * @param first The beginning of the range to insert.
         * @param last The end of the range to insert.
         * @return An iterator to the first inserted element.
         *
         * @pre The array must not be sliced (offset() == 0) when the range is not empty.
         */
        template <std::input_iterator InputIt>
            requires std::convertible_to<std::iter_value_t<InputIt>, array_traits::value_type>
        iterator insert(const_iterator pos, InputIt first, InputIt last);

        /**
         * Inserts the values of \p values before \p pos.
         *
         * Each value is encoded separately and the keys buffer is grown once for the whole range;
         * see insert(const_iterator, InputIt, InputIt) for the dictionary updates performed.
         *
         * @tparam R The range type.
         * @param pos The position before which the values are inserted.
         * @param values The range to insert.
         * @return An iterator to the first inserted element.
         *
         * @pre The array must not be sliced (offset() == 0) when the range is not empty.
         */
        template <std::ranges::input_range R>
            requires std::convertible_to<std::ranges::range_value_t<R>, array_traits::value_type>
        iterator insert(const_iterator pos, R&& values);

        /**
         * Inserts the values of an initializer list before \p pos.
         *
         * Each value is converted with to_value_type and encoded separately, with the same
         * dictionary and keys buffer updates as the range overloads.
         *
         * @tparam T The nullable value type.
         * @param pos The position before which the values are inserted.
         * @param values The values to insert.
         * @return An iterator to the first inserted element.
         *
         * @pre The array must not be sliced (offset() == 0) when the list is not empty.
         */
        template <class T>
            requires dictionary_assignable_nullable<T, bool>
        iterator insert(const_iterator pos, std::initializer_list<nullable<T>> values);

        /**
         * Removes the element at \p pos.
         *
         * The key is removed from the keys buffer, then the dictionary is compacted: entries left
         * without any reference are dropped, the dictionary child is replaced when that happens,
         * and every remaining key is renumbered. Iterators and element proxies are invalidated,
         * and an index that was pointing after \p pos now designates the following element.
         *
         * @param pos The position of the element to remove.
         * @return An iterator to the element following the removed one.
         *
         * @pre The array must not be sliced (offset() == 0).
         */
        iterator erase(const_iterator pos);

        /**
         * Removes the elements in <tt>[first, last)</tt>.
         *
         * The keys are removed in one operation, then the dictionary is compacted: entries left
         * without any reference are dropped and the remaining keys are renumbered. Iterators and
         * element proxies are invalidated.
         *
         * @param first The beginning of the range to remove.
         * @param last The end of the range to remove.
         * @return An iterator to the element following the last removed one.
         *
         * @pre The array must not be sliced (offset() == 0) when the range is not empty.
         */
        iterator erase(const_iterator first, const_iterator last);

        /**
         * Appends a value at the end of the array.
         *
         * Equivalent to insert(cend(), value): an absent value is appended to the dictionary, the
         * keys buffer grows and may reallocate, and the keys of the existing elements keep their
         * index.
         *
         * @param value The value to append.
         */
        void push_back(const value_type& value);

        /**
         * Appends a nullable value at the end of the array.
         *
         * The nullable is converted with to_value_type and appended like
         * push_back(const value_type&).
         *
         * @tparam T The nullable value type.
         * @tparam B The nullable flag type.
         * @param value The value to append.
         */
        template <class T, mpl::boolean_like B>
            requires dictionary_assignable_nullable<T, B>
        void push_back(const nullable<T, B>& value);

        /**
         * Removes the last element of the array.
         *
         * Equivalent to erase(std::prev(cend())): the dictionary is compacted and the entries
         * left without any reference are dropped.
         *
         * @pre The array must not be empty.
         * @pre The array must not be sliced (offset() == 0).
         */
        void pop_back();

        /**
         * Resizes the array, filling the new elements with null.
         *
         * Growing appends elements whose key is 0 and whose validity bit is cleared, which does
         * not add entries to the dictionary; shrinking behaves like erase and compacts the
         * dictionary.
         *
         * @param new_length The new size of the array.
         */
        void resize(size_type new_length);

        /**
         * Resizes the array, filling the new elements with \p value.
         *
         * Growing encodes \p value once, appends the entry to the dictionary when it is absent and
         * duplicates the resulting key; shrinking behaves like erase and compacts the dictionary.
         *
         * @param new_length The new size of the array.
         * @param value The value used to fill the new elements.
         */
        void resize(size_type new_length, const value_type& value);

        /**
         * Resizes the array, filling the new elements with \p value.
         *
         * The nullable is converted with to_value_type and handled like
         * resize(size_type, const value_type&).
         *
         * @tparam T The nullable value type.
         * @tparam B The nullable flag type.
         * @param new_length The new size of the array.
         * @param value The value used to fill the new elements.
         */
        template <class T, mpl::boolean_like B>
            requires dictionary_assignable_nullable<T, B>
        void resize(size_type new_length, const nullable<T, B>& value);

        /**
         * Constructs a dictionary encoded array with the given arguments.
         *
         * @tparam Args The argument types.
         * @param args Arguments forwarded to create_proxy.
         */
        template <class... Args>
            requires(mpl::excludes_copy_and_move_ctor_v<dictionary_encoded_array<IT>, Args...>)
        explicit dictionary_encoded_array(Args&&... args)
            : dictionary_encoded_array(create_proxy(std::forward<Args>(args)...))
        {
        }

        /**
         * Slices the array to keep only the elements between the given \p start and \p end.
         * A deep copy of the underlying Arrow data is created. The copied ArrowArray.offset and
         * ArrowArray.length are then updated to represent the requested range. If \p end is greater than
         * the size of the buffers, the following elements will be invalid.
         *
         * @param start The index of the first element to keep. Must be less than \p end.
         * @param end The index of the first element to discard. Must be less than the size of the buffers.
         */
        [[nodiscard]] constexpr self_type slice(size_type start, size_type end) const;

        /**
         * Slices the array to keep only the elements between the given \p start and \p end.
         * A view of the \ref array is returned. The underlying data is shared, and only the
         * ArrowArray.offset and ArrowArray.length are updated. If \p end is greater than the size of the
         * buffers, the following elements will be invalid.
         *
         * @param start The index of the first element to keep. Must be less than \p end.
         * @param end The index of the first element to discard. Must be less than the size of the buffers.
         */
        [[nodiscard]] constexpr self_type slice_view(size_type start, size_type end) const;

        /**
         * Slices the array in place to keep only the elements between the given \p start and \p end.
         * The underlying data is not modified, only the ArrowArray.offset and ArrowArray.length are updated.
         *
         * @param start The index of the first element to keep. Must be less than \p end.
         * @param end The index of the first element to discard. Must be less than the size of the buffers.
         */
        constexpr void slice_inplace(size_type start, size_type end);

        /**
         * @returns the starting offset within the buffers (i.e. ArrowArray.offset).
         */
        [[nodiscard]] constexpr size_type offset() const;

    private:

        /**
         * Creates an arrow proxy from keys buffer, values array, and validity bitmap.
         *
         * @tparam R The validity bitmap input type.
         * @tparam METADATA_RANGE The metadata container type.
         * @param keys The buffer containing dictionary keys.
         * @param values The array containing dictionary values.
         * @param bitmaps The validity bitmap.
         * @param name Optional name for the array.
         * @param metadata Optional metadata for the array.
         * @return An arrow proxy containing the dictionary encoded array data.
         */
        template <
            validity_bitmap_input R = validity_bitmap,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
        [[nodiscard]] static auto create_proxy(
            keys_buffer_type&& keys,
            array&& values,
            R&& bitmaps,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        ) -> arrow_proxy;

        /**
         * Creates an arrow proxy from keys buffer and values array.
         *
         * @tparam R The validity bitmap input type.
         * @tparam METADATA_RANGE The metadata container type.
         * @param keys The buffer containing dictionary keys.
         * @param values The array containing dictionary values.
         * @param nullable Whether the array can contain null values.
         * @param name Optional name for the array.
         * @param metadata Optional metadata for the array.
         * @return An arrow proxy containing the dictionary encoded array data.
         */
        template <
            validity_bitmap_input R = validity_bitmap,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
        [[nodiscard]] static auto create_proxy(
            keys_buffer_type&& keys,
            array&& values,
            bool nullable = true,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        ) -> arrow_proxy;

        /**
         * Internal implementation for creating an arrow proxy.
         *
         * @tparam METADATA_RANGE The metadata container type.
         * @param keys The buffer containing dictionary keys.
         * @param values The array containing dictionary values.
         * @param validity Optional validity bitmap.
         * @param name Optional name for the array.
         * @param metadata Optional metadata for the array.
         * @return An arrow proxy containing the dictionary encoded array data.
         */
        template <input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
        [[nodiscard]] static auto create_proxy_impl(
            keys_buffer_type&& keys,
            array&& values,
            std::optional<validity_bitmap> validity = std::nullopt,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        ) -> arrow_proxy;

        /**
         * Creates an arrow proxy from a key range and values array.
         *
         * @tparam KEY_RANGE The key range type.
         * @tparam R The validity bitmap input type.
         * @tparam METADATA_RANGE The metadata container type.
         * @param keys The range of keys to store.
         * @param values The array containing dictionary values.
         * @param bitmaps The validity bitmap.
         * @param name Optional name for the array.
         * @param metadata Optional metadata for the array.
         * @return An arrow proxy containing the dictionary encoded array data.
         */
        template <
            std::ranges::input_range KEY_RANGE,
            validity_bitmap_input R = validity_bitmap,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
            requires(
                !std::same_as<KEY_RANGE, keys_buffer_type>
                and std::same_as<IT, std::ranges::range_value_t<KEY_RANGE>>
            )
        [[nodiscard]] static arrow_proxy create_proxy(
            KEY_RANGE&& keys,
            array&& values,
            R&& bitmaps = validity_bitmap{validity_bitmap::default_allocator()},
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        )
        {
            keys_buffer_type keys_buffer(std::forward<KEY_RANGE>(keys));
            return create_proxy(
                std::move(keys_buffer),
                std::forward<array>(values),
                std::forward<R>(bitmaps),
                std::move(name),
                std::move(metadata)
            );
        }

        /**
         * Creates an arrow proxy from a range of nullable keys and values array.
         *
         * @tparam NULLABLE_KEY_RANGE The nullable key range type.
         * @tparam METADATA_RANGE The metadata container type.
         * @param nullable_keys The range of nullable keys to store.
         * @param values The array containing dictionary values.
         * @param name Optional name for the array.
         * @param metadata Optional metadata for the array.
         * @return An arrow proxy containing the dictionary encoded array data.
         */
        template <
            std::ranges::input_range NULLABLE_KEY_RANGE,
            input_metadata_container METADATA_RANGE = std::vector<metadata_pair>>
            requires std::is_same_v<std::ranges::range_value_t<NULLABLE_KEY_RANGE>, nullable<IT>>
        static arrow_proxy create_proxy(
            NULLABLE_KEY_RANGE&& nullable_keys,
            array&& values,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<METADATA_RANGE> metadata = std::nullopt
        );

        using keys_layout = primitive_array<IT>;
        using values_layout = cloning_ptr<array_wrapper>;
        /**
         * Gets a dummy inner value for null references.
         *
         * @return Constant reference to a dummy inner value.
         */
        [[nodiscard]] const inner_value_type& dummy_inner_value() const;

        /**
         * Gets a dummy const reference for null values.
         *
         * @return A dummy const reference.
         */
        [[nodiscard]] const_reference dummy_const_reference() const;

        /**
         * Creates a keys layout from the arrow proxy.
         *
         * @param proxy The arrow proxy containing the keys data.
         * @return A keys layout object.
         */
        [[nodiscard]] static constexpr keys_layout create_keys_layout(arrow_proxy& proxy);

        /**
         * Creates a values layout from the arrow proxy.
         *
         * @param proxy The arrow proxy containing the values data.
         * @return A values layout object.
         */
        [[nodiscard]] static values_layout create_values_layout(arrow_proxy& proxy);

        /**
         * Gets a reference to the internal arrow proxy.
         *
         * @return Reference to the arrow proxy.
         */
        [[nodiscard]] constexpr arrow_proxy& get_arrow_proxy();

        /**
         * Gets a constant reference to the internal arrow proxy.
         *
         * @return Constant reference to the arrow proxy.
         */
        [[nodiscard]] constexpr const arrow_proxy& get_arrow_proxy() const;

        /**
         * Gets the read-only element at the given index.
         *
         * @param i The index of the element to access.
         * @return Constant reference to the element at the specified index.
         */
        [[nodiscard]] const_reference element_at(size_type i) const;

        /**
         * Creates a read iterator at the given index.
         *
         * @param index The index the iterator points to.
         * @return The iterator.
         */
        [[nodiscard]] iterator make_iterator(size_type index) const;

        /**
         * Throws if the array is sliced (non-zero offset), which mutation does not support.
         *
         * @param operation The name of the attempted operation.
         */
        void throw_if_sliced_for_mutation(const char* operation) const;

        /**
         * Converts a position in the array into an index.
         *
         * @param pos The position to convert.
         * @return The index of the position.
         */
        [[nodiscard]] size_type index_of(const_iterator pos) const;

        /**
         * Converts a dictionary index into a key.
         *
         * @param index The dictionary index.
         * @return The key of the index.
         *
         * @throws std::overflow_error if the index does not fit the key type.
         */
        [[nodiscard]] IT to_key(size_type index) const;

        /**
         * Checks whether the array schema allows null values.
         *
         * @return true if the array is nullable, false otherwise.
         */
        [[nodiscard]] bool is_nullable() const;

        /**
         * Converts a nullable value to the owning value type of this array.
         *
         * @tparam T The nullable value type.
         * @tparam B The nullable flag type.
         * @param value The value to convert.
         * @return The owning value.
         */
        template <class T, mpl::boolean_like B>
        static value_type to_value_type(const nullable<T, B>& value);

        /**
         * Encodes a value into a dictionary key, appending the value to the dictionary if needed.
         *
         * @param value The value to encode.
         * @return The encoded key.
         */
        [[nodiscard]] nullable<IT> encode_value(const value_type& value);

        /**
         * Returns the key of \p value, appending it to the dictionary if it is absent.
         *
         * @param value A non-null value.
         * @return The key of the value.
         */
        IT find_or_append(const value_type& value);

        /**
         * Writes a key and its validity flag at the given index.
         *
         * @param index The index of the element.
         * @param key The key to write.
         * @param valid Whether the element is non-null.
         */
        void set_key(size_type index, IT key, bool valid);

        /**
         * Removes the dictionary entries that no non-null key references and remaps the keys.
         */
        void compact_dictionary();

        /**
         * Refreshes the cached buffer views of the internal arrow proxy.
         */
        void refresh_proxy_buffers();

        /**
         * Inserts already encoded keys at the given index.
         *
         * @param index The index before which the keys are inserted.
         * @param keys The encoded keys to insert.
         * @return An iterator to the first inserted element.
         */
        iterator insert_keys_at(size_type index, std::vector<nullable<IT>> keys);

        /**
         * Inserts already materialized values before \p pos.
         *
         * @param pos The position before which the values are inserted.
         * @param values The values to insert.
         * @return An iterator to the first inserted element.
         */
        iterator insert_values_at(const_iterator pos, const std::vector<value_type>& values);

        /** The arrow proxy containing the array data and schema. */
        arrow_proxy m_proxy;
        /** The layout for accessing dictionary keys. */
        keys_layout m_keys_layout;
        /** The layout for accessing dictionary values. */
        values_layout p_values_layout;

        friend class detail::array_access;
        friend class dictionary_reference<IT>;
    };

    /**
     * Equality comparison operator for dictionary_encoded_array.
     *
     * @tparam IT The integral type used for dictionary keys.
     * @param lhs The first dictionary_encoded_array to compare.
     * @param rhs The second dictionary_encoded_array to compare.
     * @return true if the arrays are equal, false otherwise.
     */
    template <class IT>
    constexpr bool operator==(const dictionary_encoded_array<IT>& lhs, const dictionary_encoded_array<IT>& rhs);

    /*******************************************
     * dictionary_encoded_array implementation *
     *******************************************/

    template <std::integral IT>
    dictionary_encoded_array<IT>::dictionary_encoded_array(arrow_proxy proxy)
        : m_proxy(std::move(proxy))
        , m_keys_layout(create_keys_layout(m_proxy))
        , p_values_layout(create_values_layout(m_proxy))
    {
        SPARROW_ASSERT_TRUE(data_type_is_integer(m_proxy.data_type()));
    }

    template <std::integral IT>
    constexpr dictionary_encoded_array<IT>::dictionary_encoded_array(const self_type& rhs)
        : m_proxy(rhs.m_proxy)
        , m_keys_layout(create_keys_layout(m_proxy))
        , p_values_layout(create_values_layout(m_proxy))
    {
        copy_tracker::increase(copy_tracker::key<self_type>());
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::operator=(const self_type& rhs) -> self_type&
    {
        copy_tracker::increase(copy_tracker::key<self_type>());
        if (this != &rhs)
        {
            m_proxy = rhs.m_proxy;
            m_keys_layout = create_keys_layout(m_proxy);
            p_values_layout = create_values_layout(m_proxy);
        }
        return *this;
    }

    template <std::integral IT>
    constexpr dictionary_encoded_array<IT>::dictionary_encoded_array(self_type&& rhs)
        : m_proxy(std::move(rhs.m_proxy))
        , m_keys_layout(create_keys_layout(m_proxy))
        , p_values_layout(create_values_layout(m_proxy))
    {
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::operator=(self_type&& rhs) -> self_type&
    {
        if (this != &rhs)
        {
            using std::swap;
            swap(m_proxy, rhs.m_proxy);
            m_keys_layout = create_keys_layout(m_proxy);
            p_values_layout = create_values_layout(m_proxy);
        }
        return *this;
    }

    template <std::integral IT>
    template <validity_bitmap_input VBI, input_metadata_container METADATA_RANGE>
    auto dictionary_encoded_array<IT>::create_proxy(
        keys_buffer_type&& keys,
        array&& values,
        VBI&& validity_input,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    ) -> arrow_proxy
    {
        const auto size = keys.size();
        validity_bitmap vbitmap = ensure_validity_bitmap(size, std::forward<VBI>(validity_input));
        return create_proxy_impl(
            std::forward<keys_buffer_type>(keys),
            std::forward<array>(values),
            std::make_optional<validity_bitmap>(std::move(vbitmap)),
            std::move(name),
            std::move(metadata)
        );
    }

    template <std::integral IT>
    template <validity_bitmap_input VBI, input_metadata_container METADATA_RANGE>
    auto dictionary_encoded_array<IT>::create_proxy(
        keys_buffer_type&& keys,
        array&& values,
        bool nullable,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    ) -> arrow_proxy
    {
        const auto size = keys.size();
        return create_proxy_impl(
            std::forward<keys_buffer_type>(keys),
            std::forward<array>(values),
            nullable ? std::make_optional<validity_bitmap>(nullptr, size, validity_bitmap::default_allocator())
                     : std::nullopt,
            std::move(name),
            std::move(metadata)
        );
    }

    template <std::integral IT>
    template <input_metadata_container METADATA_RANGE>
    [[nodiscard]] arrow_proxy dictionary_encoded_array<IT>::create_proxy_impl(
        keys_buffer_type&& keys,
        array&& values,
        std::optional<validity_bitmap> validity,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    )
    {
        const auto size = keys.size();
        if (validity.has_value() && validity->data() == nullptr)
        {
            validity = validity_bitmap(size, true, validity_bitmap::default_allocator());
        }
        auto [value_array, value_schema] = extract_arrow_structures(std::move(values));
        static const repeat_view<bool> children_ownership{true, 0};

        const std::optional<std::unordered_set<sparrow::ArrowFlag>>
            flags = validity.has_value()
                        ? std::make_optional<std::unordered_set<sparrow::ArrowFlag>>({ArrowFlag::NULLABLE})
                        : std::nullopt;

        // create arrow schema and array
        ArrowSchema schema = make_arrow_schema(
            data_type_to_format(detail::get_data_type_from_array<self_type>::get()),  // format
            std::move(name),                                                          // name
            std::move(metadata),                                                      // metadata
            flags,                                                                    // flags
            nullptr,                                                                  // children
            children_ownership,                                                       // children_ownership
            new ArrowSchema(std::move(value_schema)),                                 // dictionary
            true                                                                      // dictionary ownership
        );

        const size_t null_count = validity.has_value() ? validity->null_count() : 0;

        std::vector<buffer<uint8_t>> buffers;
        buffers.reserve(2);
        buffers.emplace_back(
            validity.has_value() ? std::move(*validity).extract_storage()
                                 : buffer<uint8_t>{nullptr, 0, buffer<uint8_t>::default_allocator()}
        );
        buffers.emplace_back(std::move(keys).extract_storage());
        // create arrow array
        ArrowArray arr = make_arrow_array(
            static_cast<std::int64_t>(size),        // length
            static_cast<std::int64_t>(null_count),  // Null count
            0,                                      // offset
            std::move(buffers),
            nullptr,                                 // children
            children_ownership,                      // children_ownership
            new ArrowArray(std::move(value_array)),  // dictionary
            true                                     // dictionary ownership
        );
        return arrow_proxy(std::move(arr), std::move(schema));
    }

    template <std::integral IT>
    template <std::ranges::input_range NULLABLE_KEY_RANGE, input_metadata_container METADATA_RANGE>
        requires std::is_same_v<std::ranges::range_value_t<NULLABLE_KEY_RANGE>, nullable<IT>>
    arrow_proxy dictionary_encoded_array<IT>::create_proxy(
        NULLABLE_KEY_RANGE&& nullable_keys,
        array&& values,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> metadata
    )
    {
        auto keys = nullable_keys | std::views::transform(nullable_get);
        auto is_non_null = nullable_keys
                           | std::views::transform(
                               [](const auto& v)
                               {
                                   return v.has_value();
                               }
                           );
        return create_proxy(
            std::move(keys),
            std::forward<array>(values),
            std::move(is_non_null),
            std::move(name),
            std::move(metadata)
        );
    }

    template <std::integral IT>
    constexpr std::optional<std::string_view> dictionary_encoded_array<IT>::name() const
    {
        return m_proxy.name();
    }

    template <std::integral IT>
    std::optional<key_value_view> dictionary_encoded_array<IT>::metadata() const
    {
        return m_proxy.metadata();
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::size() const -> size_type
    {
        return m_proxy.length();
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::empty() const -> bool
    {
        return size() == 0;
    }

    template <std::integral IT>
    SPARROW_CONSTEXPR_CLANG auto dictionary_encoded_array<IT>::operator[](size_type i) const -> const_reference
    {
        return element_at(i);
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::element_at(size_type i) const -> const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        const auto index = m_keys_layout[i];

        if (index.has_value())
        {
            SPARROW_ASSERT_TRUE(index.value() >= 0);
            return array_element(*p_values_layout, static_cast<std::size_t>(index.value()));
        }
        else
        {
            return dummy_const_reference();
        }
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::end() const -> const_iterator
    {
        return cend();
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::cbegin() const -> const_iterator
    {
        return const_iterator(functor_type(this), 0u);
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::cend() const -> const_iterator
    {
        return const_iterator(functor_type(this), size());
    }

    template <std::integral IT>
    [[nodiscard]] constexpr auto dictionary_encoded_array<IT>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <std::integral IT>
    [[nodiscard]] constexpr auto dictionary_encoded_array<IT>::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    template <std::integral IT>
    [[nodiscard]] constexpr auto dictionary_encoded_array<IT>::crbegin() const -> const_reverse_iterator
    {
        return rbegin();
    }

    template <std::integral IT>
    [[nodiscard]] constexpr auto dictionary_encoded_array<IT>::crend() const -> const_reverse_iterator
    {
        return rend();
    }

    template <std::integral IT>
    SPARROW_CONSTEXPR_CLANG auto dictionary_encoded_array<IT>::front() const -> const_reference
    {
        SPARROW_ASSERT_FALSE(empty());
        return operator[](0);
    }

    template <std::integral IT>
    SPARROW_CONSTEXPR_CLANG auto dictionary_encoded_array<IT>::back() const -> const_reference
    {
        SPARROW_ASSERT_FALSE(empty());
        return operator[](size() - 1);
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::dummy_inner_value() const -> const inner_value_type&
    {
        static const inner_value_type instance = array_default_element_value(*p_values_layout);
        return instance;
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::slice(size_type start, size_type end) const -> self_type
    {
        SPARROW_ASSERT_TRUE(start <= end);
        return self_type{get_arrow_proxy().slice(start, end)};
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::slice_view(size_type start, size_type end) const -> self_type
    {
        SPARROW_ASSERT_TRUE(start <= end);
        return self_type{get_arrow_proxy().slice_view(start, end)};
    }

    template <std::integral IT>
    constexpr void dictionary_encoded_array<IT>::slice_inplace(size_type start, size_type end)
    {
        SPARROW_ASSERT_TRUE(start <= end);
        get_arrow_proxy().slice_inplace(start, end);
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::offset() const -> size_type
    {
        return static_cast<size_type>(get_arrow_proxy().offset());
    }

    /*template <std::integral IT>
    auto dictionary_encoded_array<IT>::dummy_inner_const_reference() const -> inner_const_reference
    {
        static const inner_const_reference instance =
            std::visit([](const auto& val) -> inner_const_reference { return val; }, dummy_inner_value());
        return instance;
    }*/

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::dummy_const_reference() const -> const_reference
    {
        static const const_reference instance = std::visit(
            [](const auto& val) -> const_reference
            {
                using inner_ref = typename arrow_traits<std::decay_t<decltype(val)>>::const_reference;
                return const_reference{nullable<inner_ref>(inner_ref(val), false)};
            },
            dummy_inner_value()
        );
        return instance;
    }

    template <std::integral IT>
    typename dictionary_encoded_array<IT>::values_layout
    dictionary_encoded_array<IT>::create_values_layout(arrow_proxy& proxy)
    {
        const auto& dictionary = proxy.dictionary();
        SPARROW_ASSERT_TRUE(dictionary);
        arrow_proxy ar_dictionary{&(dictionary->array()), &(dictionary->schema())};
        return array_factory(std::move(ar_dictionary));
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::create_keys_layout(arrow_proxy& proxy) -> keys_layout
    {
        return keys_layout{arrow_proxy{&proxy.array(), &proxy.schema()}};
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::get_arrow_proxy() -> arrow_proxy&
    {
        return m_proxy;
    }

    template <std::integral IT>
    constexpr auto dictionary_encoded_array<IT>::get_arrow_proxy() const -> const arrow_proxy&
    {
        return m_proxy;
    }

    template <class IT>
    constexpr bool operator==(const dictionary_encoded_array<IT>& lhs, const dictionary_encoded_array<IT>& rhs)
    {
        return std::ranges::equal(lhs, rhs);
    }

    /*********************************************
     * dictionary_encoded_array mutation implementation *
     *********************************************/

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::operator[](size_type i) -> reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        if (!is_nullable())
        {
            throw std::logic_error(
                "dictionary_encoded_array::operator[]: mutable access requires a nullable array"
            );
        }
        auto& bitmap = detail::array_access::get_arrow_proxy(m_keys_layout).bitmap();
        SPARROW_ASSERT_TRUE(bitmap.has_value());
        return reference(dictionary_reference<IT>(this, i), (*bitmap)[i]);
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::set_value(size_type index, const value_type& value)
    {
        SPARROW_ASSERT_TRUE(index < size());
        throw_if_sliced_for_mutation("dictionary_encoded_array::set_value");
        const bool valid = value.has_value();
        if (!valid && !is_nullable())
        {
            throw std::invalid_argument(
                "dictionary_encoded_array::set_value: cannot store null values in a non-nullable array"
            );
        }
        const IT key = valid ? find_or_append(value) : IT{};
        set_key(index, key, valid);
        compact_dictionary();
        refresh_proxy_buffers();
    }

    template <std::integral IT>
    template <class T, mpl::boolean_like B>
        requires dictionary_assignable_nullable<T, B>
    void dictionary_encoded_array<IT>::set_value(size_type index, const nullable<T, B>& value)
    {
        set_value(index, to_value_type(value));
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::set_value(size_type index, nullval_t)
    {
        set_value(index, value_type(nullable<null_type>{}));
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::insert(const_iterator pos, const value_type& value, size_type count)
        -> iterator
    {
        const auto index = index_of(pos);
        if (count == 0)
        {
            return make_iterator(index);
        }
        throw_if_sliced_for_mutation("dictionary_encoded_array::insert");
        return insert_keys_at(index, std::vector<nullable<IT>>(count, encode_value(value)));
    }

    template <std::integral IT>
    template <class T, mpl::boolean_like B>
        requires dictionary_assignable_nullable<T, B>
    auto dictionary_encoded_array<IT>::insert(const_iterator pos, const nullable<T, B>& value, size_type count)
        -> iterator
    {
        return insert(pos, to_value_type(value), count);
    }

    template <std::integral IT>
    template <std::input_iterator InputIt>
        requires std::convertible_to<std::iter_value_t<InputIt>, array_traits::value_type>
    auto dictionary_encoded_array<IT>::insert(const_iterator pos, InputIt first, InputIt last) -> iterator
    {
        const std::vector<value_type> values(first, last);
        return insert_values_at(pos, values);
    }

    template <std::integral IT>
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, array_traits::value_type>
    auto dictionary_encoded_array<IT>::insert(const_iterator pos, R&& values) -> iterator
    {
        return insert(pos, std::ranges::begin(values), std::ranges::end(values));
    }

    template <std::integral IT>
    template <class T>
        requires dictionary_assignable_nullable<T, bool>
    auto dictionary_encoded_array<IT>::insert(const_iterator pos, std::initializer_list<nullable<T>> values)
        -> iterator
    {
        std::vector<value_type> materialized;
        materialized.reserve(values.size());
        for (const auto& value : values)
        {
            materialized.push_back(to_value_type(value));
        }
        return insert_values_at(pos, materialized);
    }

    template <std::integral IT>
    auto
    dictionary_encoded_array<IT>::insert_values_at(const_iterator pos, const std::vector<value_type>& values)
        -> iterator
    {
        const auto index = index_of(pos);
        if (values.empty())
        {
            return make_iterator(index);
        }
        throw_if_sliced_for_mutation("dictionary_encoded_array::insert");
        std::vector<nullable<IT>> keys;
        keys.reserve(values.size());
        for (const auto& value : values)
        {
            keys.push_back(encode_value(value));
        }
        return insert_keys_at(index, std::move(keys));
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::insert_keys_at(size_type index, std::vector<nullable<IT>> keys)
        -> iterator
    {
        if (keys.empty())
        {
            return make_iterator(index);
        }
        m_keys_layout.insert(
            std::next(m_keys_layout.cbegin(), static_cast<std::ptrdiff_t>(index)),
            keys.begin(),
            keys.end()
        );
        refresh_proxy_buffers();
        return make_iterator(index);
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::erase(const_iterator pos) -> iterator
    {
        SPARROW_ASSERT_TRUE(cbegin() <= pos);
        SPARROW_ASSERT_TRUE(pos < cend());
        return erase(pos, std::next(pos));
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::erase(const_iterator first, const_iterator last) -> iterator
    {
        SPARROW_ASSERT_TRUE(cbegin() <= first);
        SPARROW_ASSERT_TRUE(last <= cend());
        SPARROW_ASSERT_TRUE(first <= last);
        const auto index = index_of(first);
        const auto count = static_cast<size_type>(std::distance(first, last));
        if (count == 0)
        {
            return make_iterator(index);
        }
        throw_if_sliced_for_mutation("dictionary_encoded_array::erase");
        m_keys_layout.erase(
            std::next(m_keys_layout.cbegin(), static_cast<std::ptrdiff_t>(index)),
            std::next(m_keys_layout.cbegin(), static_cast<std::ptrdiff_t>(index + count))
        );
        compact_dictionary();
        refresh_proxy_buffers();
        return make_iterator(index);
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::push_back(const value_type& value)
    {
        insert(cend(), value);
    }

    template <std::integral IT>
    template <class T, mpl::boolean_like B>
        requires dictionary_assignable_nullable<T, B>
    void dictionary_encoded_array<IT>::push_back(const nullable<T, B>& value)
    {
        insert(cend(), to_value_type(value));
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::pop_back()
    {
        SPARROW_ASSERT_FALSE(empty());
        erase(std::prev(cend()));
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::resize(size_type new_length)
    {
        resize(new_length, value_type(nullable<null_type>{}));
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::resize(size_type new_length, const value_type& value)
    {
        const size_type current_size = size();
        if (new_length < current_size)
        {
            erase(std::next(cbegin(), static_cast<std::ptrdiff_t>(new_length)), cend());
        }
        else if (new_length > current_size)
        {
            insert(cend(), value, new_length - current_size);
        }
    }

    template <std::integral IT>
    template <class T, mpl::boolean_like B>
        requires dictionary_assignable_nullable<T, B>
    void dictionary_encoded_array<IT>::resize(size_type new_length, const nullable<T, B>& value)
    {
        resize(new_length, to_value_type(value));
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::make_iterator(size_type index) const -> iterator
    {
        return iterator(functor_type(this), index);
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::throw_if_sliced_for_mutation(const char* operation) const
    {
        if (offset() != 0)
        {
            throw std::logic_error(std::string(operation) + " does not support sliced arrays (non-zero offset)");
        }
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::index_of(const_iterator pos) const -> size_type
    {
        SPARROW_ASSERT_TRUE(cbegin() <= pos);
        SPARROW_ASSERT_TRUE(pos <= cend());
        return static_cast<size_type>(std::distance(cbegin(), pos));
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::to_key(size_type index) const -> IT
    {
        if (not std::in_range<IT>(index))
        {
            throw std::overflow_error(
                "dictionary_encoded_array: dictionary index " + std::to_string(index) + " does not fit the key type"
            );
        }
        return static_cast<IT>(index);
    }

    template <std::integral IT>
    bool dictionary_encoded_array<IT>::is_nullable() const
    {
        return (get_arrow_proxy().schema().flags & static_cast<std::int64_t>(ArrowFlag::NULLABLE)) != 0;
    }

    template <std::integral IT>
    template <class T, mpl::boolean_like B>
    auto dictionary_encoded_array<IT>::to_value_type(const nullable<T, B>& value) -> value_type
    {
        if constexpr (std::constructible_from<value_type, nullable<T, B>>)
        {
            return value_type(value);
        }
        else
        {
            return value_type(const_reference(value));
        }
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::encode_value(const value_type& value) -> nullable<IT>
    {
        if (not value.has_value())
        {
            if (!is_nullable())
            {
                throw std::invalid_argument(
                    "dictionary_encoded_array: cannot store null values in a non-nullable array"
                );
            }
            return nullable<IT>(IT{}, false);
        }
        return nullable<IT>(find_or_append(value), true);
    }

    template <std::integral IT>
    auto dictionary_encoded_array<IT>::find_or_append(const value_type& value) -> IT
    {
        SPARROW_ASSERT_TRUE(value.has_value());
        const size_type dictionary_size = array_size(*p_values_layout);
        for (size_type i = 0; i < dictionary_size; ++i)
        {
            if (detail::dictionary_value_equal(array_element(*p_values_layout, i), value))
            {
                return to_key(i);
            }
        }
        const IT key = to_key(dictionary_size);
        {
            array values = make_array_view(*p_values_layout);
            insert_element(values, dictionary_size, value);
        }
        p_values_layout = create_values_layout(m_proxy);
        return key;
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::set_key(size_type index, IT key, bool valid)
    {
        auto entry = m_keys_layout[index];
        entry.get() = key;
        if (is_nullable())
        {
            entry.null_flag() = valid;
        }
        else
        {
            SPARROW_ASSERT_TRUE(valid);
        }
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::compact_dictionary()
    {
        const size_type dictionary_size = array_size(*p_values_layout);
        const size_type item_count = size();
        if (dictionary_size == 0)
        {
            return;
        }
        auto& keys_proxy = detail::array_access::get_arrow_proxy(m_keys_layout);
        IT* keys = const_cast<IT*>(static_cast<const IT*>(keys_proxy.array().buffers[1]));
        SPARROW_ASSERT_TRUE(keys != nullptr);
        const auto& validity = keys_proxy.bitmap();
        const std::uint8_t* valid_bits = validity.has_value() ? validity->data() : nullptr;

        // A non-zero entry marks a still referenced key. The marks are then replaced by the new
        // key, shifted by one so that zero keeps meaning "no longer referenced".
        std::vector<IT> remapping(dictionary_size, IT{});
        for (size_type i = 0; i < item_count; ++i)
        {
            if (valid_bits != nullptr && !is_bit_set(valid_bits, i))
            {
                continue;
            }
            const auto key = static_cast<size_type>(keys[i]);
            SPARROW_ASSERT_TRUE(key < dictionary_size);
            remapping[key] = IT{1};
        }
        size_type compacted_size = 0;
        size_type first_kept = dictionary_size;
        size_type last_kept = 0;
        for (size_type key = 0; key < dictionary_size; ++key)
        {
            if (remapping[key] != IT{})
            {
                if (first_kept == dictionary_size)
                {
                    first_kept = key;
                }
                last_kept = key;
                remapping[key] = static_cast<IT>(++compacted_size);
            }
        }
        if (compacted_size == dictionary_size)
        {
            return;
        }

        std::pair<ArrowArray, ArrowSchema> compacted_structures;
        {
            // Both locals end before set_dictionary releases the old nested structures.
            array values = make_array_view(*p_values_layout);
            // Surviving keys that form a single run (pop_back, resize, erasing at an end, a single
            // orphaned entry) are a slice of the current dictionary: rebuilding them one by one
            // would reallocate the child buffers for every entry.
            const bool contiguous = compacted_size == 0 || last_kept - first_kept + 1 == compacted_size;
            array compacted = contiguous && compacted_size != 0
                                  ? values.slice(first_kept, last_kept + 1)
                                  : array_empty_like(values);
            if (!contiguous)
            {
                for (size_type key = 0; key < dictionary_size; ++key)
                {
                    if (remapping[key] != IT{})
                    {
                        insert_element(
                            compacted,
                            compacted.size(),
                            value_type(array_element(*p_values_layout, key))
                        );
                    }
                }
            }
            compacted_structures = extract_arrow_structures(std::move(compacted));
        }
        m_proxy.set_dictionary(std::move(compacted_structures.first), std::move(compacted_structures.second));
        p_values_layout = create_values_layout(m_proxy);

        for (size_type i = 0; i < item_count; ++i)
        {
            if (valid_bits != nullptr && !is_bit_set(valid_bits, i))
            {
                keys[i] = IT{};
            }
            else
            {
                keys[i] = static_cast<IT>(remapping[static_cast<size_type>(keys[i])] - IT{1});
            }
        }
    }

    template <std::integral IT>
    void dictionary_encoded_array<IT>::refresh_proxy_buffers()
    {
        m_proxy.set_length(m_proxy.length());
    }
}

#if defined(__cpp_lib_format)
template <std::integral IT>
struct std::formatter<sparrow::dictionary_encoded_array<IT>>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();  // Simple implementation
    }

    auto format(const sparrow::dictionary_encoded_array<IT>& ar, std::format_context& ctx) const
    {
        std::format_to(ctx.out(), "Dictionary [size={}] <", ar.size());
        std::for_each(
            ar.cbegin(),
            std::prev(ar.cend()),
            [&ctx](const auto& value)
            {
                std::format_to(ctx.out(), "{}, ", value);
            }
        );
        std::format_to(ctx.out(), "{}>", ar.back());
        return ctx.out();
    }
};

namespace sparrow
{
    template <std::integral IT>
    std::ostream& operator<<(std::ostream& os, const dictionary_encoded_array<IT>& value)
    {
        os << std::format("{}", value);
        return os;
    }
}

#endif
