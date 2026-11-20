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

#include <cstddef>
#include <string_view>
#include <version>

#include "sparrow/config/config.hpp"
#include "sparrow/layout/struct_const_reference.hpp"
#include "sparrow/layout/struct_value.hpp"

namespace sparrow
{
    class struct_array;

    /**
     * @brief Mutable reference to one element of a \ref struct_array.
     *
     * Reads expose the referenced record through a \ref struct_const_reference, so they reflect
     * the current state of the array. Assigning a \ref struct_value replaces the fields of the
     * element. The number of fields of the assigned value must match the number of children of
     * the array.
     *
     * References are invalidated by structural mutations of the array (`insert`, `erase`,
     * `resize`, ...).
     */
    class SPARROW_API struct_reference
    {
    public:

        using array_type = struct_array;
        using value_type = struct_value;
        using size_type = std::size_t;
        using reference = struct_const_reference::const_reference;
        using const_iterator = struct_const_reference::const_iterator;
        using const_reverse_iterator = struct_const_reference::const_reverse_iterator;

        /**
         * @brief Constructs a reference to the element at the given index.
         *
         * @param array The referenced array.
         * @param index The index of the referenced element.
         *
         * @pre \p array must not be null and \p index must be < array->size().
         */
        struct_reference(array_type* array, size_type index);

        /**
         * @brief Copy constructor.
         */
        struct_reference(const struct_reference&) = default;

        /**
         * @brief Move constructor.
         */
        struct_reference(struct_reference&&) noexcept = default;

        /**
         * @brief Destructor.
         */
        ~struct_reference() = default;

        /**
         * @brief Replaces the fields of the referenced element with the fields of \p value.
         *
         * @param value The record to assign.
         * @return Reference to this struct_reference.
         *
         * @throws std::logic_error if the array has a non-zero Arrow offset.
         * @throws std::invalid_argument if \p value has an incompatible number of fields.
         */
        struct_reference& operator=(const struct_value& value);

        /**
         * @brief Replaces the referenced element with the element referenced by \p rhs.
         *
         * @param rhs The reference to copy.
         * @return Reference to this struct_reference.
         */
        struct_reference& operator=(const struct_reference& rhs);

        /**
         * @brief Replaces the referenced element with the element referenced by \p rhs.
         *
         * @param rhs The reference to copy.
         * @return Reference to this struct_reference.
         */
        struct_reference& operator=(struct_reference&& rhs);

        /**
         * @brief Materializes the referenced record.
         *
         * @return An owning copy of the referenced record.
         */
        operator struct_value() const;

        /**
         * @brief Returns the number of fields of the referenced record.
         *
         * @return The number of fields.
         */
        [[nodiscard]] size_type size() const;

        /**
         * @brief Checks whether the referenced record has no field.
         *
         * @return true if the record has no field, false otherwise.
         */
        [[nodiscard]] bool empty() const;

        /**
         * @brief Gets the field at the given position.
         *
         * @param i The position of the field.
         * @return The field at position \p i.
         */
        [[nodiscard]] reference operator[](size_type i) const;

        /**
         * @brief Gets the field at the given position, checking the bounds.
         *
         * @param i The position of the field.
         * @return The field at position \p i.
         *
         * @throws std::out_of_range if \p i is not less than size().
         */
        [[nodiscard]] reference at(size_type i) const;

        /**
         * @brief Gets the field with the given name.
         *
         * @param name The name of the field.
         * @return The field named \p name.
         *
         * @throws std::out_of_range if there is no field named \p name.
         */
        [[nodiscard]] reference at(std::string_view name) const;

        /**
         * @brief Gets the first field.
         *
         * @return The first field.
         */
        [[nodiscard]] reference front() const;

        /**
         * @brief Gets the last field.
         *
         * @return The last field.
         */
        [[nodiscard]] reference back() const;

        /**
         * @brief Gets an iterator to the first field.
         *
         * @return Constant iterator to the first field.
         */
        [[nodiscard]] const_iterator begin() const;

        /**
         * @brief Gets an iterator past the last field.
         *
         * @return Constant iterator past the last field.
         */
        [[nodiscard]] const_iterator end() const;

        /**
         * @brief Gets an iterator to the first field.
         *
         * @return Constant iterator to the first field.
         */
        [[nodiscard]] const_iterator cbegin() const;

        /**
         * @brief Gets an iterator past the last field.
         *
         * @return Constant iterator past the last field.
         */
        [[nodiscard]] const_iterator cend() const;

        /**
         * @brief Gets a reverse iterator to the last field.
         *
         * @return Constant reverse iterator to the last field.
         */
        [[nodiscard]] const_reverse_iterator rbegin() const;

        /**
         * @brief Gets a reverse iterator before the first field.
         *
         * @return Constant reverse iterator before the first field.
         */
        [[nodiscard]] const_reverse_iterator rend() const;

        /**
         * @brief Gets a reverse iterator to the last field.
         *
         * @return Constant reverse iterator to the last field.
         */
        [[nodiscard]] const_reverse_iterator crbegin() const;

        /**
         * @brief Gets a reverse iterator before the first field.
         *
         * @return Constant reverse iterator before the first field.
         */
        [[nodiscard]] const_reverse_iterator crend() const;

        /**
         * @brief Gets the names of the fields.
         *
         * @return The names of the fields.
         */
        [[nodiscard]] auto names() const
        {
            return m_value.names();
        }

        /**
         * @brief Compares the referenced record with \p rhs.
         *
         * @param rhs The record to compare with.
         * @return true if the fields are equal, false otherwise.
         */
        [[nodiscard]] bool operator==(const struct_value& rhs) const;

        /**
         * @brief Compares the referenced record with the one referenced by \p rhs.
         *
         * @param rhs The reference to compare with.
         * @return true if the fields are equal, false otherwise.
         */
        [[nodiscard]] bool operator==(const struct_reference& rhs) const;

    private:

        /** The referenced array. */
        array_type* p_array = nullptr;
        /** The index of the referenced element. */
        size_type m_index = 0;
        /** The view of the referenced record, refreshed on assignment. */
        struct_const_reference m_value;
    };
}

#if defined(__cpp_lib_format)

template <>
struct std::formatter<sparrow::struct_reference>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const sparrow::struct_reference& reference, std::format_context& ctx) const
        -> decltype(ctx.out())
    {
        return std::format_to(ctx.out(), "{}", static_cast<sparrow::struct_value>(reference));
    }
};

#endif
