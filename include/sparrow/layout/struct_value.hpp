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

#include <optional>
#include <string>
#include <string_view>
#include <vector>
#if __has_include(<format>)
#    include <format>
#endif

#include "sparrow/config/config.hpp"
#include "sparrow/layout/layout_utils.hpp"
#include "sparrow/layout/list_value.hpp"
#include "sparrow/layout/map_value.hpp"
#include "sparrow/layout/struct_const_reference.hpp"
#include "sparrow/types/data_traits.hpp"
#include "sparrow/utils/functor_index_iterator.hpp"

namespace sparrow
{
    /**
     * @brief Owning snapshot of one element of a \ref struct_array.
     *
     * Unlike \ref struct_const_reference, which reads through the children of the array it was
     * created from, a \c struct_value owns its field values and its field names. Mutating the
     * array afterwards therefore leaves the value unchanged.
     *
     * Copying or moving a \c struct_value copies the fields; it never aliases the array.
     */
// MSVC warns C4251 when a dllexport class holds std::vector members, whose
// instantiation has no dll-interface. This is safe for an owning value type: the
// members are only touched by the out-of-line methods of the library.
#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4251)
#endif
    class SPARROW_API struct_value
    {
    public:

        using value_type = array_traits::value_type;
        /** Field access returns owned values: an owning record holds no reference to an array. */
        using const_reference = value_type;
        using size_type = std::size_t;
        using const_functor_type = detail::layout_bracket_functor<const struct_value, const_reference>;
        using const_iterator = functor_index_iterator<const_functor_type>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        /** Constructs the empty record (no field). */
        struct_value() = default;

        /**
         * @brief Copies the record referenced by \p reference.
         *
         * The field values and names are copied, so the result does not alias the array
         * \p reference was created from.
         *
         * @param reference The referenced record.
         */
        struct_value(const struct_const_reference& reference);

        [[nodiscard]] size_type size() const;
        [[nodiscard]] bool empty() const;

        [[nodiscard]] const_reference operator[](size_type i) const;
        [[nodiscard]] const_reference at(size_type i) const;
        [[nodiscard]] const_reference at(std::string_view name) const;

        [[nodiscard]] const_reference front() const;
        [[nodiscard]] const_reference back() const;

        [[nodiscard]] const_iterator begin() const;
        [[nodiscard]] const_iterator cbegin() const;

        [[nodiscard]] const_iterator end() const;
        [[nodiscard]] const_iterator cend() const;

        [[nodiscard]] const_reverse_iterator rbegin() const;
        [[nodiscard]] const_reverse_iterator crbegin() const;

        [[nodiscard]] const_reverse_iterator rend() const;
        [[nodiscard]] const_reverse_iterator crend() const;

        /**
         * @brief Gets the names of the fields.
         *
         * @return The names of the fields, an empty optional for an unnamed field.
         */
        [[nodiscard]] const std::vector<std::optional<std::string>>& names() const;

    private:

        std::vector<value_type> m_fields;
        std::vector<std::optional<std::string>> m_names;
    };
#ifdef _MSC_VER
#    pragma warning(pop)
#endif

    SPARROW_API
    bool operator==(const struct_value& lhs, const struct_value& rhs);
}

#if defined(__cpp_lib_format)

template <>
struct std::formatter<sparrow::struct_value>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();  // Simple implementation
    }

    SPARROW_API auto format(const sparrow::struct_value& ar, std::format_context& ctx) const
        -> decltype(ctx.out());
};

namespace sparrow
{
    SPARROW_API std::ostream& operator<<(std::ostream& os, const struct_value& value);
}

#endif
