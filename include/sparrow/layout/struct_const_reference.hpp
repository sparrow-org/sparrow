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

#include "sparrow/layout/layout_utils.hpp"
#include "sparrow/utils/functor_index_iterator.hpp"
#if __has_include(<format>)
#    include <format>
#endif

#include "sparrow/config/config.hpp"
#include "sparrow/layout/array_wrapper.hpp"
#include "sparrow/types/data_traits.hpp"
#include "sparrow/utils/memory.hpp"

namespace sparrow
{
    /**
     * @brief Non-owning reference to one element of a \ref struct_array.
     *
     * The reference holds the child arrays of the struct array and an index, so reading a field
     * reads the child array at that index. A reference therefore reflects later mutations of the
     * element it denotes. Use \ref struct_value to snapshot the record.
     *
     * References are invalidated by structural mutations of the array (`insert`, `erase`,
     * `resize`, ...).
     */
    class SPARROW_API struct_const_reference
    {
    public:

        using value_type = array_traits::value_type;
        using const_reference = array_traits::const_reference;
        using size_type = std::size_t;
        using child_ptr = cloning_ptr<array_wrapper>;
        using const_functor_type = detail::layout_bracket_functor<const struct_const_reference, const_reference>;
        using const_iterator = functor_index_iterator<const_functor_type>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        struct_const_reference() = default;
        struct_const_reference(const std::vector<child_ptr>& children, size_type index);

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

        [[nodiscard]] auto names() const
        {
            const auto result = (*p_children)
                                | std::views::transform(
                                    [](const auto& child)
                                    {
                                        return child->get_arrow_proxy().name();
                                    }
                                );
            return result;
        }

    private:

        const std::vector<child_ptr>* p_children = nullptr;
        size_type m_index = 0u;
    };

    SPARROW_API
    bool operator==(const struct_const_reference& lhs, const struct_const_reference& rhs);
}

#if defined(__cpp_lib_format)

template <>
struct std::formatter<sparrow::struct_const_reference>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();  // Simple implementation
    }

    SPARROW_API auto format(const sparrow::struct_const_reference& ar, std::format_context& ctx) const
        -> decltype(ctx.out());
};

namespace sparrow
{
    SPARROW_API std::ostream& operator<<(std::ostream& os, const struct_const_reference& value);
}

#endif
