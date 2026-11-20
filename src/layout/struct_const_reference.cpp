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

#include <algorithm>

#include "sparrow/layout/array_helper.hpp"
#include "sparrow/layout/nested_value_types.hpp"
#include "sparrow/layout/struct_const_reference.hpp"

namespace sparrow
{
    struct_const_reference::struct_const_reference(const std::vector<child_ptr>& children, size_type index)
        : p_children(&children)
        , m_index(index)
    {
    }

    auto struct_const_reference::size() const -> size_type
    {
        return p_children == nullptr ? 0 : p_children->size();
    }

    bool struct_const_reference::empty() const
    {
        return size() == 0;
    }

    auto struct_const_reference::operator[](size_type i) const -> const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return array_element(*(((*p_children)[i]).get()), m_index);
    }

    auto struct_const_reference::at(size_type i) const -> const_reference
    {
        return array_element(*((p_children->at(i)).get()), m_index);
    }

    auto struct_const_reference::at(std::string_view name) const -> const_reference
    {
        auto it = std::ranges::find_if(
            *p_children,
            [&name](const auto& child)
            {
                return child->get_arrow_proxy().name() == name;
            }
        );
        if (it != p_children->end())
        {
            return array_element(*((*it).get()), m_index);
        }
        throw std::out_of_range("Child not found");
    }

    auto struct_const_reference::front() const -> const_reference
    {
        return (*this)[0];
    }

    auto struct_const_reference::back() const -> const_reference
    {
        return (*this)[size() - 1];
    }

    bool operator==(const struct_const_reference& lhs, const struct_const_reference& rhs)
    {
        return std::ranges::equal(lhs, rhs);
    }

    auto struct_const_reference::begin() const -> const_iterator
    {
        return {const_functor_type(this), 0};
    }

    auto struct_const_reference::cbegin() const -> const_iterator
    {
        return {const_functor_type(this), 0};
    }

    auto struct_const_reference::end() const -> const_iterator
    {
        return {const_functor_type(this), size()};
    }

    auto struct_const_reference::cend() const -> const_iterator
    {
        return {const_functor_type(this), size()};
    }

    auto struct_const_reference::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    auto struct_const_reference::crbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    auto struct_const_reference::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    auto struct_const_reference::crend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }
}

#if defined(__cpp_lib_format)

auto std::formatter<sparrow::struct_const_reference>::format(const sparrow::struct_const_reference& ar, std::format_context& ctx) const
    -> decltype(ctx.out())
{
    std::format_to(ctx.out(), "<");
    if (!ar.empty())
    {
        for (std::size_t i = 0; i < ar.size() - 1; ++i)
        {
            std::format_to(ctx.out(), "{}, ", ar[i]);
        }
        std::format_to(ctx.out(), "{}", ar[ar.size() - 1]);
    }
    return std::format_to(ctx.out(), ">");
}

namespace sparrow
{
    std::ostream& operator<<(std::ostream& os, const struct_const_reference& value)
    {
        os << std::format("{}", value);
        return os;
    }
}

#endif
