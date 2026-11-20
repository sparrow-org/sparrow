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
#include <stdexcept>

#include "sparrow/layout/array_helper.hpp"
#include "sparrow/layout/struct_value.hpp"

namespace sparrow
{
    struct_value::struct_value(const struct_const_reference& reference)
    {
        const size_type field_count = reference.size();
        m_fields.reserve(field_count);
        for (size_type i = 0; i < field_count; ++i)
        {
            m_fields.push_back(reference[i]);
        }
        for (const auto& name : reference.names())
        {
            m_names.emplace_back(name);
        }
    }

    auto struct_value::size() const -> size_type
    {
        return m_fields.size();
    }

    bool struct_value::empty() const
    {
        return m_fields.empty();
    }

    auto struct_value::operator[](size_type i) const -> const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return m_fields[i];
    }

    auto struct_value::at(size_type i) const -> const_reference
    {
        return m_fields.at(i);
    }

    auto struct_value::at(std::string_view name) const -> const_reference
    {
        const auto it = std::ranges::find_if(
            m_names,
            [name](const auto& candidate)
            {
                return candidate.has_value() && *candidate == name;
            }
        );
        if (it == m_names.end())
        {
            throw std::out_of_range("Child not found");
        }
        return m_fields[static_cast<size_type>(it - m_names.begin())];
    }

    auto struct_value::front() const -> const_reference
    {
        return (*this)[0];
    }

    auto struct_value::back() const -> const_reference
    {
        return (*this)[size() - 1];
    }

    auto struct_value::names() const -> const std::vector<std::optional<std::string>>&
    {
        return m_names;
    }

    bool operator==(const struct_value& lhs, const struct_value& rhs)
    {
        return std::ranges::equal(lhs, rhs);
    }

    auto struct_value::begin() const -> const_iterator
    {
        return {const_functor_type(this), 0};
    }

    auto struct_value::cbegin() const -> const_iterator
    {
        return {const_functor_type(this), 0};
    }

    auto struct_value::end() const -> const_iterator
    {
        return {const_functor_type(this), size()};
    }

    auto struct_value::cend() const -> const_iterator
    {
        return {const_functor_type(this), size()};
    }

    auto struct_value::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    auto struct_value::crbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    auto struct_value::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    auto struct_value::crend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }
}

#if defined(__cpp_lib_format)

auto std::formatter<sparrow::struct_value>::format(const sparrow::struct_value& ar, std::format_context& ctx) const
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
    std::ostream& operator<<(std::ostream& os, const struct_value& value)
    {
        os << std::format("{}", value);
        return os;
    }
}

#endif
