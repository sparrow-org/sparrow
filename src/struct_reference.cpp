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

#include "sparrow/struct_reference.hpp"

#include <utility>

#include "sparrow/struct_array.hpp"

namespace sparrow
{
    struct_reference::struct_reference(array_type* array, size_type index)
        : p_array(array)
        , m_index(index)
        , m_value(static_cast<const array_type&>(*array).value(index))
    {
    }

    struct_reference& struct_reference::operator=(const struct_value& value)
    {
        p_array->set_value(m_index, value);
        m_value = static_cast<const array_type&>(*p_array).value(m_index);
        return *this;
    }

    struct_reference& struct_reference::operator=(const struct_reference& rhs)
    {
        if (p_array != rhs.p_array || m_index != rhs.m_index)
        {
            operator=(static_cast<struct_value>(rhs));
        }
        return *this;
    }

    struct_reference& struct_reference::operator=(struct_reference&& rhs)
    {
        return operator=(static_cast<const struct_reference&>(rhs));
    }

    struct_reference::operator struct_value() const
    {
        return m_value;
    }

    auto struct_reference::size() const -> size_type
    {
        return m_value.size();
    }

    bool struct_reference::empty() const
    {
        return m_value.empty();
    }

    auto struct_reference::operator[](size_type i) const -> reference
    {
        return m_value[i];
    }

    auto struct_reference::at(size_type i) const -> reference
    {
        return m_value.at(i);
    }

    auto struct_reference::at(std::string_view name) const -> reference
    {
        return m_value.at(name);
    }

    auto struct_reference::front() const -> reference
    {
        return m_value.front();
    }

    auto struct_reference::back() const -> reference
    {
        return m_value.back();
    }

    auto struct_reference::begin() const -> const_iterator
    {
        return m_value.begin();
    }

    auto struct_reference::end() const -> const_iterator
    {
        return m_value.end();
    }

    auto struct_reference::cbegin() const -> const_iterator
    {
        return m_value.cbegin();
    }

    auto struct_reference::cend() const -> const_iterator
    {
        return m_value.cend();
    }

    auto struct_reference::rbegin() const -> const_reverse_iterator
    {
        return m_value.rbegin();
    }

    auto struct_reference::rend() const -> const_reverse_iterator
    {
        return m_value.rend();
    }

    auto struct_reference::crbegin() const -> const_reverse_iterator
    {
        return m_value.crbegin();
    }

    auto struct_reference::crend() const -> const_reverse_iterator
    {
        return m_value.crend();
    }

    bool struct_reference::operator==(const struct_value& rhs) const
    {
        return m_value == rhs;
    }

    bool struct_reference::operator==(const struct_reference& rhs) const
    {
        return m_value == rhs.m_value;
    }
}
